"""
API routes for speech transcription service.
"""

import logging
import os
from dataclasses import asdict

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from src.config import MAX_UPLOAD_SIZE_MB

logger = logging.getLogger(__name__)

# 支持的音频 MIME 类型白名单
ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/flac",
    "audio/ogg",
    "audio/webm",
}

# 支持的文件扩展名（用于 fallback 判断）
ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".flac",
    ".ogg",
    ".webm",
}

# OpenAI response_format → internal output_format mapping
_RESPONSE_FORMAT_MAP = {
    "verbose_json": "json",
    "text": "txt",
    "vtt": "srt",
}


# API Request/Response Models
class Segment(BaseModel):
    """Segment with timestamp and optional speaker info"""

    id: int
    speaker: str | None = None  # Speaker ID (e.g., "Speaker 0")
    start: float = 0.0  # 毫秒
    end: float = 0.0  # 毫秒
    text: str


class TranscriptionResponse(BaseModel):
    """
    JSON response format with full structured data.
    Used when output_format='json'.
    """

    text: str
    duration: float | None = None
    language: str | None = None
    model: str | None = None
    segments: list[Segment] | None = Field(None, description="详细分段信息（带说话人识别）")


# Router
router = APIRouter()


@router.post("/v1/audio/transcriptions", response_model=None)
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, etc.)"),
    model: str = Form(
        "paraformer", description="Model ID (informational — actual model set by server config)"
    ),
    language: str = Form("auto", description="Language code (auto, zh, en)"),
    response_format: str | None = Form(
        None, description="OpenAI-compatible format: json, verbose_json, text, vtt, srt"
    ),
    output_format: str = Form("json", description="Output format: json (default), txt, srt"),
    with_timestamp: bool = Form(
        False, description="Include timestamps in txt output (e.g., [02:15] [Speaker 0]: ...)"
    ),
) -> TranscriptionResponse | PlainTextResponse:
    """
    Transcribe audio file with speaker diarization.

    Output Formats:
    - **json** (default): Full structured data with segments and timestamps (OpenAI verbose_json compatible)
    - **txt**: Clean text with speaker labels, suitable for RAG/LLM
    - **srt**: Standard SRT subtitle format

    OpenAI Compatibility:
    - `response_format=verbose_json` maps to `json` with segments
    - `response_format=text` maps to `txt`
    - `response_format=vtt` maps to `srt`
    - `model` parameter is accepted but informational (server uses configured model)
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # 1. 文件类型校验
    # 优先检查 MIME 类型，如果是 application/octet-stream 则 fallback 到扩展名判断
    is_valid_type = file.content_type in ALLOWED_AUDIO_TYPES

    if not is_valid_type and file.content_type == "application/octet-stream":
        # Fallback: 通过文件扩展名判断（curl 经常不设置正确的 Content-Type）
        file_ext = os.path.splitext(file.filename or "")[1].lower()
        is_valid_type = file_ext in ALLOWED_AUDIO_EXTENSIONS
        if is_valid_type:
            logger.info(
                f"[{request_id}] Accepted file by extension fallback: {file.filename} (ext={file_ext})"
            )

    if not is_valid_type:
        logger.warning(
            f"[{request_id}] Unsupported file: {file.filename} (type={file.content_type})"
        )
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Expected audio file, got: {file.content_type}"
        )

    # 2. 文件大小校验（通过底层文件对象的 seek/tell 避免读取全部内容到内存）
    file.file.seek(0, 2)  # seek to end
    file_size_bytes = file.file.tell()
    file_size_mb = file_size_bytes / (1024 * 1024)
    max_size_mb = MAX_UPLOAD_SIZE_MB

    if file_size_mb > max_size_mb:
        logger.warning(
            f"[{request_id}] File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)"
        )
        raise HTTPException(
            status_code=413, detail=f"File size exceeds maximum allowed ({max_size_mb} MB)"
        )

    file.file.seek(0)  # reset for downstream reading

    # 3. Resolve effective output format (response_format takes precedence)
    effective_format = response_format if response_format is not None else output_format
    effective_format = _RESPONSE_FORMAT_MAP.get(effective_format, effective_format)

    # 4. Capability validation — fail fast with clear 400 errors
    engine = request.app.state.engine
    caps = engine.capabilities

    if effective_format == "srt" and not caps.timestamp:
        raise HTTPException(
            status_code=400,
            detail=(
                f"SRT format requires timestamp support, but the current model "
                f"({request.app.state.model_id}) does not produce timestamps. "
                f"Use output_format=json or output_format=txt instead, "
                f"or switch to a Paraformer model."
            ),
        )

    if with_timestamp and not caps.timestamp:
        raise HTTPException(
            status_code=400,
            detail=(
                f"with_timestamp=true requires timestamp support, but the current model "
                f"({request.app.state.model_id}) does not produce timestamps. "
                f"Set with_timestamp=false, or switch to a Paraformer model."
            ),
        )

    logger.info(
        f"[{request_id}] Processing file: {file.filename} ({file_size_mb:.2f}MB, format={effective_format})"
    )

    # 5. 获取 Service
    service = request.app.state.service

    try:
        # 6. 构造参数
        params = {
            "language": language,
            "output_format": effective_format,
            "with_timestamp": with_timestamp,
        }

        # 7. 提交任务
        result = await service.submit(file, params, request_id=request_id)

        # 8. 根据格式返回不同响应
        # SRT 格式返回纯文本（字幕文件格式）
        if effective_format == "srt":
            return PlainTextResponse(
                content=result if isinstance(result, str) else result.get("text", ""),
                media_type="text/plain; charset=utf-8",
            )

        # JSON 和 TXT 格式都返回 JSON 响应（OpenAI API 兼容）
        # TXT 格式只返回 text 字段，不含 segments
        if isinstance(result, dict):
            text = result.get("text", "")
            segments_data = result.get("segments", [])
            duration = result.get("duration", 0.0)

            # 格式化 segments（仅 json 格式包含）
            segments: list[Segment] | None = None
            if effective_format == "json" and segments_data:
                segments = [
                    Segment(
                        id=i,
                        speaker=seg.get("speaker"),
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        text=seg.get("text", ""),
                    )
                    for i, seg in enumerate(segments_data)
                ]

            return TranscriptionResponse(
                text=text,
                duration=duration,
                language=language if language != "auto" else "zh",
                model=request.app.state.model_id
                if hasattr(request.app.state, "model_id")
                else "paraformer",
                segments=segments,
            )
        else:
            return TranscriptionResponse(
                text=str(result),
                language=language if language != "auto" else "zh",
                model="paraformer",
                segments=None,
            )

    except RuntimeError as e:
        if "Queue is full" in str(e):
            raise HTTPException(
                status_code=503, detail="Server is busy (Queue Full). Please try again later."
            ) from None
        logger.error(f"[{request_id}] Runtime error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal server error occurred. (Request ID: {request_id})"
        ) from None

    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal server error occurred. (Request ID: {request_id})"
        ) from None


@router.get("/v1/models/current")
async def get_current_model(request: Request) -> dict[str, object]:
    """
    Return the currently loaded model and its capabilities.
    Useful for clients to discover what formats/features are available.
    """
    engine = request.app.state.engine
    service = request.app.state.service

    return {
        "engine_type": request.app.state.engine_type,
        "model_id": request.app.state.model_id,
        "capabilities": asdict(engine.capabilities),
        "queue_size": service.queue.qsize(),
        "max_queue_size": service.queue.maxsize,
    }
