"""
API routes for speech transcription service.
"""
from fastapi import APIRouter, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import PlainTextResponse
from typing import Optional, Union
from pydantic import BaseModel, Field
import logging
from src.config import MAX_UPLOAD_SIZE_MB

logger = logging.getLogger(__name__)

# 支持的音频 MIME 类型白名单
ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/x-m4a",
    "audio/flac",
    "audio/ogg",
    "audio/webm"
}

# API Request/Response Models
class Segment(BaseModel):
    """Segment with timestamp and optional speaker info"""
    id: int
    speaker: Optional[str] = None  # Speaker ID (e.g., "Speaker 0")
    start: float = 0.0  # 毫秒
    end: float = 0.0    # 毫秒
    text: str

class TranscriptionResponse(BaseModel):
    """
    JSON response format with full structured data.
    Used when output_format='json'.
    """
    text: str
    duration: Optional[float] = None
    language: Optional[str] = None
    model: Optional[str] = None
    segments: Optional[list[Segment]] = Field(None, description="详细分段信息（带说话人识别）")

# Router
router = APIRouter()

@router.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, etc.)"),
    model: str = Form("paraformer", description="Model ID (currently uses Paraformer for speaker diarization)"),
    language: str = Form("auto", description="Language code (auto, zh, en)"),
    output_format: str = Form("json", description="Output format: 'json' (default, OpenAI compatible), 'txt' (clean text only), 'srt' (subtitle)"),
    with_timestamp: bool = Form(False, description="Include timestamps in txt output (e.g., [02:15] [Speaker 0]: ...)"),
    ):
    """
    Transcribe audio file with speaker diarization.
    
    Output Formats:
    - **txt** (default): Clean text with speaker labels, suitable for RAG/LLM
    - **json**: Full structured data with segments and timestamps
    - **srt**: Standard SRT subtitle format
    
    Examples:
    - Basic: `curl -F "file=@audio.mp3" http://localhost:50070/v1/audio/transcriptions`
    - With timestamps: `curl -F "file=@audio.mp3" -F "with_timestamp=true" ...`
    - JSON format: `curl -F "file=@audio.mp3" -F "output_format=json" ...`
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    # 1. 文件类型校验
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        logger.warning(f"[{request_id}] Unsupported file type: {file.content_type}")
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Only audio files are allowed."
        )
    
    # 2. 文件大小校验
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    max_size_mb = MAX_UPLOAD_SIZE_MB
    
    if file_size_mb > max_size_mb:
        logger.warning(f"[{request_id}] File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)")
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed ({max_size_mb} MB)"
        )
    
    await file.seek(0)
    
    logger.info(f"[{request_id}] Processing file: {file.filename} ({file_size_mb:.2f}MB, format={output_format})")
    
    # 3. 获取 Service
    service = request.app.state.service

    try:
        # 4. 构造参数
        params = {
            "language": language,
            "output_format": output_format,
            "with_timestamp": with_timestamp,
        }

        # 5. 提交任务
        result = await service.submit(file, params, request_id=request_id)
        
        # 6. 根据格式返回不同响应
        # SRT 格式返回纯文本（字幕文件格式）
        if output_format == "srt":
            return PlainTextResponse(
                content=result if isinstance(result, str) else result.get("text", ""),
                media_type="text/plain; charset=utf-8"
            )
        
        # JSON 和 TXT 格式都返回 JSON 响应（OpenAI API 兼容）
        # TXT 格式只返回 text 字段，不含 segments
        if isinstance(result, dict):
            text = result.get("text", "")
            segments_data = result.get("segments", [])
            duration = result.get("duration", 0.0)
            
            # 格式化 segments（仅 json 格式包含）
            segments = None
            if output_format == "json" and segments_data:
                segments = [
                    {
                        "id": i,
                        "speaker": seg.get("speaker"),
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", "")
                    }
                    for i, seg in enumerate(segments_data)
                ]
            
            return TranscriptionResponse(
                text=text,
                duration=duration,
                language=language if language != "auto" else "zh",
                model=request.app.state.model_id if hasattr(request.app.state, "model_id") else "paraformer",
                segments=segments
            )
        else:
            return TranscriptionResponse(
                text=result,
                language=language if language != "auto" else "zh",
                model="paraformer"
            )

    except RuntimeError as e:
        if "Queue is full" in str(e):
            raise HTTPException(status_code=503, detail="Server is busy (Queue Full). Please try again later.")
        logger.error(f"[{request_id}] Runtime error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error occurred. (Request ID: {request_id})"
        )
    
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error occurred. (Request ID: {request_id})"
        )

