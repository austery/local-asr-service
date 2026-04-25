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
from src.core.model_registry import ModelSpec, is_passthrough, list_all, lookup
from src.core.pipeline_registry import PipelineProfile, list_all_profiles, lookup_profile

logger = logging.getLogger(__name__)

ResolvedTarget = ModelSpec | PipelineProfile

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
    speaker: str | None = None
    start: float = 0.0
    end: float = 0.0
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


class ModelInfo(BaseModel):
    """Single model entry for GET /v1/models"""

    alias: str
    model_id: str
    engine_type: str
    description: str
    capabilities: dict[str, bool]
    requestable: bool


class ModelsResponse(BaseModel):
    """Response for GET /v1/models"""

    models: list[ModelInfo]
    current: str | None


# Router
router = APIRouter()


def _raise_unknown_model(model: str) -> None:
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unknown model: '{model}'. Use GET /v1/models to see built-in models, "
            "or pass a full path prefixed with 'mlx-community/' or 'iic/'."
        ),
    )


def _ensure_requestable(target: ResolvedTarget) -> None:
    if target.requestable:
        return

    raise HTTPException(
        status_code=400,
        detail=(
            f"Model '{target.alias}' is registered for future pipeline composition and "
            "is not available for direct transcription requests."
        ),
    )


def _resolve_model(model: str | None) -> ResolvedTarget | None:
    """
    Resolve the `model` form field to a ModelSpec, or None if no switch is needed.

    Returns None for passthrough values (None / "whisper-1") — signals "use current engine".
    Raises HTTPException 400 for unrecognisable model strings.
    """
    if is_passthrough(model):
        return None

    if model is None:
        return None

    try:
        target: ResolvedTarget = lookup(model)
    except ValueError:
        try:
            target = lookup_profile(model)
        except KeyError:
            _raise_unknown_model(model)

    _ensure_requestable(target)
    return target


@router.post("/v1/audio/transcriptions", response_model=None)
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, etc.)"),
    model: str | None = Form(
        None,
        description=(
            "Model alias or full model path. "
            "Pass None/'whisper-1' to use the server's current model. "
            "Examples: 'paraformer', 'qwen3-asr-mini', 'mlx-community/Qwen3-ASR-1.7B-4bit'"
        ),
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
    Transcribe audio file. Optionally specify a model to use for this request.

    Model switching:
    - Pass `model=paraformer` for multi-speaker content (enables diarization).
    - Pass `model=qwen3-asr-mini` for single-speaker content (low memory, fast).
    - Omit `model` or pass `model=whisper-1` to use the server's currently loaded model.

    Output Formats:
    - **json** (default): Full structured data with segments and timestamps
    - **txt**: Clean text with speaker labels, suitable for RAG/LLM
    - **srt**: Standard SRT subtitle format
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # 1. 文件类型校验（先做，确保文件错误优先于模型错误）
    is_valid_type = file.content_type in ALLOWED_AUDIO_TYPES
    if not is_valid_type and file.content_type == "application/octet-stream":
        file_ext = os.path.splitext(file.filename or "")[1].lower()
        is_valid_type = file_ext in ALLOWED_AUDIO_EXTENSIONS
        if is_valid_type:
            logger.info(
                f"[{request_id}] Accepted file by extension fallback: "
                f"{file.filename} (ext={file_ext})"
            )

    if not is_valid_type:
        logger.warning(
            f"[{request_id}] Unsupported file: {file.filename} (type={file.content_type})"
        )
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Expected audio file, got: {file.content_type}",
        )

    # 2. 文件大小校验
    file.file.seek(0, 2)
    file_size_bytes = file.file.tell()
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        logger.warning(
            f"[{request_id}] File too large: {file_size_mb:.2f}MB (max: {MAX_UPLOAD_SIZE_MB}MB)"
        )
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed ({MAX_UPLOAD_SIZE_MB} MB)",
        )

    file.file.seek(0)

    # 3. Resolve model → ModelSpec (or None = keep current engine)
    resolved_target = _resolve_model(model)

    # 4. Resolve effective output format
    effective_format = response_format if response_format is not None else output_format
    effective_format = _RESPONSE_FORMAT_MAP.get(effective_format, effective_format)

    # 5. Capability pre-validation (fail fast before queuing)
    #    If the request specifies a model, validate against ITS declared capabilities
    #    so the client gets an early 400 without waiting for the switch.
    #    Fall back to the current engine only for passthrough requests (model=None).
    if resolved_target is not None:
        caps = resolved_target.capabilities
    else:
        caps = request.app.state.service.capabilities

    model_label = (
        resolved_target.alias
        if resolved_target is not None
        else str(getattr(request.app.state, "model_id", "unknown"))
    )
    service = request.app.state.service

    if effective_format == "srt" and not caps.timestamp:
        raise HTTPException(
            status_code=400,
            detail=(
                f"SRT format requires timestamp support, but '{model_label}' "
                f"does not produce timestamps. "
                f"Use output_format=json or output_format=txt instead."
            ),
        )

    if with_timestamp and not caps.timestamp:
        raise HTTPException(
            status_code=400,
            detail=(
                f"with_timestamp=true requires timestamp support, but '{model_label}' "
                f"does not produce timestamps."
            ),
        )

    logger.info(
        f"[{request_id}] Processing file: {file.filename} "
        f"({file_size_mb:.2f}MB, format={effective_format}, model={model_label})"
    )

    try:
        params = {
            "language": language,
            "output_format": effective_format,
            "with_timestamp": with_timestamp,
        }

        # Determine response model alias BEFORE awaiting:
        #   - Explicit switch: use resolved_target (always correct regardless of queue ordering).
        #   - Passthrough: capture current spec now; reading it after await is racy because
        #     another concurrent request may trigger a switch while this job is queued.
        response_target: object | None = (
            resolved_target if resolved_target is not None else service.current_model_spec
        )

        result = await service.submit(
            file,
            params,
            request_id=request_id,
            model_spec=resolved_target if isinstance(resolved_target, ModelSpec) else None,
            pipeline_profile=(
                resolved_target if isinstance(resolved_target, PipelineProfile) else None
            ),
        )

        response_model = (
            response_target.alias
            if isinstance(response_target, (ModelSpec, PipelineProfile))
            else str(getattr(request.app.state, "model_id", "unknown"))
        )

        if effective_format == "srt":
            return PlainTextResponse(
                content=result if isinstance(result, str) else result.get("text", ""),
                media_type="text/plain; charset=utf-8",
            )

        if isinstance(result, dict):
            text = result.get("text", "")
            segments_data = result.get("segments", [])
            duration = result.get("duration", 0.0)

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
                model=response_model,
                segments=segments,
            )
        else:
            return TranscriptionResponse(
                text=str(result),
                language=language if language != "auto" else "zh",
                model=response_model,
                segments=None,
            )

    except RuntimeError as e:
        if "Queue is full" in str(e):
            raise HTTPException(
                status_code=503, detail="Server is busy (Queue Full). Please try again later."
            ) from None
        logger.error(f"[{request_id}] Runtime error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error occurred. (Request ID: {request_id})",
        ) from None

    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error occurred. (Request ID: {request_id})",
        ) from None


@router.get("/v1/models")
async def list_models(request: Request) -> ModelsResponse:
    """
    List all supported models and the currently loaded model.
    Only entries with `requestable=true` can be passed to POST /v1/audio/transcriptions today.
    Discovery-only entries are listed so future decoupled pipeline targets are visible.
    """
    service = request.app.state.service
    current_spec = service.current_model_spec
    current_alias = current_spec.alias if current_spec else None
    model_entries = [
        ModelInfo(
            alias=spec.alias,
            model_id=spec.model_id,
            engine_type=spec.engine_type,
            description=spec.description,
            capabilities=asdict(spec.capabilities),
            requestable=spec.requestable,
        )
        for spec in list_all()
    ]
    pipeline_entries = [
        ModelInfo(
            alias=profile.alias,
            model_id=f"{profile.transcription_alias}+{profile.diarization_alias}",
            engine_type="pipeline",
            description=profile.description,
            capabilities=asdict(profile.capabilities),
            requestable=profile.requestable,
        )
        for profile in list_all_profiles()
    ]
    models = sorted([*model_entries, *pipeline_entries], key=lambda entry: entry.alias)

    return ModelsResponse(
        models=models,
        current=current_alias,
    )


@router.get("/v1/models/current")
async def get_current_model(request: Request) -> dict[str, object]:
    """
    Return the currently loaded model and its capabilities.
    Useful for clients to discover what formats/features are available.
    """
    service = request.app.state.service
    current_spec = service.current_model_spec

    return {
        "engine_type": current_spec.engine_type if current_spec else request.app.state.engine_type,
        "model_id": current_spec.model_id if current_spec else request.app.state.model_id,
        "model_alias": current_spec.alias if current_spec else None,
        # Use current_spec.capabilities for consistency — avoids a transient mismatch
        # between current_spec and service.engine during a model switch.
        "capabilities": asdict(current_spec.capabilities) if current_spec else asdict(service.capabilities),
        "queue_size": service.queue_size,
        "max_queue_size": service.max_queue_size,
    }
