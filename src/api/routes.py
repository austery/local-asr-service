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
from src.services.transcription import PipelineQualityError

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
    requestable: bool = True


class ModelsResponse(BaseModel):
    """Response for GET /v1/models"""

    models: list[ModelInfo]
    current: str | None


# Router
router = APIRouter()


def _resolve_model(model: str | None) -> ModelSpec | None:
    """
    Resolve the `model` form field to a ModelSpec, or None if no switch is needed.

    Returns None for passthrough values (None / "whisper-1") — signals "use current engine".
    Raises HTTPException 400 for unrecognisable model strings.
    """
    if is_passthrough(model):
        return None
    try:
        return lookup(model)  # type: ignore[arg-type]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


def _resolve_pipeline_profile(model: str | None) -> PipelineProfile | None:
    """Resolve a pipeline alias, returning None for non-pipeline model values."""
    if is_passthrough(model):
        return None
    try:
        return lookup_profile(model)  # type: ignore[arg-type]
    except KeyError:
        return None


@router.post("/v1/audio/transcriptions", response_model=None)
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, etc.)"),
    model: str | None = Form(
        None,
        description=(
            "Model alias or full model path. "
            "Omit this field to use the server's current model. "
            "'whisper-1' is accepted as an OpenAI-compatible passthrough value "
            "and does not select Whisper; the fresh server default is `paraformer` "
            "unless ENGINE_TYPE or MODEL_ID overrides it. "
            "Use GET /v1/models for the live alias list. "
            "Examples: 'paraformer', 'qwen3-asr', 'qwen3-sortformer'."
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
    - Pass `model=qwen3-asr` for single-speaker quality-first content.
    - Pass `model=qwen3-sortformer` for opt-in English long-form batch speaker separation.
    - Omit `model` to use the server's currently loaded model.
    - `model=whisper-1` is accepted as an OpenAI-compatible passthrough value and
      does not select Whisper.
    - By default, a fresh server starts with `paraformer` unless `ENGINE_TYPE`,
      `MODEL_ID`, or engine-specific model environment variables override it.
    - Use `GET /v1/models` or `GET /v1/models/current` to inspect runtime aliases.

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

    # 3. Resolve model/pipeline aliases. Some pipeline profiles may remain
    # discoverable but not requestable while their runtime contract is validated.
    resolved_profile = _resolve_pipeline_profile(model)
    if resolved_profile is not None and not resolved_profile.requestable:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Pipeline profile '{resolved_profile.alias}' is discoverable "
                "but not enabled for POST yet."
            ),
        )

    resolved_spec = None if resolved_profile is not None else _resolve_model(model)

    # 4. Resolve effective output format
    effective_format = response_format if response_format is not None else output_format
    effective_format = _RESPONSE_FORMAT_MAP.get(effective_format, effective_format)

    # 5. Capability pre-validation (fail fast before queuing)
    #    If the request specifies a model, validate against ITS declared capabilities
    #    so the client gets an early 400 without waiting for the switch.
    #    Fall back to the current engine only for passthrough requests (model=None).
    if resolved_spec is not None:
        caps = resolved_spec.capabilities
    elif resolved_profile is not None:
        caps = resolved_profile.capabilities
    else:
        caps = request.app.state.service.capabilities

    model_label: str = (
        resolved_spec.alias
        if isinstance(resolved_spec, ModelSpec)
        else resolved_profile.alias
        if isinstance(resolved_profile, PipelineProfile)
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
        params: dict[str, object] = {
            "language": language,
            "output_format": effective_format,
            "with_timestamp": with_timestamp,
        }

        # Determine response model alias BEFORE awaiting:
        #   - Explicit switch: use resolved_spec (always correct regardless of queue ordering).
        #   - Passthrough: capture current spec now; reading it after await is racy because
        #     another concurrent request may trigger a switch while this job is queued.
        spec_for_response: ModelSpec | PipelineProfile | None = (
            resolved_profile
            if resolved_profile is not None
            else resolved_spec
            if resolved_spec is not None
            else service.current_model_spec
        )

        if resolved_profile is not None:
            result = await service.submit_pipeline(
                file,
                params,
                request_id=request_id,
                profile=resolved_profile,
            )
        else:
            result = await service.submit(
                file,
                params,
                request_id=request_id,
                model_spec=resolved_spec,
            )

        if isinstance(spec_for_response, ModelSpec | PipelineProfile):
            response_model = spec_for_response.alias
        else:
            response_model = str(getattr(request.app.state, "model_id", "unknown"))

        if effective_format == "srt":
            return PlainTextResponse(
                content=result if isinstance(result, str) else result.get("text", ""),
                media_type="text/plain; charset=utf-8",
            )

        response_language = language
        if isinstance(result, dict):
            text_obj = result.get("text", "")
            text = text_obj if isinstance(text_obj, str) else ""
            segments_obj = result.get("segments", [])
            segments_data = segments_obj if isinstance(segments_obj, list) else []
            duration_obj = result.get("duration", 0.0)
            duration = (
                float(duration_obj)
                if isinstance(duration_obj, int | float) and not isinstance(duration_obj, bool)
                else 0.0
            )
            result_language = result.get("language")
            if isinstance(result_language, str):
                response_language = result_language

            segments: list[Segment] | None = None
            if effective_format == "json" and segments_data:
                segments = []
                for i, seg in enumerate(segments_data):
                    if not isinstance(seg, dict):
                        continue
                    speaker = seg.get("speaker")
                    start = seg.get("start", 0.0)
                    end = seg.get("end", 0.0)
                    segment_text = seg.get("text", "")
                    segments.append(
                        Segment(
                            id=i,
                            speaker=speaker if isinstance(speaker, str) else None,
                            start=(
                                float(start)
                                if isinstance(start, int | float) and not isinstance(start, bool)
                                else 0.0
                            ),
                            end=(
                                float(end)
                                if isinstance(end, int | float) and not isinstance(end, bool)
                                else 0.0
                            ),
                            text=segment_text if isinstance(segment_text, str) else "",
                        )
                    )

            return TranscriptionResponse(
                text=text,
                duration=duration,
                language=response_language,
                model=response_model,
                segments=segments,
            )
        else:
            return TranscriptionResponse(
                text=str(result),
                language=response_language,
                model=response_model,
                segments=None,
            )

    except PipelineQualityError as e:
        logger.warning(f"[{request_id}] Pipeline quality gate failed: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e)) from None

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
    Use the returned `alias` values in the `model` field of POST /v1/audio/transcriptions.
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
            capabilities={
                k: v for k, v in asdict(spec.capabilities).items()
            },
            requestable=True,
        )
        for spec in list_all()
    ]
    profile_entries = [
        ModelInfo(
            alias=profile.alias,
            model_id=f"{profile.transcription_alias}+{profile.diarization_alias}",
            engine_type="pipeline",
            description=profile.description,
            capabilities={
                k: v for k, v in asdict(profile.capabilities).items()
            },
            requestable=profile.requestable,
        )
        for profile in list_all_profiles()
    ]

    return ModelsResponse(
        models=sorted(model_entries + profile_entries, key=lambda item: item.alias),
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
