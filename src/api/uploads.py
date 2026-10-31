"""Shared upload validation; does not load audio or model runtimes."""

import logging
import os

from fastapi import HTTPException, UploadFile

from src.config import MAX_UPLOAD_SIZE_MB

logger = logging.getLogger(__name__)

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

# Extensions accepted for octet-stream uploads.
ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".flac",
    ".ogg",
    ".webm",
}



def validate_audio_upload(file: UploadFile, request_id: str = "unknown") -> float:
    # Preserve file-error precedence before validating model selection.
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

    # Validate size before inference or model admission.
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

    return file_size_mb
