"""Explicit standalone diarization HTTP contract, separate from transcription."""

from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from src.api.uploads import validate_audio_upload
from src.core.diarization_registry import cached_snapshot, lookup_diarizer, public_diarizers
from src.core.diarization_result import DiarizationError
from src.services.transcription import TranscriptionService

router = APIRouter()


class DiarizerInfo(BaseModel):
    alias: str
    model_id: str
    task: Literal["diarization"] = "diarization"
    experimental: bool = True
    requires_explicit_selection: bool = True
    endpoint: str = "/v1/audio/diarizations"
    max_speakers: int
    weights_cached: bool


def discovered_diarizers() -> list[DiarizerInfo]:
    return [
        DiarizerInfo(
            alias=s.alias,
            model_id=s.model_id,
            max_speakers=s.max_speakers,
            weights_cached=cached_snapshot(s) is not None,
        )
        for s in public_diarizers()
    ]


class TurnResponse(BaseModel):
    speaker: str
    start: float
    end: float


class DiarizationResponse(BaseModel):
    task: Literal["diarization"] = "diarization"
    diarizer: str
    model_id: str
    revision: str
    experimental: bool = True
    duration: float
    num_speakers: int
    turns: list[TurnResponse]


@router.post("/v1/audio/diarizations", response_model=DiarizationResponse)
async def create_diarization(
    request: Request,
    file: UploadFile = File(...),
    diarizer: str = Form(
        ...,
        description="Required explicit nemotron-diar selection. Experimental speaker activity only; no text. Busy models return retryable 503.",
    ),
) -> DiarizationResponse:
    """Run only when all ASR/native work is idle and resident models offloaded.

    No ASR model switch, implicit diarizer, weight download or automatic restore.
    Missing alias returns 422; empty/auto/unknown/hidden aliases return 400.
    """
    validate_audio_upload(file)
    try:
        spec = lookup_diarizer(diarizer)
    except KeyError:
        raise HTTPException(
            400, "Select the experimental nemotron-diar alias explicitly."
        ) from None
    if not spec.public or spec.revision is None:
        raise HTTPException(400, "This diarizer is not publicly requestable.")
    service: TranscriptionService = request.app.state.service
    try:
        payload = await service.submit_diarization(file, spec)
    except DiarizationError as exc:
        headers = {"Retry-After": "5"} if exc.status == status.HTTP_503_SERVICE_UNAVAILABLE else None
        raise HTTPException(
            exc.status, detail={"code": exc.code, "message": str(exc)}, headers=headers
        ) from None
    return DiarizationResponse(
        diarizer=spec.alias,
        model_id=spec.model_id,
        revision=spec.revision,
        duration=payload.duration,
        num_speakers=len({t.speaker for t in payload.turns}),
        turns=[TurnResponse(**asdict(t)) for t in payload.turns],
    )
