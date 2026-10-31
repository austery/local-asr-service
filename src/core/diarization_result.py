"""Typed completion and explicit failure categories for standalone diarization."""

import math
from dataclasses import dataclass

from src.core.diarization_port import SpeakerTurn

MAX_SPEAKERS = 8


class DiarizationError(RuntimeError):
    def __init__(self, code: str, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class ServiceBusyError(DiarizationError):
    def __init__(self) -> None:
        super().__init__(
            "service_busy", "Native models are busy; retry after normal idle offload.", 503
        )


@dataclass(frozen=True)
class DiarizationPayload:
    duration: float
    turns: list[SpeakerTurn]


def parse_payload(value: object) -> DiarizationPayload:
    if not isinstance(value, dict):
        raise DiarizationError("invalid_output", "Expected a diarization object")
    duration = value.get("duration")
    rows = value.get("turns")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, int | float)
        or not math.isfinite(duration)
        or duration < 0
        or not isinstance(rows, list)
    ):
        raise DiarizationError("invalid_output", "Invalid duration or speaker turns")
    turns = []
    for row in rows:
        if not isinstance(row, dict):
            raise DiarizationError("invalid_output", "Invalid speaker turn")
        speaker, start, end = row.get("speaker"), row.get("start"), row.get("end")
        if (
            not isinstance(speaker, str)
            or not speaker.strip()
            or any(
                isinstance(t, bool) or not isinstance(t, int | float) or not math.isfinite(t)
                for t in (start, end)
            )
        ):
            raise DiarizationError("invalid_output", "Invalid speaker or timestamps")
        if (
            not isinstance(start, int | float)
            or not isinstance(end, int | float)
            or not 0 <= start < end <= duration
        ):
            raise DiarizationError("invalid_output", "Speaker interval exceeds the recording")
        turns.append(SpeakerTurn(speaker, float(start), float(end)))
    if len({t.speaker for t in turns}) > MAX_SPEAKERS:
        raise DiarizationError("invalid_output", "Too many speaker identities")
    return DiarizationPayload(float(duration), turns)
