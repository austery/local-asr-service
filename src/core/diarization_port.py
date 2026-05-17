from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if not self.speaker.strip():
            raise ValueError("Speaker name cannot be empty")
        if self.end <= self.start:
            raise ValueError(f"Invalid interval: end ({self.end}) must be > start ({self.start})")


class DiarizationPort(Protocol):
    def load(self) -> None: ...

    def diarize_file(self, file_path: str) -> list[SpeakerTurn]: ...

    def release(self) -> None: ...
