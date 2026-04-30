from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str
    start: float
    end: float


class DiarizationPort(Protocol):
    def load(self) -> None:
        raise NotImplementedError

    def diarize_file(self, file_path: str) -> list[SpeakerTurn]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError
