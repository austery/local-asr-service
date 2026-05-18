import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AlignedWord:
    text: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Aligned word text cannot be empty")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("Aligned word timestamps must be finite")
        if self.end < self.start:
            raise ValueError(f"Invalid interval: end ({self.end}) must be >= start ({self.start})")


@runtime_checkable
class RuntimeAlignmentItem(Protocol):
    text: str
    start_time: float
    end_time: float


class AlignmentPort(Protocol):
    def load(self) -> None: ...

    def align_file(self, file_path: str, *, text: str, language: str) -> list[AlignedWord]: ...

    def release(self) -> None: ...
