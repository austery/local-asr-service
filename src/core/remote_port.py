"""Typed file transcription boundary; no native model or HTTP ownership."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class RemoteTranscriptionError(Exception):
    """A sanitized public failure; provider bodies and transport strings stay private."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class RemoteWord:
    speaker: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class RemoteSegment:
    speaker: str
    start: float
    end: float
    text: str
    words: tuple[RemoteWord, ...]


@dataclass(frozen=True)
class RemoteTranscript:
    text: str
    duration: float
    language: str
    segments: tuple[RemoteSegment, ...]


class RemoteTranscriber(Protocol):
    async def transcribe(self, path: Path, language: str, request_id: str) -> RemoteTranscript:
        """One paid submission at most; cancellation closes owned HTTP resources."""
        ...
