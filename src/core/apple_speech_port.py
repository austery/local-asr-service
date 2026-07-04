"""Typed port contract for the Apple Speech sidecar worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AppleSpeechModule = Literal["speechTranscriber", "dictationTranscriber"]
TimingGranularity = Literal["none", "segment", "word", "unknown"]


class AppleSpeechError(RuntimeError):
    """Base error for Apple Speech sidecar integration failures."""


class AppleSpeechWorkerUnavailableError(AppleSpeechError):
    """Raised when the worker binary cannot be executed or times out."""


class AppleSpeechWorkerResponseError(AppleSpeechError):
    """Raised when the worker returns invalid JSON or an unsuccessful result."""


@dataclass(frozen=True)
class WorkerModules:
    speech_transcriber: bool
    dictation_transcriber: bool
    speech_detector: bool


@dataclass(frozen=True)
class WorkerCapabilities:
    runtime: str
    platform: str
    os_version: str
    supported: bool
    supported_locales: list[str]
    modules: WorkerModules
    notes: list[str]


@dataclass(frozen=True)
class AssetPreparationResult:
    locale: str
    module: AppleSpeechModule
    supported: bool
    allocated: bool
    downloaded: bool
    duration_ms: int


@dataclass(frozen=True)
class AppleSpeechRequest:
    input_path: Path
    locale: str
    module: AppleSpeechModule
    include_audio_ranges: bool = True
    include_volatile: bool = False


@dataclass(frozen=True)
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str
    is_final: bool
    confidence: float | None
    speaker: str | None


@dataclass(frozen=True)
class TranscriptionMetadata:
    local: bool
    apple_api: bool
    volatile_included: bool
    timing_granularity: TimingGranularity
    asset_managed_by_system: bool
    duration_ms: int


@dataclass(frozen=True)
class TranscriptionResult:
    job_id: str | None
    engine: str
    module: AppleSpeechModule
    locale: str
    text: str
    segments: list[TranscriptionSegment]
    metadata: TranscriptionMetadata
