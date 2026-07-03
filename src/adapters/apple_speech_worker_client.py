"""Subprocess client for the Apple Speech Swift worker."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

JsonObject = Mapping[str, object]


class AppleSpeechWorkerError(RuntimeError):
    """Raised when the Swift worker cannot return a valid JSON response."""


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
    module: str
    supported: bool
    allocated: bool
    downloaded: bool
    duration_ms: int


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
    timing_granularity: str
    asset_managed_by_system: bool
    duration_ms: int


@dataclass(frozen=True)
class TranscriptionResult:
    job_id: str | None
    engine: str
    module: str
    locale: str
    text: str
    segments: list[TranscriptionSegment]
    metadata: TranscriptionMetadata


class AppleSpeechWorkerClient:
    """Thin JSON-over-stdout client for `apple-speech-worker`."""

    def __init__(self, worker_path: Path, timeout_seconds: float = 120.0) -> None:
        self.worker_path = worker_path
        self.timeout_seconds = timeout_seconds

    def capabilities(self) -> WorkerCapabilities:
        payload = self._run_json(["capabilities", "--json"])
        modules = _required_object(payload, "modules")
        return WorkerCapabilities(
            runtime=_required_str(payload, "runtime"),
            platform=_required_str(payload, "platform"),
            os_version=_required_str(payload, "osVersion"),
            supported=_required_bool(payload, "supported"),
            supported_locales=_required_str_list(payload, "supportedLocales"),
            modules=WorkerModules(
                speech_transcriber=_required_bool(modules, "speechTranscriber"),
                dictation_transcriber=_required_bool(modules, "dictationTranscriber"),
                speech_detector=_required_bool(modules, "speechDetector"),
            ),
            notes=_required_str_list(payload, "notes"),
        )

    def prepare(self, locale: str, module: str) -> AssetPreparationResult:
        payload = self._run_json(
            ["prepare", "--locale", locale, "--module", module, "--json"]
        )
        return AssetPreparationResult(
            locale=_required_str(payload, "locale"),
            module=_required_str(payload, "module"),
            supported=_required_bool(payload, "supported"),
            allocated=_required_bool(payload, "allocated"),
            downloaded=_required_bool(payload, "downloaded"),
            duration_ms=_required_int(payload, "durationMs"),
        )

    def transcribe(
        self,
        input_path: Path,
        locale: str,
        module: str,
        audio_time_ranges: bool = True,
        include_volatile: bool = False,
    ) -> TranscriptionResult:
        payload = self._run_json(
            [
                "transcribe",
                "--input",
                str(input_path),
                "--locale",
                locale,
                "--module",
                module,
                "--audio-time-ranges",
                _bool_arg(audio_time_ranges),
                "--volatile",
                _bool_arg(include_volatile),
                "--json",
            ]
        )
        return _parse_transcription_result(payload)

    def _run_json(self, arguments: list[str]) -> JsonObject:
        command = [str(self.worker_path), *arguments]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AppleSpeechWorkerError(
                f"apple-speech-worker timed out after {self.timeout_seconds:.1f}s"
            ) from exc
        except OSError as exc:
            raise AppleSpeechWorkerError(
                f"failed to run apple-speech-worker at {self.worker_path}: {exc}"
            ) from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            message = stderr or f"worker exited with code {completed.returncode}"
            raise AppleSpeechWorkerError(message)

        try:
            decoded: object = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise AppleSpeechWorkerError("worker stdout is not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise AppleSpeechWorkerError("worker stdout JSON must be an object")
        return cast(JsonObject, decoded)


def _parse_transcription_result(payload: JsonObject) -> TranscriptionResult:
    metadata = _required_object(payload, "metadata")
    return TranscriptionResult(
        job_id=_optional_str(payload, "jobId"),
        engine=_required_str(payload, "engine"),
        module=_required_str(payload, "module"),
        locale=_required_str(payload, "locale"),
        text=_required_str(payload, "text"),
        segments=_parse_segments(payload),
        metadata=TranscriptionMetadata(
            local=_required_bool(metadata, "local"),
            apple_api=_required_bool(metadata, "appleApi"),
            volatile_included=_required_bool(metadata, "volatileIncluded"),
            timing_granularity=_required_str(metadata, "timingGranularity"),
            asset_managed_by_system=_required_bool(metadata, "assetManagedBySystem"),
            duration_ms=_required_int(metadata, "durationMs"),
        ),
    )


def _parse_segments(payload: JsonObject) -> list[TranscriptionSegment]:
    value = payload.get("segments")
    if not isinstance(value, list):
        raise AppleSpeechWorkerError("worker JSON field 'segments' must be a list")
    segments: list[TranscriptionSegment] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise AppleSpeechWorkerError("worker JSON segment must be an object")
        segment = cast(JsonObject, item)
        segments.append(
            TranscriptionSegment(
                id=_required_int(segment, "id"),
                start=_required_float(segment, "start"),
                end=_required_float(segment, "end"),
                text=_required_str(segment, "text"),
                is_final=_required_bool(segment, "isFinal"),
                confidence=_optional_float(segment, "confidence"),
                speaker=_optional_str(segment, "speaker"),
            )
        )
    return segments


def _required_object(payload: JsonObject, key: str) -> JsonObject:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise AppleSpeechWorkerError(f"worker JSON field '{key}' must be an object")
    return cast(JsonObject, value)


def _required_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise AppleSpeechWorkerError(f"worker JSON field '{key}' must be a string")
    return value


def _optional_str(payload: JsonObject, key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AppleSpeechWorkerError(f"worker JSON field '{key}' must be a string or null")
    return value


def _required_bool(payload: JsonObject, key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise AppleSpeechWorkerError(f"worker JSON field '{key}' must be a boolean")
    return value


def _required_int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AppleSpeechWorkerError(f"worker JSON field '{key}' must be an integer")
    return value


def _required_float(payload: JsonObject, key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AppleSpeechWorkerError(f"worker JSON field '{key}' must be a number")
    return float(value)


def _optional_float(payload: JsonObject, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AppleSpeechWorkerError(f"worker JSON field '{key}' must be a number or null")
    return float(value)


def _required_str_list(payload: JsonObject, key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AppleSpeechWorkerError(f"worker JSON field '{key}' must be a string list")
    return list(value)


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"
