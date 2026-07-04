"""Subprocess client for the Apple Speech Swift worker."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from src.core.apple_speech_port import (
    AppleSpeechError,
    AppleSpeechModule,
    AppleSpeechWorkerResponseError,
    AppleSpeechWorkerUnavailableError,
    AssetPreparationResult,
    TimingGranularity,
    TranscriptionMetadata,
    TranscriptionResult,
    TranscriptionSegment,
    WorkerCapabilities,
    WorkerModules,
)

JsonObject = Mapping[str, object]
AppleSpeechWorkerError = AppleSpeechError

__all__ = [
    "AppleSpeechWorkerClient",
    "AppleSpeechWorkerError",
    "AppleSpeechWorkerResponseError",
    "AppleSpeechWorkerUnavailableError",
]


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

    def prepare(self, locale: str, module: AppleSpeechModule) -> AssetPreparationResult:
        payload = self._run_json(
            ["prepare", "--locale", locale, "--module", module, "--json"]
        )
        return AssetPreparationResult(
            locale=_required_str(payload, "locale"),
            module=cast(AppleSpeechModule, _required_str(payload, "module")),
            supported=_required_bool(payload, "supported"),
            allocated=_required_bool(payload, "allocated"),
            downloaded=_required_bool(payload, "downloaded"),
            duration_ms=_required_int(payload, "durationMs"),
        )

    def transcribe(
        self,
        input_path: Path,
        locale: str,
        module: AppleSpeechModule,
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
            raise AppleSpeechWorkerUnavailableError(
                f"apple-speech-worker timed out after {self.timeout_seconds:.1f}s"
            ) from exc
        except OSError as exc:
            raise AppleSpeechWorkerUnavailableError(
                f"failed to run apple-speech-worker at {self.worker_path}: {exc}"
            ) from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            message = stderr or f"worker exited with code {completed.returncode}"
            raise AppleSpeechWorkerResponseError(message)

        try:
            decoded: object = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise AppleSpeechWorkerResponseError("worker stdout is not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise AppleSpeechWorkerResponseError("worker stdout JSON must be an object")
        return cast(JsonObject, decoded)


def _parse_transcription_result(payload: JsonObject) -> TranscriptionResult:
    metadata = _required_object(payload, "metadata")
    return TranscriptionResult(
        job_id=_optional_str(payload, "jobId"),
        engine=_required_str(payload, "engine"),
        module=cast(AppleSpeechModule, _required_str(payload, "module")),
        locale=_required_str(payload, "locale"),
        text=_required_str(payload, "text"),
        segments=_parse_segments(payload),
        metadata=TranscriptionMetadata(
            local=_required_bool(metadata, "local"),
            apple_api=_required_bool(metadata, "appleApi"),
            volatile_included=_required_bool(metadata, "volatileIncluded"),
            timing_granularity=cast(
                TimingGranularity,
                _required_str(metadata, "timingGranularity"),
            ),
            asset_managed_by_system=_required_bool(metadata, "assetManagedBySystem"),
            duration_ms=_required_int(metadata, "durationMs"),
        ),
    )


def _parse_segments(payload: JsonObject) -> list[TranscriptionSegment]:
    value = payload.get("segments")
    if not isinstance(value, list):
        raise AppleSpeechWorkerResponseError("worker JSON field 'segments' must be a list")
    segments: list[TranscriptionSegment] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise AppleSpeechWorkerResponseError("worker JSON segment must be an object")
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
        raise AppleSpeechWorkerResponseError(f"worker JSON field '{key}' must be an object")
    return cast(JsonObject, value)


def _required_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise AppleSpeechWorkerResponseError(f"worker JSON field '{key}' must be a string")
    return value


def _optional_str(payload: JsonObject, key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AppleSpeechWorkerResponseError(
            f"worker JSON field '{key}' must be a string or null"
        )
    return value


def _required_bool(payload: JsonObject, key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise AppleSpeechWorkerResponseError(f"worker JSON field '{key}' must be a boolean")
    return value


def _required_int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AppleSpeechWorkerResponseError(f"worker JSON field '{key}' must be an integer")
    return value


def _required_float(payload: JsonObject, key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AppleSpeechWorkerResponseError(f"worker JSON field '{key}' must be a number")
    return float(value)


def _optional_float(payload: JsonObject, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AppleSpeechWorkerResponseError(
            f"worker JSON field '{key}' must be a number or null"
        )
    return float(value)


def _required_str_list(payload: JsonObject, key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AppleSpeechWorkerResponseError(f"worker JSON field '{key}' must be a string list")
    return list(value)


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"
