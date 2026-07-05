"""ASREngine adapter for the Apple Speech sidecar worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.adapters.apple_speech_worker_client import AppleSpeechWorkerClient
from src.core.apple_speech_port import (
    AppleSpeechModule,
    TranscriptionResult,
    WorkerCapabilities,
)
from src.core.base_engine import EngineCapabilities


class AppleSpeechClient(Protocol):
    """Client interface consumed by AppleSpeechEngine."""

    def capabilities(self) -> WorkerCapabilities:
        """Return worker capabilities."""
        ...

    def transcribe(
        self,
        input_path: Path,
        locale: str,
        module: AppleSpeechModule,
        audio_time_ranges: bool = True,
        include_volatile: bool = False,
    ) -> TranscriptionResult:
        """Transcribe an audio file through the Apple Speech worker."""
        ...


@dataclass(frozen=True)
class AppleSpeechEngineConfig:
    worker_path: Path
    timeout_seconds: float = 120.0


class AppleSpeechEngine:
    """Blocking ASR engine wrapper around the Swift Apple Speech worker CLI."""

    def __init__(
        self,
        *,
        client: AppleSpeechClient,
        module: AppleSpeechModule,
    ) -> None:
        self._client = client
        self._module = module

    @classmethod
    def from_config(
        cls,
        config: AppleSpeechEngineConfig,
        module: AppleSpeechModule,
    ) -> AppleSpeechEngine:
        return cls(
            client=AppleSpeechWorkerClient(
                config.worker_path,
                timeout_seconds=config.timeout_seconds,
            ),
            module=module,
        )

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            timestamp=True,
            diarization=False,
            emotion_tags=False,
            language_detect=False,
        )

    def load(self) -> None:
        self._client.capabilities()

    def transcribe_file(
        self,
        file_path: str,
        language: str,
        output_format: str = "json",
        with_timestamp: bool = False,
        include_volatile: bool = False,
        **_kwargs: object,
    ) -> str | dict[str, object]:
        locale = self._resolve_locale(language)
        result = self._client.transcribe(
            input_path=Path(file_path),
            locale=locale,
            module=self._module,
            audio_time_ranges=True,
            include_volatile=include_volatile,
        )
        if output_format == "txt":
            return result.text
        if output_format == "srt":
            return self._format_as_srt(result)
        return self._to_service_dict(result)

    def release(self) -> None:
        return None

    def _resolve_locale(self, language: str) -> str:
        normalized = language.strip()
        if not normalized or normalized.lower() == "auto":
            raise ValueError(
                "Apple Speech requires an explicit language or locale; "
                "pass 'zh', 'zh-CN', 'en', or 'en-US'."
            )
        lowered = normalized.lower()
        if lowered == "zh":
            return "zh-CN"
        if lowered == "en":
            return "en-US"
        return normalized

    @staticmethod
    def _to_service_dict(result: TranscriptionResult) -> dict[str, object]:
        duration = 0.0
        if result.segments:
            duration = max(segment.end for segment in result.segments)
        return {
            "text": result.text,
            "segments": [
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "speaker": None,
                }
                for segment in result.segments
            ],
            "duration": duration,
            "language": result.locale,
        }

    @staticmethod
    def _format_as_srt(result: TranscriptionResult) -> str:
        lines: list[str] = []
        for index, segment in enumerate(result.segments, start=1):
            lines.extend(
                [
                    str(index),
                    f"{_seconds_to_srt_time(segment.start)} --> {_seconds_to_srt_time(segment.end)}",
                    segment.text,
                    "",
                ]
            )
        return "\n".join(lines)


def _seconds_to_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
