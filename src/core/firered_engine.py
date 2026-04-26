"""FireRed ASR runtime adapter.

Thin adapter around the FireRed ASR model via HuggingFace Transformers.
Implements the ASREngine Protocol via structural subtyping.
"""

import gc
import importlib
import logging
import math
from collections.abc import Callable
from typing import Protocol, cast

from src.core.base_engine import EngineCapabilities

logger = logging.getLogger(__name__)

_RUNTIME_MODULE = "transformers"

_FIRERED_CAPABILITIES = EngineCapabilities(
    timestamp=True,
    diarization=False,
    emotion_tags=False,
    language_detect=True,
)


class _TransformersRuntime(Protocol):
    """Structural type for the subset of transformers we use."""

    def pipeline(
        self,
        task: str,
        *,
        model: str,
        return_timestamps: bool,
    ) -> Callable[[str], object]: ...


def _load_transformers_runtime() -> _TransformersRuntime:
    """Lazy-import the transformers runtime. Raises RuntimeError if unavailable."""
    try:
        return cast(_TransformersRuntime, importlib.import_module(_RUNTIME_MODULE))
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FireRed ASR runtime is unavailable. "
            "Install transformers before using this adapter."
        ) from exc


class FireRedEngine:
    """Thin adapter around a FireRed ASR model via HuggingFace Transformers pipeline."""

    def __init__(self, model_id: str = "FireRedTeam/FireRedASR2-AED"):
        self.model_id = model_id
        self._pipeline: Callable[[str], object] | None = None

    @property
    def capabilities(self) -> EngineCapabilities:
        return _FIRERED_CAPABILITIES

    def load(self) -> None:
        if self._pipeline is not None:
            return

        runtime = _load_transformers_runtime()
        logger.info(f"🚀 Loading FireRed model '{self.model_id}'...")
        try:
            self._pipeline = runtime.pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                return_timestamps=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FireRed model '{self.model_id}': {exc}"
            ) from exc
        logger.info(f"✅ FireRed model loaded: {self.model_id}")

    def transcribe_file(
        self, file_path: str, language: str = "auto", **kwargs: object
    ) -> str | dict[str, object]:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded! Call engine.load() first.")

        # Issue #1: model_worker passes output_format=; also accept format=/response_format=
        # for callers that use the legacy kwarg names.
        raw_format = (
            kwargs.get("output_format")
            or kwargs.get("format")
            or kwargs.get("response_format", "txt")
        )
        fmt = str(raw_format) if raw_format is not None else "txt"
        if fmt in ("json", "verbose_json"):
            fmt = "json"
        elif fmt not in ("txt", "srt", "vtt"):
            fmt = "txt"

        raw = self._pipeline(file_path)
        text = raw.get("text", "") if isinstance(raw, dict) else str(raw)

        chunks: list[object] = raw.get("chunks", []) if isinstance(raw, dict) else []
        timestamped_chunks = self._collect_timestamped_chunks(chunks)

        if fmt == "json":
            segments = [
                {"start": start, "end": end, "text": chunk_text}
                for start, end, chunk_text in timestamped_chunks
            ]
            return {"text": text, "segments": segments if segments else None}

        if fmt == "srt":
            return self._format_as_srt(timestamped_chunks) or text
        if fmt == "vtt":
            return self._format_as_vtt(timestamped_chunks) or text

        return text

    @staticmethod
    def _sec_to_srt_time(sec: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        ms = max(0, int(round(sec * 1000)))
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _read_chunk_timestamps(
        self, chunk: dict[str, object]
    ) -> tuple[float | None, float | None]:
        ts = chunk.get("timestamp", (None, None))
        start = self._normalize_timestamp(
            ts[0] if isinstance(ts, (list, tuple)) and len(ts) > 0 else None
        )
        end = self._normalize_timestamp(
            ts[1] if isinstance(ts, (list, tuple)) and len(ts) > 1 else None
        )
        return start, end

    def _collect_timestamped_chunks(
        self, chunks: list[object]
    ) -> list[tuple[float, float, str]]:
        timestamped_chunks: list[tuple[float, float, str]] = []
        dropped_chunk_count = 0
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            start, end = self._read_chunk_timestamps(chunk)
            if start is None or end is None:
                dropped_chunk_count += 1
                continue
            timestamped_chunks.append((start, end, str(chunk.get("text", ""))))

        if dropped_chunk_count:
            logger.warning(
                "FireRed model '%s' dropped %d chunk(s) with missing or invalid timestamps",
                self.model_id,
                dropped_chunk_count,
            )
        return timestamped_chunks

    def _format_as_srt(self, chunks: list[tuple[float, float, str]]) -> str:
        """Build an SRT subtitle string from FireRed timestamped chunks (timestamps in seconds)."""
        lines: list[str] = []
        idx = 1
        for start, end, chunk_text in chunks:
            lines.append(str(idx))
            lines.append(f"{self._sec_to_srt_time(start)} --> {self._sec_to_srt_time(end)}")
            lines.append(chunk_text)
            lines.append("")
            idx += 1
        return "\n".join(lines)

    def _format_as_vtt(self, chunks: list[tuple[float, float, str]]) -> str:
        """Build a WebVTT subtitle string from FireRed timestamped chunks."""
        lines = ["WEBVTT", ""]
        for start, end, chunk_text in chunks:
            lines.append(
                f"{self._sec_to_srt_time(start).replace(',', '.')} --> "
                f"{self._sec_to_srt_time(end).replace(',', '.')}"
            )
            lines.append(chunk_text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _normalize_timestamp(value: object) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(timestamp):
            return None
        return timestamp

    def release(self) -> None:
        logger.info("🧹 Releasing FireRed model '%s'", self.model_id)
        self._pipeline = None
        gc.collect()
