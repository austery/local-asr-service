"""FireRed ASR runtime adapter.

Thin adapter around the FireRed ASR model via HuggingFace Transformers.
Implements the ASREngine Protocol via structural subtyping.
"""

import gc
import importlib
import logging
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
        self._pipeline = runtime.pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            return_timestamps=True,
        )
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

        if fmt == "json":
            segments: list[dict[str, object]] = []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    ts = chunk.get("timestamp", (None, None))
                    start = ts[0] if isinstance(ts, (list, tuple)) and len(ts) > 0 else None
                    end = ts[1] if isinstance(ts, (list, tuple)) and len(ts) > 1 else None
                    # Issue #3: skip chunks with any None timestamp — API Segment model
                    # requires float values; None would cause a serialization 500 error.
                    if start is None or end is None:
                        continue
                    segments.append({"start": start, "end": end, "text": chunk.get("text", "")})
            return {"text": text, "segments": segments if segments else None}

        if fmt in ("srt", "vtt"):
            return self._format_as_srt(chunks) or text

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

    def _format_as_srt(self, chunks: list[object]) -> str:
        """Build an SRT subtitle string from FireRed timestamped chunks (timestamps in seconds)."""
        lines: list[str] = []
        idx = 1
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            ts = chunk.get("timestamp", (None, None))
            start = ts[0] if isinstance(ts, (list, tuple)) and len(ts) > 0 else None
            end = ts[1] if isinstance(ts, (list, tuple)) and len(ts) > 1 else None
            if start is None or end is None:
                continue
            lines.append(str(idx))
            lines.append(f"{self._sec_to_srt_time(start)} --> {self._sec_to_srt_time(end)}")
            lines.append(chunk.get("text", ""))
            lines.append("")
            idx += 1
        return "\n".join(lines)

    def release(self) -> None:
        self._pipeline = None
        gc.collect()
