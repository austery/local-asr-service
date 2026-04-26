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

        raw_format = kwargs.get("format") or kwargs.get("response_format", "txt")
        output_format = str(raw_format) if raw_format is not None else "txt"
        if output_format in ("json", "verbose_json"):
            output_format = "json"
        elif output_format not in ("txt", "srt", "vtt"):
            output_format = "txt"

        raw = self._pipeline(file_path)
        text = raw.get("text", "") if isinstance(raw, dict) else str(raw)

        if output_format == "json":
            chunks: list[object] = raw.get("chunks", []) if isinstance(raw, dict) else []
            segments: list[dict[str, object]] = []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    ts = chunk.get("timestamp", (None, None))
                    start: object = ts[0] if isinstance(ts, (list, tuple)) and len(ts) > 0 else None
                    end: object = ts[1] if isinstance(ts, (list, tuple)) and len(ts) > 1 else None
                    segments.append({"start": start, "end": end, "text": chunk.get("text", "")})
            return {"text": text, "segments": segments}

        return text

    def release(self) -> None:
        self._pipeline = None
        gc.collect()
