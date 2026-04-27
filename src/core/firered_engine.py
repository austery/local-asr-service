"""FireRed ASR runtime adapter.

Thin adapter around the official FireRed AED runtime.
Implements the ASREngine Protocol via structural subtyping.
"""

import gc
import importlib
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from src.adapters.audio_chunking import AudioChunkingService
from src.core.base_engine import EngineCapabilities

logger = logging.getLogger(__name__)

_RUNTIME_MODULE = "fireredasr2s.fireredasr2"
_HF_HUB_MODULE = "huggingface_hub"

_FIRERED_CAPABILITIES = EngineCapabilities(
    timestamp=True,
    diarization=False,
    emotion_tags=False,
    language_detect=True,
)


class _FireRedModel(Protocol):
    def transcribe(
        self, batch_uttid: list[str], batch_wav_path: list[str]
    ) -> list[dict[str, object]]: ...


class _FireRedAsr2Factory(Protocol):
    def from_pretrained(
        self, asr_type: str, model_dir: str, config: object
    ) -> _FireRedModel: ...


class _FireRedRuntime(Protocol):
    """Structural type for the subset of fireredasr2s we use."""

    FireRedAsr2: _FireRedAsr2Factory
    FireRedAsr2Config: Callable[..., object]


class _HuggingFaceHubRuntime(Protocol):
    def snapshot_download(self, *, repo_id: str) -> str: ...


def _load_firered_runtime() -> _FireRedRuntime:
    """Lazy-import the official FireRed runtime."""
    try:
        return cast(_FireRedRuntime, importlib.import_module(_RUNTIME_MODULE))
    except ModuleNotFoundError as exc:
        missing_name = exc.name or str(exc)
        if missing_name.startswith("fireredasr2s"):
            raise RuntimeError(
                "FireRed ASR runtime is unavailable. "
                "Install fireredasr2s from the official FireRedASR2S repository "
                "before using this adapter."
            ) from exc
        raise RuntimeError(
            "FireRed ASR runtime is installed but missing a transitive dependency: "
            f"{missing_name}. Install the official FireRed runtime dependencies "
            "before using this adapter."
        ) from exc


def _load_hf_hub_runtime() -> _HuggingFaceHubRuntime:
    """Lazy-import huggingface_hub to download official FireRed checkpoints."""
    try:
        return cast(_HuggingFaceHubRuntime, importlib.import_module(_HF_HUB_MODULE))
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface_hub is unavailable. Install project dependencies before "
            "using the FireRed adapter."
        ) from exc


class FireRedEngine:
    """Thin adapter around the official FireRed AED runtime."""

    def __init__(self, model_id: str = "FireRedTeam/FireRedASR2-AED"):
        self.model_id = model_id
        self._model: _FireRedModel | None = None
        self._pipeline: Callable[[str], object] | None = None
        self._chunking_service = AudioChunkingService(max_duration_minutes=1)

    @property
    def capabilities(self) -> EngineCapabilities:
        return _FIRERED_CAPABILITIES

    def load(self) -> None:
        if self._pipeline is not None:
            return

        runtime = _load_firered_runtime()
        hub_runtime = _load_hf_hub_runtime()
        logger.info(f"🚀 Loading FireRed model '{self.model_id}'...")
        try:
            local_model_dir = hub_runtime.snapshot_download(repo_id=self.model_id)
            runtime_config = runtime.FireRedAsr2Config(
                use_gpu=False,
                return_timestamp=True,
            )
            self._model = runtime.FireRedAsr2.from_pretrained(
                "aed",
                local_model_dir,
                runtime_config,
            )
            self._pipeline = self._transcribe_with_runtime
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FireRed model '{self.model_id}': {exc}"
            ) from exc
        logger.info(f"✅ FireRed model loaded: {self.model_id}")

    def _transcribe_with_runtime(self, file_path: str) -> dict[str, object]:
        if self._model is None:
            raise RuntimeError("Model not loaded! Call engine.load() first.")

        chunk_paths = self._chunking_service.process_audio(file_path)
        merged_texts: list[str] = []
        merged_chunks: list[dict[str, object]] = []
        time_offset = 0.0

        for chunk_path in chunk_paths:
            try:
                results = self._model.transcribe([Path(chunk_path).stem], [chunk_path])
            finally:
                if chunk_path != file_path and (
                    ".chunk_" in chunk_path or chunk_path.endswith(".normalized.wav")
                ):
                    Path(chunk_path).unlink(missing_ok=True)

            if not results:
                continue

            first_result = results[0]
            chunk_text = str(first_result.get("text", "")).strip()
            if chunk_text:
                merged_texts.append(chunk_text)

            shifted_chunks = self._offset_runtime_chunks(first_result, time_offset)
            if shifted_chunks:
                merged_chunks.extend(shifted_chunks)
                time_offset = self._normalize_timestamp(shifted_chunks[-1]["timestamp"][1]) or time_offset

        return {"text": " ".join(merged_texts), "chunks": merged_chunks}

    def _offset_runtime_chunks(
        self,
        runtime_result: dict[str, object],
        time_offset: float,
    ) -> list[dict[str, object]]:
        chunks: list[dict[str, object]] = []
        raw_timestamps = runtime_result.get("timestamp", [])
        if not isinstance(raw_timestamps, list):
            return chunks

        for item in raw_timestamps:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            start = self._normalize_timestamp(item[1])
            end = self._normalize_timestamp(item[2])
            if start is None or end is None:
                continue
            chunks.append(
                {
                    "text": str(item[0]),
                    "timestamp": (start + time_offset, end + time_offset),
                }
            )

        return chunks

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
        self._model = None
        self._pipeline = None
        gc.collect()
