"""Pinned cached Nemotron adapter; imported only inside its owned worker."""

from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from src.core.diarization_port import RuntimeDiarizationSegment, SpeakerTurn
from src.core.diarization_registry import cached_snapshot, lookup_diarizer
from src.core.diarization_result import DiarizationError


class _Output(Protocol):
    segments: list[RuntimeDiarizationSegment]


class _Runtime(Protocol):
    def generate(
        self, audio: str, *, threshold: float, min_duration: float, merge_gap: float, verbose: bool
    ) -> _Output: ...


class MlxNemotronDiarizer:
    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint
        self._runtime: _Runtime | None = None

    def load(self) -> None:
        if self._runtime is not None:
            return
        checkpoint = self.checkpoint or cached_snapshot(lookup_diarizer("nemotron-diar"))
        if checkpoint is None:
            raise DiarizationError(
                "model_unavailable", "Cache the pinned Nemotron snapshot before requesting it.", 503
            )
        self._runtime = cast(
            _Runtime, import_module("mlx_audio.vad").load(str(checkpoint), strict=True)
        )

    def diarize_file(self, file_path: str) -> list[SpeakerTurn]:
        if self._runtime is None:
            raise RuntimeError("Diarizer is not loaded")
        output = self._runtime.generate(
            file_path, threshold=0.5, min_duration=0.0, merge_gap=0.0, verbose=False
        )
        return [
            SpeakerTurn(f"Speaker {s.speaker}", float(s.start), float(s.end))
            for s in output.segments
        ]

    def release(self) -> None:
        self._runtime = None
