"""Sortformer diarization runtime adapter."""

import gc
import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from src.core.diarization_port import DiarizationPort, SpeakerTurn

_RUNTIME_MODULE = "mlx_sortformer"


class _RuntimeModule(Protocol):
    def load_model(self, model_id: str) -> object: ...

    def diarize(self, model: object, file_path: str) -> Sequence[Mapping[str, object]]: ...


class SortformerEngine(DiarizationPort):
    """Thin adapter around a Sortformer diarization runtime."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model: object | None = None
        self._diarize: Callable[[object, str], Sequence[Mapping[str, object]]] | None = None

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            runtime = cast(_RuntimeModule, importlib.import_module(_RUNTIME_MODULE))
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Sortformer diarization runtime is unavailable. Install an mlx-audio version "
                "with Sortformer diarization support before using this adapter."
            ) from exc

        diarize = getattr(runtime, "diarize", None)
        if not callable(diarize):
            raise RuntimeError(
                "Sortformer diarization runtime is missing required 'diarize' callable."
            )

        model = runtime.load_model(self.model_id)
        self._model = model
        self._diarize = diarize

    def diarize_file(self, file_path: str) -> list[SpeakerTurn]:
        if self._model is None or self._diarize is None:
            raise RuntimeError("Model not loaded! Call engine.load() first.")

        return [self._to_speaker_turn(turn) for turn in self._diarize(self._model, file_path)]

    def release(self) -> None:
        self._model = None
        self._diarize = None
        gc.collect()

    @staticmethod
    def _to_speaker_turn(turn: Mapping[str, object]) -> SpeakerTurn:
        try:
            speaker = turn["speaker"]
            start = turn["start"]
            end = turn["end"]
        except KeyError as exc:
            missing_field = exc.args[0]
            raise ValueError(
                f"Sortformer diarization result is missing required field: {missing_field}"
            ) from exc

        if not isinstance(speaker, str):
            raise TypeError(f"Expected speaker to be str, got {type(speaker).__name__}")
        if isinstance(start, bool) or not isinstance(start, int | float):
            raise TypeError(f"Expected start to be numeric, got {type(start).__name__}")
        if isinstance(end, bool) or not isinstance(end, int | float):
            raise TypeError(f"Expected end to be numeric, got {type(end).__name__}")
        if not math.isfinite(float(start)):
            raise ValueError(f"Expected start to be a finite number, got {start!r}")
        if not math.isfinite(float(end)):
            raise ValueError(f"Expected end to be a finite number, got {end!r}")
        if start >= end:
            raise ValueError(
                f"Invalid timestamp range for speaker {speaker!r}: start={start} >= end={end}"
            )

        return SpeakerTurn(speaker=speaker, start=float(start), end=float(end))
