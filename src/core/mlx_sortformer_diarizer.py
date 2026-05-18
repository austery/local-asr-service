from importlib import import_module
from typing import Protocol

from src.core.diarization_port import DiarizationPort, SpeakerTurn


class _SortformerRuntime(Protocol):
    def generate(
        self,
        audio: str,
        *,
        threshold: float,
        verbose: bool,
    ) -> list[object]: ...


def _load_sortformer_runtime(model_id: str) -> _SortformerRuntime:
    vad_module = import_module("mlx_audio.vad")
    load = getattr(vad_module, "load")
    return load(model_id)


def _segment_field(segment: object, field: str) -> object:
    if isinstance(segment, dict):
        return segment[field]
    return getattr(segment, field)


class MlxSortformerDiarizer(DiarizationPort):
    def __init__(self, model_id: str = "mlx-community/diar-sortformer-4spk-v1") -> None:
        self.model_id = model_id
        self._runtime: _SortformerRuntime | None = None

    def load(self) -> None:
        if self._runtime is None:
            self._runtime = _load_sortformer_runtime(self.model_id)

    def diarize_file(self, file_path: str) -> list[SpeakerTurn]:
        if self._runtime is None:
            raise RuntimeError("Diarizer not loaded. Call load() first.")

        segments = self._runtime.generate(file_path, threshold=0.5, verbose=False)
        return [self._to_speaker_turn(segment) for segment in segments]

    def release(self) -> None:
        self._runtime = None

    def _to_speaker_turn(self, segment: object) -> SpeakerTurn:
        speaker = _segment_field(segment, "speaker")
        start = _segment_field(segment, "start")
        end = _segment_field(segment, "end")
        return SpeakerTurn(
            speaker=f"Speaker {speaker}",
            start=float(start),
            end=float(end),
        )
