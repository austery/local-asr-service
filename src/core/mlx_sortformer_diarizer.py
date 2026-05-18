from importlib import import_module
from typing import Protocol, cast

from src.core.diarization_port import DiarizationPort, RuntimeDiarizationSegment, SpeakerTurn


class _SortformerRuntime(Protocol):
    def generate(
        self,
        audio: str,
        *,
        threshold: float,
        min_duration: float,
        merge_gap: float,
        verbose: bool,
    ) -> "_RuntimeDiarizationOutput": ...


class _RuntimeDiarizationOutput(Protocol):
    segments: list[RuntimeDiarizationSegment]


def _load_sortformer_runtime(model_id: str) -> _SortformerRuntime:
    vad_module = import_module("mlx_audio.vad")
    load = vad_module.load
    return load(model_id)


def _require_runtime_segment(segment: object) -> RuntimeDiarizationSegment:
    if not isinstance(segment, RuntimeDiarizationSegment):
        raise TypeError(
            "Runtime diarization segment must expose speaker, start, and end attributes."
        )
    return cast(RuntimeDiarizationSegment, segment)


SORTFORMER_THRESHOLD = 0.35
SORTFORMER_MIN_DURATION = 0.2
SORTFORMER_MERGE_GAP = 0.3


class MlxSortformerDiarizer(DiarizationPort):
    def __init__(self, model_id: str = "mlx-community/diar_sortformer_4spk-v1-fp16") -> None:
        self.model_id = model_id
        self._runtime: _SortformerRuntime | None = None

    def load(self) -> None:
        if self._runtime is None:
            self._runtime = _load_sortformer_runtime(self.model_id)

    def diarize_file(self, file_path: str) -> list[SpeakerTurn]:
        if self._runtime is None:
            raise RuntimeError("Diarizer not loaded. Call load() first.")

        output = self._runtime.generate(
            file_path,
            threshold=SORTFORMER_THRESHOLD,
            min_duration=SORTFORMER_MIN_DURATION,
            merge_gap=SORTFORMER_MERGE_GAP,
            verbose=False,
        )
        return [self._to_speaker_turn(segment) for segment in output.segments]

    def release(self) -> None:
        self._runtime = None

    def _to_speaker_turn(self, segment: RuntimeDiarizationSegment | object) -> SpeakerTurn:
        runtime_segment = _require_runtime_segment(segment)
        if not isinstance(runtime_segment.speaker, int | str):
            raise TypeError(
                f"speaker must be int or str, got {type(runtime_segment.speaker).__name__}"
            )
        return SpeakerTurn(
            speaker=f"Speaker {runtime_segment.speaker}",
            start=float(runtime_segment.start),
            end=float(runtime_segment.end),
        )
