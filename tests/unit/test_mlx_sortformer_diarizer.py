from unittest.mock import MagicMock

import pytest

from src.core.diarization_port import SpeakerTurn
from src.core.mlx_sortformer_diarizer import MlxSortformerDiarizer


class _RuntimeSegment:
    def __init__(self, speaker: int | str, start: float, end: float) -> None:
        self.speaker = speaker
        self.start = start
        self.end = end


class _RuntimeOutput:
    def __init__(self, segments: list[object]) -> None:
        self.segments = segments


def test_diarize_file_should_require_load_first() -> None:
    diarizer = MlxSortformerDiarizer()

    with pytest.raises(RuntimeError, match="load\\(\\) first"):
        diarizer.diarize_file("sample.wav")


def test_diarize_file_should_map_attribute_segments_to_speaker_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MagicMock()
    runtime.generate.return_value = _RuntimeOutput(
        [
            _RuntimeSegment(speaker=0, start=0.0, end=1.25),
            _RuntimeSegment(speaker=1, start=1.25, end=2.5),
        ]
    )

    def fake_load_runtime(model_id: str) -> MagicMock:
        assert model_id == "mlx-community/diar_sortformer_4spk-v1-fp16"
        return runtime

    monkeypatch.setattr(
        "src.core.mlx_sortformer_diarizer._load_sortformer_runtime",
        fake_load_runtime,
    )

    diarizer = MlxSortformerDiarizer()
    diarizer.load()

    turns = diarizer.diarize_file("sample.wav")

    assert turns == [
        SpeakerTurn(speaker="Speaker 0", start=0.0, end=1.25),
        SpeakerTurn(speaker="Speaker 1", start=1.25, end=2.5),
    ]
    runtime.generate.assert_called_once_with(
        "sample.wav",
        threshold=0.35,
        min_duration=0.2,
        merge_gap=0.3,
        verbose=False,
    )


def test_diarize_file_should_raise_when_runtime_segment_shape_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MagicMock()
    runtime.generate.return_value = _RuntimeOutput(
        [{"speaker": 0, "start": 0.0, "end": 1.25}]
    )

    monkeypatch.setattr(
        "src.core.mlx_sortformer_diarizer._load_sortformer_runtime",
        lambda _model_id: runtime,
    )

    diarizer = MlxSortformerDiarizer()
    diarizer.load()

    with pytest.raises(TypeError, match="speaker, start, and end attributes"):
        diarizer.diarize_file("sample.wav")
