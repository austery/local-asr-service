from unittest.mock import MagicMock

import pytest

from src.core.diarization_port import SpeakerTurn
from src.core.mlx_sortformer_diarizer import MlxSortformerDiarizer


def test_diarize_file_should_require_load_first() -> None:
    diarizer = MlxSortformerDiarizer()

    with pytest.raises(RuntimeError, match="load\\(\\) first"):
        diarizer.diarize_file("sample.wav")


def test_diarize_file_should_map_runtime_segments_to_speaker_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = MagicMock()
    runtime.generate.return_value = [
        {"speaker": 0, "start": 0.0, "end": 1.25},
        {"speaker": 1, "start": 1.25, "end": 2.5},
    ]

    def fake_load_runtime(model_id: str) -> MagicMock:
        assert model_id == "mlx-community/diar-sortformer-4spk-v1"
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
    runtime.generate.assert_called_once_with("sample.wav", threshold=0.5, verbose=False)
