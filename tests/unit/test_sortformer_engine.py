from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.diarization_port import SpeakerTurn
from src.core.sortformer_engine import SortformerEngine


class TestSortformerEngine:
    def test_load_should_use_runtime_loader(self) -> None:
        runtime_model = MagicMock(name="runtime-model")
        runtime = SimpleNamespace(load_model=MagicMock(return_value=runtime_model), diarize=MagicMock())
        engine = SortformerEngine(model_id="mlx-community/diar_sortformer_4spk-v1-fp32")

        with patch("src.core.sortformer_engine.importlib.import_module", return_value=runtime):
            engine.load()

        runtime.load_model.assert_called_once_with("mlx-community/diar_sortformer_4spk-v1-fp32")
        assert engine._model is runtime_model

    def test_load_should_raise_actionable_error_when_runtime_is_unavailable(self) -> None:
        engine = SortformerEngine(model_id="mlx-community/diar_sortformer_4spk-v1-fp32")

        with patch(
            "src.core.sortformer_engine.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'mlx_sortformer'"),
        ):
            with pytest.raises(RuntimeError, match="mlx-audio version with Sortformer diarization support"):
                engine.load()

    def test_load_should_reset_state_when_runtime_wiring_is_incomplete(self) -> None:
        broken_runtime = SimpleNamespace(load_model=MagicMock(return_value="broken-model"))
        working_runtime_model = MagicMock(name="runtime-model")
        working_runtime = SimpleNamespace(
            load_model=MagicMock(return_value=working_runtime_model),
            diarize=MagicMock(),
        )
        engine = SortformerEngine(model_id="mlx-community/diar_sortformer_4spk-v1-fp32")

        with patch(
            "src.core.sortformer_engine.importlib.import_module",
            side_effect=[broken_runtime, working_runtime],
        ):
            with pytest.raises(RuntimeError, match="missing required 'diarize' callable"):
                engine.load()

            assert engine._model is None
            assert engine._diarize is None

            engine.load()

        assert engine._model is working_runtime_model
        assert engine._diarize is working_runtime.diarize

    def test_load_should_reject_non_callable_diarize_runtime(self) -> None:
        broken_runtime = SimpleNamespace(
            load_model=MagicMock(return_value="broken-model"),
            diarize=None,
        )
        engine = SortformerEngine(model_id="mlx-community/diar_sortformer_4spk-v1-fp32")

        with patch("src.core.sortformer_engine.importlib.import_module", return_value=broken_runtime):
            with pytest.raises(RuntimeError, match="missing required 'diarize' callable"):
                engine.load()

        assert engine._model is None
        assert engine._diarize is None
        broken_runtime.load_model.assert_not_called()

    def test_release_should_clear_loaded_model_reference(self) -> None:
        engine = SortformerEngine(model_id="mlx-community/diar_sortformer_4spk-v1-fp32")
        engine._model = object()
        engine._diarize = MagicMock()

        with patch("src.core.sortformer_engine.gc.collect") as collect:
            engine.release()

        assert engine._model is None
        assert engine._diarize is None
        collect.assert_called_once()

    def test_diarize_file_should_raise_when_model_not_loaded(self) -> None:
        engine = SortformerEngine(model_id="mlx-community/diar_sortformer_4spk-v1-fp32")

        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.diarize_file("audio.wav")

    def test_diarize_file_should_reject_non_increasing_timestamps(self) -> None:
        engine = SortformerEngine(model_id="mlx-community/diar_sortformer_4spk-v1-fp32")
        runtime_model = object()
        diarize = MagicMock(return_value=[{"speaker": "speaker_0", "start": 2.0, "end": 2.0}])
        engine._model = runtime_model
        engine._diarize = diarize

        with pytest.raises(ValueError, match="Invalid timestamp range"):
            engine.diarize_file("audio.wav")

    def test_diarize_file_should_raise_actionable_error_when_required_key_is_missing(self) -> None:
        engine = SortformerEngine(model_id="mlx-community/diar_sortformer_4spk-v1-fp32")
        runtime_model = object()
        diarize = MagicMock(return_value=[{"speaker": "speaker_0", "start": 0.0}])
        engine._model = runtime_model
        engine._diarize = diarize

        with pytest.raises(ValueError, match="missing required field: end"):
            engine.diarize_file("audio.wav")

    def test_diarize_file_should_reject_boolean_timestamps(self) -> None:
        engine = SortformerEngine(model_id="mlx-community/diar_sortformer_4spk-v1-fp32")
        runtime_model = object()
        diarize = MagicMock(return_value=[{"speaker": "speaker_0", "start": True, "end": 1.0}])
        engine._model = runtime_model
        engine._diarize = diarize

        with pytest.raises(TypeError, match="Expected start to be numeric"):
            engine.diarize_file("audio.wav")

    def test_diarize_file_should_reject_non_finite_timestamps(self) -> None:
        engine = SortformerEngine(model_id="mlx-community/diar_sortformer_4spk-v1-fp32")
        runtime_model = object()
        diarize = MagicMock(return_value=[{"speaker": "speaker_0", "start": float("nan"), "end": 1.0}])
        engine._model = runtime_model
        engine._diarize = diarize

        with pytest.raises(ValueError, match="Expected start to be a finite number"):
            engine.diarize_file("audio.wav")

    def test_diarize_file_should_map_runtime_dicts_to_speaker_turns(self) -> None:
        engine = SortformerEngine(model_id="mlx-community/diar_sortformer_4spk-v1-fp32")
        runtime_model = object()
        diarize = MagicMock(
            return_value=[
                {"speaker": "speaker_0", "start": 0.0, "end": 1.5},
                {"speaker": "speaker_1", "start": 1.5, "end": 3.0},
            ]
        )
        engine._model = runtime_model
        engine._diarize = diarize

        result = engine.diarize_file("audio.wav")

        assert result == [
            SpeakerTurn(speaker="speaker_0", start=0.0, end=1.5),
            SpeakerTurn(speaker="speaker_1", start=1.5, end=3.0),
        ]
        diarize.assert_called_once_with(runtime_model, "audio.wav")
