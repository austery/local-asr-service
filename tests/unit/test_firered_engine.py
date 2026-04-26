"""Unit tests for the FireRed ASR adapter (src/core/firered_engine.py)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.firered_engine import FireRedEngine


class TestFireRedEngineCapabilities:
    def test_capabilities_match_firered_profile(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        caps = engine.capabilities

        assert caps.timestamp is True
        assert caps.diarization is False
        assert caps.emotion_tags is False
        assert caps.language_detect is True


class TestFireRedEngineLoad:
    def test_load_creates_pipeline_via_transformers_runtime(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        mock_pipeline_instance = MagicMock(name="pipeline-instance")
        mock_pipeline_fn = MagicMock(return_value=mock_pipeline_instance)
        runtime = SimpleNamespace(pipeline=mock_pipeline_fn)

        with patch("src.core.firered_engine.importlib.import_module", return_value=runtime):
            engine.load()

        mock_pipeline_fn.assert_called_once_with(
            "automatic-speech-recognition",
            model="FireRedTeam/FireRedASR2-AED",
            return_timestamps=True,
        )
        assert engine._pipeline is mock_pipeline_instance

    def test_load_is_idempotent(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        mock_pipeline_instance = MagicMock(name="pipeline-instance")
        mock_pipeline_fn = MagicMock(return_value=mock_pipeline_instance)
        runtime = SimpleNamespace(pipeline=mock_pipeline_fn)

        with patch("src.core.firered_engine.importlib.import_module", return_value=runtime):
            engine.load()
            engine.load()

        mock_pipeline_fn.assert_called_once()

    def test_load_raises_actionable_error_when_transformers_unavailable(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")

        with patch(
            "src.core.firered_engine.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'transformers'"),
        ):
            with pytest.raises(RuntimeError, match="transformers"):
                engine.load()


class TestFireRedEngineTranscribe:
    def test_transcribe_raises_before_load(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")

        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.transcribe_file("audio.wav")

    def test_transcribe_returns_text_string_by_default(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(return_value={"text": "Hello world", "chunks": []})

        result = engine.transcribe_file("audio.wav")

        assert result == "Hello world"
        engine._pipeline.assert_called_once_with("audio.wav")

    def test_transcribe_returns_text_string_for_txt_format(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(return_value={"text": "Hello world", "chunks": []})

        result = engine.transcribe_file("audio.wav", format="txt")

        assert result == "Hello world"

    def test_transcribe_returns_json_dict_with_segments(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(
            return_value={
                "text": "Hello world",
                "chunks": [
                    {"text": "Hello", "timestamp": (0.0, 0.5)},
                    {"text": " world", "timestamp": (0.5, 1.0)},
                ],
            }
        )

        result = engine.transcribe_file("audio.wav", format="json")

        assert isinstance(result, dict)
        assert result["text"] == "Hello world"
        segments = result["segments"]
        assert len(segments) == 2
        assert segments[0] == {"start": 0.0, "end": 0.5, "text": "Hello"}
        assert segments[1] == {"start": 0.5, "end": 1.0, "text": " world"}

    def test_transcribe_verbose_json_is_treated_as_json(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(return_value={"text": "Hi", "chunks": []})

        result = engine.transcribe_file("audio.wav", response_format="verbose_json")

        assert isinstance(result, dict)
        assert result["text"] == "Hi"
        assert result["segments"] == []

    def test_transcribe_unknown_format_falls_back_to_txt(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(return_value={"text": "Fallback", "chunks": []})

        result = engine.transcribe_file("audio.wav", format="srt")

        assert result == "Fallback"


class TestFireRedEngineRelease:
    def test_release_clears_pipeline_reference(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock()

        engine.release()

        assert engine._pipeline is None

    def test_release_calls_gc_collect(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock()

        with patch("src.core.firered_engine.gc.collect") as mock_gc:
            engine.release()

        mock_gc.assert_called_once()
