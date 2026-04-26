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
        # No chunks → no valid segments; convention: return None (not []) to match funasr_engine
        assert result["segments"] is None

    def test_transcribe_unknown_format_falls_back_to_txt(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(return_value={"text": "Fallback", "chunks": []})

        result = engine.transcribe_file("audio.wav", format="xyz_unknown")

        assert result == "Fallback"

    def test_transcribe_srt_format_returns_subtitle_string(self) -> None:
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

        result = engine.transcribe_file("audio.wav", format="srt")

        assert isinstance(result, str)
        assert "00:00:00,000 --> 00:00:00,500" in result
        assert "Hello" in result
        assert "00:00:00,500 --> 00:00:01,000" in result
        assert " world" in result

    def test_transcribe_srt_falls_back_to_plain_text_when_no_valid_chunks(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(
            return_value={
                "text": "Some text",
                "chunks": [{"text": "Some text", "timestamp": (None, None)}],
            }
        )

        result = engine.transcribe_file("audio.wav", format="srt")

        assert result == "Some text"

    # Issue #1: worker passes output_format kwarg — engine must honor it
    def test_transcribe_honors_output_format_kwarg_from_worker(self) -> None:
        """The model_worker passes output_format=, not format= or response_format=.
        Regression: before the fix, output_format='json' would silently return plain text."""
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(
            return_value={
                "text": "Worker JSON",
                "chunks": [{"text": "Worker JSON", "timestamp": (0.0, 1.0)}],
            }
        )

        result = engine.transcribe_file("audio.wav", output_format="json")

        assert isinstance(result, dict), "Expected dict when output_format='json'"
        assert result["text"] == "Worker JSON"
        segments = result["segments"]
        assert len(segments) == 1
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 1.0

    # Issue #3: None timestamps must not appear in JSON segments (API model expects floats)
    def test_transcribe_json_omits_segments_when_timestamps_are_none(self) -> None:
        """When FireRed returns chunks with (None, None) timestamps, the adapter
        must not emit segments with None start/end — the API Segment model requires floats."""
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(
            return_value={
                "text": "Some text",
                "chunks": [
                    {"text": "Some", "timestamp": (None, None)},
                    {"text": " text", "timestamp": (None, None)},
                ],
            }
        )

        result = engine.transcribe_file("audio.wav", output_format="json")

        assert isinstance(result, dict)
        assert result["text"] == "Some text"
        # Segments must be None (no valid timestamps) or an empty list — not a list
        # containing dicts with None floats that would cause API serialization errors.
        segments = result["segments"]
        if segments is not None:
            for seg in segments:
                assert seg["start"] is not None, "start must not be None"
                assert seg["end"] is not None, "end must not be None"

    def test_transcribe_json_filters_chunks_with_partial_none_timestamps(self) -> None:
        """Chunks with partial None timestamps (e.g. open-ended) must be excluded
        from segments to avoid float serialization errors downstream."""
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(
            return_value={
                "text": "Hello world",
                "chunks": [
                    {"text": "Hello", "timestamp": (0.0, 0.5)},   # valid
                    {"text": " world", "timestamp": (0.5, None)},  # invalid: end is None
                ],
            }
        )

        result = engine.transcribe_file("audio.wav", output_format="json")

        assert isinstance(result, dict)
        assert result["text"] == "Hello world"
        segments = result["segments"]
        # Only the valid segment should be included
        assert segments is not None
        assert len(segments) == 1
        assert segments[0] == {"start": 0.0, "end": 0.5, "text": "Hello"}


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
