"""Unit tests for the FireRed ASR adapter (src/core/firered_engine.py)."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, call, patch

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


@pytest.fixture
def firered_load_env():
    """Shared mock runtime + hub for FireRed load tests."""
    mock_model = MagicMock(name="firered-model")
    mock_config_ctor = MagicMock(return_value="cfg")
    runtime = SimpleNamespace(
        FireRedAsr2=SimpleNamespace(from_pretrained=MagicMock(return_value=mock_model)),
        FireRedAsr2Config=mock_config_ctor,
    )
    hub = SimpleNamespace(snapshot_download=MagicMock(return_value="/tmp/firered-model"))

    def import_module(name: str) -> object:
        if name == "fireredasr2s.fireredasr2":
            return runtime
        if name == "huggingface_hub":
            return hub
        raise ModuleNotFoundError(name)

    return SimpleNamespace(
        runtime=runtime, hub=hub, mock_model=mock_model, import_module=import_module
    )


class TestFireRedEngineLoad:
    def test_load_downloads_snapshot_and_builds_official_aed_runtime(
        self, firered_load_env: SimpleNamespace
    ) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        env = firered_load_env

        with patch("src.core.firered_engine.importlib.import_module", side_effect=env.import_module):
            engine.load()

        env.hub.snapshot_download.assert_called_once_with(repo_id="FireRedTeam/FireRedASR2-AED")
        env.runtime.FireRedAsr2Config.assert_called_once_with(use_gpu=False, return_timestamp=True)
        env.runtime.FireRedAsr2.from_pretrained.assert_called_once_with(
            "aed",
            "/tmp/firered-model",
            "cfg",
        )
        assert engine._model is env.mock_model
        assert engine._pipeline is not None

    def test_load_is_idempotent(self, firered_load_env: SimpleNamespace) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        env = firered_load_env

        with patch("src.core.firered_engine.importlib.import_module", side_effect=env.import_module):
            engine.load()
            engine.load()

        env.hub.snapshot_download.assert_called_once()
        env.runtime.FireRedAsr2.from_pretrained.assert_called_once()

    def test_load_raises_actionable_error_when_firered_runtime_unavailable(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")

        with patch(
            "src.core.firered_engine.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'fireredasr2s'"),
        ):
            with pytest.raises(RuntimeError, match="fireredasr2s"):
                engine.load()

    def test_load_surfaces_missing_transitive_runtime_dependency(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")

        def import_module(name: str) -> object:
            if name == "fireredasr2s.fireredasr2":
                raise ModuleNotFoundError("No module named 'kaldi_native_fbank'")
            raise AssertionError(f"unexpected import: {name}")

        with patch("src.core.firered_engine.importlib.import_module", side_effect=import_module):
            with pytest.raises(RuntimeError, match="kaldi_native_fbank"):
                engine.load()

    def test_load_should_include_model_id_when_pipeline_creation_fails(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        runtime = SimpleNamespace(
            FireRedAsr2=SimpleNamespace(from_pretrained=MagicMock(side_effect=RuntimeError("boom"))),
            FireRedAsr2Config=MagicMock(return_value="cfg"),
        )
        hub = SimpleNamespace(snapshot_download=MagicMock(return_value="/tmp/firered-model"))

        def import_module(name: str) -> object:
            if name == "fireredasr2s.fireredasr2":
                return runtime
            if name == "huggingface_hub":
                return hub
            raise ModuleNotFoundError(name)

        with patch("src.core.firered_engine.importlib.import_module", side_effect=import_module):
            with pytest.raises(RuntimeError, match="FireRedTeam/FireRedASR2-AED"):
                engine.load()


class TestFireRedEngineTranscribe:
    def test_transcribe_chunks_long_audio_before_runtime_call(self) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            [{"text": "hello", "timestamp": [("hello", 0.0, 0.4)]}],
            [{"text": "world", "timestamp": [("world", 0.0, 0.5)]}],
        ]
        chunking_service = MagicMock()
        chunking_service.process_audio.return_value = [
            "/tmp/audio.chunk_000.wav",
            "/tmp/audio.chunk_001.wav",
        ]

        with patch("src.core.firered_engine.AudioChunkingService", return_value=chunking_service):
            engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
            engine._model = mock_model
            engine._pipeline = engine._transcribe_with_runtime
            result = engine.transcribe_file("/tmp/original.mp3", output_format="json")

        assert result["text"] == "hello world"
        assert result["segments"] == [
            {"start": 0.0, "end": 0.4, "text": "hello"},
            {"start": 0.4, "end": 0.9, "text": "world"},
        ]
        chunking_service.process_audio.assert_called_once_with("/tmp/original.mp3")
        assert mock_model.transcribe.call_args_list == [
            call(ANY, ["/tmp/audio.chunk_000.wav"]),
            call(ANY, ["/tmp/audio.chunk_001.wav"]),
        ]

    def test_firered_engine_uses_conservative_chunking_threshold(self) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = [{"text": "hello", "timestamp": []}]

        with patch("src.core.firered_engine.AudioChunkingService") as chunking_cls:
            engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
            engine._model = mock_model
            engine._pipeline = engine._transcribe_with_runtime
            chunking_cls.return_value.process_audio.return_value = [
                "/tmp/audio.chunk_000.wav"
            ]
            engine.transcribe_file("/tmp/original.mp3", output_format="json")

        chunking_cls.assert_called_once_with(max_duration_minutes=1)

    def test_transcribe_normalizes_audio_before_official_runtime(self) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = [
            {
                "text": "normalized text",
                "timestamp": [("normalized", 0.0, 0.5)],
            }
        ]
        chunking_service = MagicMock()
        chunking_service.process_audio.return_value = ["/tmp/audio.normalized.wav"]

        with patch("src.core.firered_engine.AudioChunkingService", return_value=chunking_service):
            engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
            engine._model = mock_model
            engine._pipeline = engine._transcribe_with_runtime
            result = engine.transcribe_file("audio.mp3", output_format="json")

        chunking_service.process_audio.assert_called_once_with("audio.mp3")
        mock_model.transcribe.assert_called_once_with(
            [ANY],
            ["/tmp/audio.normalized.wav"],
        )
        assert isinstance(result, dict)
        assert result["text"] == "normalized text"

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

    def test_transcribe_vtt_format_returns_webvtt_string(self) -> None:
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

        result = engine.transcribe_file("audio.wav", format="vtt")

        assert isinstance(result, str)
        assert result.startswith("WEBVTT\n\n")
        assert "00:00:00.000 --> 00:00:00.500" in result
        assert "00:00:00,000 --> 00:00:00,500" not in result

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

    def test_transcribe_json_normalizes_numeric_timestamps_to_plain_float(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(
            return_value={
                "text": "Hello world",
                "chunks": [
                    {"text": "Hello", "timestamp": (Decimal("0.0"), Decimal("0.5"))},
                ],
            }
        )

        result = engine.transcribe_file("audio.wav", output_format="json")

        assert isinstance(result, dict)
        segments = result["segments"]
        assert segments is not None
        assert segments[0] == {"start": 0.0, "end": 0.5, "text": "Hello"}
        assert isinstance(segments[0]["start"], float)
        assert isinstance(segments[0]["end"], float)

    def test_transcribe_json_filters_bool_timestamps(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(
            return_value={
                "text": "Hello world",
                "chunks": [
                    {"text": "Hello", "timestamp": (True, 0.5)},
                ],
            }
        )

        result = engine.transcribe_file("audio.wav", output_format="json")

        assert isinstance(result, dict)
        assert result["segments"] is None

    def test_transcribe_json_filters_non_finite_timestamps(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(
            return_value={
                "text": "Hello world",
                "chunks": [
                    {"text": "Hello", "timestamp": (0.0, float("inf"))},
                ],
            }
        )

        result = engine.transcribe_file("audio.wav", output_format="json")

        assert isinstance(result, dict)
        assert result["segments"] is None

    def test_transcribe_json_logs_when_chunks_are_dropped_due_to_missing_timestamps(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock(
            return_value={
                "text": "Hello world",
                "chunks": [
                    {"text": "Hello", "timestamp": (0.0, 0.5)},
                    {"text": " world", "timestamp": (0.5, None)},
                ],
            }
        )

        with patch("src.core.firered_engine.logger.warning") as mock_warning:
            result = engine.transcribe_file("audio.wav", output_format="json")

        assert isinstance(result, dict)
        mock_warning.assert_called_once()
        assert "timestamp" in mock_warning.call_args.args[0].lower()


class TestFireRedEngineRelease:
    def test_release_clears_pipeline_reference(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock()
        engine._model = MagicMock()

        engine.release()

        assert engine._pipeline is None
        assert engine._model is None

    def test_release_calls_gc_collect(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock()

        with patch("src.core.firered_engine.gc.collect") as mock_gc:
            engine.release()

        mock_gc.assert_called_once()

    def test_release_logs_model_release(self) -> None:
        engine = FireRedEngine(model_id="FireRedTeam/FireRedASR2-AED")
        engine._pipeline = MagicMock()

        with patch("src.core.firered_engine.logger.info") as mock_info:
            engine.release()

        mock_info.assert_called_once()
