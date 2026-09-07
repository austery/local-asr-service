"""Public MOSS adapter behavior with a stubbed upstream runtime."""

import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.base_engine import TranscriptionInputError, TranscriptionOutputError
from src.core.mlx_engine import MlxAudioEngine
from src.core.model_registry import MOSS_MODEL_ID, MOSS_MODEL_REVISION, lookup

_Runtime = tuple[MlxAudioEngine, MagicMock, MagicMock]

def _response(end: float = 2.0) -> SimpleNamespace:
    return SimpleNamespace(
        text=f"[0.00][S01]Hello.[1.00] [1.00][S02]Hi.[{end:.2f}]",
        segments=[
            {"start": 0.0, "end": 1.0, "speaker_id": "S01", "text": "[S01] Hello."},
            {"start": 1.0, "end": end, "speaker_id": "S02", "text": "[S02] Hi."},
        ],
        generation_tokens=20,
    )


@pytest.fixture
def runtime() -> Iterator[_Runtime]:
    with (
        patch("src.core.mlx_engine.AudioChunkingService") as chunker,
        patch("src.core.mlx_engine.generate_transcription", return_value=_response()) as generate,
    ):
        chunker.return_value.get_audio_duration.return_value = 2.0
        engine = MlxAudioEngine(MOSS_MODEL_ID)
        engine.model = object()
        yield engine, generate, chunker.return_value


def test_alias_and_engine_declare_the_same_native_capabilities(runtime: _Runtime) -> None:
    engine, _, _ = runtime
    spec = lookup("moss-transcribe-diarize")
    assert lookup(MOSS_MODEL_ID) == spec
    assert spec.engine_type == "mlx"
    assert spec.capabilities == engine.capabilities
    assert engine.capabilities.timestamp and engine.capabilities.diarization
    assert not engine.capabilities.language_detect


def test_whole_recording_uses_tested_budget_and_normalizes_speakers(runtime: _Runtime) -> None:
    engine, generate, chunker = runtime
    result = engine.transcribe_file("recording.mp3", "en", output_format="json")
    assert result == {
        "text": "Hello. Hi.", "duration": 2.0, "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "speaker": "S01", "text": "Hello."},
            {"start": 1.0, "end": 2.0, "speaker": "S02", "text": "Hi."},
        ],
    }
    chunker.process_audio.assert_not_called()
    assert generate.call_count == 1
    kwargs = generate.call_args.kwargs
    assert kwargs["audio"] == "recording.mp3"
    assert kwargs["max_tokens"] == 32768 and kwargs["prefill_step_size"] == 4096
    assert kwargs["temperature"] == 0.0
    assert "language" not in kwargs
    assert not Path(kwargs["output_path"]).parent.exists()


@pytest.mark.parametrize("duration", [1800.001, 3600.0, 0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_duration_before_inference(runtime: _Runtime, duration: float) -> None:
    engine, generate, chunker = runtime
    chunker.get_audio_duration.return_value = duration
    with pytest.raises(TranscriptionInputError, match="1800 seconds"):
        engine.transcribe_file("recording.wav", "en", output_format="json")
    generate.assert_not_called()
    chunker.process_audio.assert_not_called()


def test_accepts_exact_thirty_minute_boundary(runtime: _Runtime) -> None:
    engine, generate, chunker = runtime
    chunker.get_audio_duration.return_value = 1800.0
    generate.return_value = _response(1800.0)
    assert engine.transcribe_file("recording.wav", "en", output_format="json")["duration"] == 1800.0


@pytest.mark.parametrize("language", ["auto", "", "zh", "fr"])
def test_rejects_language_outside_validated_scope(runtime: _Runtime, language: str) -> None:
    engine, generate, _ = runtime
    with pytest.raises(TranscriptionInputError, match="language=en"):
        engine.transcribe_file("recording.wav", language)
    generate.assert_not_called()


@pytest.mark.parametrize("language", ["en", "en-US", "en_US", "eng", "English"])
def test_accepts_explicit_english_aliases(runtime: _Runtime, language: str) -> None:
    engine, _, _ = runtime
    assert engine.transcribe_file("recording.wav", language, response_format="verbose_json")["language"] == "en"


@pytest.mark.parametrize("fault", ["raw_tail", "raw_prefix", "budget", "no_speaker", "nan", "reverse", "overshoot", "unparsed", "empty"])
def test_rejects_malformed_output_and_recovers_on_next_call(runtime: _Runtime, fault: str) -> None:
    engine, generate, _ = runtime
    response = _response()
    if fault == "raw_tail":
        response.text += "[2.10][S"
    elif fault == "raw_prefix":
        response.text = "unparsed words " + response.text
    elif fault == "budget":
        response.generation_tokens = 32768
    elif fault == "no_speaker":
        del response.segments[0]["speaker_id"]
    elif fault == "nan":
        response.segments[0]["start"] = float("nan")
    elif fault == "reverse":
        response.segments.reverse()
    elif fault == "overshoot":
        response = _response(2.5)
    elif fault == "unparsed":
        response.text = response.text.replace("Hello.", "Different.")
    else:
        response.segments = []
    generate.return_value = response
    with pytest.raises(TranscriptionOutputError):
        engine.transcribe_file("recording.wav", "en", output_format="json")
    assert not Path(generate.call_args.kwargs["output_path"]).parent.exists()
    generate.return_value = _response()
    assert engine.transcribe_file("recording.wav", "en", output_format="json")["text"] == "Hello. Hi."


@pytest.mark.parametrize("placement", ["head", "middle", "tail"])
def test_rejects_large_uncovered_spans(runtime: _Runtime, placement: str) -> None:
    engine, generate, chunker = runtime
    chunker.get_audio_duration.return_value = 22.0
    response = _response()
    if placement == "head":
        response.text = "[20.00][S01]Hello.[21.00] [21.00][S02]Hi.[22.00]"
        for segment in response.segments:
            segment["start"] += 20.0
            segment["end"] += 20.0
    elif placement == "middle":
        response.text = "[0.00][S01]Hello.[1.00] [21.00][S02]Hi.[22.00]"
        response.segments[1].update(start=21.0, end=22.0)
    generate.return_value = response
    with pytest.raises(TranscriptionOutputError, match="uncovered"):
        engine.transcribe_file("recording.wav", "en", output_format="json")


def test_clamps_small_endpoint_rounding_error(runtime: _Runtime) -> None:
    engine, generate, _ = runtime
    generate.return_value = _response(2.10)
    result = engine.transcribe_file("recording.wav", "en", output_format="json")
    assert result["segments"][-1]["end"] == 2.0


def test_overlap_keeps_native_speaker_labels(runtime: _Runtime) -> None:
    engine, generate, _ = runtime
    response = _response()
    response.text = response.text.replace("[1.00][S02]", "[0.50][S02]")
    response.segments[1]["start"] = 0.5
    generate.return_value = response
    result = engine.transcribe_file("recording.wav", "en", output_format="json")
    assert [segment["speaker"] for segment in result["segments"]] == ["S01", "S02"]


@pytest.mark.parametrize(
    ("output_format", "with_timestamp", "expected"),
    [
        ("txt", False, "[S01]: Hello.\n[S02]: Hi."),
        ("text", True, "[00:00] [S01]: Hello.\n[00:01] [S02]: Hi."),
        ("srt", False, "1\n00:00:00,000 --> 00:00:01,000\n[S01]: Hello.\n\n2\n00:00:01,000 --> 00:00:02,000\n[S02]: Hi.\n"),
    ],
)
def test_formats_clean_speaker_text_and_subtitles(runtime: _Runtime, output_format: str, with_timestamp: bool, expected: str) -> None:
    engine, _, _ = runtime
    assert engine.transcribe_file("recording.wav", "en", output_format=output_format, with_timestamp=with_timestamp) == expected


def test_cleans_up_upstream_output_when_generation_fails(runtime: _Runtime) -> None:
    engine, generate, _ = runtime
    generate.side_effect = RuntimeError("upstream failure")
    with pytest.raises(RuntimeError, match="upstream failure"):
        engine.transcribe_file("recording.wav", "en")
    assert not Path(generate.call_args.kwargs["output_path"]).parent.exists()


def test_moss_deadline_terminates_only_its_subprocess() -> None:
    script = """
import time
from src.workers import model_worker
model_worker.MOSS_INFERENCE_TIMEOUT_SECONDS = 0.05
with model_worker._inference_deadline(model_worker.MOSS_MODEL_ID):
    time.sleep(5)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=10)
    assert result.returncode == -signal.SIGALRM


def test_completed_moss_call_cancels_the_alarm() -> None:
    script = """
import time
from src.workers import model_worker
model_worker.MOSS_INFERENCE_TIMEOUT_SECONDS = 0.05
with model_worker._inference_deadline(model_worker.MOSS_MODEL_ID):
    pass
time.sleep(0.1)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode()


def test_loads_the_evaluated_moss_checkpoint(runtime: _Runtime) -> None:
    engine, _, _ = runtime
    engine.model = None
    with patch("src.core.mlx_engine.load_model") as load:
        engine.load()
    load.assert_called_once_with(MOSS_MODEL_ID, revision=MOSS_MODEL_REVISION)
