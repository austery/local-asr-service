import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.adapters.apple_speech_worker_client import (
    AppleSpeechWorkerClient,
    AppleSpeechWorkerError,
)


def test_capabilities_parses_worker_json() -> None:
    completed = subprocess.CompletedProcess(
        args=["worker"],
        returncode=0,
        stdout=(
            '{"runtime":"apple-speech","platform":"macOS","osVersion":"26.5",'
            '"supported":true,"supportedLocales":["en-US"],'
            '"modules":{"speechTranscriber":true,'
            '"dictationTranscriber":true,"speechDetector":true},"notes":[]}'
        ),
        stderr="",
    )

    with patch("src.adapters.apple_speech_worker_client.subprocess.run", return_value=completed):
        client = AppleSpeechWorkerClient(Path("/tmp/apple-speech-worker"))
        result = client.capabilities()

    assert result.runtime == "apple-speech"
    assert result.supported is True
    assert result.modules.speech_transcriber is True
    assert result.supported_locales == ["en-US"]


def test_prepare_passes_locale_module_and_json_flag() -> None:
    completed = subprocess.CompletedProcess(
        args=["worker"],
        returncode=0,
        stdout=(
            '{"locale":"zh-CN","module":"speechTranscriber","supported":true,'
            '"allocated":true,"downloaded":true,"durationMs":61}'
        ),
        stderr="",
    )

    with patch("src.adapters.apple_speech_worker_client.subprocess.run", return_value=completed) as run:
        client = AppleSpeechWorkerClient(Path("/tmp/apple-speech-worker"))
        result = client.prepare(locale="zh-CN", module="speechTranscriber")

    run.assert_called_once()
    assert run.call_args.args[0] == [
        "/tmp/apple-speech-worker",
        "prepare",
        "--locale",
        "zh-CN",
        "--module",
        "speechTranscriber",
        "--json",
    ]
    assert result.downloaded is True


def test_transcribe_passes_input_and_returns_segments() -> None:
    completed = subprocess.CompletedProcess(
        args=["worker"],
        returncode=0,
        stdout=(
            '{"jobId":null,"engine":"apple-speech","module":"speechTranscriber",'
            '"locale":"en-US","text":"hello world",'
            '"segments":[{"id":0,"start":0.0,"end":1.0,"text":"hello world",'
            '"isFinal":true,"confidence":null,"speaker":null}],'
            '"metadata":{"local":true,"appleApi":true,"volatileIncluded":false,'
            '"timingGranularity":"segment","assetManagedBySystem":true,"durationMs":42}}'
        ),
        stderr="",
    )

    with patch("src.adapters.apple_speech_worker_client.subprocess.run", return_value=completed) as run:
        client = AppleSpeechWorkerClient(Path("/tmp/apple-speech-worker"))
        result = client.transcribe(
            input_path=Path("/tmp/audio.wav"),
            locale="en-US",
            module="speechTranscriber",
        )

    assert "--audio-time-ranges" in run.call_args.args[0]
    assert result.text == "hello world"
    assert result.segments[0].start == 0.0
    assert result.metadata.timing_granularity == "segment"


def test_worker_nonzero_raises_with_stderr() -> None:
    completed = subprocess.CompletedProcess(
        args=["worker"],
        returncode=1,
        stdout="",
        stderr="asset unsupported",
    )

    with patch("src.adapters.apple_speech_worker_client.subprocess.run", return_value=completed):
        client = AppleSpeechWorkerClient(Path("/tmp/apple-speech-worker"))
        with pytest.raises(AppleSpeechWorkerError, match="asset unsupported"):
            client.capabilities()


def test_worker_invalid_json_raises() -> None:
    completed = subprocess.CompletedProcess(
        args=["worker"],
        returncode=0,
        stdout="Loading...\n{}",
        stderr="",
    )

    with patch("src.adapters.apple_speech_worker_client.subprocess.run", return_value=completed):
        client = AppleSpeechWorkerClient(Path("/tmp/apple-speech-worker"))
        with pytest.raises(AppleSpeechWorkerError, match="valid JSON"):
            client.capabilities()


def test_worker_timeout_raises() -> None:
    with patch(
        "src.adapters.apple_speech_worker_client.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["worker"], timeout=1.0),
    ):
        client = AppleSpeechWorkerClient(Path("/tmp/apple-speech-worker"), timeout_seconds=1.0)
        with pytest.raises(AppleSpeechWorkerError, match="timed out"):
            client.capabilities()


def test_worker_os_error_raises_with_worker_path() -> None:
    with patch(
        "src.adapters.apple_speech_worker_client.subprocess.run",
        side_effect=FileNotFoundError("missing worker"),
    ):
        client = AppleSpeechWorkerClient(Path("/tmp/missing-apple-speech-worker"))
        with pytest.raises(
            AppleSpeechWorkerError,
            match="/tmp/missing-apple-speech-worker",
        ):
            client.capabilities()
