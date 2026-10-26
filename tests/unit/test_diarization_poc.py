"""Public benchmark input/guard behavior, without loading model frameworks."""

import argparse
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import psutil
import pytest

from benchmarks import diarization_poc as runner


def test_audio_reader_preserves_pcm_scale_and_rejects_wrong_rate(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(np.array([-32768, 0, 16384], dtype="<i2").tobytes())
    np.testing.assert_array_equal(runner.read_audio(audio), [-1, 0, 0.5])
    with wave.open(str(audio), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(bytes(10))
    with pytest.raises(ValueError, match="16 kHz"):
        runner.read_audio(audio)


def test_supervisor_reaps_a_child_that_exits_during_memory_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MagicMock(pid=123, returncode=0)
    process.poll.side_effect = [None, 0]
    child = MagicMock()
    child.memory_info.side_effect = psutil.ZombieProcess(123)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(runner.psutil, "Process", lambda pid: child)
    monkeypatch.setattr(
        runner.psutil, "virtual_memory", lambda: argparse.Namespace(available=32 * 1024**3)
    )
    kill = MagicMock()
    monkeypatch.setattr(runner.os, "killpg", kill)
    runner.supervise(
        argparse.Namespace(output=tmp_path / "out.json", protect_pid=[], memory_gib=8, timeout=3)
    )
    process.wait.assert_called_once_with(timeout=5)
    kill.assert_not_called()


def test_production_inference_prevents_starting_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MagicMock()
    service.is_running.return_value = True
    service.create_time.return_value = 10.0
    worker = MagicMock(pid=8)
    worker.status.return_value = psutil.STATUS_RUNNING
    worker.cmdline.return_value = ["python", "-c", "spawn_main"]
    service.children.return_value = [worker]
    monkeypatch.setattr(runner.psutil, "Process", lambda pid: service)
    with pytest.raises(RuntimeError, match="Production inference active"):
        runner.protect({7: 10.0})


@pytest.mark.parametrize("suffix", [".json", ".npz", ".rttm", ".log", ".supervision.json"])
def test_existing_evidence_is_never_overwritten(tmp_path: Path, suffix: str) -> None:
    out = tmp_path / "out.json"
    preserved = out.with_suffix(suffix)
    preserved.write_text("unique previous result")
    with pytest.raises(FileExistsError):
        runner.supervise(argparse.Namespace(output=out))
    assert preserved.read_text() == "unique previous result"


def test_memory_reserve_drop_stops_only_owned_group_and_retains_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = MagicMock(pid=123, returncode=0)
    process.poll.side_effect = [None, None, 0]
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: process)
    readings = iter([32, 32, 19])
    monkeypatch.setattr(
        runner.psutil,
        "virtual_memory",
        lambda: argparse.Namespace(available=next(readings) * 1024**3),
    )
    kill = MagicMock()
    monkeypatch.setattr(runner.os, "killpg", kill)
    output = tmp_path / "stopped.json"
    with pytest.raises(MemoryError, match="Available memory fell"):
        runner.supervise(argparse.Namespace(output=output, protect_pid=[], memory_gib=8, timeout=3))
    kill.assert_called_once_with(123, runner.signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=10)
    assert not output.exists()
    assert "Available memory fell" in output.with_suffix(".supervision.json").read_text()
