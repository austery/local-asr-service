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


@pytest.mark.parametrize(
    "suffix", [".json", ".npz", ".rttm", ".log", ".supervision.json", ".memory.json"]
)
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


@pytest.mark.parametrize("filename", ["existing.txt", "existing"])
def test_non_json_output_cannot_overwrite_an_existing_file(tmp_path: Path, filename: str) -> None:
    output = tmp_path / filename
    output.write_text("unique data")
    with pytest.raises(ValueError, match=".json suffix"):
        runner.supervise(argparse.Namespace(output=output))
    assert output.read_text() == "unique data"


def test_mlx_guard_terminates_owned_child_when_allocations_exceed_budget(tmp_path: Path) -> None:
    import subprocess
    import sys

    diagnostic = tmp_path / "memory.json"
    code = """
import sys,time
from pathlib import Path
from benchmarks.diarization_poc import allocation_guard
class FakeMLX:
    def get_active_memory(self) -> int: return 9*1024**3
    def get_cache_memory(self) -> int: return 0
    def get_peak_memory(self) -> int: return 9*1024**3
with allocation_guard(FakeMLX(), Path(sys.argv[1]), 8*1024**3):
    time.sleep(5)
"""
    completed = subprocess.run([sys.executable, "-c", code, str(diagnostic)], timeout=10)
    assert completed.returncode == -runner.signal.SIGTERM
    assert "MLX allocation exceeded owned budget" in diagnostic.read_text()


@pytest.mark.parametrize("reference_case", ["missing_recording", "invalid_uem", "valid"])
def test_worker_preserves_completed_inference_when_scoring_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_case: str
) -> None:
    import json
    import os
    import sys
    from types import SimpleNamespace

    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(bytes(32000))
    reference = tmp_path / "reference.rttm"
    recording_id = "other" if reference_case == "missing_recording" else "clip"
    reference.write_text(f"SPEAKER {recording_id} 1 0 0.75 <NA> <NA> A <NA> <NA>\n")
    output = tmp_path / "result.json"
    probabilities = np.full((100, 8), 0.25, dtype=np.float32)
    prediction = SimpleNamespace(
        segments=[SimpleNamespace(speaker="0", start=0.0, end=0.75)],
        speaker_probs=probabilities,
    )
    model = MagicMock()
    model.generate.return_value = prediction
    mx = MagicMock()
    mx.get_peak_memory.return_value = 1024
    mx.get_active_memory.return_value = 1024
    mx.get_cache_memory.return_value = 0
    modules = {"mlx.core": mx, "mlx_audio.vad": SimpleNamespace(load=lambda *a, **k: model)}
    monkeypatch.setattr(runner, "import_module", lambda name: modules[name])
    monkeypatch.setattr(runner, "version", lambda name: "test")
    argv = [
        "benchmark",
        "--worker",
        "--model",
        "nemotron",
        "--checkpoint",
        str(tmp_path),
        "--audio",
        str(audio),
        "--output",
        str(output),
        "--reference",
        str(reference),
        "--recording-id",
        "clip",
        "--protect-pid",
        str(os.getpid()),
    ]
    if reference_case == "invalid_uem":
        uem = tmp_path / "bad.uem"
        uem.write_text("clip 1 nan inf\n")
        argv += ["--uem", str(uem)]
    monkeypatch.setattr(sys, "argv", argv)
    if reference_case == "valid":
        runner.main()
    else:
        with pytest.raises(ValueError):
            runner.main()
    record = json.loads(output.read_text())
    assert record["raw_turns"] == [{"speaker": "0", "start": 0.0, "end": 0.75}]
    assert record["audio_sha256"]
    assert record["duration_s"] == 1
    assert record["versions"]["mlx-audio"] == "test"
    np.testing.assert_array_equal(
        np.load(output.with_suffix(".npz"))["probabilities"], probabilities
    )
    assert "SPEAKER clip 1 0.000000 0.750000" in output.with_suffix(".rttm").read_text()
    if reference_case == "valid":
        assert record["status"] == "ok"
        assert record["scores"][0]["der"] == 0
    else:
        assert record["status"] == "error"
        assert record["error_type"] == "ValueError"
        assert record["scores"] == []


@pytest.mark.parametrize("during_spawn", [False, True])
def test_sigterm_reaps_detached_worker_even_during_spawn(
    tmp_path: Path, during_spawn: bool
) -> None:
    import json
    import subprocess
    import sys
    import time

    worker_pid_file = tmp_path / "worker.pid"
    output = tmp_path / "terminated.json"
    code = """
import argparse,os,signal,subprocess,sys,time
from pathlib import Path
from benchmarks import diarization_poc as runner
original_spawn = subprocess.Popen
def spawn(*args, **kwargs):
    child = original_spawn([sys.executable, '-c', 'import time;time.sleep(60)'], **kwargs)
    Path(sys.argv[1]).write_text(str(child.pid))
    if sys.argv[3] == 'True': os.kill(os.getpid(), signal.SIGTERM)
    return child
runner.subprocess.Popen = spawn
runner.psutil.virtual_memory = lambda: argparse.Namespace(available=32*1024**3)
previous = signal.getsignal(signal.SIGTERM)
try:
    runner.supervise(argparse.Namespace(output=Path(sys.argv[2]),protect_pid=[],memory_gib=1,timeout=30))
finally:
    assert signal.getsignal(signal.SIGTERM) == previous
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", code, str(worker_pid_file), str(output), str(during_spawn)]
    )
    worker: psutil.Process | None = None
    try:
        deadline = time.monotonic() + 10
        while not worker_pid_file.exists() and time.monotonic() < deadline:
            assert parent.poll() is None, "Supervisor failed before spawning its worker"
            time.sleep(0.02)
        assert worker_pid_file.exists(), "Worker was never spawned"
        worker_pid = int(worker_pid_file.read_text())
        try:
            worker = psutil.Process(worker_pid)
        except psutil.NoSuchProcess:
            pass
        if not during_spawn:
            parent.terminate()
        assert parent.wait(timeout=15) == 128 + runner.signal.SIGTERM
        assert not psutil.pid_exists(worker_pid), "Detached worker survived supervisor termination"
        evidence = json.loads(output.with_suffix(".supervision.json").read_text())
        assert evidence["returncode"] == -runner.signal.SIGTERM
        assert evidence["stop_reason"] == str(128 + runner.signal.SIGTERM)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if worker is not None and worker.is_running():
            worker.kill()
            worker.wait(timeout=5)
