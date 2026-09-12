"""Real subprocess lifetime tests, each protected by an outer process watchdog."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", [
    "normal", "idle", "crash", "load_error", "invalid", "timeout", "kill", "large_queue", "stopped",
    "partial_start_exit", "partial_start_wait", "partial_result_exit", "partial_result_wait",
    "partial_start_control", "partial_result_control",
])
def test_real_process_and_feeder_reclamation(mode: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tests.helpers.worker_session_probe", mode],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "three lifetimes reclaimed" in result.stdout
    assert "resource_tracker" not in result.stderr
    assert "never retrieved" not in result.stderr
