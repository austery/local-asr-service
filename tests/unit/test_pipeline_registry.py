import subprocess
import sys
from pathlib import Path

import pytest

from src.core.pipeline_registry import list_all_profiles, lookup_profile


def test_should_reject_retired_qwen3_sortformer_profile() -> None:
    with pytest.raises(KeyError, match="Unknown pipeline profile"):
        lookup_profile("qwen3-sortformer")


def test_should_not_advertise_retired_profiles() -> None:
    assert list_all_profiles() == []


def test_should_raise_for_unknown_profile() -> None:
    with pytest.raises(KeyError, match="Unknown pipeline profile"):
        lookup_profile("missing-profile")


def test_retired_probe_exits_before_reading_media_or_writing_output(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "must-not-be-created"
    result = subprocess.run(
        [sys.executable, str(root / "scripts/probe_qwen3_sortformer_longform.py"),
         "--audio", str(tmp_path / "missing.wav"), "--output", str(output)],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode != 0
    assert "qwen3-sortformer is retired" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()
