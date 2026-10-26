"""Thin adapter/typed payload validation without real ML frameworks."""

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.diarization_registry import cached_snapshot, lookup_diarizer
from src.core.diarization_result import DiarizationError, parse_payload
from src.core.mlx_nemotron_diarizer import MlxNemotronDiarizer
from src.workers.model_worker import create_diarizer


@pytest.mark.parametrize(
    "start,end", [(-1, 1), (0, 0), (0, 2), (math.nan, 1), (0, math.inf), (True, 1)]
)
def test_invalid_native_turns_are_not_silently_clipped(start: float, end: float) -> None:
    with pytest.raises(DiarizationError):
        parse_payload(
            {"duration": 1, "turns": [{"speaker": "Speaker 0", "start": start, "end": end}]}
        )


def test_overlap_and_empty_valid_payloads_remain_intact() -> None:
    data = parse_payload(
        {
            "duration": 2,
            "turns": [
                {"speaker": "A", "start": 0, "end": 1.5},
                {"speaker": "B", "start": 1, "end": 2},
            ],
        }
    )
    assert len(data.turns) == 2 and data.turns[1].start < data.turns[0].end
    assert parse_payload({"duration": 0, "turns": []}).turns == []


def test_adapter_loads_only_local_checkpoint_with_upstream_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = MagicMock()
    native.generate.return_value = SimpleNamespace(
        segments=[SimpleNamespace(speaker=2, start=0, end=1)]
    )
    load = MagicMock(return_value=native)
    monkeypatch.setattr(
        "src.core.mlx_nemotron_diarizer.import_module", lambda name: SimpleNamespace(load=load)
    )
    adapter = MlxNemotronDiarizer(tmp_path)
    adapter.load()
    adapter.load()
    load.assert_called_once_with(str(tmp_path), strict=True)
    assert adapter.diarize_file("clip.wav")[0].speaker == "Speaker 2"
    native.generate.assert_called_once_with(
        "clip.wav", threshold=0.5, min_duration=0.0, merge_gap=0.0, verbose=False
    )
    adapter.release()
    with pytest.raises(RuntimeError):
        adapter.diarize_file("clip.wav")


def test_factories_distinguish_two_mlx_backends() -> None:
    from src.core.mlx_sortformer_diarizer import MlxSortformerDiarizer

    assert isinstance(create_diarizer("nemotron-diar"), MlxNemotronDiarizer)
    assert isinstance(create_diarizer("sortformer-diar"), MlxSortformerDiarizer)


def test_cache_discovery_is_pinned_and_never_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    for name in ["config.json", "model.safetensors"]:
        (tmp_path / name).touch()

    def local(repo: str, filename: str, *, revision: str) -> str:
        calls.append((repo, filename, revision))
        return str(tmp_path / filename)

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", local)
    spec = lookup_diarizer("nemotron-diar")
    assert cached_snapshot(spec) == tmp_path
    assert {c[2] for c in calls} == {"59ed2dbfc1346dcea9d423c71306a3a2499c568f"}
    (tmp_path / "model.safetensors").unlink()
    assert cached_snapshot(spec) is None


def test_decode_budget_rejects_truncation_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.workers import diarization_worker as worker

    monkeypatch.setattr(worker, "cached_snapshot", lambda spec: tmp_path)
    monkeypatch.setattr(worker, "DECODED_BUDGET", 100)

    def decode(*args: object, **kwargs: object) -> SimpleNamespace:
        (tmp_path / "decoded.wav").write_bytes(bytes(101))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.subprocess, "run", decode)
    load = MagicMock()
    monkeypatch.setattr(worker, "MlxNemotronDiarizer", load)
    with pytest.raises(DiarizationError) as error:
        worker.execute(tmp_path / "audio.wav", tmp_path / "output.json", tmp_path)
    assert error.value.code == "input_limit" and error.value.status == 400
    load.assert_not_called()


def test_discovery_does_not_import_model_frameworks() -> None:
    import subprocess
    import sys

    code = """
import sys
from src.api.diarization import discovered_diarizers
entries=discovered_diarizers()
assert entries[0].alias=='nemotron-diar'
assert not any(name=='mlx.core' or name=='torch' or name.startswith('mlx_audio.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)
