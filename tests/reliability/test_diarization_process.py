"""Real owned processes with deterministic lightweight workers, never ML inference."""

import asyncio
import sys
from pathlib import Path

import psutil
import pytest

from src.core.diarization_result import DiarizationError
from src.services.diarization_session import DiarizationSession


@pytest.fixture
def worker(tmp_path: Path) -> Path:
    source = tmp_path / "worker.py"
    source.write_text("""
import json,os,sys,time
from pathlib import Path
mode,audio,output,checkpoint=sys.argv[1:]
output=Path(output)
output.with_suffix('.pid').write_text(str(os.getpid()))
if mode=='child':
    import subprocess
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    output.with_suffix('.child').write_text(str(child.pid))
    time.sleep(60)
elif mode=='hang':time.sleep(60)
elif mode=='badmemory':
    output.with_suffix('.memory.json').write_text('invalid JSON')
    raise SystemExit(1)
elif mode=='bad':output.write_text('not JSON')
elif mode=='bounds':output.write_text(json.dumps({'duration':1,'turns':[{'speaker':'A','start':0,'end':2}]}))
else:output.write_text(json.dumps({'duration':1,'turns':[]}))
""")
    return source


async def wait_pid(output: Path) -> int:
    for _ in range(200):
        if output.with_suffix(".pid").exists():
            return int(output.with_suffix(".pid").read_text())
        await asyncio.sleep(0.01)
    raise AssertionError("Owned worker never started")


@pytest.mark.parametrize("mode", ["ok", "bad", "bounds", "hang", "badmemory"])
async def test_result_errors_and_deadline_reap_owned_worker(
    worker: Path, tmp_path: Path, mode: str
) -> None:
    output = tmp_path / "output.json"
    session = DiarizationSession(
        (sys.executable, str(worker), mode), timeout=0.3 if mode == "hang" else 3, reserve_bytes=0
    )
    if mode == "ok":
        assert (await session.run(worker, output, tmp_path)).turns == []
    else:
        with pytest.raises(DiarizationError) as error:
            await session.run(worker, output, tmp_path)
        assert error.value.status == (504 if mode == "hang" else 502)
    assert not psutil.pid_exists(await wait_pid(output))
    await session.close()


async def test_repeated_cancellation_reaps_started_worker(worker: Path, tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    session = DiarizationSession((sys.executable, str(worker), "hang"), reserve_bytes=0)
    task = asyncio.create_task(session.run(worker, output, tmp_path))
    pid = await wait_pid(output)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not psutil.pid_exists(pid)


async def test_cancellation_during_spawn_retains_child_until_disposal(
    worker: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.json"
    entered, release = asyncio.Event(), asyncio.Event()
    original = asyncio.create_subprocess_exec
    pids = []

    async def spawn(*args: str, **kwargs: object) -> asyncio.subprocess.Process:
        process = await original(*args, **kwargs)
        pids.append(process.pid)
        entered.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    session = DiarizationSession((sys.executable, str(worker), "hang"), reserve_bytes=0)
    task = asyncio.create_task(session.run(worker, output, tmp_path))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not psutil.pid_exists(pids[0])


async def test_low_available_memory_rejects_before_start(
    worker: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        "src.services.diarization_session.psutil.virtual_memory",
        lambda: SimpleNamespace(available=0),
    )
    output = tmp_path / "output.json"
    with pytest.raises(DiarizationError) as error:
        await DiarizationSession((sys.executable, str(worker), "hang")).run(
            worker, output, tmp_path
        )
    assert error.value.status == 503
    assert not output.with_suffix(".pid").exists()


async def test_cancellation_disposes_decode_descendant_group(worker: Path, tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    session = DiarizationSession((sys.executable, str(worker), "child"), reserve_bytes=0)
    task = asyncio.create_task(session.run(worker, output, tmp_path))
    pid = await wait_pid(output)
    for _ in range(200):
        if output.with_suffix(".child").exists():
            break
        await asyncio.sleep(0.01)
    child = int(output.with_suffix(".child").read_text())
    try:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not psutil.pid_exists(pid)
        assert not psutil.pid_exists(child)
    finally:
        await session.close()


async def test_spawn_failure_is_typed_and_session_is_disposable(tmp_path: Path) -> None:
    session = DiarizationSession((str(tmp_path / "missing-python"),), reserve_bytes=0)
    with pytest.raises(DiarizationError) as error:
        await session.run(tmp_path / "input", tmp_path / "result.json", tmp_path)
    assert error.value.code == "worker_failed" and error.value.status == 502
    await session.close()
