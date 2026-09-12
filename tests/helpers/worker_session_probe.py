"""Watchdog-isolated real-process acceptance; invoked by pytest in a fresh Python."""

import asyncio
import multiprocessing
import os
import signal
import sys
import threading

import src.workers.transport as process_transport
from src.services.worker_session import WorkerSession
from src.workers.model_worker import WorkerJob
from src.workers.transport import ProcessTransport, ShutdownTimeouts, WorkerConfig
from tests.helpers.session_worker import run_probe_worker


async def heartbeat(ticks: list[int]) -> None:
    while True:
        ticks.append(1)
        await asyncio.sleep(0.01)


async def probe(mode: str) -> None:
    process_transport.run_worker = run_probe_worker
    timeouts = ShutdownTimeouts(graceful=0.1, terminate=0.1, kill=1)
    session = WorkerSession(lambda c: ProcessTransport(c, timeouts=timeouts), startup_timeout=0.5)
    ticks: list[int] = []
    ticker = asyncio.create_task(heartbeat(ticks))
    for _ in range(3):
        ticks_before = len(ticks)
        try:
            await session.start(WorkerConfig("probe", mode, 0.1 if mode == "idle" else 0))
        except RuntimeError as exc:
            if mode == "partial_start_control":
                assert "failed to load model" in str(exc)
            elif mode == "partial_start_wait":
                assert "timeout" in str(exc)
            assert mode in {"load_error", "invalid", "timeout", "partial_start_exit", "partial_start_wait", "partial_start_control"}
        else:
            await exercise_started(session, mode)
        if mode.endswith("wait"):
            assert len(ticks) - ticks_before >= 2, "heartbeat stalled during partial frame"
        await session.close()
        assert not session.alive
        assert not multiprocessing.active_children(), "unreaped child"
        assert not any(t.name in {"QueueFeederThread", "ASRResultReader"} for t in threading.enumerate()), "IPC thread leak"
    ticker.cancel()
    assert ticks, "event loop did not make progress"
    print(f"{mode}: three lifetimes reclaimed")


async def exercise_started(session: WorkerSession, mode: str) -> None:
    if mode == "stopped":
        await asyncio.sleep(0.05)
        child, = multiprocessing.active_children()
        assert child.pid is not None
        os.kill(child.pid, signal.SIGSTOP)
    elif mode == "idle":
        async with asyncio.timeout(2):
            while session.alive:
                await asyncio.sleep(0.01)
    else:
        params: dict[str, object] = {"payload": "x" * 2_000_000} if mode == "large_queue" else {}
        future = session.enqueue(WorkerJob("probe-result", "probe.wav", params))
        if mode in {"kill", "large_queue"}:
            future.cancel()
        elif mode in {"crash", "partial_result_exit", "partial_result_wait"}:
            try:
                await asyncio.wait_for(future, 2)
            except (RuntimeError, TimeoutError) as exc:
                assert isinstance(exc, TimeoutError if mode == "partial_result_wait" else RuntimeError)
            else:
                raise AssertionError("crash did not fail the waiter")
        else:
            assert await asyncio.wait_for(future, 2) == ("x" * 2_000_000 if mode == "partial_result_control" else "probe result")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    asyncio.run(probe(sys.argv[1]))
