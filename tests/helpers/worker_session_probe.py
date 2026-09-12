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


async def probe(mode: str) -> None:
    process_transport.run_worker = run_probe_worker
    timeouts = ShutdownTimeouts(graceful=0.1, terminate=0.1, kill=1)
    session = WorkerSession(lambda c: ProcessTransport(c, timeouts=timeouts), startup_timeout=0.5)
    for index in range(3):
        try:
            await session.start(WorkerConfig("probe", mode, 0.1 if mode == "idle" else 0))
        except RuntimeError:
            assert mode in {"load_error", "invalid", "timeout"}
        else:
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
                future = session.enqueue(WorkerJob(str(index), "probe.wav", params))
                if mode in {"kill", "large_queue"}:
                    future.cancel()
                elif mode == "crash":
                    try:
                        await asyncio.wait_for(future, 2)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("crash did not fail the waiter")
                else:
                    assert await asyncio.wait_for(future, 2) == "probe result"
        await session.close()
        assert not session.alive
        assert not multiprocessing.active_children(), "unreaped child"
        assert not any(t.name == "QueueFeederThread" for t in threading.enumerate()), "feeder leak"
    print(f"{mode}: three lifetimes reclaimed")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    asyncio.run(probe(sys.argv[1]))
