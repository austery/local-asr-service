"""Small spawn-safe subprocess target for testing the real process transport."""

from __future__ import annotations

import os
import queue
import signal
from multiprocessing import Queue

from src.workers.model_worker import WorkerJob


def run_probe_worker(
    jobs: Queue[WorkerJob | None], results: Queue[object], engine_type: str,
    model_id: str, idle_timeout: float,
) -> None:
    if model_id == "load_error":
        results.put(("LOAD_ERROR", "probe load error"))
        return
    if model_id == "invalid":
        results.put(("BOGUS", None))
        return
    if model_id in {"kill", "large_queue"}:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if model_id != "timeout":
        results.put(("READY", None))
    if model_id in {"timeout", "kill", "large_queue"}:
        while True:
            signal.pause()
    while True:
        try:
            job = jobs.get(timeout=idle_timeout or None)
        except queue.Empty:
            results.put(("IDLE_EXIT", None))
            return
        if job is None:
            return
        if model_id == "crash":
            os._exit(7)
        results.put(("RESULT", job.uid, "probe result"))
