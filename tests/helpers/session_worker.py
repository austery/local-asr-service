"""Small spawn-safe subprocess target for testing the real process transport."""

from __future__ import annotations

import os
import queue
import signal
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Protocol, cast
from unittest.mock import patch

from src.workers.model_worker import WorkerJob


def run_probe_worker(
    jobs: Queue[WorkerJob | None], results: Queue[object], engine_type: str,
    model_id: str, idle_timeout: float,
) -> None:
    if model_id.startswith("partial_start"):
        write_partial_result(results, model_id)
        return
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
        if model_id.startswith("partial_result"):
            write_partial_result(results, model_id)
            return
        if model_id == "crash":
            os._exit(7)
        results.put(("RESULT", job.uid, "probe result"))


class _RawResultQueue(Protocol):
    _writer: Connection


class _FrameWriter(Protocol):
    def _send(self, data: bytes | memoryview) -> None: ...


def write_partial_result(results: Queue[object], mode: str) -> None:
    # Use the genuine Queue serializer; interrupt its separate frame-header write.
    writer = cast(_RawResultQueue, results)._writer
    send = cast(_FrameWriter, writer)._send

    def send_frame(data: bytes | memoryview) -> None:
        send(data)
        if len(data) == 4 and not mode.endswith("control"):
            if mode.endswith("exit"):
                os._exit(9)
            while True:
                signal.pause()

    with patch.object(writer, "_send", side_effect=send_frame):
        message = ("LOAD_ERROR", "x" * 2_000_000) if "start" in mode else ("RESULT", "probe-result", "x" * 2_000_000)
        results.put(message)
        results.close()
        results.join_thread()
