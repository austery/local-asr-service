"""Process/queue Adapter used by the resident worker session."""

from __future__ import annotations

import multiprocessing
import queue
import threading
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Protocol, cast

from src.workers.model_worker import WorkerJob, run_worker


@dataclass(frozen=True)
class WorkerConfig:
    engine_type: str
    model_id: str
    idle_timeout: float


@dataclass(frozen=True)
class ShutdownTimeouts:
    graceful: float = 5.0
    terminate: float = 3.0
    kill: float = 2.0
    reader: float = 2.0


class WorkerTransport(Protocol):
    """Internal Seam: admission, complete-message polling, synchronous disposal.

    receive raises queue.Empty while awaiting data and RuntimeError on terminal
    failure, after delivering prior complete messages. Process death alone must
    not discard frames still being decoded by the receiver.
    """

    def start(self) -> None: ...
    def send(self, job: WorkerJob) -> None: ...
    def receive(self) -> object: ...
    def is_alive(self) -> bool: ...
    def close(self) -> None: ...


class _QueueInternals(Protocol):
    """CPython Queue disposal details absent from the public typing stubs."""

    _ignore_epipe: bool
    _reader: Connection
    _writer: Connection


@dataclass(frozen=True)
class _ReceiveFailure:
    error: Exception


class ProcessTransport:
    """Own both queues and the child until it has been reaped and IPC disposed.

    close runs off the event loop. A failed close retains ownership for retry;
    callers must not start a replacement until close succeeds.
    """

    def __init__(
        self, config: WorkerConfig, *, timeouts: ShutdownTimeouts = ShutdownTimeouts(),
    ) -> None:
        self._jobs: multiprocessing.Queue[WorkerJob | None] = multiprocessing.Queue()
        # CPython captures this flag when it starts the feeder. Once the child
        # is reaped, closing our reader must stop a blocked writer with EPIPE.
        # This private Queue detail stays inside this Adapter and is exercised
        # by spawn-process probes (including a child killed holding _rlock).
        cast(_QueueInternals, self._jobs)._ignore_epipe = True
        self._results: multiprocessing.Queue[object] = multiprocessing.Queue()
        self._process = multiprocessing.Process(
            target=run_worker,
            args=(self._jobs, self._results, config.engine_type, config.model_id, config.idle_timeout),
            daemon=True,
        )
        self._timeouts = timeouts
        self._closed = False
        self._received: queue.Queue[object] = queue.Queue()
        self._receive_thread: threading.Thread | None = None

    def start(self) -> None:
        self._process.start()
        # Only the child writes results. Closing the parent's duplicate writer
        # makes child exit produce EOF, including midway through a framed message.
        cast(_QueueInternals, self._results)._writer.close()
        reader = threading.Thread(target=self._receive_messages, name="ASRResultReader", daemon=True)
        reader.start()
        self._receive_thread = reader

    def send(self, job: WorkerJob) -> None:
        self._jobs.put_nowait(job)

    def receive(self) -> object:
        message = self._received.get_nowait()
        if isinstance(message, _ReceiveFailure):
            raise RuntimeError("Worker result channel closed or failed") from message.error
        return message

    def _receive_messages(self) -> None:
        # Queue.get_nowait only polls for the beginning of a frame; decoding its
        # body can still block. Only this owned thread reads the process pipe.
        try:
            while True:
                self._received.put(self._results.get())
        except Exception as exc:
            self._received.put(_ReceiveFailure(exc))

    def is_alive(self) -> bool:
        return not self._closed and self._process.is_alive()

    def close(self) -> None:
        if self._closed:
            return
        self._reap()
        self._join_receiver()
        # No live consumer remains. Reading abandoned data can deadlock on a
        # lock held by the killed child, or on a partially consumed frame.
        # Close the reader instead: the parent's feeder exits on EPIPE, and
        # queue.close wakes it if it was waiting for work. Never cancel its join.
        for channel in (self._jobs, self._results):
            internals = cast(_QueueInternals, channel)
            internals._reader.close()
            channel.close()
            channel.join_thread()
            internals._writer.close()
        self._process.close()
        self._closed = True
        del self._jobs
        del self._results

    def _join_receiver(self) -> None:
        # Reaping closes the last writer and wakes a partial/idle read with EOF.
        # Never close a descriptor underneath a live receive thread or abandon it.
        if self._receive_thread is not None:
            self._receive_thread.join(timeout=self._timeouts.reader)
            if self._receive_thread.is_alive():
                raise RuntimeError("Worker result reader did not exit; retaining transport ownership")
            self._receive_thread = None

    def _reap(self) -> None:
        if self._process.pid is None:
            return
        if self._process.is_alive():
            self._jobs.put_nowait(None)
        self._process.join(timeout=self._timeouts.graceful)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=self._timeouts.terminate)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=self._timeouts.kill)
        if self._process.is_alive():
            raise RuntimeError("Worker survived SIGKILL; refusing to release its process handle")
