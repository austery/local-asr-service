"""One resident worker lifetime, including its IPC reader and completion waiters."""

import asyncio
import logging
import queue
from collections.abc import Callable, Coroutine
from contextlib import suppress

from src.workers.model_worker import WorkerJob
from src.workers.transport import ProcessTransport, WorkerConfig, WorkerTransport

_POLL_SECONDS = 0.01
_STARTUP_MESSAGE_SIZE = 2
_RESULT_MESSAGE_SIZE = 3
_TYPED_ERROR_SIZE = 4


class WorkerRemoteError(RuntimeError):
    """Exception raised in the worker, reconstructed without importing its class."""

    def __init__(self, exc_type_name: str, message: str) -> None:
        super().__init__(message)
        self.exc_type_name = exc_type_name


class WorkerSession:
    """Ensure a worker, enqueue jobs, and close it without exposing process state.

    start/close serialize lifetime changes. enqueue is synchronous and must follow
    start under the caller's admission lock. Cancelling a waiter releases its slot,
    not native inference. A replacement never starts before disposal completes.
    """

    def __init__(
        self, transport_factory: Callable[[WorkerConfig], WorkerTransport] = ProcessTransport,
        *, startup_timeout: float = 120.0,
    ) -> None:
        self._factory = transport_factory
        self._startup_timeout = startup_timeout
        self._transport: WorkerTransport | None = None
        self._config: WorkerConfig | None = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[object]] = {}
        self._lifetime_lock = asyncio.Lock()

    @property
    def alive(self) -> bool:
        return self._config is not None and self._transport is not None and self._transport.is_alive()

    @property
    def pending_count(self) -> int:
        return sum(not future.done() for future in self._pending.values())

    async def start(self, config: WorkerConfig) -> None:
        async with self._lifetime_lock:
            if self._config == config and self.alive:
                return
            await self._close()
            transport = self._factory(config)
            self._transport = transport
            try:
                transport.start()
                await self._wait_ready(transport)
            except BaseException:
                await self._close()
                raise
            self._config = config
            self._reader = asyncio.create_task(self._read_results(transport))

    def enqueue(self, job: WorkerJob) -> asyncio.Future[object]:
        if not self.alive or self._transport is None:
            raise RuntimeError("Worker is not ready")
        if job.uid in self._pending:
            raise RuntimeError(f"Duplicate in-flight request ID: {job.uid}")
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        self._pending[job.uid] = future
        future.add_done_callback(lambda done: self._forget(job.uid, done))
        try:
            self._transport.send(job)
        except BaseException:
            future.cancel()
            self._forget(job.uid, future)
            raise
        return future

    async def close(self) -> None:
        await _finish_cleanup(self._locked_close())

    async def _locked_close(self) -> None:
        async with self._lifetime_lock:
            await self._dispose()

    async def _close(self) -> None:
        await _finish_cleanup(self._dispose())

    async def _dispose(self) -> None:
        self._config = None
        self._fail_pending(RuntimeError("Worker terminated (model switch or shutdown)"))
        if self._reader is not None:
            self._reader.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        if self._transport is not None:
            await asyncio.to_thread(self._transport.close)
            self._transport = None

    async def _wait_ready(self, transport: WorkerTransport) -> None:
        deadline = asyncio.get_running_loop().time() + self._startup_timeout
        while True:
            try:
                message = transport.receive()
            except queue.Empty:
                if not transport.is_alive():
                    raise RuntimeError("Worker died during startup") from None
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError("Worker failed to start within startup timeout") from None
                await asyncio.sleep(_POLL_SECONDS)
                continue
            if message == ("READY", None):
                return
            if isinstance(message, tuple) and len(message) == _STARTUP_MESSAGE_SIZE and message[0] == "LOAD_ERROR":
                raise RuntimeError(f"Worker failed to load model: {message[1]}")
            raise RuntimeError(f"Worker sent unexpected startup message: {message!r}")

    async def _read_results(self, transport: WorkerTransport) -> None:
        try:
            while True:
                try:
                    message = transport.receive()
                except queue.Empty:
                    if not transport.is_alive():
                        raise RuntimeError("Worker process died unexpectedly") from None
                    await asyncio.sleep(_POLL_SECONDS)
                    continue
                if message == ("IDLE_EXIT", None):
                    raise RuntimeError("Worker exited while idle")
                self._deliver(message)
        except Exception as exc:
            self._config = None
            self._fail_pending(exc)
            # Dispose even with no subsequent request. start/close join this task
            # before reusing the slot; cleanup failure keeps the transport owned.
            try:
                await _finish_cleanup(asyncio.to_thread(transport.close))
            except Exception:
                # The next start/close retries disposal and exposes any failure.
                logging.getLogger(__name__).exception("Worker disposal failed; retaining transport")

    def _deliver(self, message: object) -> None:
        if not isinstance(message, tuple) or len(message) < _RESULT_MESSAGE_SIZE:
            raise RuntimeError(f"Invalid worker result message: {message!r}")
        kind, uid = message[:2]
        if not isinstance(uid, str):
            raise RuntimeError("Worker result has no request ID")
        error: RuntimeError | None
        if kind == "RESULT" and len(message) == _RESULT_MESSAGE_SIZE:
            error = None
        elif kind == "ERROR" and len(message) == _TYPED_ERROR_SIZE:
            error = WorkerRemoteError(str(message[2]), str(message[3]))
        elif kind == "ERROR" and len(message) == _RESULT_MESSAGE_SIZE:
            error = RuntimeError(str(message[2]))
        else:
            raise RuntimeError(f"Invalid worker result message: {message!r}")
        future = self._pending.pop(uid, None)
        if future is not None and not future.done():
            if error is None:
                future.set_result(message[2])
            else:
                future.set_exception(error)

    def _forget(self, uid: str, future: asyncio.Future[object]) -> None:
        if self._pending.get(uid) is future:
            self._pending.pop(uid)

    def _fail_pending(self, error: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()


async def _finish_cleanup(operation: Coroutine[object, object, None]) -> None:
    """Delay caller cancellation until owned cleanup has finished, including retries."""
    cleanup = asyncio.create_task(operation)
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    cleanup.result()
    if cancelled:
        raise asyncio.CancelledError
