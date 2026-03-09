import asyncio
import logging
import multiprocessing
import os
import queue as _stdlib_queue
import shutil
import tempfile
from typing import Any

from fastapi import UploadFile

from src.core.base_engine import EngineCapabilities
from src.core.model_registry import ModelSpec
from src.workers.model_worker import WorkerJob, run_worker


class TranscriptionService:
    """
    Manages a ModelWorker child process via multiprocessing.Queue IPC.

    The worker subprocess self-terminates on idle timeout, allowing the OS to
    reclaim ML framework memory (MPS/CUDA) that cannot be freed in-process.
    """

    def __init__(
        self,
        engine_type: str,
        model_id: str,
        max_queue_size: int = 50,
        initial_model_spec: ModelSpec | None = None,
        idle_timeout: int = 60,
    ) -> None:
        self._engine_type = engine_type
        self._model_id = model_id
        self._current_model_spec = initial_model_spec
        self._idle_timeout = idle_timeout
        self._max_queue_size = max_queue_size

        self._worker: multiprocessing.Process | None = None
        self._job_queue: multiprocessing.Queue[WorkerJob | None] | None = None
        self._result_queue: multiprocessing.Queue[tuple[Any, ...]] | None = None
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._temp_dirs: dict[str, str] = {}
        self._spawn_lock: asyncio.Lock = asyncio.Lock()
        self._result_reader_task: asyncio.Task[None] | None = None
        self.is_running = False

        self.logger = logging.getLogger(__name__)

    @property
    def current_model_spec(self) -> ModelSpec | None:
        return self._current_model_spec

    @property
    def model_loaded(self) -> bool:
        """True if worker subprocess is alive."""
        return self._worker is not None and self._worker.is_alive()

    @property
    def capabilities(self) -> EngineCapabilities:
        """Return capabilities from current model spec — no live engine needed."""
        if self._current_model_spec is not None:
            return self._current_model_spec.capabilities
        return EngineCapabilities()

    @property
    def queue_size(self) -> int:
        """Number of jobs currently in-flight (queued or processing)."""
        return len(self._pending)

    @property
    def max_queue_size(self) -> int:
        """Maximum allowed concurrent in-flight jobs."""
        return self._max_queue_size

    async def start_worker(self) -> None:
        """Mark service as running. Worker spawns lazily on first request."""
        self.is_running = True
        self.logger.info("🚦 Service initialized (worker spawns on first request).")

    async def stop_worker(self) -> None:
        """Gracefully stop worker subprocess and result reader."""
        self.is_running = False
        await self._shutdown_worker()
        if self._result_reader_task and not self._result_reader_task.done():
            self._result_reader_task.cancel()
            try:
                await self._result_reader_task
            except asyncio.CancelledError:
                pass

    async def submit(
        self,
        file: UploadFile,
        params: dict[str, Any],
        request_id: str = "unknown",
        model_spec: ModelSpec | None = None,
    ) -> str | dict[str, Any]:
        if len(self._pending) >= self._max_queue_size:
            self.logger.warning(f"[{request_id}] Queue full, rejecting request")
            raise RuntimeError("Service busy: Queue is full.")

        temp_dir = tempfile.mkdtemp(prefix="asr_task_")
        try:
            file_ext = os.path.splitext(file.filename or "upload.wav")[1] or ".wav"
            temp_path = os.path.join(temp_dir, f"original{file_ext}")
            with open(temp_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)

            loop = asyncio.get_running_loop()
            future: asyncio.Future[str | dict[str, Any]] = loop.create_future()

            async with self._spawn_lock:
                if model_spec is not None and model_spec != self._current_model_spec:
                    await self._switch_worker(model_spec)
                elif not self.model_loaded:
                    await self._spawn_worker()

                if self._job_queue is None:
                    raise RuntimeError("Job queue is None after successful spawn — this is a bug")

                # Register AFTER the worker is ready so that _shutdown_worker (called
                # during a concurrent model switch) does not cancel this request's
                # future or delete its temp dir before the job is even queued.
                self._pending[request_id] = future
                self._temp_dirs[request_id] = temp_dir

                self._job_queue.put_nowait(WorkerJob(
                    uid=request_id,
                    temp_file_path=temp_path,
                    params=params,
                ))

            return await future

        except BaseException:
            self._pending.pop(request_id, None)
            self._temp_dirs.pop(request_id, None)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    async def _spawn_worker(self, model_spec: ModelSpec | None = None) -> None:
        for old_q in (self._job_queue, self._result_queue):
            if old_q is not None:
                try:
                    old_q.close()
                    old_q.join_thread()
                except Exception:
                    pass

        self._job_queue = multiprocessing.Queue()
        self._result_queue = multiprocessing.Queue()

        effective_spec = model_spec or self._current_model_spec
        engine_type = effective_spec.engine_type if effective_spec else self._engine_type
        model_id = effective_spec.model_id if effective_spec else self._model_id

        self._worker = multiprocessing.Process(
            target=run_worker,
            args=(
                self._job_queue,
                self._result_queue,
                engine_type,
                model_id,
                self._idle_timeout,
            ),
            daemon=True,
        )
        self._worker.start()

        loop = asyncio.get_running_loop()
        try:
            msg: tuple[Any, ...] = await asyncio.wait_for(
                loop.run_in_executor(None, self._result_queue.get),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            self._worker.terminate()
            self._worker = None
            raise RuntimeError("Worker failed to start within 120s timeout.")

        if msg[0] == "LOAD_ERROR":
            self._worker.terminate()
            self._worker = None
            raise RuntimeError(f"Worker failed to load model: {msg[1]}")

        if msg[0] != "READY":
            self._worker.terminate()
            self._worker = None
            raise RuntimeError(f"Worker sent unexpected startup message: {msg!r}")

        if model_spec is not None:
            self._current_model_spec = model_spec

        if self._result_reader_task is None or self._result_reader_task.done():
            self._result_reader_task = asyncio.create_task(self._result_reader_loop())

        self.logger.info("✅ Worker subprocess ready.")

    async def _switch_worker(self, new_spec: ModelSpec) -> None:
        old_alias = self._current_model_spec.alias if self._current_model_spec else "unknown"
        self.logger.info(f"🔄 Switching worker model: {old_alias} → {new_spec.alias}")
        await self._shutdown_worker()
        await self._spawn_worker(new_spec)

    async def _shutdown_worker(self) -> None:
        for uid, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(RuntimeError("Worker terminated (model switch or shutdown)"))
        self._pending.clear()
        for temp_dir in self._temp_dirs.values():
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()

        if self._worker is None:
            return

        if self._job_queue is not None:
            try:
                self._job_queue.put(None)
            except Exception:
                pass

        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._worker.join(timeout=5)),  # type: ignore[union-attr]
                timeout=6.0,
            )
        except (asyncio.TimeoutError, Exception):
            pass

        if self._worker.is_alive():
            self._worker.terminate()

        self._worker = None

    async def _result_reader_loop(self) -> None:
        """Polls result_queue (non-blocking) every 50ms, resolves pending Futures."""
        _liveness_ticks = 0
        while self.is_running:
            if self._result_queue is None:
                await asyncio.sleep(0.05)
                continue
            try:
                msg = self._result_queue.get_nowait()
            except _stdlib_queue.Empty:
                _liveness_ticks += 1
                if _liveness_ticks >= 20:  # check liveness ~every 1s
                    _liveness_ticks = 0
                    if self._worker is not None and not self._worker.is_alive() and self._pending:
                        exit_code = self._worker.exitcode
                        self.logger.error(
                            "Worker process died unexpectedly (exit code %s) with %d pending job(s) — failing all",
                            exit_code,
                            len(self._pending),
                        )
                        self._fail_all_pending(
                            RuntimeError(f"Worker process died unexpectedly (exit code {exit_code})")
                        )
                        self._worker = None
                await asyncio.sleep(0.05)
                continue

            try:
                msg_type: str = msg[0]
                if msg_type == "RESULT":
                    self._resolve_future(msg[1], result=msg[2])
                elif msg_type == "ERROR":
                    self._resolve_future(msg[1], error=RuntimeError(msg[2]))
                elif msg_type == "IDLE_EXIT":
                    self.logger.info("💤 Worker exited due to idle timeout — memory reclaimed by OS")
                    if self._worker:
                        worker = self._worker
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, lambda: worker.join(timeout=1))
                    self._worker = None
            except Exception:
                self.logger.exception("Unexpected error processing IPC message: %r", msg)

    def _fail_all_pending(self, error: Exception) -> None:
        """Fail all in-flight futures with the given error (e.g., after worker crash)."""
        for uid, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(error)
        self._pending.clear()
        for temp_dir in self._temp_dirs.values():
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()

    def _resolve_future(self, uid: str, result: Any = None, error: Exception | None = None) -> None:
        future = self._pending.pop(uid, None)
        self._cleanup_temp(uid)
        if future is None or future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    def _cleanup_temp(self, uid: str) -> None:
        temp_dir = self._temp_dirs.pop(uid, None)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
