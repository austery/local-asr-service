import asyncio
import logging
import multiprocessing
import os
import queue as _stdlib_queue
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Literal

from fastapi import UploadFile

from src.config import (
    APPLE_SPEECH_MAX_CONCURRENCY,
    APPLE_SPEECH_WORKER_PATH,
    APPLE_SPEECH_WORKER_TIMEOUT_SEC,
)
from src.core.apple_speech_engine import AppleSpeechEngine, AppleSpeechEngineConfig
from src.core.apple_speech_port import AppleSpeechModule
from src.core.base_engine import EngineCapabilities
from src.core.model_registry import ModelSpec
from src.services.execution import (
    ExecutionPlan,
    ExecutionResult,
    TranscriptionResult,
    TranscriptionResultDict,
)
from src.workers.model_worker import WorkerJob, run_worker

WorkerStartupMessage = tuple[Literal["READY"], None] | tuple[Literal["LOAD_ERROR"], str]
WorkerResultMessage = (
    tuple[Literal["RESULT"], str, object]
    | tuple[Literal["ERROR"], str, str]
    | tuple[Literal["ERROR"], str, str, str]
    | tuple[Literal["IDLE_EXIT"], None]
)
WorkerMessage = WorkerStartupMessage | WorkerResultMessage


class WorkerRemoteError(RuntimeError):
    """Exception raised by the worker process and reconstructed in the parent."""

    def __init__(self, exc_type_name: str, message: str) -> None:
        super().__init__(message)
        self.exc_type_name = exc_type_name


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
        self._result_queue: multiprocessing.Queue[WorkerMessage] | None = None
        self._pending: dict[str, asyncio.Future[object]] = {}
        self._temp_dirs: dict[str, str] = {}
        self._spawn_lock: asyncio.Lock = asyncio.Lock()
        self._result_reader_task: asyncio.Task[None] | None = None
        self._sidecar_pending: set[str] = set()
        self._sidecar_semaphore = asyncio.Semaphore(APPLE_SPEECH_MAX_CONCURRENCY)
        self._apple_speech_engines: dict[str, AppleSpeechEngine] = {}
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
        """Number of jobs currently in-flight across worker and sidecar paths."""
        return self._active_job_count()

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
            with suppress(asyncio.CancelledError):
                await self._result_reader_task

    async def submit(
        self,
        file: UploadFile,
        params: dict[str, object],
        request_id: str = "unknown",
        model_spec: ModelSpec | None = None,
    ) -> ExecutionResult:
        params = dict(params)
        explicit_plan = ExecutionPlan.select(model_spec, self._model_id) if model_spec is not None else None
        if explicit_plan is not None:
            explicit_plan.validate(params)
        if self._active_job_count() >= self._max_queue_size:
            self.logger.warning(f"[{request_id}] Queue full, rejecting request")
            raise RuntimeError("Service busy: Queue is full.")

        temp_dir = tempfile.mkdtemp(prefix="asr_task_")
        try:
            file_ext = os.path.splitext(file.filename or "upload.wav")[1] or ".wav"
            temp_path = os.path.join(temp_dir, f"original{file_ext}")
            with open(temp_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)

            # Explicit sidecar requests do not wait for resident-worker startup.
            if explicit_plan is not None and self._is_apple_speech_spec(explicit_plan.spec):
                result = await self._submit_apple_speech_job(
                    temp_file_path=temp_path, params=params, request_id=request_id,
                )
                return ExecutionResult(self._coerce_transcription_result(result), explicit_plan.model)
            else:
                return await self._submit_resident_job(
                    temp_file_path=temp_path,
                    params=params,
                    request_id=request_id,
                    plan=explicit_plan,
                    temp_dir=temp_dir,
                )
        finally:
            self._discard_request_state(request_id)
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _submit_resident_job(
        self,
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
        plan: ExecutionPlan | None,
        temp_dir: str | None = None,
    ) -> ExecutionResult:
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        try:
            # Select passthrough under the same lock that protects model switches.
            # Release it after enqueue, so waiting for inference never serializes
            # admission of other requests or hides them from queue_size.
            async with self._spawn_lock:
                selected = plan or ExecutionPlan.select(self._current_model_spec, self._model_id)
                selected.validate(params)
                route_to_sidecar = self._is_apple_speech_spec(selected.spec)
                if not route_to_sidecar:
                    await self._enqueue_worker_job(
                        future=future,
                        temp_file_path=temp_file_path,
                        params=params,
                        request_id=request_id,
                        model_spec=plan.spec if plan is not None else None,
                        temp_dir=temp_dir,
                    )
            if route_to_sidecar:
                result = await self._submit_apple_speech_job(
                    temp_file_path=temp_file_path, params=params, request_id=request_id,
                )
            else:
                result = await future
            return ExecutionResult(self._coerce_transcription_result(result), selected.model)
        except BaseException:
            self._discard_request_state(request_id)
            raise

    def _active_job_count(self) -> int:
        return len(self._pending) + len(self._sidecar_pending)

    @staticmethod
    def _is_apple_speech_spec(model_spec: ModelSpec | None) -> bool:
        return model_spec is not None and model_spec.engine_type == "apple-speech"

    def _get_apple_speech_engine(self) -> AppleSpeechEngine:
        module: AppleSpeechModule = "speechTranscriber"
        engine = self._apple_speech_engines.get(module)
        if engine is None:
            engine = AppleSpeechEngine.from_config(
                AppleSpeechEngineConfig(
                    worker_path=Path(APPLE_SPEECH_WORKER_PATH),
                    timeout_seconds=APPLE_SPEECH_WORKER_TIMEOUT_SEC,
                ),
                module=module,
            )
            engine.load()
            self._apple_speech_engines[module] = engine
        return engine

    async def _submit_apple_speech_job(
        self,
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
    ) -> object:
        if self._active_job_count() >= self._max_queue_size:
            self.logger.warning(f"[{request_id}] Queue full, rejecting Apple Speech request")
            raise RuntimeError("Service busy: Queue is full.")

        self._sidecar_pending.add(request_id)
        try:
            async with self._sidecar_semaphore:
                engine = self._get_apple_speech_engine()
                language = params.get("language", "auto")
                output_format = params.get("output_format", "json")
                with_timestamp = params.get("with_timestamp", False)
                return await asyncio.to_thread(
                    engine.transcribe_file,
                    temp_file_path,
                    language=language if isinstance(language, str) else "auto",
                    output_format=output_format if isinstance(output_format, str) else "json",
                    with_timestamp=with_timestamp if isinstance(with_timestamp, bool) else False,
                )
        finally:
            self._sidecar_pending.discard(request_id)

    async def _enqueue_worker_job(
        self,
        future: asyncio.Future[object],
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
        model_spec: ModelSpec | None,
        temp_dir: str | None,
    ) -> None:
        """Enqueue a transcription while the caller holds _spawn_lock."""
        if self._active_job_count() >= self._max_queue_size:
            self.logger.warning(f"[{request_id}] Queue full, rejecting worker job")
            raise RuntimeError("Service busy: Queue is full.")

        if model_spec is not None and model_spec != self._current_model_spec:
            await self._switch_worker(model_spec)
        elif not self.model_loaded:
            await self._spawn_worker()

        if self._job_queue is None:
            raise RuntimeError("Job queue is None after successful spawn — this is a bug")

        # Register only after the worker is ready: switching shuts down the old
        # worker and clears its requests, which must not include this new job.
        self._pending[request_id] = future
        if temp_dir is not None:
            self._temp_dirs[request_id] = temp_dir
        self._job_queue.put_nowait(WorkerJob(
            uid=request_id,
            temp_file_path=temp_file_path,
            params=params,
            job_kind="transcribe",
            requested_model_spec_alias=model_spec.alias if model_spec is not None else None,
        ))

    @staticmethod
    def _coerce_transcription_result(result: object) -> TranscriptionResult:
        if isinstance(result, str):
            return result
        if not isinstance(result, dict):
            raise TypeError(f"Expected transcription result as str or dict, got {type(result).__name__}")

        coerced: TranscriptionResultDict = {}
        text = result.get("text", "")
        coerced["text"] = text if isinstance(text, str) else ""

        segments = result.get("segments")
        if segments is None or isinstance(segments, list):
            coerced["segments"] = segments
        else:
            raise TypeError("Expected transcription result 'segments' to be a list or None")

        duration = result.get("duration")
        if isinstance(duration, int | float) and not isinstance(duration, bool):
            coerced["duration"] = float(duration)

        language = result.get("language")
        if isinstance(language, str):
            coerced["language"] = language

        return coerced


    def _discard_request_state(self, request_id: str) -> None:
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.cancel()
        temp_dir = self._temp_dirs.pop(request_id, None)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


    async def _stop_result_reader_task(self) -> None:
        task = self._result_reader_task
        if task is None:
            return
        if task.done():
            self._result_reader_task = None
            return
        if task is asyncio.current_task():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        self._result_reader_task = None

    async def _spawn_worker(self, model_spec: ModelSpec | None = None) -> None:
        await self._stop_result_reader_task()

        for old_q in (self._job_queue, self._result_queue):
            if old_q is not None:
                try:
                    old_q.close()
                    old_q.join_thread()
                except Exception:
                    self.logger.warning("Failed to close stale worker queue during respawn", exc_info=True)

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
            msg = await asyncio.wait_for(
                loop.run_in_executor(None, self._result_queue.get),
                timeout=120.0,
            )
        except TimeoutError as exc:
            self._worker.terminate()
            self._worker = None
            raise RuntimeError("Worker failed to start within 120s timeout.") from exc

        if not isinstance(msg, tuple) or len(msg) != 2 or not isinstance(msg[0], str):
            self._worker.terminate()
            self._worker = None
            raise RuntimeError(f"Worker sent unexpected startup message: {msg!r}")

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
        # Apple Speech is sidecar-only: no resident subprocess to spawn. Releasing
        # the previous resident worker above is still required for memory safety.
        if new_spec.engine_type == "apple-speech":
            self._current_model_spec = new_spec
            return
        await self._spawn_worker(new_spec)

    async def _shutdown_worker(self) -> None:
        """Gracefully shutdown worker subprocess and clean up IPC resources.

        Critical: This method MUST properly reap the child process and close
        all multiprocessing.Queue instances to avoid zombie processes and
        resource_tracker hangs that prevent the main process from exiting.
        """
        for _uid, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(RuntimeError("Worker terminated (model switch or shutdown)"))
        self._pending.clear()
        for temp_dir in self._temp_dirs.values():
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()

        if self._worker is None:
            return

        # 1. Send shutdown sentinel (allows graceful engine.release())
        if self._job_queue is not None:
            try:
                self._job_queue.put(None, timeout=1.0)
            except Exception as exc:
                self.logger.warning("Failed to send shutdown sentinel to worker: %s", exc)

        # 2. Wait for graceful exit
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._worker.join(timeout=5)),  # type: ignore[union-attr]
                timeout=6.0,
            )
        except TimeoutError:
            self.logger.warning("Worker did not exit gracefully within 5s timeout")

        # 3. Force-kill if still alive
        if self._worker.is_alive():
            self.logger.warning("Sending SIGTERM to worker subprocess")
            self._worker.terminate()

            # CRITICAL: join() after terminate() to reap the zombie process.
            # Without this, the process becomes a zombie and resource_tracker
            # cannot exit, causing the main process to hang indefinitely.
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self._worker.join(timeout=3)),  # type: ignore[union-attr]
                    timeout=4.0,
                )
            except TimeoutError:
                self.logger.error("Worker did not respond to SIGTERM within 3s, using SIGKILL")
                self._worker.kill()  # SIGKILL (last resort)
                # Final join (no timeout wrap — must wait for kill to complete)
                await loop.run_in_executor(None, lambda: self._worker.join(timeout=2))  # type: ignore[union-attr]

        self._worker = None

        # 4. Clean up IPC queues to stop feeder threads
        # multiprocessing.Queue has a background "feeder thread" that serializes
        # and writes data to the underlying pipe. If not explicitly cleaned up,
        # this thread may block waiting for the pipe to flush, and resource_tracker
        # will not exit until all Queue resources are properly closed.
        for q in (self._job_queue, self._result_queue):
            if q is not None:
                try:
                    q.close()
                    q.join_thread()  # Wait for feeder thread to finish
                except Exception as exc:
                    self.logger.warning("Failed to clean up queue: %s", exc)

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
                    if len(msg) == 4:
                        self._resolve_future(msg[1], error=WorkerRemoteError(msg[2], msg[3]))
                    else:
                        self._resolve_future(msg[1], error=RuntimeError(msg[2]))
                elif msg_type == "IDLE_EXIT":
                    self.logger.info("💤 Worker exited due to idle timeout — memory reclaimed by OS")
                    if self._worker:
                        worker = self._worker
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None, lambda current_worker=worker: current_worker.join(timeout=1)
                        )
                    self._worker = None
            except Exception:
                self.logger.exception("Unexpected error processing IPC message: %r", msg)

    def _fail_all_pending(self, error: Exception) -> None:
        """Fail all in-flight futures with the given error (e.g., after worker crash)."""
        for _uid, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(error)
        self._pending.clear()
        for temp_dir in self._temp_dirs.values():
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()

    def _resolve_future(
        self,
        uid: str,
        result: object | None = None,
        error: Exception | None = None,
    ) -> None:
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
