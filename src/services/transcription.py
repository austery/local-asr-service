import asyncio
import logging
import multiprocessing
import os
import queue as _stdlib_queue
import shutil
import tempfile
from contextlib import suppress
from typing import Literal, TypedDict

from fastapi import UploadFile

from src.adapters.segment_alignment import align_speakers
from src.core.base_engine import EngineCapabilities
from src.core.diarization_port import SpeakerTurn
from src.core.model_registry import ModelSpec
from src.core.pipeline_registry import PipelineProfile
from src.workers.model_worker import WorkerJob, run_worker


class TranscriptionResultDict(TypedDict, total=False):
    text: str
    segments: list[dict[str, object]] | None
    duration: float
    language: str


TranscriptionResult = str | TranscriptionResultDict
WorkerStartupMessage = tuple[Literal["READY"], None] | tuple[Literal["LOAD_ERROR"], str]
WorkerResultMessage = (
    tuple[Literal["RESULT"], str, object]
    | tuple[Literal["ERROR"], str, str]
    | tuple[Literal["IDLE_EXIT"], None]
)
WorkerMessage = WorkerStartupMessage | WorkerResultMessage


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
            with suppress(asyncio.CancelledError):
                await self._result_reader_task

    async def submit(
        self,
        file: UploadFile,
        params: dict[str, object],
        request_id: str = "unknown",
        model_spec: ModelSpec | None = None,
    ) -> TranscriptionResult:
        if len(self._pending) >= self._max_queue_size:
            self.logger.warning(f"[{request_id}] Queue full, rejecting request")
            raise RuntimeError("Service busy: Queue is full.")

        temp_dir = tempfile.mkdtemp(prefix="asr_task_")
        try:
            file_ext = os.path.splitext(file.filename or "upload.wav")[1] or ".wav"
            temp_path = os.path.join(temp_dir, f"original{file_ext}")
            with open(temp_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)

            result = await self._submit_worker_job(
                temp_file_path=temp_path,
                params=params,
                request_id=request_id,
                model_spec=model_spec,
                temp_dir=temp_dir,
            )
            return self._coerce_transcription_result(result)

        except BaseException:
            self._pending.pop(request_id, None)
            self._temp_dirs.pop(request_id, None)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    async def submit_pipeline(
        self,
        file: UploadFile,
        params: dict[str, object],
        request_id: str,
        profile: PipelineProfile,
    ) -> TranscriptionResult:
        if not profile.requestable:
            raise RuntimeError(f"Pipeline profile '{profile.alias}' is not enabled for requests.")
        if len(self._pending) >= self._max_queue_size:
            self.logger.warning(f"[{request_id}] Queue full, rejecting pipeline request")
            raise RuntimeError("Service busy: Queue is full.")

        temp_dir = tempfile.mkdtemp(prefix="asr_pipeline_")
        try:
            file_ext = os.path.splitext(file.filename or "upload.wav")[1] or ".wav"
            temp_path = os.path.join(temp_dir, f"original{file_ext}")
            with open(temp_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)

            return await self._run_decoupled_pipeline(
                temp_file_path=temp_path,
                params=params,
                request_id=request_id,
                profile=profile,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _submit_worker_job(
        self,
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
        model_spec: ModelSpec | None,
        temp_dir: str | None = None,
    ) -> object:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()

        try:
            async with self._spawn_lock:
                if len(self._pending) >= self._max_queue_size:
                    self.logger.warning(f"[{request_id}] Queue full, rejecting worker job")
                    raise RuntimeError("Service busy: Queue is full.")

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
                if temp_dir is not None:
                    self._temp_dirs[request_id] = temp_dir

                self._job_queue.put_nowait(WorkerJob(
                    uid=request_id,
                    temp_file_path=temp_file_path,
                    params=params,
                ))
        except BaseException:
            self._pending.pop(request_id, None)
            self._temp_dirs.pop(request_id, None)
            raise

        return await future

    async def _transcribe_with_alias(
        self,
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
        alias: str,
    ) -> TranscriptionResult:
        result = await self._submit_worker_job(
            temp_file_path=temp_file_path,
            params=params,
            request_id=f"{request_id}:transcribe",
            model_spec=self._lookup_model_spec(alias),
        )
        return self._coerce_transcription_result(result)

    async def _diarize_with_alias(
        self,
        temp_file_path: str,
        request_id: str,
        alias: str,
    ) -> list[SpeakerTurn]:
        result = await self._submit_worker_job(
            temp_file_path=temp_file_path,
            params={},
            request_id=f"{request_id}:diarize",
            model_spec=self._lookup_model_spec(alias),
        )
        if not isinstance(result, list) or not all(isinstance(turn, SpeakerTurn) for turn in result):
            raise TypeError(f"Expected diarization result as list[SpeakerTurn], got {type(result).__name__}")
        return result

    def _lookup_model_spec(self, alias: str) -> ModelSpec:
        from src.core.model_registry import lookup

        return lookup(alias)

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

    async def _run_decoupled_pipeline(
        self,
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
        profile: PipelineProfile,
    ) -> TranscriptionResult:
        previous_spec = self._current_model_spec
        try:
            transcript_result = await self._transcribe_with_alias(
                temp_file_path,
                params,
                request_id,
                profile.transcription_alias,
            )
            if not isinstance(transcript_result, dict):
                return transcript_result

            try:
                speaker_turns = await self._diarize_with_alias(
                    temp_file_path,
                    request_id,
                    profile.diarization_alias,
                )
            except Exception as exc:
                if self._is_worker_lifecycle_error(exc):
                    raise
                self.logger.warning(
                    "[%s] Decoupled pipeline diarization failed for profile %s; returning transcript-only result: %s",
                    request_id,
                    profile.alias,
                    exc,
                )
                return transcript_result

            segments = transcript_result.get("segments")
            if not isinstance(segments, list):
                return transcript_result

            try:
                aligned_segments = align_speakers(
                    self._coerce_segment_list(segments),
                    speaker_turns,
                )
            except (TypeError, ValueError) as exc:
                self.logger.warning(
                    "[%s] Decoupled pipeline alignment failed for profile %s; returning transcript-only result: %s",
                    request_id,
                    profile.alias,
                    exc,
                )
                return transcript_result

            return {**transcript_result, "segments": aligned_segments}
        finally:
            await self._restore_resident_model(previous_spec)

    @staticmethod
    def _coerce_segment_list(segments: list[object]) -> list[dict[str, object]]:
        if not all(isinstance(segment, dict) for segment in segments):
            raise TypeError("Expected transcription segments to be dictionaries")
        return segments

    async def _restore_resident_model(self, previous_spec: ModelSpec | None) -> None:
        async with self._spawn_lock:
            current_spec = self._current_model_spec
            if current_spec == previous_spec:
                return
            if previous_spec is None:
                await self._shutdown_worker()
                self._current_model_spec = None
                return
            await self._switch_worker(previous_spec)

    @staticmethod
    def _is_worker_lifecycle_error(exc: Exception) -> bool:
        return isinstance(exc, RuntimeError) and str(exc).startswith(
            (
                "Worker terminated",
                "Worker process died unexpectedly",
                "Worker failed to start",
                "Worker failed to load model",
                "Worker sent unexpected startup message",
            )
        )

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
