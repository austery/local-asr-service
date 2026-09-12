import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path

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
from src.services.worker_session import WorkerSession
from src.workers.model_worker import WorkerJob
from src.workers.transport import WorkerConfig


class TranscriptionService:
    """
    Selects and admits requests to the resident session or Apple sidecar.

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
        *,
        worker_session: WorkerSession | None = None,
    ) -> None:
        self._engine_type = engine_type
        self._model_id = model_id
        self._current_model_spec = initial_model_spec
        self._idle_timeout = idle_timeout
        self._max_queue_size = max_queue_size

        self._session = worker_session or WorkerSession()
        self._spawn_lock = asyncio.Lock()
        self._stopping = False
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
        return self._session.alive


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
        self._stopping = False
        self.logger.info("🚦 Service initialized (worker spawns on first request).")

    async def stop_worker(self) -> None:
        """Gracefully stop worker subprocess and result reader."""
        self.is_running = False
        self._stopping = True
        await self._session.close()

    async def submit(
        self,
        file: UploadFile,
        params: dict[str, object],
        request_id: str = "unknown",
        model_spec: ModelSpec | None = None,
    ) -> ExecutionResult:
        if self._stopping:
            raise RuntimeError("Service is stopping")
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
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _submit_resident_job(
        self,
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
        plan: ExecutionPlan | None,
    ) -> ExecutionResult:
        # Select passthrough under the same lock that protects model switches.
        # Release it after enqueue, so waiting for inference never serializes
        # admission of other requests or hides them from queue_size.
        async with self._spawn_lock:
            if self._stopping:
                raise RuntimeError("Service is stopping")
            selected = plan or ExecutionPlan.select(self._current_model_spec, self._model_id)
            selected.validate(params)
            route_to_sidecar = self._is_apple_speech_spec(selected.spec)
            if not route_to_sidecar:
                future = await self._enqueue_worker_job(
                    temp_file_path=temp_file_path,
                    params=params,
                    request_id=request_id,
                    model_spec=plan.spec if plan is not None else None,
                )
        if route_to_sidecar:
            result = await self._submit_apple_speech_job(
                temp_file_path=temp_file_path, params=params, request_id=request_id,
            )
        else:
            result = await future
        return ExecutionResult(self._coerce_transcription_result(result), selected.model)

    def _active_job_count(self) -> int:
        return self._session.pending_count + len(self._sidecar_pending)

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
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
        model_spec: ModelSpec | None,
    ) -> asyncio.Future[object]:
        """Enqueue a transcription while the caller holds _spawn_lock."""
        if self._active_job_count() >= self._max_queue_size:
            self.logger.warning(f"[{request_id}] Queue full, rejecting worker job")
            raise RuntimeError("Service busy: Queue is full.")

        if model_spec is not None and model_spec != self._current_model_spec:
            await self._switch_worker(model_spec)
        elif not self.model_loaded:
            await self._spawn_worker()

        # Registration belongs to the ready session, after any old worker exits.
        return self._session.enqueue(WorkerJob(
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


    async def _spawn_worker(self, model_spec: ModelSpec | None = None) -> None:
        effective_spec = model_spec or self._current_model_spec
        config = WorkerConfig(
            effective_spec.engine_type if effective_spec else self._engine_type,
            effective_spec.model_id if effective_spec else self._model_id,
            self._idle_timeout,
        )
        await self._session.start(config)
        if model_spec is not None:
            self._current_model_spec = model_spec

    async def _switch_worker(self, new_spec: ModelSpec) -> None:
        # Preserve explicit ModelSpec switching even when aliases share a model ID.
        await self._session.close()
        if new_spec.engine_type == "apple-speech":
            self._current_model_spec = new_spec
            return
        await self._spawn_worker(new_spec)
