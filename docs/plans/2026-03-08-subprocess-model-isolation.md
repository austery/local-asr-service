# Subprocess Model Isolation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move FunASR/MLX model loading into a managed child subprocess so that process termination on idle forces the OS to reclaim all MPS memory (including PyTorch's Metal heap), achieving < 500 MB idle footprint for the main FastAPI process.

**Architecture:** FastAPI main process stays alive permanently (<200 MB). An `ModelWorker` subprocess is spawned on the first transcription request. It loads the model, runs inference, and self-terminates after `MODEL_IDLE_TIMEOUT_SEC` seconds of inactivity. `TranscriptionService` is refactored from "holds an engine" to "manages a worker subprocess via `multiprocessing.Queue`".

**Tech Stack:** Python `multiprocessing` (stdlib), `asyncio`, existing `ASREngine` protocol, `pytest-asyncio` for tests.

---

## Affected Files

| File | Action |
|------|--------|
| `src/workers/__init__.py` | Create (empty) |
| `src/workers/model_worker.py` | Create — subprocess entry point |
| `src/services/transcription.py` | Refactor — subprocess manager |
| `src/main.py` | Update — no startup engine load |
| `src/api/routes.py` | Fix — `service.engine` → `service.capabilities` |
| `tests/unit/test_worker.py` | Create — worker unit tests |
| `tests/unit/test_idle_offload.py` | Rewrite — mock subprocess |
| `tests/unit/test_service.py` | Update — new constructor |
| `tests/unit/test_dynamic_switching.py` | Update — new constructor |
| `docs/specs/SPEC-009-Idle-Model-Offload.md` | Update status |

---

## Task 1: Create `src/workers/model_worker.py`

**Files:**
- Create: `src/workers/__init__.py`
- Create: `src/workers/model_worker.py`

### Step 1: Write the failing test first

Create `tests/unit/test_worker.py`:

```python
"""Unit tests for model_worker subprocess entry point."""
import queue
import multiprocessing
from unittest.mock import MagicMock, patch, call
import pytest

from src.workers.model_worker import WorkerJob, run_worker


def _make_job(uid="job-1", path="/tmp/test.wav", params=None):
    return WorkerJob(
        uid=uid,
        temp_file_path=path,
        params=params or {"language": "auto", "output_format": "txt", "with_timestamp": False},
    )


class TestRunWorker:
    def test_sends_ready_after_load(self):
        """Worker must put READY on result_queue after engine.load() succeeds."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()
        job_q.put(None)  # immediate shutdown after READY

        mock_engine = MagicMock()

        with patch("src.workers.model_worker.create_engine", return_value=mock_engine):
            run_worker(job_q, result_q, engine_type="mlx", model_id="test-model", idle_timeout=0)

        msg = result_q.get_nowait()
        assert msg == ("READY", None)
        mock_engine.load.assert_called_once()

    def test_sends_load_error_if_load_fails(self):
        """Worker must put LOAD_ERROR and exit if model load raises."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        mock_engine = MagicMock()
        mock_engine.load.side_effect = RuntimeError("out of memory")

        with patch("src.workers.model_worker.create_engine", return_value=mock_engine):
            with pytest.raises(SystemExit):
                run_worker(job_q, result_q, engine_type="mlx", model_id="test-model", idle_timeout=0)

        msg = result_q.get_nowait()
        assert msg[0] == "LOAD_ERROR"
        assert "out of memory" in msg[1]

    def test_transcribes_job_and_puts_result(self):
        """Worker processes a job and puts (RESULT, uid, output) on result_queue."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        job = _make_job(uid="abc-123")
        job_q.put(job)
        job_q.put(None)  # shutdown

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = {"text": "hello world", "segments": None}

        with patch("src.workers.model_worker.create_engine", return_value=mock_engine):
            run_worker(job_q, result_q, engine_type="funasr", model_id="para", idle_timeout=0)

        ready = result_q.get_nowait()
        assert ready == ("READY", None)
        result_msg = result_q.get_nowait()
        assert result_msg[0] == "RESULT"
        assert result_msg[1] == "abc-123"

    def test_puts_error_on_transcription_failure(self):
        """Worker puts (ERROR, uid, msg) when transcribe_file raises."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        job = _make_job(uid="fail-1")
        job_q.put(job)
        job_q.put(None)

        mock_engine = MagicMock()
        mock_engine.transcribe_file.side_effect = RuntimeError("GPU crash")

        with patch("src.workers.model_worker.create_engine", return_value=mock_engine):
            run_worker(job_q, result_q, engine_type="funasr", model_id="para", idle_timeout=0)

        result_q.get_nowait()  # READY
        err_msg = result_q.get_nowait()
        assert err_msg[0] == "ERROR"
        assert err_msg[1] == "fail-1"
        assert "GPU crash" in err_msg[2]

    def test_idle_exit_when_timeout_reached(self):
        """Worker puts IDLE_EXIT and releases engine when queue.get() times out."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()
        # No jobs — let idle timeout fire immediately (timeout=0.01s)

        mock_engine = MagicMock()

        with patch("src.workers.model_worker.create_engine", return_value=mock_engine):
            with pytest.raises(SystemExit):
                run_worker(job_q, result_q, engine_type="mlx", model_id="test", idle_timeout=0.01)

        ready = result_q.get_nowait()
        assert ready == ("READY", None)
        idle_exit = result_q.get_nowait()
        assert idle_exit == ("IDLE_EXIT", None)
        mock_engine.release.assert_called_once()
```

### Step 2: Run test to verify it fails

```bash
cd /Users/leipeng/Documents/Projects/local-asr-service
uv run pytest tests/unit/test_worker.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.workers'`

### Step 3: Create `src/workers/__init__.py`

```python
```
(empty file)

### Step 4: Create `src/workers/model_worker.py`

```python
"""
Model worker subprocess entry point.

Runs in a child process. Loads the ASR engine, processes jobs from job_queue,
and self-terminates after idle_timeout seconds of inactivity — allowing the OS
to reclaim all MPS/Metal memory.

IPC protocol (result_queue messages):
  ("READY", None)              — engine loaded, ready for jobs
  ("LOAD_ERROR", error_str)    — engine.load() failed; process will exit
  ("RESULT", uid, result)      — transcription succeeded
  ("ERROR", uid, error_str)    — transcription raised an exception
  ("IDLE_EXIT", None)          — idle timeout reached; process exiting
"""

import logging
import queue
import sys
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("model_worker")


@dataclass
class WorkerJob:
    uid: str
    temp_file_path: str
    params: dict[str, Any]
    requested_model_spec_alias: str | None = field(default=None)


def run_worker(
    job_queue: Any,
    result_queue: Any,
    engine_type: str,
    model_id: str,
    idle_timeout: float,
) -> None:
    """Subprocess entry point. Call via multiprocessing.Process(target=run_worker, args=(...))."""
    logging.basicConfig(
        level="INFO",
        format="[worker] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import inside subprocess to avoid pickling the engine object
    from src.core.factory import create_engine as _create_engine

    engine = _create_engine(engine_type=engine_type, model_id=model_id)

    try:
        engine.load()
    except Exception as exc:
        result_queue.put(("LOAD_ERROR", str(exc)))
        sys.exit(1)

    result_queue.put(("READY", None))
    log.info(f"Worker ready — engine={engine_type} model={model_id} idle_timeout={idle_timeout}s")

    while True:
        try:
            timeout = idle_timeout if idle_timeout and idle_timeout > 0 else None
            job: WorkerJob | None = job_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            log.info(f"Idle timeout ({idle_timeout}s) reached — releasing model and exiting")
            try:
                engine.release()
            except Exception:
                pass
            result_queue.put(("IDLE_EXIT", None))
            sys.exit(0)

        if job is None:
            log.info("Received shutdown sentinel — releasing model and exiting")
            try:
                engine.release()
            except Exception:
                pass
            sys.exit(0)

        try:
            result = engine.transcribe_file(
                job.temp_file_path,
                language=job.params.get("language", "auto"),
                output_format=job.params.get("output_format", "txt"),
                with_timestamp=job.params.get("with_timestamp", False),
                use_itn=True,
            )
            result_queue.put(("RESULT", job.uid, result))
        except Exception as exc:
            log.exception(f"Transcription failed for job {job.uid}: {exc}")
            result_queue.put(("ERROR", job.uid, str(exc)))
```

### Step 5: Run tests to verify they pass

```bash
uv run pytest tests/unit/test_worker.py -v
```

Expected: All 5 tests pass.

### Step 6: Commit

```bash
git add src/workers/__init__.py src/workers/model_worker.py tests/unit/test_worker.py
git commit -m "feat(worker): add ModelWorker subprocess for memory-isolated ASR inference

Implements SPEC-009 v2: model runs in a child process so that
idle termination lets the OS reclaim all MPS/Metal memory.

IPC via multiprocessing.Queue. Worker self-exits on idle timeout
or shutdown sentinel."
```

---

## Task 2: Refactor `TranscriptionService`

**Files:**
- Modify: `src/services/transcription.py`

The service changes from "holds an engine" to "manages a worker subprocess". The public `submit()` API is unchanged from the caller's perspective.

### Step 1: Write the failing tests

Rewrite `tests/unit/test_idle_offload.py`:

```python
"""
Unit tests for TranscriptionService subprocess management (SPEC-009 v2).

Mocks the worker subprocess — tests verify that the service correctly
spawns, communicates with, and tracks the lifecycle of the worker process.
"""
import asyncio
import queue as _stdlib_queue
from io import BytesIO
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from fastapi import UploadFile

from src.services.transcription import TranscriptionService
from src.workers.model_worker import WorkerJob


def _make_upload() -> UploadFile:
    return UploadFile(file=BytesIO(b"audio"), filename="test.wav")


def _make_mock_queues(ready_result=None, job_result=None):
    """
    Returns (job_queue_mock, result_queue_mock) with pre-configured get() behavior.
    result_queue.get_nowait() returns items from a list in order, then raises Empty.
    """
    responses = [("READY", None)]
    if job_result:
        responses.append(job_result)

    call_count = {"n": 0}

    def get_nowait():
        if call_count["n"] < len(responses):
            item = responses[call_count["n"]]
            call_count["n"] += 1
            return item
        raise _stdlib_queue.Empty()

    result_q = MagicMock()
    result_q.get_nowait.side_effect = get_nowait
    result_q.get.return_value = responses[0]  # blocking get for READY

    job_q = MagicMock()

    return job_q, result_q


@pytest.fixture
def funasr_spec():
    from src.core.model_registry import lookup
    return lookup("paraformer")


@pytest.fixture
def mlx_spec():
    from src.core.model_registry import lookup
    return lookup("qwen3-asr")


@pytest.mark.asyncio
class TestTranscriptionServiceSubprocess:

    async def test_worker_is_spawned_on_first_request(self, funasr_spec):
        """Service spawns worker subprocess when first job is submitted."""
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
            idle_timeout=60,
        )
        await service.start_worker()

        with patch("src.services.transcription.multiprocessing.Process") as MockProcess, \
             patch("src.services.transcription.multiprocessing.Queue") as MockQueue:

            mock_proc = MagicMock()
            mock_proc.is_alive.return_value = True
            MockProcess.return_value = mock_proc

            result_q = MagicMock()
            result_q.get.return_value = ("READY", None)
            MockQueue.side_effect = [MagicMock(), result_q]

            service._worker = None  # force spawn path
            await service._spawn_worker()

            MockProcess.assert_called_once()
            mock_proc.start.assert_called_once()

        await service.stop_worker()

    async def test_model_loaded_is_false_when_no_worker(self, funasr_spec):
        """model_loaded reports False when no worker subprocess is alive."""
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
            idle_timeout=60,
        )
        assert service.model_loaded is False

    async def test_model_loaded_is_true_when_worker_alive(self, funasr_spec):
        """model_loaded reports True when worker subprocess is running."""
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
            idle_timeout=60,
        )
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        service._worker = mock_proc
        assert service.model_loaded is True

    async def test_capabilities_from_model_spec(self, funasr_spec):
        """capabilities property returns current_model_spec.capabilities."""
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
            idle_timeout=60,
        )
        assert service.capabilities == funasr_spec.capabilities

    async def test_idle_exit_clears_worker_reference(self, funasr_spec):
        """When result reader receives IDLE_EXIT, worker reference is set to None."""
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
            idle_timeout=60,
        )
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        service._worker = mock_proc
        service.is_running = True

        import queue as _q
        result_q = MagicMock()
        # First call returns IDLE_EXIT, then Empty forever
        call_n = {"n": 0}
        def getnowait():
            if call_n["n"] == 0:
                call_n["n"] += 1
                return ("IDLE_EXIT", None)
            raise _q.Empty()
        result_q.get_nowait.side_effect = getnowait
        service._result_queue = result_q

        # Run one iteration of the reader loop then cancel
        task = asyncio.create_task(service._result_reader_loop())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert service._worker is None
```

### Step 2: Run to confirm failure

```bash
uv run pytest tests/unit/test_idle_offload.py -v
```

Expected: `ImportError` or `TypeError` (wrong constructor signature).

### Step 3: Rewrite `src/services/transcription.py`

```python
"""
TranscriptionService — subprocess-based orchestrator (SPEC-009 v2).

Manages a ModelWorker child process that holds the ASR model in memory.
When the worker exits (idle timeout or error), the OS reclaims all memory,
including the PyTorch MPS heap that cannot be released from within a process.
"""
import asyncio
import logging
import multiprocessing
import os
import queue as _stdlib_queue
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import UploadFile

from src.core.base_engine import EngineCapabilities
from src.core.model_registry import ModelSpec
from src.workers.model_worker import WorkerJob, run_worker


class TranscriptionService:
    """
    转录服务调度器 (SPEC-009 v2: subprocess isolation)。
    职责：
    1. 管理 ModelWorker 子进程生命周期
    2. 通过 multiprocessing.Queue 进行 IPC
    3. 将 asyncio.Future 与 IPC 结果桥接
    4. 支持动态换模 (SPEC-108): kill old worker, spawn new
    5. 管理临时文件生命周期
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
        self._job_queue: multiprocessing.Queue | None = None  # type: ignore[type-arg]
        self._result_queue: multiprocessing.Queue | None = None  # type: ignore[type-arg]
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._spawn_lock: asyncio.Lock = asyncio.Lock()
        self._result_reader_task: asyncio.Task[None] | None = None
        self._temp_dirs: dict[str, str] = {}  # uid → temp_dir

        self.is_running = False
        self.logger = logging.getLogger(__name__)

        if idle_timeout > 0:
            self.logger.info(f"💤 Idle offload enabled: worker exits after {idle_timeout}s of inactivity")
        else:
            self.logger.info("💤 Idle offload disabled (MODEL_IDLE_TIMEOUT_SEC=0)")

    # ── Public properties ─────────────────────────────────────────────────

    @property
    def current_model_spec(self) -> ModelSpec | None:
        return self._current_model_spec

    @property
    def model_loaded(self) -> bool:
        """True if worker subprocess is alive."""
        return self._worker is not None and self._worker.is_alive()

    @property
    def capabilities(self) -> EngineCapabilities:
        """Return capabilities from current model spec (no live engine needed)."""
        if self._current_model_spec is not None:
            return self._current_model_spec.capabilities
        return EngineCapabilities()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start_worker(self) -> None:
        """Mark service as running. Worker is spawned lazily on first request."""
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

    # ── Submit API (unchanged from callers' perspective) ──────────────────

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
                import shutil as _shutil
                _shutil.copyfileobj(file.file, buf)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str | dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        self._temp_dirs[request_id] = temp_dir

        try:
            async with self._spawn_lock:
                needs_switch = (
                    model_spec is not None and model_spec != self._current_model_spec
                )
                if needs_switch:
                    await self._switch_worker(model_spec)
                elif not self.model_loaded:
                    await self._spawn_worker()

            job = WorkerJob(uid=request_id, temp_file_path=temp_path, params=params)
            assert self._job_queue is not None
            self._job_queue.put_nowait(job)

            return await future

        except Exception as exc:
            self._pending.pop(request_id, None)
            self._cleanup_temp(request_id)
            raise exc

    # ── Worker spawn / switch / shutdown ─────────────────────────────────

    async def _spawn_worker(self, model_spec: ModelSpec | None = None) -> None:
        """Spawn a new worker subprocess and wait for its READY signal."""
        effective_spec = model_spec or self._current_model_spec
        engine_type = effective_spec.engine_type if effective_spec else self._engine_type
        model_id = effective_spec.model_id if effective_spec else self._model_id

        self.logger.info(f"🚀 Spawning worker (engine={engine_type}, model={model_id})")
        spawn_start = time.time()

        self._job_queue = multiprocessing.Queue()
        self._result_queue = multiprocessing.Queue()

        proc = multiprocessing.Process(
            target=run_worker,
            args=(self._job_queue, self._result_queue, engine_type, model_id, self._idle_timeout),
            daemon=True,
        )
        proc.start()
        self._worker = proc

        if model_spec is not None:
            self._current_model_spec = model_spec

        # Wait for READY or LOAD_ERROR (blocking in executor — can take 10-60s)
        try:
            msg = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, self._result_queue.get),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            proc.terminate()
            self._worker = None
            raise RuntimeError("Worker failed to start within 120s (model load timeout)")

        if msg[0] == "LOAD_ERROR":
            proc.terminate()
            self._worker = None
            raise RuntimeError(f"Worker failed to load model: {msg[1]}")

        elapsed = time.time() - spawn_start
        self.logger.info(f"✅ Worker ready in {elapsed:.1f}s")

        # Start result reader if not already running
        if self._result_reader_task is None or self._result_reader_task.done():
            self._result_reader_task = asyncio.create_task(self._result_reader_loop())

    async def _switch_worker(self, new_spec: ModelSpec) -> None:
        """Kill current worker and spawn a new one with the new model spec."""
        old_alias = self._current_model_spec.alias if self._current_model_spec else "unknown"
        self.logger.info(f"🔄 Switching model: {old_alias} → {new_spec.alias}")
        await self._shutdown_worker()
        await self._spawn_worker(new_spec)

    async def _shutdown_worker(self) -> None:
        """Gracefully terminate the current worker process."""
        if self._worker is None:
            return
        if self._job_queue is not None:
            try:
                self._job_queue.put_nowait(None)  # graceful shutdown sentinel
            except Exception:
                pass
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._worker.join(timeout=5)),  # type: ignore[union-attr]
                timeout=6.0,
            )
        except asyncio.TimeoutError:
            pass
        if self._worker.is_alive():
            self._worker.terminate()
        self._worker = None

    # ── Result reader loop ────────────────────────────────────────────────

    async def _result_reader_loop(self) -> None:
        """
        Polls result_queue (non-blocking) and resolves pending asyncio Futures.
        Runs as a background asyncio task for the lifetime of the service.
        """
        while self.is_running:
            if self._result_queue is None:
                await asyncio.sleep(0.05)
                continue
            try:
                msg = self._result_queue.get_nowait()
            except _stdlib_queue.Empty:
                await asyncio.sleep(0.05)
                continue

            msg_type: str = msg[0]

            if msg_type == "RESULT":
                uid, result = msg[1], msg[2]
                self._resolve_future(uid, result=result)

            elif msg_type == "ERROR":
                uid, err_str = msg[1], msg[2]
                self._resolve_future(uid, error=RuntimeError(err_str))

            elif msg_type == "IDLE_EXIT":
                self.logger.info("💤 Worker exited due to idle timeout — memory reclaimed by OS")
                if self._worker:
                    self._worker.join(timeout=1)
                self._worker = None

    def _resolve_future(
        self,
        uid: str,
        result: Any = None,
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
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
```

### Step 4: Run the new idle offload tests

```bash
uv run pytest tests/unit/test_idle_offload.py -v
```

Expected: All 5 tests pass.

### Step 5: Commit

```bash
git add src/services/transcription.py tests/unit/test_idle_offload.py
git commit -m "refactor(service): replace in-process idle offload with subprocess isolation

SPEC-009 v2: TranscriptionService now manages a ModelWorker child process
instead of holding an ASREngine directly. Worker self-terminates on idle
timeout; OS reclaims all MPS memory on process exit.

Breaking change: constructor now takes engine_type/model_id instead of engine.
main.py and routes.py will be updated in subsequent commits."
```

---

## Task 3: Update `src/main.py`

**Files:**
- Modify: `src/main.py`

### Step 1: Identify all changes needed

Current `lifespan()`:
1. `engine = create_engine()` → remove
2. `engine.load()` → remove
3. `TranscriptionService(engine=engine, ...)` → `TranscriptionService(engine_type=ENGINE_TYPE, model_id=get_model_id(), ...)`
4. `app.state.engine = engine` → remove (or keep None for backwards compat)
5. `app.state.service.engine.release()` in shutdown → remove

### Step 2: Update `main.py` lifespan

Replace the entire lifespan function body:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("🌱 System starting up...")
    logger.info(f"📋 Engine type: {ENGINE_TYPE}")
    logger.info(f"📋 Model ID: {get_model_id()}")
    logger.info(f"💤 Idle timeout: {MODEL_IDLE_TIMEOUT_SEC}s (0 = disabled)")
    logger.warning("⚠️  Running with workers=1 (REQUIRED for Mac Silicon to prevent OOM)")

    # Resolve startup model spec for tracking
    startup_model_id = get_model_id()
    try:
        initial_spec = lookup(startup_model_id)
    except ValueError:
        initial_spec = None
        logger.warning(f"⚠️  Startup model '{startup_model_id}' not in registry; model tracking disabled.")

    # Initialize service — worker subprocess spawns on first request
    service = TranscriptionService(
        engine_type=ENGINE_TYPE,
        model_id=startup_model_id,
        max_queue_size=MAX_QUEUE_SIZE,
        initial_model_spec=initial_spec,
        idle_timeout=MODEL_IDLE_TIMEOUT_SEC,
    )
    await service.start_worker()

    app.state.service = service
    app.state.engine_type = ENGINE_TYPE
    app.state.model_id = startup_model_id

    logger.info("✅ System ready! Worker spawns on first transcription request.")

    yield  # --- service running ---

    logger.info("🛑 System shutting down...")
    if hasattr(app.state, "service"):
        await app.state.service.stop_worker()
```

Also remove unused imports: `create_engine`, and update `TranscriptionService` import (no change needed).

### Step 3: Run smoke test

```bash
uv run pytest tests/unit/test_config_factory.py -v
```

Expected: All pass (config module unchanged).

### Step 4: Commit

```bash
git add src/main.py
git commit -m "refactor(main): remove startup engine load; worker spawns lazily on first request

Service no longer loads the 16-23GB model at startup.
Memory footprint at startup: ~150MB instead of ~23GB."
```

---

## Task 4: Fix `src/api/routes.py`

**Files:**
- Modify: `src/api/routes.py`

### Step 1: Fix both `service.engine` references

**Location 1** (line ~197):
```python
# BEFORE:
caps = request.app.state.service.engine.capabilities

# AFTER:
caps = request.app.state.service.capabilities
```

**Location 2** (line ~356):
```python
# BEFORE:
"capabilities": asdict(current_spec.capabilities) if current_spec else asdict(service.engine.capabilities),

# AFTER:
"capabilities": asdict(current_spec.capabilities) if current_spec else asdict(service.capabilities),
```

### Step 2: Run routes-related tests

```bash
uv run pytest tests/unit/test_adapters.py tests/unit/test_security.py -v
```

Expected: All pass.

### Step 3: Commit

```bash
git add src/api/routes.py
git commit -m "fix(routes): replace service.engine.capabilities with service.capabilities

TranscriptionService no longer holds a direct engine reference.
Capabilities are now derived from current_model_spec."
```

---

## Task 5: Update existing unit tests

**Files:**
- Modify: `tests/unit/test_service.py`
- Modify: `tests/unit/test_dynamic_switching.py`

### Step 1: Check what breaks

```bash
uv run pytest tests/unit/test_service.py tests/unit/test_dynamic_switching.py -v 2>&1 | head -40
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'engine'`

### Step 2: Update `test_service.py` constructor calls

Find all `TranscriptionService(engine=mock_engine, ...)` and replace with:
```python
from src.core.model_registry import lookup

funasr_spec = lookup("paraformer")

service = TranscriptionService(
    engine_type="funasr",
    model_id=funasr_spec.model_id,
    max_queue_size=2,
    initial_model_spec=funasr_spec,
    idle_timeout=0,
)
```

Since `test_service.py` tests the `submit()` flow (which now goes through a subprocess), mock `_spawn_worker` and pre-inject mock queues:

```python
async def test_submit_success(self, funasr_spec):
    service = TranscriptionService(
        engine_type="funasr",
        model_id=funasr_spec.model_id,
        max_queue_size=2,
        initial_model_spec=funasr_spec,
        idle_timeout=0,
    )
    await service.start_worker()

    # Inject fake subprocess infrastructure
    import queue as _q, multiprocessing
    service._job_queue = multiprocessing.Queue()
    service._result_queue = multiprocessing.Queue()
    mock_proc = MagicMock(); mock_proc.is_alive.return_value = True
    service._worker = mock_proc

    # Simulate worker result arriving asynchronously
    async def fake_spawn():
        pass

    with patch.object(service, "_spawn_worker", side_effect=fake_spawn):
        # Pre-populate result before submit so reader picks it up
        async def deliver_result():
            await asyncio.sleep(0.05)
            service._result_queue.put(("RESULT", "test-req", {"text": "hello", "segments": None, "duration": 1.0}))

        asyncio.create_task(deliver_result())
        service._result_reader_task = asyncio.create_task(service._result_reader_loop())
        service.is_running = True

        result = await service.submit(_make_upload_file(), {"language": "auto", "output_format": "json"}, request_id="test-req")

    assert result["text"] == "hello"
    await service.stop_worker()
```

Apply the same pattern to all tests in `test_service.py` and `test_dynamic_switching.py`.

For dynamic switching tests, mock `_switch_worker`:
```python
with patch.object(service, "_switch_worker", new_callable=AsyncMock) as mock_switch:
    # test that switch is called when different model_spec submitted
    ...
    mock_switch.assert_called_once_with(new_spec)
```

### Step 3: Run all unit tests

```bash
uv run pytest tests/unit/ -v --tb=short
```

Expected: All tests pass (previously 126; final count may vary slightly with new test_worker.py added).

### Step 4: Commit

```bash
git add tests/unit/test_service.py tests/unit/test_dynamic_switching.py
git commit -m "test: update service and dynamic switching tests for subprocess architecture"
```

---

## Task 6: Update SPEC-009

**Files:**
- Modify: `docs/specs/SPEC-009-Idle-Model-Offload.md`

### Step 1: Update status and add v2 note

Change frontmatter:
```yaml
status: ✅ 已完成 v2 (Completed — Subprocess Isolation)
lastUpdateDate: 2026-03-08
```

Add a section after `## 9. Status History`:
```markdown
### v2 Revision (2026-03-08)

**Problem with v1**: `torch.mps.empty_cache()` + `gc.collect()` after `engine.release()` only reduced
memory from 23 GB to ~15-18 GB. PyTorch's MPS allocator retains a Metal heap pool that cannot be
released from within the process.

**v2 Solution**: Moved the ASR model into a child subprocess (`src/workers/model_worker.py`).
`TranscriptionService` manages the subprocess lifecycle. On idle timeout, the subprocess exits and
the OS forcibly reclaims all memory including the MPS heap. Achieves < 500 MB idle memory.

**Design doc**: `docs/plans/2026-03-08-subprocess-model-isolation-design.md`
```

### Step 2: Commit

```bash
git add docs/specs/SPEC-009-Idle-Model-Offload.md
git commit -m "docs(spec-009): update status to v2 completed; document subprocess isolation approach"
```

---

## Task 7: Full test suite + smoke verification

### Step 1: Run all unit tests

```bash
uv run pytest tests/unit/ -v
```

Expected: All tests pass.

### Step 2: Run integration tests (if available)

```bash
uv run pytest tests/ -m "not e2e" -v --tb=short
```

### Step 3: Manual smoke test (optional but recommended)

```bash
# Start the service
uv run uvicorn src.main:app --host 0.0.0.0 --port 50070

# In another terminal — check startup memory (should be ~150MB, not 23GB)
ps aux | grep python

# Send a transcription request
curl -X POST http://localhost:50070/v1/audio/transcriptions \
  -F "file=@/path/to/sample.wav" \
  -F "language=auto"

# Observe: worker spawns, memory rises to ~23GB
# Wait 70s with no requests
# Observe: memory drops back to ~150MB in Activity Monitor
```

### Step 4: Final commit (if any cleanup needed)

```bash
git add -A
git commit -m "chore: final cleanup for subprocess model isolation (SPEC-009 v2)"
```
