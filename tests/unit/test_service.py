"""Unit tests for TranscriptionService core behaviors (SPEC-009).

Tests: success result delivery, txt format, queue full, temp file cleanup, error handling.
Uses injected mock worker infrastructure (no real subprocess spawned).
"""
import asyncio
import multiprocessing
import os
import queue as _stdlib_queue
import tempfile as _tempfile
from contextlib import suppress
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import UploadFile

from src.core.model_registry import ModelSpec, lookup
from src.services.transcription import TranscriptionService, WorkerRemoteError
from src.workers.model_worker import WorkerJob


@pytest.fixture
def funasr_spec():
    return lookup("paraformer")


def _make_upload() -> UploadFile:
    return UploadFile(file=BytesIO(b"fake audio content"), filename="test.wav")


def _setup_service(spec, max_queue_size: int = 2) -> TranscriptionService:
    """Create a service with injected mock worker — no subprocess spawned."""
    svc = TranscriptionService(
        engine_type=spec.engine_type,
        model_id=spec.model_id,
        max_queue_size=max_queue_size,
        initial_model_spec=spec,
        idle_timeout=0,
    )
    svc.is_running = True
    mock_proc = MagicMock()
    mock_proc.is_alive.return_value = True
    svc._worker = mock_proc
    svc._job_queue = multiprocessing.Queue()
    svc._result_queue = multiprocessing.Queue()
    return svc


async def _stop_service(svc: TranscriptionService) -> None:
    svc.is_running = False
    if svc._result_reader_task and not svc._result_reader_task.done():
        svc._result_reader_task.cancel()
        with suppress(asyncio.CancelledError):
            await svc._result_reader_task


@pytest.mark.asyncio
class TestTranscriptionService:

    async def test_submit_success(self, funasr_spec):
        """RESULT message from worker resolves the submit() future with correct data."""
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())
        expected = {"text": "Mocked Transcription", "segments": [], "duration": 1.0}

        async def deliver() -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("RESULT", "req-1", expected))

        asyncio.create_task(deliver())
        try:
            result = await asyncio.wait_for(
                svc.submit(_make_upload(), {"language": "zh", "output_format": "json"}, request_id="req-1"),
                timeout=5.0,
            )
        finally:
            await _stop_service(svc)

        assert result["text"] == "Mocked Transcription"
        assert "segments" in result

    async def test_submit_txt_format(self, funasr_spec):
        """Plain-text result (str) is returned as-is from the worker."""
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())

        async def deliver() -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("RESULT", "req-2", "[Speaker 0]: Mocked Transcription"))

        asyncio.create_task(deliver())
        try:
            result = await asyncio.wait_for(
                svc.submit(_make_upload(), {"output_format": "txt"}, request_id="req-2"),
                timeout=5.0,
            )
        finally:
            await _stop_service(svc)

        assert result == "[Speaker 0]: Mocked Transcription"

    async def test_queue_full(self, funasr_spec):
        """submit() raises RuntimeError immediately when pending dict is at capacity."""
        svc = _setup_service(funasr_spec, max_queue_size=2)
        loop = asyncio.get_running_loop()
        svc._pending["x"] = loop.create_future()
        svc._pending["y"] = loop.create_future()

        with pytest.raises(RuntimeError, match="Queue is full"):
            await svc.submit(_make_upload(), {})

    async def test_temp_file_lifecycle(self, funasr_spec):
        """Temp directory is created before the job and deleted after result arrives."""
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())

        created_dirs: list[str] = []
        original_mkdtemp = _tempfile.mkdtemp

        def capture_mkdtemp(*args: object, **kwargs: object) -> str:
            path = original_mkdtemp(*args, **kwargs)
            created_dirs.append(path)
            return path

        async def deliver() -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("RESULT", "req-3", {"text": "ok", "segments": None, "duration": 0.5}))

        asyncio.create_task(deliver())
        try:
            with patch("src.services.transcription.tempfile.mkdtemp", side_effect=capture_mkdtemp):
                await asyncio.wait_for(
                    svc.submit(_make_upload(), {}, request_id="req-3"),
                    timeout=5.0,
                )
        finally:
            await _stop_service(svc)

        assert len(created_dirs) == 1, "Expected exactly one temp dir to be created"
        assert not os.path.exists(created_dirs[0]), "Temp dir must be deleted after job completes"

    async def test_worker_error_handling(self, funasr_spec):
        """ERROR message from worker raises RuntimeError in submit()."""
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())

        async def deliver() -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("ERROR", "req-4", "Model Error"))

        asyncio.create_task(deliver())
        try:
            with pytest.raises(RuntimeError, match="Model Error"):
                await asyncio.wait_for(
                    svc.submit(_make_upload(), {}, request_id="req-4"),
                    timeout=5.0,
                )
        finally:
            await _stop_service(svc)

    async def test_worker_error_handling_preserves_remote_exception_type(self, funasr_spec):
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())

        async def deliver() -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("ERROR", "req-typed", "ValueError", "bad job shape"))

        asyncio.create_task(deliver())
        try:
            with pytest.raises(WorkerRemoteError) as exc_info:
                await asyncio.wait_for(
                    svc.submit(_make_upload(), {}, request_id="req-typed"),
                    timeout=5.0,
                )
        finally:
            await _stop_service(svc)

        assert exc_info.value.exc_type_name == "ValueError"
        assert "bad job shape" in str(exc_info.value)


@pytest.mark.asyncio
async def test_submit_resident_job_enforces_queue_limit_for_internal_callers(funasr_spec):
    svc = _setup_service(funasr_spec, max_queue_size=1)
    loop = asyncio.get_running_loop()
    svc._pending["existing"] = loop.create_future()

    with pytest.raises(RuntimeError, match="Queue is full"):
        await asyncio.wait_for(
            svc._submit_resident_job(
                temp_file_path="audio.wav",
                params={},
                request_id="req-internal",
                model_spec=funasr_spec,
            ),
            timeout=0.2,
        )


@pytest.mark.asyncio
async def test_submit_resident_job_cleans_pending_when_wait_is_cancelled(funasr_spec):
    svc = _setup_service(funasr_spec)
    request_id = "req-cancelled"

    task = asyncio.create_task(
        svc._submit_resident_job(
            temp_file_path="audio.wav",
            params={},
            request_id=request_id,
            model_spec=funasr_spec,
        )
    )
    try:
        for _ in range(20):
            if request_id in svc._pending:
                break
            await asyncio.sleep(0.01)
        assert request_id in svc._pending

        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        assert request_id not in svc._pending
    finally:
        if not task.done():
            task.cancel()
        await _stop_service(svc)


@pytest.mark.asyncio
async def test_spawn_worker_cancels_existing_result_reader_before_startup_handshake(funasr_spec):
    class FakeQueue:
        def get(self):
            return ("READY", None)

        def get_nowait(self):
            raise _stdlib_queue.Empty

        def close(self) -> None:
            return None

        def join_thread(self) -> None:
            return None

        def put(self, item, timeout=None) -> None:
            return None

        def put_nowait(self, item) -> None:
            return None

    svc = TranscriptionService(
        engine_type=funasr_spec.engine_type,
        model_id=funasr_spec.model_id,
        max_queue_size=2,
        initial_model_spec=funasr_spec,
        idle_timeout=0,
    )
    svc.is_running = True
    old_reader = asyncio.create_task(asyncio.sleep(60))
    svc._result_reader_task = old_reader
    mock_process = MagicMock()
    mock_process.is_alive.return_value = True

    try:
        with (
            patch("src.services.transcription.multiprocessing.Process", return_value=mock_process),
            patch("src.services.transcription.multiprocessing.Queue", side_effect=[FakeQueue(), FakeQueue()]),
        ):
            await svc._spawn_worker(funasr_spec)

        assert old_reader.cancelled()
        assert svc._result_reader_task is not old_reader
        assert svc._result_reader_task is not None
    finally:
        await _stop_service(svc)


async def _next_job(svc: TranscriptionService) -> WorkerJob:
    assert svc._job_queue is not None
    job = await asyncio.to_thread(svc._job_queue.get, True, 1.0)
    assert isinstance(job, WorkerJob)
    return job


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_model", [False, True])
async def test_cancelled_submission_cleans_upload_and_allows_next_request(
    funasr_spec: ModelSpec, explicit_model: bool,
) -> None:
    svc = _setup_service(funasr_spec)
    selected = funasr_spec if explicit_model else None
    task = asyncio.create_task(svc.submit(_make_upload(), {}, "cancel-me", selected))
    job = await _next_job(svc)
    upload = Path(job.temp_file_path)
    assert upload.exists()
    try:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not upload.parent.exists()
        assert svc.queue_size == 0

        next_task = asyncio.create_task(svc.submit(_make_upload(), {}, "next", selected))
        next_job = await _next_job(svc)
        svc._resolve_future(next_job.uid, result={"text": "recovered"})
        result = await asyncio.wait_for(next_task, timeout=1.0)
        assert isinstance(result, dict) and result["text"] == "recovered"
        assert not Path(next_job.temp_file_path).parent.exists()
    finally:
        task.cancel()
        await _stop_service(svc)


@pytest.mark.asyncio
async def test_pending_submission_rejects_overflow_before_first_result(
    funasr_spec: ModelSpec,
) -> None:
    svc = _setup_service(funasr_spec, max_queue_size=1)
    task = asyncio.create_task(svc.submit(_make_upload(), {}, "first"))
    job = await _next_job(svc)
    try:
        assert svc.queue_size == 1
        with pytest.raises(RuntimeError, match="Queue is full"):
            await svc.submit(_make_upload(), {}, "overflow")
        assert not task.done()
        svc._resolve_future(job.uid, result="first result")
        assert await asyncio.wait_for(task, timeout=1.0) == "first result"
        assert svc.queue_size == 0
    finally:
        task.cancel()
        await _stop_service(svc)


@pytest.mark.asyncio
async def test_cancel_while_waiting_for_startup_cleans_unregistered_upload(
    funasr_spec: ModelSpec, tmp_path: Path,
) -> None:
    svc = _setup_service(funasr_spec)
    uploaded = asyncio.Event()
    task_dir = tmp_path / "request"

    def make_temp_dir(prefix: str) -> str:
        task_dir.mkdir()
        uploaded.set()
        return str(task_dir)

    await svc._spawn_lock.acquire()
    with patch("src.services.transcription.tempfile.mkdtemp", side_effect=make_temp_dir):
        task = asyncio.create_task(svc.submit(_make_upload(), {}, "waiting"))
        try:
            await asyncio.wait_for(uploaded.wait(), timeout=1.0)
            assert (task_dir / "original.wav").exists()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not task_dir.exists()
            assert svc.queue_size == 0
        finally:
            svc._spawn_lock.release()
            task.cancel()
            await _stop_service(svc)


@pytest.mark.asyncio
async def test_enqueue_failure_cleans_registered_upload(funasr_spec: ModelSpec) -> None:
    svc = _setup_service(funasr_spec)
    captured_uploads: list[Path] = []

    def reject_job(job: WorkerJob) -> None:
        captured_uploads.append(Path(job.temp_file_path))
        raise RuntimeError("IPC enqueue failed")

    assert svc._job_queue is not None
    with patch.object(svc._job_queue, "put_nowait", side_effect=reject_job):
        with pytest.raises(RuntimeError, match="IPC enqueue failed"):
            await svc.submit(_make_upload(), {}, "enqueue-failure")
    assert captured_uploads
    assert not captured_uploads[0].parent.exists()
    assert svc.queue_size == 0
