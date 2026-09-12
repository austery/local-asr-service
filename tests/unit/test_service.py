"""Submission behavior through the real session and a deterministic transport."""

import asyncio
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import UploadFile

from src.core.model_registry import lookup
from src.services.worker_session import WorkerRemoteError
from tests.helpers.worker_transport import WorkerHarness


def upload() -> UploadFile:
    return UploadFile(file=BytesIO(b"audio"), filename="test.wav")


@pytest.mark.parametrize("payload", [
    {"text": "hello", "segments": [], "duration": 1.0},
    "[Speaker 0]: hello",
])
async def test_submit_delivers_result_and_cleans_upload(payload: object) -> None:
    harness = WorkerHarness()
    service = harness.service()
    task = asyncio.create_task(service.submit(upload(), {}, "result"))
    transport = await harness.next_transport()
    job = await transport.next_job()
    path = Path(job.temp_file_path)
    assert path.read_bytes() == b"audio"
    transport.messages.put(("RESULT", job.uid, payload))
    try:
        assert (await asyncio.wait_for(task, 1)).payload == payload
        assert service.queue_size == 0
        assert not path.parent.exists()
    finally:
        await service.stop_worker()


@pytest.mark.parametrize("error", [
    ("ERROR", "failed", "Model Error"),
    ("ERROR", "failed", "ValueError", "Model Error"),
])
async def test_worker_error_handling_and_upload_cleanup(error: tuple[str, ...]) -> None:
    harness = WorkerHarness()
    service = harness.service()
    task = asyncio.create_task(service.submit(upload(), {}, "failed"))
    transport = await harness.next_transport()
    job = await transport.next_job()
    transport.messages.put(error)
    try:
        with pytest.raises(RuntimeError, match="Model Error") as caught:
            await asyncio.wait_for(task, 1)
        if len(error) == 4:
            assert isinstance(caught.value, WorkerRemoteError)
            assert caught.value.exc_type_name == "ValueError"
        assert service.queue_size == 0
        assert not Path(job.temp_file_path).parent.exists()
    finally:
        await service.stop_worker()


@pytest.mark.parametrize("explicit_model", [False, True])
async def test_cancelled_submission_cleans_upload_and_allows_next_request(explicit_model: bool) -> None:
    harness = WorkerHarness()
    service = harness.service()
    selected = lookup("paraformer") if explicit_model else None
    task = asyncio.create_task(service.submit(upload(), {}, "cancel-me", selected))
    transport = await harness.next_transport()
    job = await transport.next_job()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not Path(job.temp_file_path).parent.exists()
    assert service.queue_size == 0
    try:
        # A late result from the cancelled request cannot resolve another waiter.
        transport.messages.put(("RESULT", job.uid, "late"))
        next_task = asyncio.create_task(service.submit(upload(), {}, "next", selected))
        next_job = await transport.next_job()
        transport.messages.put(("RESULT", next_job.uid, "recovered"))
        assert (await asyncio.wait_for(next_task, 1)).payload == "recovered"
        assert not Path(next_job.temp_file_path).parent.exists()
    finally:
        await service.stop_worker()


@pytest.mark.parametrize("internal", [False, True])
async def test_pending_submission_rejects_overflow_before_first_result(internal: bool) -> None:
    harness = WorkerHarness()
    service = harness.service(capacity=1)
    first = asyncio.create_task(service.submit(upload(), {}, "first"))
    transport = await harness.next_transport()
    job = await transport.next_job()
    try:
        assert service.queue_size == 1
        with pytest.raises(RuntimeError, match="Queue is full"):
            if internal:
                await service._submit_resident_job("audio.wav", {}, "overflow", None)
            else:
                await service.submit(upload(), {}, "overflow")
        assert not first.done()
        transport.messages.put(("RESULT", job.uid, "first result"))
        assert (await asyncio.wait_for(first, 1)).payload == "first result"
    finally:
        await service.stop_worker()


async def test_cancel_while_waiting_for_startup_cleans_unregistered_upload(tmp_path: Path) -> None:
    harness = WorkerHarness()
    service = harness.service()
    uploaded = asyncio.Event()
    task_dir = tmp_path / "request"

    def make_temp_dir(prefix: str) -> str:
        task_dir.mkdir()
        uploaded.set()
        return str(task_dir)

    await service._spawn_lock.acquire()
    with patch("src.services.transcription.tempfile.mkdtemp", side_effect=make_temp_dir):
        task = asyncio.create_task(service.submit(upload(), {}, "waiting"))
        try:
            await asyncio.wait_for(uploaded.wait(), 1)
            assert (task_dir / "original.wav").exists()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not task_dir.exists()
            assert service.queue_size == 0
        finally:
            service._spawn_lock.release()
            await service.stop_worker()


async def test_enqueue_failure_cleans_registered_upload() -> None:
    harness = WorkerHarness()
    harness.configure = lambda t: setattr(t, "send_error", RuntimeError("IPC enqueue failed"))
    service = harness.service()
    try:
        with pytest.raises(RuntimeError, match="IPC enqueue failed"):
            await service.submit(upload(), {}, "enqueue-failure")
        job = await harness.transports[0].next_job()
        assert not Path(job.temp_file_path).parent.exists()
        assert service.queue_size == 0
    finally:
        await service.stop_worker()


async def test_stop_serializes_with_startup_and_rejects_later_admission() -> None:
    harness = WorkerHarness()
    harness.configure = lambda t: setattr(t, "ready", False)
    service = harness.service()
    request = asyncio.create_task(service.submit(upload(), {}, "startup"))
    transport = await harness.next_transport()
    stop = asyncio.create_task(service.stop_worker())
    # Releasing startup lets the already-admitted request enqueue, then stop
    # acquires session lifetime ownership and terminates it. No worker may escape shutdown.
    transport.messages.put(("READY", None))
    with pytest.raises(RuntimeError, match="terminated"):
        await asyncio.wait_for(request, 1)
    await asyncio.wait_for(stop, 1)
    assert transport.closed
    with pytest.raises(RuntimeError, match="stopping"):
        await service.submit(upload(), {}, "late")
