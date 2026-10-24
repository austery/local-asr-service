"""Model selection and disposal ordering through real admission and session."""

import asyncio
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import UploadFile

from src.core.model_registry import lookup
from tests.helpers.worker_transport import WorkerHarness


def upload() -> UploadFile:
    return UploadFile(file=BytesIO(b"audio"), filename="test.wav")


async def test_same_model_requests_reuse_worker() -> None:
    harness = WorkerHarness()
    service = harness.service()
    try:
        for uid in ("first", "second"):
            task = asyncio.create_task(service.submit(upload(), {}, uid, lookup("paraformer")))
            transport = await harness.next_transport() if uid == "first" else harness.transports[0]
            job = await transport.next_job()
            transport.messages.put(("RESULT", job.uid, uid))
            assert (await asyncio.wait_for(task, 1)).payload == uid
        assert len(harness.transports) == 1
    finally:
        await service.stop_worker()


@pytest.mark.parametrize("same_weights", [False, True])
async def test_switch_reaps_old_worker_and_fails_old_jobs_before_new_admission(
    same_weights: bool,
) -> None:
    harness = WorkerHarness()
    service = harness.service(lookup("qwen3-asr" if same_weights else "paraformer"))
    target = replace(lookup("qwen3-asr"), alias="same-weights") if same_weights else lookup("qwen3-asr")
    old = asyncio.create_task(service.submit(upload(), {}, "old"))
    first = await harness.next_transport()
    old_job = await first.next_job()
    switched = asyncio.create_task(service.submit(upload(), {}, "new", target))
    second = await harness.next_transport()
    job = await second.next_job()
    try:
        assert first.closed
        assert second.config.model_id == lookup("qwen3-asr").model_id
        assert Path(job.temp_file_path).exists()
        with pytest.raises(RuntimeError, match="terminated"):
            await old
        assert not Path(old_job.temp_file_path).parent.exists()
        second.messages.put(("RESULT", job.uid, "switched result"))
        result = await asyncio.wait_for(switched, 1)
        assert result.payload == "switched result"
        assert result.model == target.alias
        assert service.current_model_spec == target
    finally:
        await service.stop_worker()


async def test_failed_switch_cleans_upload_and_recovers_previous_selection(tmp_path: Path) -> None:
    harness = WorkerHarness()
    harness.configure = lambda t: setattr(t, "start_error", RuntimeError("switch failed"))
    service = harness.service()
    task_dir = tmp_path / "request"
    task_dir.mkdir()
    try:
        with patch("src.services.transcription.tempfile.mkdtemp", return_value=str(task_dir)):
            with pytest.raises(RuntimeError, match="switch failed"):
                await service.submit(upload(), {}, "failed", lookup("qwen3-asr"))
        assert not task_dir.exists()
        assert harness.transports[0].closed
        assert service.queue_size == 0
        assert service.current_model_spec == lookup("paraformer")
        harness.configure = None
        task = asyncio.create_task(service.submit(upload(), {}, "retry"))
        await harness.next_transport()  # failed generation
        transport = await harness.next_transport()
        job = await transport.next_job()
        assert transport.config.model_id == lookup("paraformer").model_id
        transport.messages.put(("RESULT", job.uid, "recovered"))
        assert (await asyncio.wait_for(task, 1)).payload == "recovered"
    finally:
        await service.stop_worker()


async def test_switch_to_apple_speech_releases_resident_without_spawning() -> None:
    harness = WorkerHarness()
    service = harness.service()
    task = asyncio.create_task(service.submit(upload(), {}, "resident"))
    transport = await harness.next_transport()
    await transport.next_job()
    await service._switch_worker(lookup("apple-speech"))
    with pytest.raises(RuntimeError, match="terminated"):
        await task
    assert transport.closed
    assert len(harness.transports) == 1
    assert not service.model_loaded
    assert service.current_model_spec == lookup("apple-speech")
    await service.stop_worker()


async def test_second_passthrough_enqueues_while_first_still_transcribing() -> None:
    harness = WorkerHarness()
    service = harness.service(capacity=2)
    first = asyncio.create_task(service.submit(upload(), {}, "first"))
    transport = await harness.next_transport()
    first_job = await transport.next_job()
    second = asyncio.create_task(service.submit(upload(), {}, "second"))
    second_job = await transport.next_job()
    try:
        assert service.queue_size == 2
        assert not first.done()
        transport.messages.put(("RESULT", second_job.uid, "two"))
        transport.messages.put(("RESULT", first_job.uid, "one"))
        assert (await asyncio.wait_for(first, 1)).payload == "one"
        assert (await asyncio.wait_for(second, 1)).payload == "two"
    finally:
        await service.stop_worker()
