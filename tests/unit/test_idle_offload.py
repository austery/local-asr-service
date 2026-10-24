"""Idle/crash offload behavior through submission, without ML models."""

import asyncio
from io import BytesIO

import pytest
from fastapi import UploadFile

from src.core.model_registry import lookup
from tests.helpers.worker_transport import WorkerHarness


async def test_model_state_and_capabilities_follow_lazy_startup() -> None:
    harness = WorkerHarness()
    service = harness.service()
    assert not service.model_loaded
    assert service.capabilities == lookup("paraformer").capabilities
    task = asyncio.create_task(service.submit(UploadFile(file=BytesIO(b"audio")), {}, "first"))
    transport = await harness.next_transport()
    job = await transport.next_job()
    assert service.model_loaded
    transport.messages.put(("RESULT", job.uid, "done"))
    await asyncio.wait_for(task, 1)
    await service.stop_worker()
    assert not service.model_loaded


@pytest.mark.parametrize("idle", [True, False])
async def test_exit_fails_pending_and_next_request_spawns_fresh_worker(idle: bool) -> None:
    harness = WorkerHarness()
    service = harness.service()
    task = asyncio.create_task(service.submit(UploadFile(file=BytesIO(b"audio")), {}, "first"))
    old = await harness.next_transport()
    await old.next_job()
    if idle:
        old.messages.put(("IDLE_EXIT", None))
    else:
        old.running = False
    try:
        with pytest.raises(RuntimeError, match="idle|unexpectedly"):
            await asyncio.wait_for(task, 1)
        assert not service.model_loaded
        next_task = asyncio.create_task(service.submit(UploadFile(file=BytesIO(b"audio")), {}, "next"))
        new = await harness.next_transport()
        job = await new.next_job()
        new.messages.put(("RESULT", job.uid, "recovered"))
        assert (await asyncio.wait_for(next_task, 1)).payload == "recovered"
        assert old.closed
    finally:
        await service.stop_worker()
