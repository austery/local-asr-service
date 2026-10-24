"""Concurrent backpressure and recovery using the real session."""

import asyncio
from io import BytesIO

import pytest
from fastapi import UploadFile

from tests.helpers.worker_transport import WorkerHarness


def upload() -> UploadFile:
    return UploadFile(file=BytesIO(b"audio"))


async def test_queue_backpressure() -> None:
    harness = WorkerHarness()
    service = harness.service(capacity=5)
    tasks = [asyncio.create_task(service.submit(upload(), {}, str(i))) for i in range(5)]
    transport = await harness.next_transport()
    jobs = [await transport.next_job() for _ in tasks]
    try:
        assert service.queue_size == 5
        with pytest.raises(RuntimeError, match="Queue is full"):
            await service.submit(upload(), {}, "overflow")
        for job in jobs:
            transport.messages.put(("RESULT", job.uid, job.uid))
        await asyncio.wait_for(asyncio.gather(*tasks), 1)
        assert service.queue_size == 0
    finally:
        await service.stop_worker()


async def test_worker_recovery_after_multiple_job_errors() -> None:
    harness = WorkerHarness()
    service = harness.service()
    try:
        for i in range(4):
            task = asyncio.create_task(service.submit(upload(), {}, str(i)))
            transport = await harness.next_transport() if i == 0 else harness.transports[0]
            job = await transport.next_job()
            transport.messages.put(("ERROR" if i < 3 else "RESULT", job.uid, "outcome"))
            if i < 3:
                with pytest.raises(RuntimeError, match="outcome"):
                    await asyncio.wait_for(task, 1)
            else:
                assert (await asyncio.wait_for(task, 1)).payload == "outcome"
        assert service.model_loaded
        assert len(harness.transports) == 1
    finally:
        await service.stop_worker()
