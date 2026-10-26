"""Selection and native resource ownership through public service methods."""

import asyncio
import threading
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile

from src.core.diarization_registry import lookup_diarizer
from src.core.diarization_result import DiarizationError, DiarizationPayload, ServiceBusyError
from src.core.model_registry import lookup
from src.services.transcription import TranscriptionService
from src.services.worker_session import WorkerSession
from tests.helpers.worker_transport import WorkerHarness


def upload() -> UploadFile:
    return UploadFile(file=BytesIO(b"audio"), filename="test.wav")


class ControlledDiarization:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release_run = asyncio.Event()
        self.closing = asyncio.Event()
        self.release_close = asyncio.Event()
        self.release_close.set()
        self.path: Path | None = None
        self.close_error = False

    async def run(self, audio: Path, output: Path, checkpoint: Path) -> DiarizationPayload:
        self.path = audio
        self.started.set()
        await self.release_run.wait()
        return DiarizationPayload(1, [])

    async def close(self) -> None:
        self.closing.set()
        await self.release_close.wait()
        if self.close_error:
            raise RuntimeError("disposal failed")
        self.release_run.set()


def service_for(
    fake: ControlledDiarization, harness: WorkerHarness | None = None
) -> TranscriptionService:
    spec = lookup("paraformer")
    return TranscriptionService(
        spec.engine_type,
        spec.model_id,
        initial_model_spec=spec,
        worker_session=WorkerSession(harness.factory) if harness else None,
        diarization_session_factory=lambda: fake,
    )


@pytest.fixture
def cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.services.transcription.cached_snapshot", lambda spec: tmp_path)


@pytest.mark.usefixtures("cached")
async def test_explicit_experiment_blocks_every_asr_path_and_preserves_selection() -> None:
    fake = ControlledDiarization()
    service = service_for(fake)
    before = service.current_model_spec
    task = asyncio.create_task(
        service.submit_diarization(upload(), lookup_diarizer("nemotron-diar"))
    )
    await asyncio.wait_for(fake.started.wait(), 1)
    try:
        for model in (None, lookup("qwen3-asr"), lookup("apple-speech")):
            with pytest.raises(ServiceBusyError):
                await service.submit(upload(), {"language": "en"}, model_spec=model)
        with pytest.raises(ServiceBusyError):
            await service.submit_diarization(upload(), lookup_diarizer("nemotron-diar"))
        assert service.current_model_spec is before
        assert fake.path is not None and fake.path.is_file()
        fake.release_run.set()
        assert (await task).turns == []
        assert not fake.path.parent.exists()
        assert service.current_model_spec is before
    finally:
        fake.release_run.set()
        await service.stop_worker()
        await task


@pytest.mark.usefixtures("cached")
async def test_resident_worker_blocks_experiment_even_after_request_completes() -> None:
    fake, harness = ControlledDiarization(), WorkerHarness()
    service = service_for(fake, harness)
    task = asyncio.create_task(service.submit(upload(), {}, "ordinary"))
    transport = await harness.next_transport()
    job = await transport.next_job()
    try:
        with pytest.raises(ServiceBusyError):
            await service.submit_diarization(upload(), lookup_diarizer("nemotron-diar"))
        transport.messages.put(("RESULT", job.uid, "done"))
        await task
        with pytest.raises(ServiceBusyError):
            await service.submit_diarization(upload(), lookup_diarizer("nemotron-diar"))
        assert not fake.started.is_set()
    finally:
        await service.stop_worker()


@pytest.mark.usefixtures("cached")
async def test_cancel_retains_lease_and_upload_through_repeated_cancellation() -> None:
    fake = ControlledDiarization()
    fake.release_close.clear()
    service = service_for(fake)
    task = asyncio.create_task(
        service.submit_diarization(upload(), lookup_diarizer("nemotron-diar"))
    )
    await asyncio.wait_for(fake.started.wait(), 1)
    task.cancel()
    await asyncio.wait_for(fake.closing.wait(), 1)
    task.cancel()
    try:
        with pytest.raises(ServiceBusyError):
            await service.submit(upload(), {})
        assert fake.path is not None and fake.path.exists()
        assert not task.done()
    finally:
        fake.release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await service.stop_worker()
    assert not fake.path.parent.exists()


@pytest.mark.usefixtures("cached")
async def test_cleanup_failure_blocks_replacement_until_shutdown_retries() -> None:
    fake = ControlledDiarization()
    fake.close_error = True
    fake.release_run.set()
    service = service_for(fake)
    with pytest.raises(RuntimeError, match="disposal failed"):
        await service.submit_diarization(upload(), lookup_diarizer("nemotron-diar"))
    try:
        with pytest.raises(ServiceBusyError):
            await service.submit(upload(), {})
        assert fake.path is not None and fake.path.exists()
    finally:
        fake.close_error = False
        await service.stop_worker()
    assert not fake.path.parent.exists()


@pytest.mark.parametrize("passthrough", [False, True])
@pytest.mark.usefixtures("cached")
async def test_cancelled_apple_waiter_does_not_release_native_ownership(
    passthrough: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ControlledDiarization()
    service = service_for(fake)
    started, finished = threading.Event(), threading.Event()
    engine = MagicMock()

    def transcribe(*args: object, **kwargs: object) -> str:
        started.set()
        assert finished.wait(5)
        return "apple result"

    engine.transcribe_file.side_effect = transcribe
    monkeypatch.setattr(service, "_get_apple_speech_engine", lambda: engine)
    if passthrough:
        # Initial selection is configured, not changed by a test-private switch.
        spec = lookup("apple-speech")
        service = TranscriptionService(
            spec.engine_type,
            spec.model_id,
            initial_model_spec=spec,
            diarization_session_factory=lambda: fake,
        )
        monkeypatch.setattr(service, "_get_apple_speech_engine", lambda: engine)
    task = asyncio.create_task(
        service.submit(
            upload(), {"language": "en"}, model_spec=None if passthrough else lookup("apple-speech")
        )
    )
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    try:
        with pytest.raises(ServiceBusyError):
            await service.submit_diarization(upload(), lookup_diarizer("nemotron-diar"))
        assert not task.done()
    finally:
        finished.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await service.stop_worker()


async def test_missing_cache_does_not_create_session(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ControlledDiarization()
    service = service_for(fake)
    monkeypatch.setattr("src.services.transcription.cached_snapshot", lambda spec: None)
    with pytest.raises(DiarizationError) as error:
        await service.submit_diarization(upload(), lookup_diarizer("nemotron-diar"))
    assert error.value.code == "model_unavailable"
    assert not fake.started.is_set()
    await service.stop_worker()


@pytest.mark.usefixtures("cached")
async def test_shutdown_waits_for_experiment_cleanup_and_rejects_new_admission() -> None:
    fake = ControlledDiarization()
    fake.release_close.clear()
    service = service_for(fake)
    request = asyncio.create_task(
        service.submit_diarization(upload(), lookup_diarizer("nemotron-diar"))
    )
    await asyncio.wait_for(fake.started.wait(), 1)
    stopping = asyncio.create_task(service.stop_worker())
    await asyncio.wait_for(fake.closing.wait(), 1)
    try:
        with pytest.raises(RuntimeError, match="stopping"):
            await service.submit(upload(), {})
        assert not stopping.done()
        assert fake.path is not None and fake.path.exists()
    finally:
        fake.release_close.set()
        await asyncio.wait_for(stopping, 2)
        await asyncio.wait_for(request, 2)
    assert not fake.path.parent.exists()


@pytest.mark.usefixtures("cached")
async def test_naturally_exited_asr_is_disposed_before_diarization_starts() -> None:
    fake, harness = ControlledDiarization(), WorkerHarness()
    service = service_for(fake, harness)
    first = asyncio.create_task(service.submit(upload(), {}, "first"))
    transport = await harness.next_transport()
    job = await transport.next_job()
    transport.messages.put(("RESULT", job.uid, "done"))
    await first
    transport.running = False
    transport.messages.put(("IDLE_EXIT", None))
    task = asyncio.create_task(
        service.submit_diarization(upload(), lookup_diarizer("nemotron-diar"))
    )
    try:
        await asyncio.wait_for(fake.started.wait(), 1)
        assert transport.closed
        assert service.current_model_spec == lookup("paraformer")
        fake.release_run.set()
        await task
    finally:
        fake.release_run.set()
        await service.stop_worker()


@pytest.mark.usefixtures("cached")
async def test_failed_resident_disposal_blocks_new_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, harness = ControlledDiarization(), WorkerHarness()
    service = service_for(fake, harness)
    task = asyncio.create_task(service.submit(upload(), {}, "first"))
    transport = await harness.next_transport()
    job = await transport.next_job()
    transport.messages.put(("RESULT", job.uid, "done"))
    await task
    transport.running = False
    original = transport.close

    def fail_close() -> None:
        raise RuntimeError("owned transport cleanup failed")

    monkeypatch.setattr(transport, "close", fail_close)
    try:
        with pytest.raises(ServiceBusyError):
            await service.submit_diarization(upload(), lookup_diarizer("nemotron-diar"))
        assert not fake.started.is_set()
    finally:
        monkeypatch.setattr(transport, "close", original)
        await service.stop_worker()
