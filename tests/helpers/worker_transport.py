"""Deterministic transport Adapter; session and admission remain real in tests."""

import asyncio
import queue
from collections.abc import Callable

from src.core.model_registry import ModelSpec, lookup
from src.services.transcription import TranscriptionService
from src.services.worker_session import WorkerSession
from src.workers.model_worker import WorkerJob
from src.workers.transport import WorkerConfig


class FakeTransport:
    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self.messages: queue.Queue[object] = queue.Queue()
        self.jobs: asyncio.Queue[WorkerJob] = asyncio.Queue()
        self.running = False
        self.closed = False
        self.ready = True
        self.start_error: Exception | None = None
        self.send_error: Exception | None = None
        self.on_send: Callable[[WorkerJob], None] | None = None

    def start(self) -> None:
        self.running = True
        if self.start_error is not None:
            raise self.start_error
        if self.ready:
            self.messages.put(("READY", None))

    def send(self, job: WorkerJob) -> None:
        self.jobs.put_nowait(job)
        if self.send_error is not None:
            raise self.send_error
        if self.on_send is not None:
            self.on_send(job)

    def receive(self) -> object:
        try:
            return self.messages.get_nowait()
        except queue.Empty:
            if not self.running:
                raise RuntimeError("Worker process died unexpectedly") from None
            raise

    def is_alive(self) -> bool:
        return self.running

    def close(self) -> None:
        self.running = False
        self.closed = True

    async def next_job(self) -> WorkerJob:
        return await asyncio.wait_for(self.jobs.get(), timeout=1)


class WorkerHarness:
    def __init__(self) -> None:
        self.transports: list[FakeTransport] = []
        self.created: asyncio.Queue[FakeTransport] = asyncio.Queue()
        self.configure: Callable[[FakeTransport], None] | None = None

    def factory(self, config: WorkerConfig) -> FakeTransport:
        assert all(t.closed for t in self.transports), "replacement before prior disposal"
        transport = FakeTransport(config)
        if self.configure is not None:
            self.configure(transport)
        self.transports.append(transport)
        self.created.put_nowait(transport)
        return transport

    def service(self, spec: ModelSpec | None = None, capacity: int = 5) -> TranscriptionService:
        spec = spec or lookup("paraformer")
        return TranscriptionService(
            spec.engine_type, spec.model_id, capacity, spec,
            worker_session=WorkerSession(self.factory),
        )

    async def next_transport(self) -> FakeTransport:
        return await asyncio.wait_for(self.created.get(), timeout=1)
