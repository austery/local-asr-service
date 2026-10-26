"""Controlled native experiment for admission and actual ASGI disconnect tests."""

import asyncio
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from starlette.types import ASGIApp

from src.core.diarization_result import DiarizationPayload
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


def start_disconnecting_request(app: "ASGIApp", disconnect: asyncio.Event) -> asyncio.Task[None]:
    """Real multipart/ASGI disconnect messages through the complete middleware."""
    from starlette.types import Message, Scope

    boundary = "diarization-disconnect-test"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="diarizer"\r\n\r\nnemotron-diar\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.wav"\r\nContent-Type: audio/wav\r\n\r\naudio\r\n--{boundary}--\r\n'
    ).encode()
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        pass

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/diarizations",
        "raw_path": b"/v1/audio/diarizations",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 1000),
        "server": ("test", 80),
    }

    async def run_request() -> None:
        await app(scope, receive, send)

    return asyncio.create_task(run_request())
