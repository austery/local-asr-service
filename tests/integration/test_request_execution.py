"""Exercise HTTP and actual admission together with an event-controlled worker."""

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, UploadFile
from httpx import ASGITransport, AsyncClient, Response

from src.api.routes import router
from src.core.base_engine import TranscriptionInputError
from src.core.model_registry import ModelSpec, lookup
from src.services.execution import ExecutionResult
from src.services.transcription import TranscriptionService
from src.workers.model_worker import WorkerJob


@dataclass
class WorkerControl:
    starting: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    admitted: asyncio.Event = field(default_factory=asyncio.Event)
    jobs: list[tuple[str, WorkerJob]] = field(default_factory=list)
    fail_start: bool = False


class ControlledService(TranscriptionService):
    """Replace only worker transport; retain routing, validation, enqueue, and cleanup."""

    def __init__(self, spec: ModelSpec, control: WorkerControl) -> None:
        super().__init__(spec.engine_type, spec.model_id, initial_model_spec=spec)
        self.control = control

    async def submit(
        self, file: UploadFile, params: dict[str, object], request_id: str = "unknown",
        model_spec: ModelSpec | None = None,
    ) -> ExecutionResult:
        if request_id != "switch":
            self.control.admitted.set()
        return await super().submit(file, params, request_id, model_spec)

    async def _shutdown_worker(self) -> None:
        self._worker = None

    async def _spawn_worker(self, model_spec: ModelSpec | None = None) -> None:
        self.control.starting.set()
        await self.control.release.wait()
        if self.control.fail_start:
            self.control.fail_start = False
            raise RuntimeError("controlled startup failure")
        self._current_model_spec = model_spec or self.current_model_spec
        worker = MagicMock()
        worker.is_alive.return_value = True
        self._worker = worker
        jobs = MagicMock()
        jobs.put_nowait.side_effect = self._deliver
        self._job_queue = jobs

    def _deliver(self, job: WorkerJob) -> None:
        assert self.current_model_spec is not None
        self.control.jobs.append((self.current_model_spec.alias, job))
        asyncio.get_running_loop().call_soon(
            self._resolve_future, job.uid, {"text": "controlled result"},
        )


def _app(service: TranscriptionService) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.service = service
    app.state.model_id = "startup-fallback"
    app.state.engine_type = "funasr"
    return app


async def _switch_and_request(
    old: str, new: str, data: dict[str, str], *, fail_start: bool = False,
) -> tuple[Response, WorkerControl]:
    control = WorkerControl(fail_start=fail_start)
    service = ControlledService(lookup(old), control)
    switch = asyncio.create_task(service.submit(
        UploadFile(file=BytesIO(b"audio"), filename="switch.wav"), {}, "switch", lookup(new),
    ))
    request: asyncio.Task[Response] | None = None
    entered: asyncio.Task[bool] | None = None
    try:
        await asyncio.wait_for(control.starting.wait(), timeout=1)
        async with AsyncClient(transport=ASGITransport(app=_app(service)), base_url="http://test") as client:
            request = asyncio.create_task(client.post(
                "/v1/audio/transcriptions", data=data,
                files={"file": ("request.wav", b"audio", "audio/wav")},
            ))
            entered = asyncio.create_task(control.admitted.wait())
            # On the old implementation invalid requests could fail in the route
            # before reaching submit. Observe either outcome without timing sleeps.
            done, _ = await asyncio.wait({request, entered}, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            assert done, "Request never reached admission or returned an HTTP error"
            control.release.set()
            response = await asyncio.wait_for(request, timeout=1)
        if fail_start:
            with pytest.raises(RuntimeError, match="controlled startup failure"):
                await switch
        else:
            await switch
        assert service.queue_size == 0
        assert all(not Path(job.temp_file_path).parent.exists() for _, job in control.jobs)
        return response, control
    finally:
        control.release.set()
        for task in (switch, request, entered):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError, RuntimeError):
                    await task


@pytest.mark.asyncio
@pytest.mark.parametrize("placeholder", ["", "whisper-1"])
async def test_response_reports_the_model_selected_after_switch(placeholder: str) -> None:
    response, control = await _switch_and_request(
        "paraformer", "sensevoice-small", {"model": placeholder},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "sensevoice-small"
    assert control.jobs[-1][0] == "sensevoice-small"


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [{"with_timestamp": "true"}, {"output_format": "srt"}])
async def test_new_model_without_timestamps_rejects_before_inference(data: dict[str, str]) -> None:
    response, control = await _switch_and_request("paraformer", "sensevoice-small", data)
    assert response.status_code == 400
    assert "sensevoice-small" in response.json()["detail"]
    assert [job.uid for _, job in control.jobs] == ["switch"]


@pytest.mark.asyncio
async def test_new_model_with_timestamps_accepts_previously_unsupported_request() -> None:
    response, control = await _switch_and_request(
        "sensevoice-small", "paraformer", {"with_timestamp": "true"},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "paraformer"
    assert control.jobs[-1][0] == "paraformer"


@pytest.mark.asyncio
async def test_failed_switch_uses_prior_selection_for_passthrough() -> None:
    response, control = await _switch_and_request(
        "paraformer", "sensevoice-small", {"with_timestamp": "true"}, fail_start=True,
    )
    assert response.status_code == 200
    assert response.json()["model"] == "paraformer"
    assert [model for model, _ in control.jobs] == ["paraformer"]


@pytest.mark.asyncio
async def test_explicit_selection_is_stable_while_waiting_for_another_switch() -> None:
    response, control = await _switch_and_request(
        "paraformer", "sensevoice-small", {"model": "qwen3-asr"},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "qwen3-asr"
    assert [model for model, _ in control.jobs] == ["sensevoice-small", "qwen3-asr"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("alias", "params"), [
    ("sensevoice-small", {"output_format": "srt"}),
    ("apple-speech", {"language": "auto"}),
    ("mlx-community/custom", {"with_timestamp": True}),
])
async def test_explicit_invalid_request_never_waits_for_worker(
    alias: str, params: dict[str, object],
) -> None:
    control = WorkerControl()
    service = ControlledService(lookup("paraformer"), control)
    async with service._spawn_lock:
        with pytest.raises(TranscriptionInputError):
            await asyncio.wait_for(service.submit(
                UploadFile(file=BytesIO(b"audio"), filename="request.wav"), params,
                "invalid", lookup(alias),
            ), timeout=1)
    assert not control.starting.is_set()
    assert control.jobs == []


@pytest.mark.asyncio
async def test_passthrough_apple_language_validation_precedes_sidecar_execution() -> None:
    control = WorkerControl()
    service = ControlledService(lookup("apple-speech"), control)
    async with AsyncClient(transport=ASGITransport(app=_app(service)), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions", data={"language": "auto"},
            files={"file": ("request.wav", b"audio", "audio/wav")},
        )
    assert response.status_code == 400
    assert "explicit language" in response.json()["detail"]
    assert control.jobs == []


@pytest.mark.asyncio
async def test_custom_model_returns_its_full_path_identity() -> None:
    control = WorkerControl()
    control.release.set()
    service = ControlledService(lookup("paraformer"), control)
    async with AsyncClient(transport=ASGITransport(app=_app(service)), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions", data={"model": "mlx-community/custom"},
            files={"file": ("request.wav", b"audio", "audio/wav")},
        )
    assert response.status_code == 200
    assert response.json()["model"] == "mlx-community/custom"


@pytest.mark.asyncio
async def test_completion_identity_survives_a_later_runtime_change() -> None:
    class LaterSwitchService(ControlledService):
        def _resolve_future(
            self, uid: str, result: object | None = None, error: Exception | None = None,
        ) -> None:
            super()._resolve_future(uid, result, error)
            self._current_model_spec = lookup("sensevoice-small")

    control = WorkerControl()
    control.release.set()
    service = LaterSwitchService(lookup("paraformer"), control)
    async with AsyncClient(transport=ASGITransport(app=_app(service)), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions", data={"model": "qwen3-asr"},
            files={"file": ("request.wav", b"audio", "audio/wav")},
        )
    assert service.current_model_spec == lookup("sensevoice-small")
    assert response.status_code == 200
    assert response.json()["model"] == "qwen3-asr"
