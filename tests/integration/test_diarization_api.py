"""Experimental discovery and required standalone selection; no model inference."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.diarization_port import SpeakerTurn
from src.core.diarization_result import DiarizationError, DiarizationPayload, ServiceBusyError
from src.core.model_registry import lookup
from src.main import app
from src.services.execution import ExecutionResult
from src.services.transcription import TranscriptionService


@pytest.fixture
def api() -> Iterator[tuple[TestClient, MagicMock]]:
    service = MagicMock(spec=TranscriptionService)
    service.current_model_spec = lookup("paraformer")
    service.start_worker = AsyncMock()
    service.stop_worker = AsyncMock()
    type(service).capabilities = PropertyMock(return_value=service.current_model_spec.capabilities)
    type(service).queue_size = PropertyMock(return_value=0)
    type(service).max_queue_size = PropertyMock(return_value=50)
    service.submit = AsyncMock(return_value=ExecutionResult({"text": "ordinary"}, "paraformer"))
    service.submit_diarization = AsyncMock(
        return_value=DiarizationPayload(
            2, [SpeakerTurn("Speaker 0", 0, 1.5), SpeakerTurn("Speaker 1", 1, 2)]
        )
    )
    with patch("src.main.TranscriptionService", return_value=service):
        with TestClient(app) as client:
            yield client, service


def test_discovery_keeps_asr_list_and_adds_experimental_group(
    api: tuple[TestClient, MagicMock],
) -> None:
    client, service = api
    payload = client.get("/v1/models").json()
    assert payload["current"] == "paraformer"
    assert "nemotron-diar" not in {m["alias"] for m in payload["models"]}
    entry = payload["diarizers"][0]
    assert (
        entry["alias"] == "nemotron-diar"
        and entry["experimental"]
        and entry["requires_explicit_selection"]
    )
    service.submit_diarization.assert_not_called()
    assert client.get("/openapi.json").json()["paths"]["/v1/audio/diarizations"]


@pytest.mark.parametrize("choice", [None, "", "   "])
def test_ordinary_empty_selection_is_disabled(
    api: tuple[TestClient, MagicMock], choice: str | None
) -> None:
    client, service = api
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"audio", "audio/wav")},
        data={} if choice is None else {"diarizer": choice},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "paraformer"
    service.submit_diarization.assert_not_called()


@pytest.mark.parametrize(
    "choice,status",
    [
        (None, 422),
        ("", 400),
        ("auto", 400),
        ("unknown", 400),
        ("sortformer-diar", 400),
        ("nemotron-diar", 200),
    ],
)
def test_standalone_requires_explicit_public_alias(
    api: tuple[TestClient, MagicMock], choice: str | None, status: int
) -> None:
    client, service = api
    response = client.post(
        "/v1/audio/diarizations",
        files={"file": ("a.m4a", b"audio", "audio/mp4")},
        data={} if choice is None else {"diarizer": choice},
    )
    assert response.status_code == status
    if status == 200:
        result = response.json()
        assert (
            result["num_speakers"] == 2 and result["turns"][1]["start"] < result["turns"][0]["end"]
        )
        assert result["diarizer"] == "nemotron-diar" and result["experimental"]
        assert "text" not in result
        assert client.get("/v1/models").json()["current"] == "paraformer"
        service.submit.assert_not_called()
    else:
        service.submit_diarization.assert_not_called()


@pytest.mark.parametrize(
    "field,choice",
    [
        ("diarizer", "nemotron-diar"),
        ("model", "nemotron-diar"),
        ("model", "mlx-community/Nemotron-3-Diarization"),
    ],
)
def test_wrong_endpoint_fails_before_any_execution(
    api: tuple[TestClient, MagicMock], field: str, choice: str
) -> None:
    client, service = api
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"audio", "audio/wav")},
        data={field: choice},
    )
    assert response.status_code == 400
    service.submit.assert_not_called()
    service.submit_diarization.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        ServiceBusyError(),
        DiarizationError("model_unavailable", "cache missing", 503),
        DiarizationError("invalid_output", "bad", 502),
        DiarizationError("timeout", "late", 504),
    ],
)
def test_explicit_error_categories_and_retry_header(
    api: tuple[TestClient, MagicMock], failure: DiarizationError
) -> None:
    client, service = api
    service.submit_diarization.side_effect = failure
    response = client.post(
        "/v1/audio/diarizations",
        files={"file": ("a.wav", b"audio", "audio/wav")},
        data={"diarizer": "nemotron-diar"},
    )
    assert response.status_code == failure.status
    assert response.json()["detail"]["code"] == failure.code
    if failure.status == 503:
        assert response.headers["Retry-After"] == "5"


def test_busy_ordinary_transcription_is_503_not_runtime_500(
    api: tuple[TestClient, MagicMock],
) -> None:
    client, service = api
    service.submit.side_effect = ServiceBusyError()
    response = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", b"audio", "audio/wav")}
    )
    assert response.status_code == 503 and response.headers["Retry-After"] == "5"


async def test_asgi_disconnect_waits_for_disposal_and_upload_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Send real ASGI disconnect, not just cancellation of a Python waiter."""
    import asyncio
    from pathlib import Path

    from tests.helpers.diarization_runtime import ControlledDiarization, service_for

    fake = ControlledDiarization()
    fake.release_close.clear()
    service = service_for(fake)
    app.state.service = service
    monkeypatch.setattr("src.services.transcription.cached_snapshot", lambda spec: tmp_path)
    from tests.helpers.diarization_runtime import start_disconnecting_request

    disconnect = asyncio.Event()
    request = start_disconnecting_request(app, disconnect)
    try:
        await asyncio.wait_for(fake.started.wait(), 1)
        disconnect.set()
        await asyncio.wait_for(fake.closing.wait(), 1)
        with pytest.raises(ServiceBusyError):
            from tests.helpers.diarization_runtime import upload

            await service.submit(upload(), {})
        assert fake.path is not None and Path(fake.path).exists()
        assert not request.done()
    finally:
        fake.release_close.set()
        if not request.done():
            request.cancel()
        await asyncio.wait_for(asyncio.gather(request, return_exceptions=True), 2)
        await service.stop_worker()
    assert not fake.path.parent.exists()


async def test_disconnect_reaps_real_stubborn_worker_before_releasing_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import json
    import sys
    from io import BytesIO

    import psutil
    from fastapi import UploadFile

    from src.services.diarization_session import DiarizationSession
    from tests.helpers.diarization_runtime import start_disconnecting_request

    marker = tmp_path / "started.json"
    closing = tmp_path / "closing"
    child = tmp_path / "native.py"
    child.write_text(f"""
import json,os,signal,sys,time
from pathlib import Path
signal.signal(signal.SIGTERM,lambda signum,frame:Path({str(closing)!r}).write_text('terminating'))
Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid(),'upload':sys.argv[1]}}))
time.sleep(60)
""")
    spec = lookup("paraformer")
    service = TranscriptionService(
        spec.engine_type,
        spec.model_id,
        initial_model_spec=spec,
        diarization_session_factory=lambda: DiarizationSession(
            (sys.executable, str(child)), reserve_bytes=0, timeout=10
        ),
    )
    app.state.service = service
    monkeypatch.setattr("src.services.transcription.cached_snapshot", lambda spec: tmp_path)
    disconnect = asyncio.Event()
    request = start_disconnecting_request(app, disconnect)
    owned: dict[str, object] = {}
    try:
        for _ in range(200):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        owned = json.loads(marker.read_text())
        disconnect.set()
        for _ in range(200):
            if closing.exists():
                break
            await asyncio.sleep(0.01)
        assert closing.exists()
        with pytest.raises(ServiceBusyError):
            await service.submit(UploadFile(file=BytesIO(b"audio"), filename="a.wav"), {})
        assert Path(str(owned["upload"])).exists()
        await asyncio.wait_for(request, 8)
        assert not psutil.pid_exists(int(str(owned["pid"])))
        assert not Path(str(owned["upload"])).parent.exists()
        assert service.current_model_spec is spec
    finally:
        disconnect.set()
        if not request.done():
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        await service.stop_worker()
