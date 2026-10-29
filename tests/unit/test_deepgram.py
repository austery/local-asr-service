"""Public cloud behavior with a local HTTP transport; never submit paid audio."""

import asyncio
import copy
from collections.abc import AsyncIterator, Callable, Coroutine
from io import BytesIO
from pathlib import Path
from typing import cast
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, UploadFile

from src.adapters.deepgram_http import DeepgramAdapter
from src.adapters.deepgram_output import normalize_deepgram
from src.api.routes import router
from src.core.model_registry import lookup
from src.core.remote_port import RemoteTranscript, RemoteTranscriptionError
from src.services.remote_session import RemoteSession
from src.services.transcription import TranscriptionService
from src.services.worker_session import WorkerSession
from tests.helpers.worker_transport import WorkerHarness

ALIAS = "deepgram-nova-3-diarize"


def response_data() -> dict[str, object]:
    word = {"word": "Hello", "start": 0.0, "end": 1.0, "speaker": 0}
    return {
        "metadata": {"duration": 2.0, "request_id": str(uuid4())},
        "results": {
            "channels": [{"alternatives": [{"transcript": "Hello", "words": [word]}]}],
            "utterances": [{"start": 0.0, "end": 1.0, "speaker": 0, "transcript": "Hello", "words": [word]}],
        },
    }


@pytest.fixture(autouse=True)
def enable_for_fake_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.services.remote_session.DEEPGRAM_ENABLED", True)
    monkeypatch.setattr("src.services.remote_session.DEEPGRAM_API_KEY", "fake-test-key")


def mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def utterances(data: dict[str, object]) -> list[dict[str, object]]:
    value = mapping(data["results"])["utterances"]
    assert isinstance(value, list)
    return cast(list[dict[str, object]], value)


def words(utterance: dict[str, object]) -> list[dict[str, object]]:
    value = utterance["words"]
    assert isinstance(value, list)
    return cast(list[dict[str, object]], value)


def upload() -> UploadFile:
    return UploadFile(file=BytesIO(b"audio"), filename="test.wav")


def service_with_http(handler: Callable[[httpx.Request], httpx.Response] | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]], capacity: int = 50) -> TranscriptionService:
    adapter = DeepgramAdapter("fake-test-key", transport=httpx.MockTransport(handler))
    return TranscriptionService("funasr", "local", capacity, lookup("paraformer"), remote_session=RemoteSession(adapter))


def app_for(service: TranscriptionService) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.service = service
    return app


@pytest.mark.parametrize("format_name", ["verbose_json", "json", "text", "srt"])
async def test_http_contract_and_local_selection_are_preserved(format_name: str) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert await request.aread() == b"audio"
        assert dict(request.url.params) == {"model": "nova-3", "language": "en", "utterances": "true", "diarize_model": "v2"}
        assert request.headers["authorization"] == "Token fake-test-key"
        return httpx.Response(200, json=response_data())

    service = service_with_http(handler)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_for(service)), base_url="http://test") as client:
        response = await client.post("/v1/audio/transcriptions", files={"file": ("test.wav", b"audio", "audio/wav")}, data={"model": ALIAS, "language": "en", "response_format": format_name})
        assert response.status_code == 200, response.text
        if format_name == "srt":
            assert "00:00:00,000 --> 00:00:01,000" in response.text
            assert "[Speaker 0]: Hello" in response.text
        else:
            assert response.json()["model"] == ALIAS
            if format_name == "text":
                assert response.json()["text"] == "[Speaker 0]: Hello"
            else:
                assert response.json()["segments"][0] == {"id": 0, "speaker": "Speaker 0", "start": 0.0, "end": 1.0, "text": "Hello"}
        assert (await client.get("/v1/models/current")).json()["model_alias"] == "paraformer"
    assert len(requests) == 1
    assert not service.model_loaded
    assert service.current_model_spec == lookup("paraformer")
    assert service.queue_size == 0
    await service.stop_worker()


@pytest.mark.parametrize("language", ["auto", "multi", "fr", "zh-CN", ""])
async def test_unsupported_language_never_sends(language: str) -> None:
    service = service_with_http(lambda r: pytest.fail("network must not run"))
    with pytest.raises(RemoteTranscriptionError) as caught:
        await service.submit(upload(), {"language": language}, model_spec=lookup(ALIAS))
    assert caught.value.status_code == 400
    assert service.queue_size == 0


async def test_disabled_and_key_only_discovery_fail_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.services.remote_session.DEEPGRAM_ENABLED", False)
    service = service_with_http(lambda r: pytest.fail("network must not run"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_for(service)), base_url="http://test") as client:
        entry = next(m for m in (await client.get("/v1/models")).json()["models"] if m["alias"] == ALIAS)
        assert entry["requestable"] is False
        assert entry["provider"] == "deepgram"
        assert "acceptance" in entry["unavailable_reason"]
        response = await client.post("/v1/audio/transcriptions", files={"file": ("x.wav", b"audio")}, data={"model": ALIAS, "language": "en"})
        assert response.status_code == 503
        assert "fake-test-key" not in response.text


@pytest.mark.parametrize("size,filename,format_name,status", [(200 * 1024 * 1024 + 1, "x.wav", "json", 413), (10, "x.exe", "json", 400), (0, "x.wav", "json", 400), (10, "x.wav", "nonsense", 400)])
def test_preflight_limits(size: int, filename: str, format_name: str, status: int) -> None:
    with pytest.raises(RemoteTranscriptionError) as caught:
        RemoteSession().preflight(filename, size, {"language": "en", "output_format": format_name})
    assert caught.value.status_code == status


@pytest.mark.parametrize("field,value", [("speaker", None), ("speaker", True), ("speaker", -1), ("speaker", "0"), ("start", float("nan")), ("end", float("inf")), ("end", 0.0), ("transcript", 7), ("words", [])])
def test_malformed_utterances_fail_closed(field: str, value: object) -> None:
    data = response_data()
    utterances(data)[0][field] = value
    with pytest.raises(RemoteTranscriptionError, match="invalid diarized output"):
        normalize_deepgram(data, "en")


@pytest.mark.parametrize("field", ["speaker", "start", "end", "word"])
def test_missing_word_fields_fail_closed(field: str) -> None:
    data = response_data()
    del words(utterances(data)[0])[0][field]
    with pytest.raises(RemoteTranscriptionError):
        normalize_deepgram(data, "zh")


def test_silence_and_overlap_preserve_provider_structure() -> None:
    silence = {"metadata": {"duration": 2.0}, "results": {"channels": [{"alternatives": [{"transcript": "", "words": []}]}], "utterances": []}}
    assert normalize_deepgram(silence, "zh").segments == ()
    data = response_data()
    second = copy.deepcopy(utterances(data)[0])
    second["speaker"] = 1
    words(second)[0]["speaker"] = 1
    utterances(data).append(second)
    assert [s.speaker for s in normalize_deepgram(data, "en").segments] == ["Speaker 0", "Speaker 1"]
    mapping(data["results"])["utterances"] = []
    with pytest.raises(RemoteTranscriptionError):
        normalize_deepgram(data, "en")


@pytest.mark.parametrize("status,expected", [(302, 502), (401, 502), (403, 502), (429, 503), (500, 502)])
async def test_provider_failures_are_sanitized_and_never_retried(tmp_path: Path, status: int, expected: int) -> None:
    calls = 0
    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="fake-test-key private transcript", headers={"location": "https://unintended.example"})
    path = tmp_path / "x.wav"
    path.write_bytes(b"audio")
    with pytest.raises(RemoteTranscriptionError) as caught:
        await DeepgramAdapter("fake-test-key", transport=httpx.MockTransport(handler)).transcribe(path, "en", str(uuid4()))
    assert caught.value.status_code == expected
    assert "fake-test-key" not in str(caught.value)
    assert "private transcript" not in str(caught.value)
    assert calls == 1


class BlockingProvider:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closing = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.path: Path | None = None
        self.calls = 0

    async def transcribe(self, path: Path, language: str, request_id: str) -> RemoteTranscript:
        self.path = path
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
            return normalize_deepgram(response_data(), language)
        finally:
            self.closing.set()
            await self.allow_close.wait()
            assert path.exists(), "upload deleted before provider disposal"


async def test_shared_capacity_includes_native_startup_and_cloud_waiters() -> None:
    harness = WorkerHarness()
    harness.configure = lambda t: setattr(t, "ready", False)
    provider = BlockingProvider()
    service = TranscriptionService("funasr", "local", 2, lookup("paraformer"), worker_session=WorkerSession(harness.factory), remote_session=RemoteSession(provider))
    native = asyncio.create_task(service.submit(upload(), {}, "local"))
    transport = await harness.next_transport()
    cloud = asyncio.create_task(service.submit(upload(), {"language": "en"}, "cloud", lookup(ALIAS)))
    await provider.started.wait()
    assert service.queue_size == 2
    with pytest.raises(RuntimeError, match="Queue is full"):
        await service.submit(upload(), {"language": "en"}, "overflow", lookup(ALIAS))
    provider.allow_close.set()
    cloud.cancel()
    native.cancel()
    await asyncio.gather(cloud, native, return_exceptions=True)
    transport.messages.put(("READY", None))
    await service.stop_worker()


@pytest.mark.parametrize("shutdown", [False, True])
async def test_cancellation_and_shutdown_await_provider_close_before_upload_cleanup(shutdown: bool) -> None:
    provider = BlockingProvider()
    service = TranscriptionService("funasr", "local", remote_session=RemoteSession(provider))
    task = asyncio.create_task(service.submit(upload(), {"language": "en"}, "cloud", lookup(ALIAS)))
    await provider.started.wait()
    if shutdown:
        stopping = asyncio.create_task(service.stop_worker())
    else:
        task.cancel()
    await provider.closing.wait()
    assert provider.path is not None
    assert provider.path.exists()
    if shutdown:
        assert not stopping.done()
        stopping.cancel()
    else:
        task.cancel()  # repeated cancellation must still retain upload ownership
    provider.allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    if shutdown:
        with pytest.raises(asyncio.CancelledError):
            await stopping
        with pytest.raises(RuntimeError, match="stopping"):
            await service.submit(upload(), {"language": "en"}, model_spec=lookup(ALIAS))
    assert not provider.path.parent.exists()
    assert service.queue_size == 0


async def test_queue_deadline_sends_nothing_and_active_timeout_is_unknown(tmp_path: Path) -> None:
    provider = BlockingProvider()
    provider.allow_close.set()
    session = RemoteSession(provider)
    path = tmp_path / "x.wav"
    path.write_bytes(b"audio")
    first = asyncio.create_task(session.submit(path, "en", "one", session.deadline()))
    await provider.started.wait()
    with pytest.raises(RemoteTranscriptionError, match="expired in queue"):
        await session.submit(path, "en", "two", asyncio.get_running_loop().time() + 0.01)
    assert provider.calls == 1
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    with pytest.raises(RemoteTranscriptionError, match="result unknown"):
        await session.submit(path, "en", "three", asyncio.get_running_loop().time() + 0.01)
    assert provider.calls == 2
    await session.close()


@pytest.mark.parametrize("passthrough", [None, "", "whisper-1"])
async def test_cloud_completion_does_not_change_passthrough(passthrough: str | None) -> None:
    from src.api.routes import _resolve_model
    harness = WorkerHarness()
    service = harness.service()
    adapter = DeepgramAdapter("fake", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response_data())))
    service._remote = RemoteSession(adapter)
    cloud = await service.submit(upload(), {"language": "en"}, model_spec=lookup(ALIAS))
    assert cloud.model == ALIAS
    assert harness.transports == []
    local = asyncio.create_task(service.submit(upload(), {}, "local", _resolve_model(passthrough)))
    transport = await harness.next_transport()
    job = await transport.next_job()
    assert transport.config.engine_type == "funasr"
    transport.messages.put(("RESULT", job.uid, "local"))
    assert (await local).model == "paraformer"
    await service.stop_worker()


def test_cloud_startup_rejected() -> None:
    with pytest.raises(ValueError, match="startup MODEL_ID"):
        TranscriptionService("funasr", "nova-3", initial_model_spec=lookup(ALIAS))


async def test_missing_key_is_actionable_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.services.remote_session.DEEPGRAM_API_KEY", "")
    service = service_with_http(lambda r: pytest.fail("network must not run"))
    with pytest.raises(RemoteTranscriptionError, match="API_KEY is missing") as caught:
        await service.submit(upload(), {"language": "en"}, model_spec=lookup(ALIAS))
    assert caught.value.status_code == 503


@pytest.mark.parametrize("error", [httpx.ConnectTimeout("private fake-test-key"), httpx.WriteTimeout("private fake-test-key"), httpx.ReadTimeout("private fake-test-key"), httpx.ReadError("private fake-test-key")])
async def test_transport_errors_report_unknown_without_resending(error: httpx.HTTPError) -> None:
    calls = 0
    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await request.aread()
        raise error
    service = service_with_http(handler)
    with pytest.raises(RemoteTranscriptionError, match="result unknown") as caught:
        await service.submit(upload(), {"language": "en"}, model_spec=lookup(ALIAS))
    assert "fake-test-key" not in str(caught.value)
    assert "private" not in str(caught.value)
    assert calls == 1
    assert service.queue_size == 0


@pytest.mark.parametrize("kind", ["malformed_json", "malformed_shape", "too_large", "compressed"])
async def test_bad_http_output_is_bounded_and_sanitized(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.adapters.deepgram_http._MAX_RESPONSE_BYTES", 128)
    async def handler(request: httpx.Request) -> httpx.Response:
        if kind == "malformed_json":
            return httpx.Response(200, content=b"private fake-test-key")
        if kind == "malformed_shape":
            return httpx.Response(200, json={"private": "fake-test-key"})
        if kind == "compressed":
            # Use an unrecognized encoding to avoid MockTransport decoding eagerly.
            return httpx.Response(200, content=b"body", headers={"content-encoding": "unexpected"})
        return httpx.Response(200, content=b"x" * 129)
    service = service_with_http(handler)
    with pytest.raises(RemoteTranscriptionError) as caught:
        await service.submit(upload(), {"language": "en"}, model_spec=lookup(ALIAS))
    assert caught.value.status_code == 502
    assert "private" not in str(caught.value)
    assert "fake-test-key" not in str(caught.value)


async def test_http_status_mapping_for_unknown_failure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("upstream credentials or body must never surface")
    service = service_with_http(handler)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_for(service)), base_url="http://test") as client:
        response = await client.post("/v1/audio/transcriptions", files={"file": ("x.wav", b"audio")}, data={"model": ALIAS, "language": "en"})
        assert response.status_code == 504
        assert "result unknown" in response.json()["detail"]
        assert "credentials" not in response.text


async def test_serial_cloud_capacity_and_queued_cancel_never_uploads() -> None:
    provider = BlockingProvider()
    service = TranscriptionService("funasr", "local", 2, remote_session=RemoteSession(provider))
    first = asyncio.create_task(service.submit(upload(), {"language": "en"}, "one", lookup(ALIAS)))
    await provider.started.wait()
    second = asyncio.create_task(service.submit(upload(), {"language": "en"}, "two", lookup(ALIAS)))
    await asyncio.sleep(0)
    assert service.queue_size == 2
    assert provider.calls == 1
    with pytest.raises(RuntimeError, match="Queue is full"):
        await service.submit(upload(), {}, "native-overflow")
    with pytest.raises(RuntimeError, match="Queue is full"):
        await service.submit(upload(), {"language": "en"}, "apple-overflow", lookup("apple-speech"))
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert provider.calls == 1
    provider.allow_close.set()
    provider.release.set()
    assert (await first).model == ALIAS
    await service.stop_worker()


async def test_expired_admission_deadline_never_calls_provider(tmp_path: Path) -> None:
    provider = BlockingProvider()
    provider.allow_close.set()
    session = RemoteSession(provider)
    path = tmp_path / "x.wav"
    path.write_bytes(b"audio")
    with pytest.raises(RemoteTranscriptionError, match="no upload was sent"):
        await session.submit(path, "en", "expired", asyncio.get_running_loop().time() - 1)
    assert provider.calls == 0
    await session.close()


class OwnedHTTPTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.reading = asyncio.Event()
        self.closed = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.request: httpx.Request | None = None
        self.path: Path | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        await request.aread()
        self.started.set()
        await self.reading.wait()
        return httpx.Response(200, json=response_data())

    async def aclose(self) -> None:
        self.closed.set()
        await self.allow_close.wait()
        assert self.path is not None and self.path.exists()


async def test_actual_client_disposal_survives_simultaneous_cancel_and_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = OwnedHTTPTransport()
    provider = DeepgramAdapter("fake-test-key", transport=transport)
    service = TranscriptionService("funasr", "local", remote_session=RemoteSession(provider))
    from src.adapters import deepgram_http
    original_audio = deepgram_http._audio
    async def tracked_audio(path: Path) -> AsyncIterator[bytes]:
        transport.path = path
        async for chunk in original_audio(path):
            yield chunk
    monkeypatch.setattr(deepgram_http, "_audio", tracked_audio)
    task = asyncio.create_task(service.submit(upload(), {"language": "en"}, "cloud", lookup(ALIAS)))
    await transport.started.wait()
    task.cancel()
    await transport.closed.wait()
    stop = asyncio.create_task(service.stop_worker())
    await asyncio.sleep(0)
    assert not task.done()
    assert transport.path is not None
    assert transport.path.exists()
    stop.cancel()
    task.cancel()
    transport.allow_close.set()
    for pending in (task, stop):
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert not transport.path.parent.exists()
    assert transport.request is not None
    assert transport.request.extensions["timeout"] == {"connect": 10, "read": 660, "write": 60, "pool": 10}
    assert transport.request.url.host == "api.deepgram.com"


async def test_only_uuid_request_evidence_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    local = str(uuid4())
    remote = str(uuid4())
    async def handler(request: httpx.Request) -> httpx.Response:
        data = response_data()
        mapping(data["metadata"])["request_id"] = "private fake-test-key"
        return httpx.Response(200, json=data, headers={"dg-request-id": remote})
    service = service_with_http(handler)
    with caplog.at_level("INFO", logger="src.adapters.deepgram_http"):
        await service.submit(upload(), {"language": "en"}, local, lookup(ALIAS))
    messages = [r.message for r in caplog.records if r.name == "src.adapters.deepgram_http"]
    assert messages == [f"Deepgram evidence local_request_id={local} provider_request_id={remote}"]
    assert "fake-test-key" not in "".join(messages)


async def test_text_timestamps_and_zh_selection() -> None:
    service = service_with_http(lambda r: httpx.Response(200, json=response_data()))
    execution = await service.submit(upload(), {"language": "zh", "output_format": "txt", "with_timestamp": True}, model_spec=lookup(ALIAS))
    assert execution.payload == "[00:00] [Speaker 0]: Hello"
    assert execution.model == ALIAS


async def test_main_lifespan_rejects_cloud_startup_without_starting_service(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.main import lifespan
    monkeypatch.setattr("src.main.get_model_id", lambda: ALIAS)
    with pytest.raises(ValueError, match="startup MODEL_ID"):
        async with lifespan(FastAPI()):
            pytest.fail("startup must reject cloud model")


async def test_cloud_does_not_close_or_replace_an_already_loaded_native_worker() -> None:
    harness = WorkerHarness()
    service = harness.service()
    local = asyncio.create_task(service.submit(upload(), {}, "warm-local"))
    transport = await harness.next_transport()
    job = await transport.next_job()
    transport.messages.put(("RESULT", job.uid, "local result"))
    await local
    service._remote = RemoteSession(DeepgramAdapter("fake", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response_data()))))
    execution = await service.submit(upload(), {"language": "en"}, model_spec=lookup(ALIAS))
    assert execution.model == ALIAS
    assert harness.transports == [transport]
    assert transport.running and not transport.closed
    assert service.model_loaded
    assert service.current_model_spec == lookup("paraformer")
    await service.stop_worker()


async def test_service_lifecycle_can_reopen_remote_admission_after_disposal() -> None:
    service = service_with_http(lambda r: httpx.Response(200, json=response_data()))
    await service.start_worker()
    await service.submit(upload(), {"language": "en"}, model_spec=lookup(ALIAS))
    await service.stop_worker()
    await service.start_worker()
    execution = await service.submit(upload(), {"language": "en"}, model_spec=lookup(ALIAS))
    assert execution.model == ALIAS
    assert not service.model_loaded
    await service.stop_worker()
