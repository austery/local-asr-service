"""
Integration tests for model listing and per-request model selection (SPEC-108, cases MA-1..MA-7).

Uses FastAPI TestClient with a patched TranscriptionService — no real model loading.
The API layer (routes.py) is tested in isolation; subprocess worker logic is covered
by unit tests.
"""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.base_engine import EngineCapabilities
from src.core.model_registry import lookup as real_lookup
from src.main import app
from src.services.transcription import TranscriptionService


def _make_mock_service(
    capabilities: EngineCapabilities,
    submit_result: object,
    current_model_spec: object = None,
) -> MagicMock:
    """Build a MagicMock that duck-types TranscriptionService for API-layer tests."""
    service = MagicMock(spec=TranscriptionService)
    type(service).capabilities = PropertyMock(return_value=capabilities)
    service.current_model_spec = current_model_spec
    service.submit = AsyncMock(return_value=submit_result)
    service.start_worker = AsyncMock()
    service.stop_worker = AsyncMock()
    type(service).queue_size = PropertyMock(return_value=0)
    type(service).max_queue_size = PropertyMock(return_value=50)
    return service


@pytest.fixture
def client():
    """TestClient with qwen3-asr as startup model (timestamp, no diarization)."""
    qwen_spec = real_lookup("qwen3-asr")
    mock_service = _make_mock_service(
        qwen_spec.capabilities,
        {"text": "test result", "segments": None, "duration": 1.0},
        current_model_spec=qwen_spec,
    )
    with patch("src.main.TranscriptionService", return_value=mock_service):
        with patch("src.main.lookup", return_value=qwen_spec):
            with TestClient(app) as c:
                yield c


@pytest.fixture
def funasr_client():
    """TestClient where startup resolves to paraformer (diarization capable)."""
    paraformer_spec = real_lookup("paraformer")
    mock_service = _make_mock_service(
        paraformer_spec.capabilities,
        {"text": "funasr result", "segments": [], "duration": 1.0},
        current_model_spec=paraformer_spec,
    )
    with patch("src.main.TranscriptionService", return_value=mock_service):
        with patch("src.main.lookup", return_value=paraformer_spec):
            with TestClient(app) as c:
                yield c


def _audio_file() -> tuple[str, BytesIO, str]:
    return ("file", BytesIO(b"RIFF\x00\x00\x00\x00WAVEfmt "), "audio/wav")


# MA-1
def test_should_return_model_list_on_get_models(client) -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert "models" in body
    aliases = [m["alias"] for m in body["models"]]
    assert "paraformer" in aliases
    assert "qwen3-asr" in aliases
    assert "sensevoice-small" in aliases
    


# MA-2
def test_should_include_current_model_in_get_models_response(client) -> None:
    response = client.get("/v1/models")

    body = response.json()
    # Client fixture resolves startup model to qwen3-asr
    assert body["current"] == "qwen3-asr"


# MA-3
def test_should_succeed_when_valid_alias_provided(client) -> None:
    # With the subprocess architecture, model switching is handled inside submit().
    # The API layer only needs to resolve the alias and pass the spec to submit().
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-asr", "language": "zh"},
        files={"file": _audio_file()},
    )

    assert response.status_code == 200


# MA-4
def test_should_return_400_when_unknown_model_provided(client) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "not-a-real-model", "language": "zh"},
        files={"file": _audio_file()},
    )

    assert response.status_code == 400
    assert "Unknown model" in response.json()["detail"]


# MA-5
def test_should_use_current_model_when_model_field_omitted(client) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        data={"language": "zh"},
        files={"file": _audio_file()},
    )

    assert response.status_code == 200


# MA-6
def test_should_return_400_when_srt_requested_with_no_timestamp_model(client) -> None:
    """Capability pre-validation: model with no timestamp support + SRT output → 400."""
    from src.core.base_engine import EngineCapabilities
    from src.core.model_registry import ModelSpec

    no_ts_spec = ModelSpec(
        alias="no-ts-model",
        model_id="mlx-community/no-ts-model",
        engine_type="mlx",
        description="Test model with no timestamp support.",
        capabilities=EngineCapabilities(timestamp=False, diarization=False),
    )

    with patch("src.api.routes.lookup", return_value=no_ts_spec):
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "no-ts-model", "output_format": "srt"},
            files={"file": _audio_file()},
        )

    assert response.status_code == 400
    assert "timestamp" in response.json()["detail"].lower()


# MA-7
def test_should_use_current_model_when_whisper_1_provided(client) -> None:
    """whisper-1 is OpenAI's default placeholder — treated as 'use current model'."""
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "language": "zh"},
        files={"file": _audio_file()},
    )

    assert response.status_code == 200


# MA-8
def test_should_return_400_when_with_timestamp_requested_with_no_timestamp_model(client) -> None:
    """Capability pre-validation: resolved spec has no timestamp + with_timestamp=True → 400.
    Exercises the resolved_spec.capabilities branch (C3 fix), not the current engine branch."""
    from src.core.base_engine import EngineCapabilities
    from src.core.model_registry import ModelSpec

    no_ts_spec = ModelSpec(
        alias="no-ts-model",
        model_id="mlx-community/no-ts-model",
        engine_type="mlx",
        description="Test model with no timestamp support.",
        capabilities=EngineCapabilities(timestamp=False, diarization=False),
    )

    with patch("src.api.routes.lookup", return_value=no_ts_spec):
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "no-ts-model", "with_timestamp": "true"},
            files={"file": _audio_file()},
        )

    assert response.status_code == 400
    assert "with_timestamp" in response.json()["detail"].lower()


# GET /v1/models/current — response includes model_alias field
def test_get_current_model_includes_alias(client) -> None:
    response = client.get("/v1/models/current")

    assert response.status_code == 200
    body = response.json()
    assert "model_alias" in body
    assert body["model_alias"] == "qwen3-asr"
