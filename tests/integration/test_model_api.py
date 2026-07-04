"""
Integration tests for model listing and per-request model selection (SPEC-108, cases MA-1..MA-7).

Uses FastAPI TestClient with a patched TranscriptionService — no real model loading.
The API layer (routes.py) is tested in isolation; subprocess worker logic is covered
by unit tests.
"""

from dataclasses import replace
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.base_engine import EngineCapabilities
from src.core.model_registry import lookup as real_lookup
from src.core.pipeline_registry import lookup_profile
from src.main import app
from src.services.transcription import PipelineQualityError, TranscriptionService


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
    service.submit_pipeline = AsyncMock(return_value=submit_result)
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
    with (
        patch("src.main.TranscriptionService", return_value=mock_service),
        patch("src.main.lookup", return_value=qwen_spec),
        TestClient(app) as c,
    ):
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
    with (
        patch("src.main.TranscriptionService", return_value=mock_service),
        patch("src.main.lookup", return_value=paraformer_spec),
        TestClient(app) as c,
    ):
        yield c


def _audio_file() -> tuple[str, BytesIO, str]:
    return ("file", BytesIO(b"RIFF\x00\x00\x00\x00WAVEfmt "), "audio/wav")


def test_openapi_transcription_docs_should_use_current_model_aliases() -> None:
    schema = app.openapi()
    operation = schema["paths"]["/v1/audio/transcriptions"]["post"]
    description = operation["description"]
    body_schema_ref = operation["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    body_schema_name = body_schema_ref.rsplit("/", 1)[-1]
    model_description = schema["components"]["schemas"][body_schema_name]["properties"]["model"][
        "description"
    ]

    assert "model=qwen3-asr" in description
    assert "model=qwen3-sortformer" in description
    assert "qwen3-asr-mini" not in description
    assert "qwen3-asr-mini" not in model_description
    assert "'qwen3-asr'" in model_description
    assert "'qwen3-sortformer'" in model_description
    assert "GET /v1/models" in description
    assert "GET /v1/models" in model_description
    assert "whisper-1" in description
    assert "passthrough" in description
    assert "does not select Whisper" in description
    assert "fresh server starts with `paraformer`" in description
    assert "passthrough" in model_description
    assert "fresh server default is `paraformer`" in model_description


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



def test_models_endpoint_should_include_requestable_pipeline_profiles(client) -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    models = {item["alias"]: item for item in body["models"]}
    assert "qwen3-sortformer" in models
    assert models["qwen3-sortformer"]["capabilities"]["diarization"] is True
    assert models["qwen3-sortformer"]["requestable"] is True


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


def test_should_return_501_for_non_requestable_pipeline_profile() -> None:
    qwen_spec = real_lookup("qwen3-asr")
    mock_service = _make_mock_service(
        qwen_spec.capabilities,
        {"text": "unused", "segments": None, "duration": 1.0},
        current_model_spec=qwen_spec,
    )
    non_requestable_profile = replace(lookup_profile("qwen3-sortformer"), requestable=False)

    with (
        patch("src.main.TranscriptionService", return_value=mock_service),
        patch("src.main.lookup", return_value=qwen_spec),
        patch("src.api.routes.lookup_profile", return_value=non_requestable_profile),
        TestClient(app) as c,
    ):
        response = c.post(
            "/v1/audio/transcriptions",
            data={"model": "qwen3-sortformer"},
            files={"file": _audio_file()},
        )

    assert response.status_code == 501
    assert "not enabled" in response.json()["detail"]
    mock_service.submit.assert_not_called()
    mock_service.submit_pipeline.assert_not_awaited()


def test_should_submit_qwen3_sortformer_pipeline_by_default() -> None:
    qwen_spec = real_lookup("qwen3-asr")
    mock_service = _make_mock_service(
        qwen_spec.capabilities,
        {
            "text": "pipeline result",
            "segments": [{"text": "hello", "start": 0.0, "end": 1.0, "speaker": "Speaker A"}],
            "duration": 1.0,
        },
        current_model_spec=qwen_spec,
    )

    with (
        patch("src.main.TranscriptionService", return_value=mock_service),
        patch("src.main.lookup", return_value=qwen_spec),
        TestClient(app) as c,
    ):
        response = c.post(
            "/v1/audio/transcriptions",
            data={"model": "qwen3-sortformer", "output_format": "json"},
            files={"file": _audio_file()},
        )

    assert response.status_code == 200
    assert response.json()["model"] == "qwen3-sortformer"
    mock_service.submit.assert_not_called()
    mock_service.submit_pipeline.assert_awaited_once()


def test_should_submit_pipeline_profile_when_explicitly_requestable() -> None:
    qwen_spec = real_lookup("qwen3-asr")
    mock_service = _make_mock_service(
        qwen_spec.capabilities,
        {
            "text": "pipeline result",
            "segments": [{"text": "hello", "start": 0.0, "end": 1.0, "speaker": "Speaker A"}],
            "duration": 1.0,
        },
        current_model_spec=qwen_spec,
    )
    requestable_profile = replace(lookup_profile("qwen3-sortformer"), requestable=True)

    with (
        patch("src.main.TranscriptionService", return_value=mock_service),
        patch("src.main.lookup", return_value=qwen_spec),
        patch("src.api.routes.lookup_profile", return_value=requestable_profile),
        TestClient(app) as c,
    ):
        response = c.post(
            "/v1/audio/transcriptions",
            data={"model": "qwen3-sortformer", "output_format": "json"},
            files={"file": _audio_file()},
        )

    assert response.status_code == 200
    assert response.json()["model"] == "qwen3-sortformer"
    mock_service.submit.assert_not_called()
    mock_service.submit_pipeline.assert_awaited_once()


def test_should_return_422_when_pipeline_quality_gate_fails() -> None:
    qwen_spec = real_lookup("qwen3-asr")
    mock_service = _make_mock_service(
        qwen_spec.capabilities,
        {"text": "unused", "segments": None, "duration": 1.0},
        current_model_spec=qwen_spec,
    )
    mock_service.submit_pipeline = AsyncMock(
        side_effect=PipelineQualityError("alignment quality gate failed: tail timestamp collapse")
    )
    requestable_profile = replace(lookup_profile("qwen3-sortformer"), requestable=True)

    with (
        patch("src.main.TranscriptionService", return_value=mock_service),
        patch("src.main.lookup", return_value=qwen_spec),
        patch("src.api.routes.lookup_profile", return_value=requestable_profile),
        TestClient(app) as c,
    ):
        response = c.post(
            "/v1/audio/transcriptions",
            data={"model": "qwen3-sortformer", "output_format": "json"},
            files={"file": _audio_file()},
        )

    assert response.status_code == 422
    assert "alignment quality gate failed" in response.json()["detail"]


# MA-5
def test_should_use_current_model_when_model_field_omitted(client) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        data={"language": "zh"},
        files={"file": _audio_file()},
    )

    assert response.status_code == 200


def test_should_preserve_detected_language_when_request_uses_auto() -> None:
    qwen_spec = real_lookup("qwen3-asr")
    mock_service = _make_mock_service(
        qwen_spec.capabilities,
        {"text": "test result", "segments": None, "duration": 1.0, "language": "en"},
        current_model_spec=qwen_spec,
    )
    with (
        patch("src.main.TranscriptionService", return_value=mock_service),
        patch("src.main.lookup", return_value=qwen_spec),
        TestClient(app) as c,
    ):
        response = c.post(
            "/v1/audio/transcriptions",
            data={"language": "auto"},
            files={"file": _audio_file()},
        )

    assert response.status_code == 200
    assert response.json()["language"] == "en"


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
