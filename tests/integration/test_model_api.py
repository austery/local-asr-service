"""
Integration tests for model listing and per-request model selection (SPEC-108, cases MA-1..MA-7).

Uses FastAPI TestClient with a patched engine factory — no real model loading.
Follows the same pattern as test_api.py: patch "src.main.create_engine" so
the real lifespan runs but uses a mock engine, giving us proper app.state setup.
"""

from io import BytesIO
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.base_engine import EngineCapabilities
from src.main import app


@pytest.fixture
def mock_create_engine():
    """Patch the engine factory so lifespan runs without loading a real model."""
    with patch("src.main.create_engine") as mock:
        yield mock


@pytest.fixture
def client(mock_create_engine):
    """TestClient with a mock MLX engine (Qwen3-ASR-like caps: timestamp, no diarization)."""
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine

    type(mock_engine).capabilities = PropertyMock(
        return_value=EngineCapabilities(
            timestamp=True, diarization=False, emotion_tags=False, language_detect=True
        )
    )
    mock_engine.transcribe_file.return_value = {"text": "test result", "segments": None}

    # Patch lookup in main.py so startup resolves to qwen3-asr instead of paraformer
    # (avoids dependency on the default env var ENGINE_TYPE=funasr)
    from src.core.model_registry import lookup as real_lookup

    qwen_spec = real_lookup("qwen3-asr")
    with patch("src.main.lookup", return_value=qwen_spec):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def funasr_client(mock_create_engine):
    """TestClient where startup resolves to paraformer (diarization capable)."""
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine

    type(mock_engine).capabilities = PropertyMock(
        return_value=EngineCapabilities(
            timestamp=True, diarization=True, emotion_tags=False, language_detect=True
        )
    )
    mock_engine.transcribe_file.return_value = {"text": "funasr result", "segments": []}

    from src.core.model_registry import lookup as real_lookup

    paraformer_spec = real_lookup("paraformer")
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
def test_should_succeed_when_valid_alias_provided(client, mock_create_engine) -> None:
    mock_new_engine = MagicMock()
    type(mock_new_engine).capabilities = PropertyMock(
        return_value=EngineCapabilities(timestamp=True, diarization=False)
    )
    mock_new_engine.transcribe_file.return_value = {"text": "ok", "segments": None}

    with patch(
        "src.services.transcription.create_engine_for_spec", return_value=mock_new_engine
    ):
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
