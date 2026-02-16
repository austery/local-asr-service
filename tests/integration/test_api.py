import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, PropertyMock
from src.core.base_engine import EngineCapabilities
from src.main import app
from src.services.transcription import TranscriptionService

# We mock the Engine Factory so no real model is loaded during tests
@pytest.fixture
def mock_create_engine():
    with patch("src.main.create_engine") as mock:
        yield mock

@pytest.fixture
def client(mock_create_engine):
    """
    Create TestClient with a mock engine that has Paraformer-like capabilities
    (timestamp=True, diarization=True) — the default production config.
    """
    mock_instance = MagicMock()
    mock_create_engine.return_value = mock_instance

    # Default: Paraformer capabilities (full features)
    type(mock_instance).capabilities = PropertyMock(
        return_value=EngineCapabilities(timestamp=True, diarization=True, language_detect=True)
    )

    mock_instance.transcribe_file.return_value = {
        "text": "Integration Test Result",
        "segments": [
            {"speaker": "Speaker 0", "text": "Integration", "start": 0, "end": 1000},
            {"speaker": "Speaker 0", "text": "Test Result", "start": 1000, "end": 2000}
        ]
    }

    with TestClient(app) as c:
        yield c

@pytest.fixture
def sensevoice_client(mock_create_engine):
    """
    Create TestClient with a SenseVoice-like engine (no timestamp, no diarization).
    Used to test capability validation (400 errors).
    """
    mock_instance = MagicMock()
    mock_create_engine.return_value = mock_instance

    type(mock_instance).capabilities = PropertyMock(
        return_value=EngineCapabilities(emotion_tags=True, language_detect=True)
    )

    mock_instance.transcribe_file.return_value = {
        "text": "SenseVoice Result",
        "segments": None,
    }

    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """测试健康检查接口"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_transcribe_endpoint_json(client):
    """测试转录接口 - JSON 格式 (OpenAI 兼容)"""
    files = {
        "file": ("test.wav", b"fake audio bytes", "audio/wav")
    }
    data = {
        "language": "zh",
        "output_format": "json"  # 默认值，显式指定
    }

    response = client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 200
    result = response.json()

    # 验证 OpenAI 兼容格式
    assert result["text"] == "Integration Test Result"
    assert "duration" in result
    assert "language" in result
    assert "model" in result
    # JSON 格式应该包含 segments
    assert "segments" in result
    assert result["segments"] is not None

def test_transcribe_endpoint_txt(client):
    """测试转录接口 - TXT 格式 (也返回 JSON 结构以兼容 OpenAI API)"""
    files = {
        "file": ("test.wav", b"fake audio bytes", "audio/wav")
    }
    data = {
        "language": "zh",
        "output_format": "txt"
    }

    response = client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 200
    result = response.json()

    # TXT 格式也返回 JSON 结构，但不包含 segments
    assert result["text"] == "Integration Test Result"
    assert "duration" in result
    # TXT 格式不应该包含 segments
    assert result["segments"] is None

def test_transcribe_no_file(client):
    """测试缺少文件的情况"""
    response = client.post("/v1/audio/transcriptions", data={"language": "zh"})
    assert response.status_code == 422 # Validation Error

def test_transcribe_default_format(client):
    """测试不传 output_format 参数时的默认行为 (应为 JSON)"""
    files = {
        "file": ("test.wav", b"fake audio bytes", "audio/wav")
    }
    data = {
        "language": "zh"
        # 不传 output_format，应该默认为 json
    }

    response = client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 200
    result = response.json()

    # 默认应该返回 JSON 格式
    assert result["text"] == "Integration Test Result"
    assert "duration" in result
    assert "segments" in result  # JSON 格式应该包含 segments


# === Capability Validation Tests ===

def test_srt_without_timestamp_returns_400(sensevoice_client):
    """SRT format with a model that lacks timestamps → 400."""
    files = {"file": ("test.wav", b"fake audio bytes", "audio/wav")}
    data = {"output_format": "srt"}

    response = sensevoice_client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 400
    assert "timestamp" in response.json()["detail"].lower()

def test_with_timestamp_without_capability_returns_400(sensevoice_client):
    """with_timestamp=true with a model that lacks timestamps → 400."""
    files = {"file": ("test.wav", b"fake audio bytes", "audio/wav")}
    data = {"with_timestamp": "true", "output_format": "txt"}

    response = sensevoice_client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 400
    assert "timestamp" in response.json()["detail"].lower()


# === response_format (OpenAI alias) Tests ===

def test_response_format_verbose_json(client):
    """response_format=verbose_json maps to json with segments."""
    files = {"file": ("test.wav", b"fake audio bytes", "audio/wav")}
    data = {"response_format": "verbose_json"}

    response = client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert result["text"] == "Integration Test Result"
    assert result["segments"] is not None

def test_response_format_text(client):
    """response_format=text maps to txt (no segments)."""
    files = {"file": ("test.wav", b"fake audio bytes", "audio/wav")}
    data = {"response_format": "text"}

    response = client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert result["segments"] is None

def test_response_format_overrides_output_format(client):
    """response_format takes precedence over output_format when both are provided."""
    files = {"file": ("test.wav", b"fake audio bytes", "audio/wav")}
    # response_format=text → txt (no segments), even though output_format=json
    data = {"response_format": "text", "output_format": "json"}

    response = client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    # response_format wins → txt → no segments
    assert result["segments"] is None


# === GET /v1/models/current Tests ===

def test_get_current_model(client):
    """GET /v1/models/current returns model info and capabilities."""
    response = client.get("/v1/models/current")

    assert response.status_code == 200
    result = response.json()
    assert "engine_type" in result
    assert "model_id" in result
    assert "capabilities" in result
    caps = result["capabilities"]
    assert "timestamp" in caps
    assert "diarization" in caps
    assert "emotion_tags" in caps
    assert "language_detect" in caps
    assert "queue_size" in result
    assert "max_queue_size" in result
