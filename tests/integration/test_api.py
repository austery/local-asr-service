import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from src.core.base_engine import EngineCapabilities
from src.main import app
from src.services.transcription import TranscriptionService


_PARAFORMER_RESULT = {
    "text": "Integration Test Result",
    "segments": [
        {"speaker": "Speaker 0", "text": "Integration", "start": 0, "end": 1000},
        {"speaker": "Speaker 0", "text": "Test Result", "start": 1000, "end": 2000},
    ],
    "duration": 2.0,
}
_SENSEVOICE_RESULT = {"text": "SenseVoice Result", "segments": None, "duration": 1.0}


def _make_mock_service(
    capabilities: EngineCapabilities,
    submit_result: object,
) -> MagicMock:
    """Build a MagicMock that duck-types TranscriptionService for API-layer tests."""
    service = MagicMock(spec=TranscriptionService)
    type(service).capabilities = PropertyMock(return_value=capabilities)
    service.current_model_spec = None
    service.submit = AsyncMock(return_value=submit_result)
    service.start_worker = AsyncMock()
    service.stop_worker = AsyncMock()
    type(service).queue_size = PropertyMock(return_value=0)
    type(service).max_queue_size = PropertyMock(return_value=50)
    return service


@pytest.fixture
def client():
    """TestClient with Paraformer-like capabilities (timestamp + diarization)."""
    mock_service = _make_mock_service(
        EngineCapabilities(timestamp=True, diarization=True, language_detect=True),
        _PARAFORMER_RESULT,
    )
    with patch("src.main.TranscriptionService", return_value=mock_service):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def sensevoice_client():
    """TestClient with SenseVoice-like capabilities (no timestamp, no diarization)."""
    mock_service = _make_mock_service(
        EngineCapabilities(emotion_tags=True, language_detect=True),
        _SENSEVOICE_RESULT,
    )
    with patch("src.main.TranscriptionService", return_value=mock_service):
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
