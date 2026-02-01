import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.main import app
from src.services.transcription import TranscriptionService

# 我们需要 Mock 掉 Engine Factory，防止测试时加载真实模型
@pytest.fixture
def mock_create_engine():
    with patch("src.main.create_engine") as mock:
        yield mock

@pytest.fixture
def client(mock_create_engine):
    """
    创建 TestClient。
    注意：TestClient 会触发 lifespan (启动事件)。
    我们通过 mock_create_engine 确保 lifespan 中初始化的 Engine 是假的。
    """
    # 1. Setup Mock Engine 实例
    mock_instance = MagicMock()
    mock_create_engine.return_value = mock_instance
    
    # Mock 推理结果 (返回 FunASR 格式的结构化数据)
    mock_instance.transcribe_file.return_value = {
        "text": "Integration Test Result",
        "segments": [
            {"speaker": "Speaker 0", "text": "Integration", "start": 0, "end": 1000},
            {"speaker": "Speaker 0", "text": "Test Result", "start": 1000, "end": 2000}
        ]
    }
    
    # 2. 启动 Client
    # 使用 with 语句触发 lifespan (startup/shutdown)
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
