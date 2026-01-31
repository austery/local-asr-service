import pytest
import os
import wave
import struct
from fastapi.testclient import TestClient
from src.main import app

# 这是一个 "慢速" 测试，因为它会真的加载模型
# 我们可以用 pytest marker 来标记它
@pytest.mark.e2e
class TestEndToEnd:
    
    @pytest.fixture(scope="class")
    def real_client(self):
        """
        创建一个真正的 Client，不 Mock 任何东西。
        这会触发 lifespan -> 加载真实模型 (第一次运行会下载 500MB+)。
        """
        # 注意：为了防止测试跑太久，我们这里假设环境里已经有模型，或者用户愿意等
        with TestClient(app) as client:
            yield client

    @pytest.fixture
    def dummy_wav(self, tmp_path):
        """生成一个 1秒钟的静音 WAV 文件"""
        file_path = tmp_path / "silence.wav"
        with wave.open(str(file_path), 'w') as wav_file:
            wav_file.setnchannels(1) # 单声道
            wav_file.setsampwidth(2) # 16-bit
            wav_file.setframerate(16000) # 16kHz
            # 写入 16000 个 0 (1秒静音)
            data = struct.pack('<' + ('h'*16000), *[0]*16000)
            wav_file.writeframes(data)
        return file_path

    def test_real_transcription_flow(self, real_client, dummy_wav):
        """
        端到端测试：
        1. 上传真实 WAV 文件
        2. 经过真实 Service 队列
        3. 调用真实 Engine (MPS/CPU)
        4. 返回结果
        """
        with open(dummy_wav, "rb") as f:
            response = real_client.post(
                "/v1/audio/transcriptions",
                files={"file": ("silence.wav", f, "audio/wav")},
                data={"language": "zh", "clean_tags": "true"}
            )
        
        assert response.status_code == 200
        result = response.json()
        
        # 对于静音文件，SenseVoice 可能返回空字符串，或者一些幻觉文本
        # 我们主要验证流程跑通了，没有报错
        assert "text" in result
        assert result["duration"] > 0
