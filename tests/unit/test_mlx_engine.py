import pytest
from unittest.mock import MagicMock, patch


class TestMlxAudioEngine:
    """
    测试 src/core/mlx_engine.py
    重点：Mock 掉 mlx_audio 模块，避免下载或加载真实模型
    """

    @pytest.fixture
    def mock_load_model(self):
        """Mock mlx_audio.stt.utils.load_model"""
        with patch("src.core.mlx_engine.load_model") as mock:
            yield mock

    @pytest.fixture
    def mock_generate_transcription(self):
        """Mock mlx_audio.stt.generate.generate_transcription"""
        with patch("src.core.mlx_engine.generate_transcription") as mock:
            yield mock

    @pytest.fixture
    def mock_gc(self):
        """Mock gc 模块"""
        with patch("src.core.mlx_engine.gc") as mock:
            yield mock

    def test_initialization(self):
        """测试引擎初始化"""
        from src.core.mlx_engine import MlxAudioEngine
        
        engine = MlxAudioEngine(model_id="test/mlx-model")
        assert engine.model_id == "test/mlx-model"
        assert engine.model is None

    def test_default_model_id(self):
        """测试默认模型 ID"""
        from src.core.mlx_engine import MlxAudioEngine
        
        engine = MlxAudioEngine()
        assert engine.model_id == "mlx-community/VibeVoice-ASR-4bit"

    def test_load_model(self, mock_load_model):
        """测试模型加载逻辑"""
        from src.core.mlx_engine import MlxAudioEngine
        
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        engine = MlxAudioEngine(model_id="mlx-community/test-model")
        engine.load()
        
        mock_load_model.assert_called_once_with("mlx-community/test-model")
        assert engine.model is mock_model

    def test_load_model_idempotency(self, mock_load_model):
        """测试重复加载（幂等性）"""
        from src.core.mlx_engine import MlxAudioEngine
        
        mock_load_model.return_value = MagicMock()
        
        engine = MlxAudioEngine()
        engine.load()
        engine.load()  # 第二次调用
        
        assert mock_load_model.call_count == 1

    def test_transcribe_not_loaded(self):
        """测试未加载模型直接推理应报错"""
        from src.core.mlx_engine import MlxAudioEngine
        
        engine = MlxAudioEngine()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.transcribe_file("dummy.wav")

    def test_transcribe_success(self, mock_load_model, mock_generate_transcription):
        """测试正常推理流程"""
        from src.core.mlx_engine import MlxAudioEngine
        
        # Setup
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_result = MagicMock()
        mock_result.text = "  Hello from MLX  "
        mock_generate_transcription.return_value = mock_result
        
        # Execute
        engine = MlxAudioEngine()
        engine.load()
        result = engine.transcribe_file("test.wav", language="en")
        
        # Verify
        assert result == "Hello from MLX"  # stripped
        mock_generate_transcription.assert_called_once_with(
            model=mock_model,
            audio_path="test.wav",
            verbose=False
        )

    def test_transcribe_with_verbose(self, mock_load_model, mock_generate_transcription):
        """测试 verbose 参数传递"""
        from src.core.mlx_engine import MlxAudioEngine
        
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_result = MagicMock()
        mock_result.text = "Test"
        mock_generate_transcription.return_value = mock_result
        
        engine = MlxAudioEngine()
        engine.load()
        engine.transcribe_file("test.wav", verbose=True)
        
        mock_generate_transcription.assert_called_once_with(
            model=mock_model,
            audio_path="test.wav",
            verbose=True
        )

    def test_release_resources(self, mock_load_model, mock_gc):
        """测试资源释放逻辑"""
        from src.core.mlx_engine import MlxAudioEngine
        
        mock_load_model.return_value = MagicMock()
        
        engine = MlxAudioEngine()
        engine.load()
        assert engine.model is not None
        
        engine.release()
        
        assert engine.model is None
        mock_gc.collect.assert_called_once()

    def test_release_when_not_loaded(self, mock_gc):
        """测试未加载时释放不报错"""
        from src.core.mlx_engine import MlxAudioEngine
        
        engine = MlxAudioEngine()
        engine.release()  # 不应该抛出异常
        
        mock_gc.collect.assert_not_called()
