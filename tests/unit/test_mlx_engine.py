import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestMlxAudioEngine:
    """
    测试 src/core/mlx_engine.py
    重点：Mock 掉 mlx_audio 模块和音频切片服务
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
    def mock_chunking_service(self):
        """Mock AudioChunkingService"""
        with patch("src.core.mlx_engine.AudioChunkingService") as mock:
            mock_instance = MagicMock()
            mock_instance.process_audio = AsyncMock(return_value=["test.wav"])
            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def mock_gc(self):
        """Mock gc 模块"""
        with patch("src.core.mlx_engine.gc") as mock:
            yield mock

    def test_initialization(self, mock_chunking_service):
        """测试引擎初始化"""
        from src.core.mlx_engine import MlxAudioEngine
        
        engine = MlxAudioEngine(model_id="test/mlx-model")
        assert engine.model_id == "test/mlx-model"
        assert engine.model is None

    def test_default_model_id(self, mock_chunking_service):
        """测试默认模型 ID"""
        from src.core.mlx_engine import MlxAudioEngine
        
        engine = MlxAudioEngine()
        assert engine.model_id == "mlx-community/Qwen3-ASR-1.7B-4bit"

    def test_load_model(self, mock_load_model, mock_chunking_service):
        """测试模型加载逻辑"""
        from src.core.mlx_engine import MlxAudioEngine
        
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        engine = MlxAudioEngine(model_id="mlx-community/test-model")
        engine.load()
        
        mock_load_model.assert_called_once_with("mlx-community/test-model")
        assert engine.model is mock_model

    def test_load_model_idempotency(self, mock_load_model, mock_chunking_service):
        """测试重复加载（幂等性）"""
        from src.core.mlx_engine import MlxAudioEngine
        
        mock_load_model.return_value = MagicMock()
        
        engine = MlxAudioEngine()
        engine.load()
        engine.load()  # 第二次调用
        
        # 应该只加载一次
        mock_load_model.assert_called_once()

    def test_transcribe_without_load(self, mock_chunking_service):
        """测试未加载模型直接推理应报错"""
        from src.core.mlx_engine import MlxAudioEngine
        
        engine = MlxAudioEngine()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.transcribe_file("dummy.wav")

    def test_transcribe_success_single_chunk(
        self,
        mock_load_model,
        mock_generate_transcription,
        mock_chunking_service
    ):
        """测试正常推理流程（单个文件，无切片）"""
        from src.core.mlx_engine import MlxAudioEngine
        
        # Setup
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Mock chunking service 返回单个文件（无切片）
        mock_chunking_service.return_value.process_audio = AsyncMock(
            return_value=["test.wav"]
        )
        
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
            audio="test.wav",
            format="txt",
            verbose=False
        )

    def test_transcribe_success_multiple_chunks(
        self,
        mock_load_model,
        mock_generate_transcription,
        mock_chunking_service
    ):
        """测试长音频切片后推理（多个切片）"""
        from src.core.mlx_engine import MlxAudioEngine
        
        # Setup
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Mock chunking service 返回3个切片
        mock_chunking_service.return_value.process_audio = AsyncMock(
            return_value=["chunk_0.wav", "chunk_1.wav", "chunk_2.wav"]
        )
        
        # Mock 每个切片的转录结果
        mock_results = [
            MagicMock(text="First part"),
            MagicMock(text="Second part"),
            MagicMock(text="Third part"),
        ]
        mock_generate_transcription.side_effect = mock_results
        
        # Execute
        engine = MlxAudioEngine()
        engine.load()
        result = engine.transcribe_file("long_audio.wav")
        
        # Verify
        assert result == "First part Second part Third part"
        assert mock_generate_transcription.call_count == 3

    def test_transcribe_with_verbose(
        self,
        mock_load_model,
        mock_generate_transcription,
        mock_chunking_service
    ):
        """测试 verbose 参数传递"""
        from src.core.mlx_engine import MlxAudioEngine
        
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_chunking_service.return_value.process_audio = AsyncMock(
            return_value=["test.wav"]
        )
        
        mock_result = MagicMock()
        mock_result.text = "Test"
        mock_generate_transcription.return_value = mock_result
        
        engine = MlxAudioEngine()
        engine.load()
        engine.transcribe_file("test.wav", verbose=True)
        
        mock_generate_transcription.assert_called_once_with(
            model=mock_model,
            audio="test.wav",
            format="txt",
            verbose=True
        )

    def test_release(self, mock_load_model, mock_gc, mock_chunking_service):
        """测试资源释放"""
        from src.core.mlx_engine import MlxAudioEngine
        
        mock_load_model.return_value = MagicMock()
        
        engine = MlxAudioEngine()
        engine.load()
        assert engine.model is not None
        
        engine.release()
        
        assert engine.model is None
        mock_gc.collect.assert_called_once()
