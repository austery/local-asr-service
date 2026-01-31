import pytest
from unittest.mock import MagicMock, patch
from src.core.funasr_engine import FunASREngine

class TestFunASREngine:
    """
    测试 src/core/funasr_engine.py
    重点：Mock 掉 funasr.AutoModel，避免下载或加载真实模型
    """

    @pytest.fixture
    def mock_auto_model(self):
        """Mock funasr.AutoModel 类"""
        with patch("src.core.funasr_engine.AutoModel") as mock:
            yield mock

    @pytest.fixture
    def mock_torch(self):
        """Mock torch 模块"""
        with patch("src.core.funasr_engine.torch") as mock:
            yield mock

    @pytest.fixture
    def mock_gc(self):
        """Mock gc 模块"""
        with patch("src.core.funasr_engine.gc") as mock:
            yield mock

    def test_initialization(self):
        """测试引擎初始化"""
        engine = FunASREngine(model_id="test/model", device="cpu")
        assert engine.model_id == "test/model"
        assert engine.device == "cpu"
        assert engine.model is None

    def test_load_model(self, mock_auto_model):
        """测试模型加载逻辑"""
        engine = FunASREngine(device="cpu")
        
        # 执行加载
        engine.load()
        
        # 验证 AutoModel 是否被正确调用
        mock_auto_model.assert_called_once()
        call_kwargs = mock_auto_model.call_args.kwargs
        assert call_kwargs["model"] == "iic/SenseVoiceSmall"
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["disable_update"] is True
        
        # 验证 engine.model 是否被赋值
        assert engine.model is not None

    def test_load_model_idempotency(self, mock_auto_model):
        """测试重复加载（幂等性）"""
        engine = FunASREngine()
        engine.load()
        engine.load() # 第二次调用
        
        # 应该只初始化一次
        assert mock_auto_model.call_count == 1

    def test_transcribe_not_loaded(self):
        """测试未加载模型直接推理应报错"""
        engine = FunASREngine()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.transcribe_file("dummy.wav")

    def test_transcribe_success(self, mock_auto_model):
        """测试正常推理流程"""
        # 1. Setup Mock
        mock_instance = MagicMock()
        mock_auto_model.return_value = mock_instance
        
        # 模拟 generate 返回值: list of dict
        mock_instance.generate.return_value = [{"text": "Hello World"}]
        
        # 2. Load Engine
        engine = FunASREngine()
        engine.load()
        
        # 3. Execute
        result = engine.transcribe_file("test.wav", language="en")
        
        # 4. Assertions
        assert result == "Hello World"
        
        # 验证 generate 调用参数
        mock_instance.generate.assert_called_once()
        call_kwargs = mock_instance.generate.call_args.kwargs
        assert call_kwargs["input"] == "test.wav"
        assert call_kwargs["language"] == "en"
        assert call_kwargs["use_itn"] is True

    def test_transcribe_language_fallback(self, mock_auto_model):
        """测试语言参数回退逻辑"""
        mock_instance = MagicMock()
        mock_auto_model.return_value = mock_instance
        mock_instance.generate.return_value = [{"text": "..."}]
        
        engine = FunASREngine()
        engine.load()
        
        # 传入不支持的语言 "fr" (法语)
        engine.transcribe_file("test.wav", language="fr")
        
        # 验证是否回退到 "auto"
        call_kwargs = mock_instance.generate.call_args.kwargs
        assert call_kwargs["language"] == "auto"

    def test_transcribe_mps_cleanup(self, mock_auto_model, mock_torch):
        """测试 MPS 环境下的显存清理"""
        # Setup
        mock_torch.backends.mps.is_available.return_value = True
        mock_instance = MagicMock()
        mock_auto_model.return_value = mock_instance
        mock_instance.generate.return_value = [{"text": "MPS Test"}]
        
        # Initialize with MPS
        engine = FunASREngine(device="mps")
        engine.load()
        
        # Execute
        engine.transcribe_file("test.wav")
        
        # Verify cleanup
        mock_torch.mps.empty_cache.assert_called_once()
        # Ensure CUDA cleanup was NOT called
        mock_torch.cuda.empty_cache.assert_not_called()

    def test_transcribe_cuda_cleanup(self, mock_auto_model, mock_torch):
        """测试 CUDA 环境下的显存清理"""
        # Setup
        mock_instance = MagicMock()
        mock_auto_model.return_value = mock_instance
        mock_instance.generate.return_value = [{"text": "CUDA Test"}]
        
        # Initialize with CUDA
        engine = FunASREngine(device="cuda")
        engine.load()
        
        # Execute
        engine.transcribe_file("test.wav")
        
        # Verify cleanup
        mock_torch.cuda.empty_cache.assert_called_once()
        mock_torch.mps.empty_cache.assert_not_called()

    def test_release_resources(self, mock_auto_model, mock_torch, mock_gc):
        """测试资源释放逻辑"""
        # Setup
        engine = FunASREngine(device="mps")
        engine.load()
        assert engine.model is not None
        
        # Execute release
        engine.release()
        
        # Verify
        assert engine.model is None
        mock_torch.mps.empty_cache.assert_called()
        mock_gc.collect.assert_called_once()