import pytest
from unittest.mock import patch


class TestConfig:
    """测试 src/config.py 配置模块"""

    def test_default_engine_type(self):
        """测试默认引擎类型"""
        with patch.dict("os.environ", {}, clear=True):
            # 重新加载模块以获取新的环境变量值
            import importlib
            import src.config
            importlib.reload(src.config)
            
            assert src.config.ENGINE_TYPE == "funasr"

    def test_custom_engine_type(self):
        """测试自定义引擎类型"""
        with patch.dict("os.environ", {"ENGINE_TYPE": "mlx"}):
            import importlib
            import src.config
            importlib.reload(src.config)
            
            assert src.config.ENGINE_TYPE == "mlx"

    def test_get_model_id_funasr_default(self):
        """测试 FunASR 默认模型 ID"""
        with patch.dict("os.environ", {"ENGINE_TYPE": "funasr"}, clear=True):
            import importlib
            import src.config
            importlib.reload(src.config)
            
            assert src.config.get_model_id() == "iic/SenseVoiceSmall"

    def test_get_model_id_mlx_default(self):
        """测试 MLX 默认模型 ID"""
        with patch.dict("os.environ", {"ENGINE_TYPE": "mlx"}, clear=True):
            import importlib
            import src.config
            importlib.reload(src.config)
            
            assert src.config.get_model_id() == "mlx-community/VibeVoice-ASR-4bit"

    def test_get_model_id_override(self):
        """测试 MODEL_ID 环境变量覆盖"""
        with patch.dict("os.environ", {
            "ENGINE_TYPE": "funasr",
            "MODEL_ID": "custom/model"
        }):
            import importlib
            import src.config
            importlib.reload(src.config)
            
            assert src.config.get_model_id() == "custom/model"

    def test_service_config_defaults(self):
        """测试服务配置默认值"""
        with patch.dict("os.environ", {}, clear=True):
            import importlib
            import src.config
            importlib.reload(src.config)
            
            assert src.config.HOST == "0.0.0.0"
            assert src.config.PORT == 50070
            assert src.config.MAX_QUEUE_SIZE == 50


class TestFactory:
    """测试 src/core/factory.py 工厂模块"""

    def test_create_funasr_engine(self):
        """测试创建 FunASR 引擎"""
        with patch.dict("os.environ", {"ENGINE_TYPE": "funasr"}, clear=True):
            import importlib
            import src.config
            importlib.reload(src.config)
            
            from src.core.factory import create_engine
            from src.core.funasr_engine import FunASREngine
            
            engine = create_engine()
            assert isinstance(engine, FunASREngine)
            assert engine.model_id == "iic/SenseVoiceSmall"

    def test_create_mlx_engine(self):
        """测试创建 MLX 引擎"""
        with patch.dict("os.environ", {"ENGINE_TYPE": "mlx"}, clear=True):
            import importlib
            import src.config
            importlib.reload(src.config)
            
            # 重新加载 factory 以使用新的 config
            import src.core.factory
            importlib.reload(src.core.factory)
            
            from src.core.factory import create_engine
            from src.core.mlx_engine import MlxAudioEngine
            
            engine = create_engine()
            assert isinstance(engine, MlxAudioEngine)
            assert engine.model_id == "mlx-community/VibeVoice-ASR-4bit"

    def test_create_engine_with_custom_model(self):
        """测试使用自定义模型创建引擎"""
        with patch.dict("os.environ", {
            "ENGINE_TYPE": "mlx",
            "MODEL_ID": "mlx-community/custom-model"
        }, clear=True):
            import importlib
            import src.config
            import src.core.factory
            importlib.reload(src.config)
            importlib.reload(src.core.factory)
            
            from src.core.factory import create_engine
            
            engine = create_engine()
            assert engine.model_id == "mlx-community/custom-model"

    def test_create_engine_invalid_type(self):
        """测试无效引擎类型"""
        with patch.dict("os.environ", {"ENGINE_TYPE": "invalid"}, clear=True):
            import importlib
            import src.config
            import src.core.factory
            importlib.reload(src.config)
            importlib.reload(src.core.factory)
            
            from src.core.factory import create_engine
            
            with pytest.raises(ValueError, match="Unsupported ENGINE_TYPE"):
                create_engine()
