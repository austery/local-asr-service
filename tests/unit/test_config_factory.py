import pytest
from unittest.mock import patch, MagicMock
import os
from src.core.funasr_engine import DEFAULT_MODEL_ID


class TestConfig:
    """测试 src/config.py 配置模块"""

    def test_default_values_via_env(self):
        """测试通过环境变量设置默认值"""
        # 保存当前环境
        old_env = os.environ.copy()
        try:
            # 清理环境变量
            for key in ["ENGINE_TYPE", "MODEL_ID", "FUNASR_MODEL_ID", "MLX_MODEL_ID"]:
                os.environ.pop(key, None)
            
            # 重新加载配置
            import importlib
            import src.config
            with patch("src.config.load_dotenv"):  # Skip .env loading
                importlib.reload(src.config)
            
            # 测试默认值
            # NOTE: SPEC-007 更新默认模型为 Paraformer (支持说话人分离)
            assert src.config.FUNASR_MODEL_ID == DEFAULT_MODEL_ID
            assert src.config.MLX_MODEL_ID == "mlx-community/Qwen3-ASR-1.7B-4bit"
        finally:
            os.environ.update(old_env)


class TestFactory:
    """测试 src/core/factory.py 工厂模块"""

    def test_create_engines(self):
        """测试引擎创建（不依赖环境变量reload）"""
        from src.core.funasr_engine import FunASREngine
        from src.core.mlx_engine import MlxAudioEngine
        
        # 直接测试引擎类
        funasr_engine = FunASREngine(model_id=DEFAULT_MODEL_ID)
        assert funasr_engine.model_id == DEFAULT_MODEL_ID
        
        mlx_engine = MlxAudioEngine(model_id="mlx-community/Qwen3-ASR-1.7B-4bit")
        assert mlx_engine.model_id == "mlx-community/Qwen3-ASR-1.7B-4bit"
