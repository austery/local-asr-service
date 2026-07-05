import os
from typing import get_args
from unittest.mock import patch

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
            with patch("dotenv.load_dotenv"):  # Skip .env loading before src.config import/reload
                import src.config
                importlib.reload(src.config)

            # 测试默认值
            # NOTE: SPEC-007 更新默认模型为 Paraformer (支持说话人分离)
            assert src.config.FUNASR_MODEL_ID == DEFAULT_MODEL_ID
            assert src.config.MLX_MODEL_ID == "mlx-community/Qwen3-ASR-1.7B-8bit"
        finally:
            os.environ.update(old_env)

    def test_startup_engine_type_should_exclude_sidecar_only_runtimes(self) -> None:
        import src.config

        assert get_args(src.config.EngineType) == ("funasr", "mlx")


class TestFactory:
    """测试 src/core/factory.py 工厂模块"""

    @patch("src.adapters.audio_chunking.subprocess.run")
    def test_create_engines(self, mock_run):
        """测试引擎创建（不依赖环境变量reload）"""
        from src.core.funasr_engine import FunASREngine
        from src.core.mlx_engine import MlxAudioEngine

        # 直接测试引擎类
        funasr_engine = FunASREngine(model_id=DEFAULT_MODEL_ID)
        assert funasr_engine.model_id == DEFAULT_MODEL_ID

        mlx_engine = MlxAudioEngine(model_id="mlx-community/Qwen3-ASR-1.7B-8bit")
        assert mlx_engine.model_id == "mlx-community/Qwen3-ASR-1.7B-8bit"
