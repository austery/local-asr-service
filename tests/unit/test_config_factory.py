import importlib
import os
from unittest.mock import MagicMock, patch

from src.core.funasr_engine import DEFAULT_MODEL_ID


class TestConfig:
    """测试 src/config.py 配置模块"""

    def test_default_values_via_env(self):
        """测试通过环境变量设置默认值"""
        old_env = os.environ.copy()
        try:
            for key in ["ENGINE_TYPE", "MODEL_ID", "FUNASR_MODEL_ID", "MLX_MODEL_ID"]:
                os.environ.pop(key, None)

            import src.config

            with patch("src.config.load_dotenv"):
                importlib.reload(src.config)

            assert src.config.FUNASR_MODEL_ID == DEFAULT_MODEL_ID
            assert src.config.MLX_MODEL_ID == "mlx-community/Qwen3-ASR-1.7B-4bit"
        finally:
            os.environ.clear()
            os.environ.update(old_env)


class TestFactory:
    """测试 src/core/factory.py 工厂模块"""

    def test_create_engines(self):
        """测试引擎创建（不依赖环境变量reload）"""
        from src.core.funasr_engine import FunASREngine
        from src.core.mlx_engine import MlxAudioEngine

        funasr_engine = FunASREngine(model_id=DEFAULT_MODEL_ID)
        assert funasr_engine.model_id == DEFAULT_MODEL_ID

        mlx_engine = MlxAudioEngine(model_id="mlx-community/Qwen3-ASR-1.7B-4bit")
        assert mlx_engine.model_id == "mlx-community/Qwen3-ASR-1.7B-4bit"

    def test_should_route_sortformer_model_ids_to_sortformer_engine(self) -> None:
        from src.core.factory import _create_by_type

        sortformer_engine = MagicMock(name="sortformer-engine")

        with patch("src.core.sortformer_engine.SortformerEngine", return_value=sortformer_engine) as sortformer_cls:
            result = _create_by_type("mlx", "mlx-community/diar_sortformer_4spk-v1-fp32")

        assert result is sortformer_engine
        sortformer_cls.assert_called_once_with(model_id="mlx-community/diar_sortformer_4spk-v1-fp32")

    def test_should_keep_standard_mlx_models_on_mlx_audio_engine(self) -> None:
        from src.core.factory import _create_by_type

        mlx_engine = MagicMock(name="mlx-engine")

        with patch("src.core.mlx_engine.MlxAudioEngine", return_value=mlx_engine) as mlx_cls:
            result = _create_by_type("mlx", "mlx-community/Qwen3-ASR-1.7B-8bit")

        assert result is mlx_engine
        mlx_cls.assert_called_once_with(model_id="mlx-community/Qwen3-ASR-1.7B-8bit")
