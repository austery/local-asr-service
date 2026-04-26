import importlib
import os
from unittest.mock import MagicMock, patch

from src.core.funasr_engine import DEFAULT_MODEL_ID

_FIRERED_DEFAULT_MODEL = "FireRedTeam/FireRedASR2-AED"


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

    # Issue #2: ENGINE_TYPE=firered must not fall through to FUNASR default
    def test_get_model_id_returns_firered_default_when_engine_type_is_firered(self):
        """With ENGINE_TYPE=firered and no MODEL_ID override, get_model_id() must
        return the FireRed default, NOT the FunASR default.

        Before the fix, the funasr fallback branch would be reached and the FunASR
        model ID would be passed into FireRedEngine, causing a broken startup."""
        old_env = os.environ.copy()
        try:
            os.environ["ENGINE_TYPE"] = "firered"
            os.environ.pop("MODEL_ID", None)
            os.environ.pop("FIRERED_MODEL_ID", None)

            import src.config

            with patch("src.config.load_dotenv"):
                importlib.reload(src.config)

            result = src.config.get_model_id()
            assert result == _FIRERED_DEFAULT_MODEL, (
                f"Expected FireRed default '{_FIRERED_DEFAULT_MODEL}', got '{result}'. "
                "get_model_id() has no firered branch and falls through to funasr."
            )
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_firered_model_id_env_var_is_respected(self):
        """FIRERED_MODEL_ID env var must override the built-in default."""
        old_env = os.environ.copy()
        custom = "FireRedTeam/custom-model"
        try:
            os.environ["ENGINE_TYPE"] = "firered"
            os.environ.pop("MODEL_ID", None)
            os.environ["FIRERED_MODEL_ID"] = custom

            import src.config

            with patch("src.config.load_dotenv"):
                importlib.reload(src.config)

            result = src.config.get_model_id()
            assert result == custom
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_engine_type_literal_includes_firered(self):
        """EngineType must include 'firered' so type-checkers and validation accept it."""
        import src.config
        import typing

        args = typing.get_args(src.config.EngineType)
        assert "firered" in args, (
            f"'firered' not in EngineType args {args}. "
            "Add 'firered' to the Literal in src/config.py."
        )


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

    def test_should_route_firered_engine_type_to_firered_engine(self) -> None:
        from src.core.factory import _create_by_type

        firered_engine = MagicMock(name="firered-engine")

        with patch(
            "src.core.firered_engine.FireRedEngine", return_value=firered_engine
        ) as firered_cls:
            result = _create_by_type("firered", "FireRedTeam/FireRedASR2-AED")

        assert result is firered_engine
        firered_cls.assert_called_once_with(model_id="FireRedTeam/FireRedASR2-AED")
