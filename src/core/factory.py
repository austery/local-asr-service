"""
引擎工厂模块。
根据配置创建对应的 ASR 引擎实例。
"""

from src.config import ENGINE_TYPE, get_model_id
from src.core.base_engine import ASREngine


def create_engine() -> ASREngine:
    """
    根据 ENGINE_TYPE 环境变量创建引擎实例。

    Returns:
        ASREngine 实例 (FunASREngine 或 MlxAudioEngine)

    Raises:
        ValueError: 不支持的引擎类型
    """
    model_id = get_model_id()

    if ENGINE_TYPE == "funasr":
        from src.core.funasr_engine import FunASREngine

        print(f"🏭 Creating FunASR engine with model: {model_id}")
        return FunASREngine(model_id=model_id)

    elif ENGINE_TYPE == "mlx":
        from src.core.mlx_engine import MlxAudioEngine

        print(f"🏭 Creating MLX Audio engine with model: {model_id}")
        return MlxAudioEngine(model_id=model_id)

    else:
        raise ValueError(f"Unsupported ENGINE_TYPE: {ENGINE_TYPE}. Must be 'funasr' or 'mlx'.")
