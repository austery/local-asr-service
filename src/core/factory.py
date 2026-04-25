"""
引擎工厂模块。
根据配置或 ModelSpec 创建对应的 ASR 引擎实例。
"""

import logging
from typing import TypeAlias

from src.config import ENGINE_TYPE, get_model_id
from src.core.base_engine import ASREngine
from src.core.diarization_port import DiarizationPort
from src.core.model_registry import ModelSpec

logger = logging.getLogger(__name__)

EngineInstance: TypeAlias = ASREngine | DiarizationPort
_SORTFORMER_MODEL_PREFIX = "mlx-community/diar_sortformer"


def create_engine() -> EngineInstance:
    """
    根据 ENGINE_TYPE 环境变量创建引擎实例（服务启动时调用）。
    """
    return _create_by_type(ENGINE_TYPE, get_model_id())


def create_engine_for_spec(spec: ModelSpec) -> EngineInstance:
    """
    根据 ModelSpec 创建引擎实例（动态换模时调用）。
    """
    return _create_by_type(spec.engine_type, spec.model_id)


def _create_by_type(engine_type: str, model_id: str) -> EngineInstance:
    if engine_type == "funasr":
        from src.core.funasr_engine import FunASREngine

        logger.info(f"🏭 Creating FunASR engine with model: {model_id}")
        return FunASREngine(model_id=model_id)

    if engine_type == "mlx":
        if model_id.startswith(_SORTFORMER_MODEL_PREFIX):
            from src.core.sortformer_engine import SortformerEngine

            logger.info(f"🏭 Creating Sortformer engine with model: {model_id}")
            return SortformerEngine(model_id=model_id)

        from src.core.mlx_engine import MlxAudioEngine

        logger.info(f"🏭 Creating MLX Audio engine with model: {model_id}")
        return MlxAudioEngine(model_id=model_id)

    raise ValueError(f"Unsupported engine_type: '{engine_type}'. Must be 'funasr' or 'mlx'.")
