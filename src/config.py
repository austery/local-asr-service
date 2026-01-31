"""
统一配置管理模块。
通过环境变量控制服务行为。
"""
import os
from typing import Literal

# 引擎类型
EngineType = Literal["funasr", "mlx"]

# === 引擎配置 ===
ENGINE_TYPE: EngineType = os.getenv("ENGINE_TYPE", "funasr")  # type: ignore

# === 模型配置 ===
# FunASR 默认模型
FUNASR_MODEL_ID = os.getenv("FUNASR_MODEL_ID", "iic/SenseVoiceSmall")

# MLX 默认模型
MLX_MODEL_ID = os.getenv("MLX_MODEL_ID", "mlx-community/VibeVoice-ASR-4bit")

# 通用 MODEL_ID（优先级高于引擎特定配置）
MODEL_ID = os.getenv("MODEL_ID", None)

def get_model_id() -> str:
    """获取当前引擎应使用的模型 ID"""
    if MODEL_ID:
        return MODEL_ID
    if ENGINE_TYPE == "mlx":
        return MLX_MODEL_ID
    return FUNASR_MODEL_ID

# === 服务配置 ===
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "50070"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "50"))

# === 日志配置 ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
