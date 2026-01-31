"""
统一配置管理模块。
通过环境变量控制服务行为。
支持从 .env 文件加载配置。
"""
import os
from typing import Literal
from pathlib import Path

# 加载 .env 文件（如果存在）
from dotenv import load_dotenv

# 查找 .env 文件：优先使用项目根目录的 .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # 尝试从当前工作目录加载
    load_dotenv()

# 引擎类型
EngineType = Literal["funasr", "mlx"]

# === 引擎配置 ===
ENGINE_TYPE: EngineType = os.getenv("ENGINE_TYPE", "funasr")  # type: ignore

# === 模型配置 ===
# FunASR 默认模型
FUNASR_MODEL_ID = os.getenv("FUNASR_MODEL_ID", "iic/SenseVoiceSmall")

# MLX 默认模型
MLX_MODEL_ID = os.getenv("MLX_MODEL_ID", "mlx-community/Qwen3-ASR-1.7B-4bit")

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

# === 音频处理配置 ===
# 最大音频时长（分钟），超过则自动切片（仅 MLX 引擎）
MAX_AUDIO_DURATION_MINUTES = int(os.getenv("MAX_AUDIO_DURATION_MINUTES", "50"))

# 静音检测配置
SILENCE_THRESHOLD_SEC = float(os.getenv("SILENCE_THRESHOLD_SEC", "0.5"))
SILENCE_NOISE_DB = os.getenv("SILENCE_NOISE_DB", "-30dB")

# 音频归一化配置
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "64k")

# 重叠切片配置（fallback 策略）
CHUNK_OVERLAP_SECONDS = int(os.getenv("CHUNK_OVERLAP_SECONDS", "15"))

# Tokenizers 并行警告抑制
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
