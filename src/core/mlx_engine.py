"""
MLX Audio 推理引擎封装类。
支持 VibeVoice、Whisper、Qwen3-ASR 等 mlx-audio 兼容模型。
"""
import time
import gc
from typing import Optional

from mlx_audio.stt.utils import load_model
from mlx_audio.stt.generate import generate_transcription


class MlxAudioEngine:
    """
    MLX Audio 通用推理引擎。
    支持所有 mlx-audio 兼容的 STT 模型。
    实现 ASREngine Protocol。
    """

    def __init__(self, model_id: str = "mlx-community/VibeVoice-ASR-4bit"):
        self.model_id = model_id
        self.model = None
        print(f"⚙️ MLX Engine initialized. Model: {self.model_id}")

    def load(self) -> None:
        """
        加载模型。
        mlx-audio 会自动处理：
        1. 检查本地缓存 (~/.cache/huggingface)
        2. 如果不存在，自动下载
        3. 加载到 MLX 统一内存
        """
        if self.model is not None:
            print("⚠️ Model already loaded. Skipping.")
            return

        print(f"🚀 Loading MLX model '{self.model_id}'...")
        print("   (If this is the first run, it will download the model automatically. Please wait.)")

        try:
            start_time = time.time()
            self.model = load_model(self.model_id)
            duration = time.time() - start_time
            print(f"✅ MLX Model loaded successfully in {duration:.2f}s")
        except Exception as e:
            print(f"❌ Failed to load MLX model: {e}")
            raise e

    def transcribe_file(self, file_path: str, language: str = "auto", **kwargs) -> str:
        """
        执行推理，返回转录文本。
        
        Args:
            file_path: 音频文件路径
            language: 语言代码 (当前 mlx-audio 部分模型支持)
            **kwargs: 其他参数（如 verbose）
            
        Returns:
            转录文本
        """
        if not self.model:
            raise RuntimeError("Model not loaded! Call engine.load() first.")

        verbose = kwargs.get("verbose", False)

        try:
            result = generate_transcription(
                model=self.model,
                audio_path=file_path,
                verbose=verbose
            )
            # generate_transcription 返回的对象有 .text 属性
            return result.text.strip() if hasattr(result, 'text') else str(result).strip()
        except Exception as e:
            print(f"❌ MLX transcription failed: {e}")
            raise e

    def release(self) -> None:
        """
        释放资源。
        MLX 使用统一内存，主要通过 Python GC 清理。
        """
        if self.model:
            print(f"♻️ Releasing MLX model '{self.model_id}'...")
            del self.model
            self.model = None
            gc.collect()
            print("✅ MLX Model released.")
