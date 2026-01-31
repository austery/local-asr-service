"""
MLX Audio 推理引擎封装类。
支持 VibeVoice、Whisper、Qwen3-ASR 等 mlx-audio 兼容模型。
支持自动音频切片（长音频超过限制时）。
"""
import time
import gc
from pathlib import Path
from typing import Optional

from mlx_audio.stt.utils import load_model
from mlx_audio.stt.generate import generate_transcription

from src.adapters.audio_chunking import AudioChunkingService


class MlxAudioEngine:
    """
    MLX Audio 通用推理引擎。
    支持所有 mlx-audio 兼容的 STT 模型。
    实现 ASREngine Protocol。
    """

    def __init__(self, model_id: str = "mlx-community/VibeVoice-ASR-4bit"):
        self.model_id = model_id
        self.model = None
        self.chunking_service = AudioChunkingService()
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
        自动处理长音频切片。
        
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
            # 步骤1: 检查音频是否需要切片
            import asyncio
            chunks = asyncio.run(self.chunking_service.process_audio(file_path))
            
            # 步骤2: 转录所有切片
            transcripts = []
            for i, chunk_path in enumerate(chunks):
                print(f"🎙️ Transcribing chunk {i + 1}/{len(chunks)}...")
                try:
                    result = generate_transcription(
                        model=self.model,
                        audio=chunk_path,
                        verbose=verbose
                    )
                    text = result.text.strip() if hasattr(result, 'text') else str(result).strip()
                    transcripts.append(text)
                finally:
                    # 清理临时切片文件（但保留原始归一化文件）
                    if chunk_path != chunks[0] or len(chunks) > 1:
                        # 只删除切片文件，不删除原始归一化文件（如果只有一个文件）
                        if ".chunk_" in chunk_path or len(chunks) > 1:
                            Path(chunk_path).unlink(missing_ok=True)
            
            # 步骤3: 合并结果
            final_text = " ".join(transcripts)
            
            if len(chunks) > 1:
                print(f"✅ Successfully merged {len(chunks)} chunks")
            
            return final_text
            
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
