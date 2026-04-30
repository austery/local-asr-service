"""
MLX Audio 推理引擎封装类。
支持 Qwen3-ASR、Whisper、Parakeet 等 mlx-audio 兼容模型。
支持自动音频切片（长音频超过限制时）。
支持说话人分离（某些模型特性）。
"""

import gc
import time
from pathlib import Path
from typing import Any

from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.utils import load_model

from src.adapters.audio_chunking import AudioChunkingService
from src.core.base_engine import EngineCapabilities

# Per-model capability profiles (prefix-matched, longest prefix wins)
_MLX_MODEL_CAPABILITIES: dict[str, EngineCapabilities] = {
    "mlx-community/Qwen3-ASR": EngineCapabilities(
        timestamp=True,
        language_detect=True,
    ),
    "mlx-community/whisper": EngineCapabilities(
        timestamp=True,
        language_detect=True,
    ),
    "mlx-community/parakeet": EngineCapabilities(
        timestamp=True,
    ),
}

# Conservative default for unknown MLX models
_MLX_DEFAULT_CAPS = EngineCapabilities()

_QWEN3_LANGUAGE_ALIASES: dict[str, str] = {
    "auto": "English",
    "en": "English",
    "eng": "English",
    "english": "English",
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
}

_QWEN3_SUPPORTED_LANGUAGES = frozenset({
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
})


def _resolve_mlx_capabilities(model_id: str) -> EngineCapabilities:
    """Resolve capabilities via longest-prefix match against model_id."""
    best_match = ""
    best_caps = _MLX_DEFAULT_CAPS
    for prefix, caps in _MLX_MODEL_CAPABILITIES.items():
        if model_id.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
            best_caps = caps
    return best_caps


def _is_qwen3_asr_model(model_id: str) -> bool:
    return "qwen3-asr" in model_id.lower()


def _normalize_mlx_language(model_id: str, language: str) -> str:
    if not _is_qwen3_asr_model(model_id):
        return language
    normalized_key = language.strip().lower()
    normalized = _QWEN3_LANGUAGE_ALIASES.get(normalized_key, language.strip())
    if normalized not in _QWEN3_SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported Qwen3-ASR language: {language}. "
            f"Expected one of: {', '.join(sorted(_QWEN3_SUPPORTED_LANGUAGES))}"
        )
    return normalized


class MlxAudioEngine:
    """
    MLX Audio 通用推理引擎。
    支持所有 mlx-audio 兼容的 STT 模型。
    实现 ASREngine Protocol。
    """

    def __init__(self, model_id: str = "mlx-community/Qwen3-ASR-1.7B-4bit"):
        self.model_id = model_id
        self._capabilities = _resolve_mlx_capabilities(model_id)
        self.model = None
        self.chunking_service = AudioChunkingService()
        print(f"⚙️ MLX Engine initialized. Model: {self.model_id}")

    @property
    def capabilities(self) -> EngineCapabilities:
        return self._capabilities

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
        print(
            "   (If this is the first run, it will download the model automatically. Please wait.)"
        )

        try:
            start_time = time.time()
            self.model = load_model(self.model_id)
            duration = time.time() - start_time
            print(f"✅ MLX Model loaded successfully in {duration:.2f}s")
        except Exception as e:
            print(f"❌ Failed to load MLX model: {e}")
            raise e

    def transcribe_file(
        self, file_path: str, language: str = "auto", **kwargs: Any
    ) -> str | dict[str, Any]:
        """
        执行推理，返回转录结果。
        自动处理长音频切片。
        支持多种输出格式（txt, json, srt, vtt）。

        Args:
            file_path: 音频文件路径
            language: 语言代码 (当前 mlx-audio 部分模型支持)
            **kwargs: 其他参数
                - verbose: bool - 详细输出
                - format: str - 输出格式 (txt, json, srt, vtt)
                - response_format: str - OpenAI 兼容的响应格式参数

        Returns:
            txt 格式: 转录文本字符串
            json 格式: 包含 text 和 segments 的字典（说话人信息）
        """
        if not self.model:
            raise RuntimeError("Model not loaded! Call engine.load() first.")

        verbose = kwargs.get("verbose", False)
        # 支持两种参数名：format (mlx-audio) 和 response_format (OpenAI)
        output_format = kwargs.get("format") or kwargs.get("response_format", "txt")

        # 标准化格式名称
        if output_format in ["json", "verbose_json"]:
            output_format = "json"
        elif output_format not in ["txt", "srt", "vtt"]:
            output_format = "txt"  # 默认文本格式

        try:
            # 步骤1: 检查音频是否需要切片
            chunks = self.chunking_service.process_audio(file_path)

            # 步骤2: 转录所有切片
            results = []
            normalized_language = _normalize_mlx_language(self.model_id, language)
            for i, chunk_path in enumerate(chunks):
                print(f"🎙️ Transcribing chunk {i + 1}/{len(chunks)} (format: {output_format})...")
                try:
                    result = generate_transcription(
                        model=self.model,
                        audio=chunk_path,
                        format=output_format,
                        verbose=verbose,
                        language=normalized_language,
                    )
                    results.append(result)
                finally:
                    # 清理临时切片文件
                    if (chunk_path != chunks[0] or len(chunks) > 1) and (
                        ".chunk_" in chunk_path or len(chunks) > 1
                    ):
                        Path(chunk_path).unlink(missing_ok=True)

            # 步骤3: 根据格式合并结果
            if output_format == "json":
                final_result = self._merge_json_results(results)
            else:
                # txt, srt, vtt 格式
                texts = []
                for result in results:
                    text = result.text.strip() if hasattr(result, "text") else str(result).strip()
                    texts.append(text)
                final_result = " ".join(texts)

            if len(chunks) > 1:
                print(f"✅ Successfully merged {len(chunks)} chunks")

            return final_result

        except Exception as e:
            print(f"❌ MLX transcription failed: {e}")
            raise e

    def _merge_json_results(self, results: list[Any]) -> dict[str, Any]:
        """
        合并多个 JSON 格式的转录结果。

        Args:
            results: mlx-audio 返回的结果对象列表

        Returns:
            合并后的字典，包含 text 和 segments
        """
        if not results:
            return {"text": "", "segments": []}

        # 如果只有一个结果，直接转换
        if len(results) == 1:
            return self._result_to_dict(results[0])

        # 合并多个结果
        all_text = []
        all_segments = []
        time_offset = 0.0

        for _i, result in enumerate(results):
            result_dict = self._result_to_dict(result)

            # 累加文本
            all_text.append(result_dict.get("text", ""))

            # 调整时间戳并合并 segments
            segments = result_dict.get("segments", [])
            for segment in segments:
                adjusted_segment = segment.copy()
                if "start" in adjusted_segment:
                    adjusted_segment["start"] += time_offset
                if "end" in adjusted_segment:
                    adjusted_segment["end"] += time_offset
                all_segments.append(adjusted_segment)

            # 更新时间偏移（使用最后一个 segment 的结束时间）
            if segments:
                last_segment = segments[-1]
                last_timestamp = last_segment.get("end", last_segment.get("start"))
                if isinstance(last_timestamp, int | float) and not isinstance(last_timestamp, bool):
                    time_offset += float(last_timestamp)

        return {"text": " ".join(all_text), "segments": all_segments}

    def _result_to_dict(self, result: Any) -> dict[str, Any]:
        """
        将 mlx-audio 结果对象转换为字典。

        Args:
            result: mlx-audio 返回的结果对象

        Returns:
            包含 text 和 segments 的字典
        """
        result_dict: dict[str, Any] = {"text": "", "segments": []}

        # 提取文本
        if hasattr(result, "text"):
            text = result.text
            result_dict["text"] = text.strip() if isinstance(text, str) else ""
        elif isinstance(result, dict):
            text = result.get("text", "")
            result_dict["text"] = text if isinstance(text, str) else ""

        # 提取 segments（说话人信息）
        if hasattr(result, "segments"):
            segments = result.segments
            if isinstance(segments, list):
                result_dict["segments"] = [self._normalize_segment(seg) for seg in segments]
        elif isinstance(result, dict) and "segments" in result:
            result_dict["segments"] = [self._normalize_segment(seg) for seg in result["segments"]]

        return result_dict

    def _normalize_segment(self, segment: Any) -> dict[str, Any]:
        """
        标准化 segment 格式。

        Args:
            segment: 原始 segment 对象或字典

        Returns:
            标准化的 segment 字典
        """
        normalized = {}

        # 处理字典格式
        if isinstance(segment, dict):
            normalized = segment.copy()
        # 处理对象格式
        else:
            for attr in ["speaker", "start", "end", "text"]:
                if hasattr(segment, attr):
                    normalized[attr] = getattr(segment, attr)

        return normalized

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
