import gc
import logging
import time
from typing import Any

import funasr.auto.auto_model as _auto_model
import funasr.models.campplus.utils as _campplus_utils
import torch
from funasr import AutoModel

from src.core.base_engine import EngineCapabilities

_log = logging.getLogger(__name__)

# Monkey-patch for FunASR bug: distribute_spk crashes when sv_output contains
# entries with spk_st=None or spk_ed=None (happens on short/ambiguous segments).
# See: funasr/models/campplus/utils.py:203
def _patched_distribute_spk(sentence_list: list, sd_time_list: list) -> list:
    valid = [(st, ed, spk) for st, ed, spk in sd_time_list if st is not None and ed is not None]
    discarded = len(sd_time_list) - len(valid)
    if discarded > 0:
        _log.warning(
            "distribute_spk patch: filtered %d/%d speaker segments with None timestamps; "
            "speaker attribution for affected sentences may be inaccurate.",
            discarded,
            len(sd_time_list),
        )
    sd_time_ms = [(st * 1000, ed * 1000, spk) for st, ed, spk in valid]
    for d in sentence_list:
        sentence_start = d["start"]
        sentence_end = d["end"]
        sentence_spk = 0
        max_overlap = 0
        for spk_st, spk_ed, spk in sd_time_ms:
            overlap = max(min(sentence_end, spk_ed) - max(sentence_start, spk_st), 0)
            if overlap > max_overlap:
                max_overlap = overlap
                sentence_spk = spk
        d["spk"] = int(sentence_spk)
    return sentence_list

# Patch the source module attribute (guards against direct attribute-access call sites).
_campplus_utils.distribute_spk = _patched_distribute_spk

# Patch the auto_model namespace directly — this is the actual call site.
# auto_model.py uses `from funasr.models.campplus.utils import distribute_spk`,
# so it holds a direct reference to the old function that must be replaced here.
# Guard against FunASR refactors where this attribute may no longer exist — a
# silent set on a missing attribute would make the patch appear to work while
# the original bug remains active.
if not hasattr(_auto_model, "distribute_spk"):
    _log.warning(
        "funasr.auto.auto_model has no 'distribute_spk' attribute; "
        "the None-timestamp bug fix patch was NOT applied to the call site. "
        "FunASR may have been updated — verify the patch is still needed."
    )
else:
    _auto_model.distribute_spk = _patched_distribute_spk

# 推荐使用的 Paraformer 模型 ID (支持时间戳，必须用于说话人分离)
# SEACO-Paraformer 是目前阿里最成熟的串联模型，中文识别 SOTA
DEFAULT_MODEL_ID = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

# Per-model capability profiles (prefix-matched, longest prefix wins)
_FUNASR_MODEL_CAPABILITIES: dict[str, EngineCapabilities] = {
    "iic/speech_seaco_paraformer": EngineCapabilities(
        timestamp=True,
        diarization=True,
        language_detect=True,
    ),
    "iic/speech_paraformer": EngineCapabilities(
        timestamp=True,
        diarization=True,
        language_detect=True,
    ),
    "iic/SenseVoiceSmall": EngineCapabilities(
        emotion_tags=True,
        language_detect=True,
    ),
    "iic/SenseVoiceLarge": EngineCapabilities(
        emotion_tags=True,
        language_detect=True,
    ),
}

# Conservative default for unknown FunASR models
_FUNASR_DEFAULT_CAPS = EngineCapabilities()


def _resolve_capabilities(model_id: str) -> EngineCapabilities:
    """Resolve capabilities via longest-prefix match against model_id."""
    best_match = ""
    best_caps = _FUNASR_DEFAULT_CAPS
    for prefix, caps in _FUNASR_MODEL_CAPABILITIES.items():
        if model_id.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
            best_caps = caps
    return best_caps


class FunASREngine:
    """
    FunASR 推理引擎封装类。
    负责模型的生命周期管理（加载、推理、资源释放）。
    实现 ASREngine Protocol。

    支持两种模式：
    - Paraformer 模型：完整管道 (VAD + ASR + Punc + CAM++ 说话人分离)
    - SenseVoice 等模型：纯转录模式 (VAD + ASR + Punc，无说话人分离)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self._capabilities = _resolve_capabilities(model_id)
        # 自动检测 Apple Silicon (MPS) 环境
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        print(f"⚙️ Engine initialized. Target device: {self.device}")
        if not self._capabilities.diarization:
            print(f"ℹ️  Model '{model_id}' does not support diarization. Speaker model disabled.")

    @property
    def capabilities(self) -> EngineCapabilities:
        return self._capabilities

    def load(self) -> None:
        """
        加载模型。
        这一步会触发 FunASR 的自动检查机制：
        1. 检查本地缓存 (~/.cache/modelscope)
        2. 如果不存在，自动下载
        3. 加载到内存/显存
        """
        if self.model is not None:
            print("⚠️ Model already loaded. Skipping.")
            return

        print(f"🚀 Loading model '{self.model_id}' on {self.device}...")
        print(
            "   (If this is the first run, it will download the model automatically. Please wait.)"
        )

        try:
            start_time = time.time()

            # 根据模型能力决定加载的管道组件
            # Paraformer: VAD + ASR + Punc + CAM++ (完整说话人分离)
            # SenseVoice 等: VAD + ASR + Punc (无说话人分离，因为不支持时间戳)
            model_kwargs: dict[str, object] = dict(
                model=self.model_id,
                vad_model="fsmn-vad",  # 语音活动检测，用于切分长音频
                vad_kwargs={"max_single_segment_time": 30000},  # 30秒切片优化
                punc_model="ct-punc",  # 标点符号模型
                device=self.device,
                disable_update=True,  # 禁止每次都去 check update，加快启动速度
                log_level="ERROR",  # 减少刷屏日志
            )
            if self._capabilities.diarization:
                model_kwargs["spk_model"] = "cam++"  # 声纹识别模型（说话人分离）

            self.model = AutoModel(**model_kwargs)

            duration = time.time() - start_time
            print(f"✅ Model loaded successfully in {duration:.2f}s")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise e

    def transcribe_file(  # type: ignore[override]
        self,
        file_path: str,
        language: str = "auto",
        output_format: str = "json",  # 选项: 'json', 'txt', 'srt'
        with_timestamp: bool = False,  # txt/srt 中是否包含时间戳
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        """
        执行推理，支持多种输出格式。

        Args:
            file_path: 音频文件路径
            language: 语言代码 (auto, zh, en, yue, ja, ko)
            output_format: 输出格式
                - 'json': 返回完整结构化数据 (推荐存储，包含时间戳和说话人)
                - 'txt': 返回人类易读文本 (适合 RAG/LLM)
                - 'srt': 返回 SRT 字幕格式
            with_timestamp: 如果是 txt 格式，是否在行首保留 [00:12] 时间标记

        Returns:
            根据 output_format 返回不同格式:
            - json: {"text": str, "segments": List[Dict]}
            - txt: str
            - srt: str

        注意：这是同步阻塞方法，必须在 Service 层通过线程池调用。
        """
        if not self.model:
            raise RuntimeError("Model not loaded! Call engine.load() first.")

        # 从 kwargs 提取 FunASR 特定参数
        use_itn = kwargs.get("use_itn", True)

        # 调用 FunASR 推理
        res = self.model.generate(
            input=file_path,
            cache={},
            use_itn=use_itn,  # 逆文本标准化 (一百 -> 100)
            batch_size_s=60,  # 批处理大小 (60秒音频切片)
            merge_vad=True,  # 自动合并短句
            merge_length_s=15,
        )

        # 解析结果
        result_data = res[0] if res else {}
        text = result_data.get("text", "")

        # SenseVoice 输出包含特殊标签 (<|zh|><|NEUTRAL|> 等)，需要清洗
        if self._capabilities.emotion_tags:
            from src.adapters.text import clean_sensevoice_tags

            text = clean_sensevoice_tags(text)

        sentence_info = result_data.get("sentence_info", [])

        # === 内存优化：打扫战场 ===
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        # 兜底：如果模型没返回分句信息，直接返回全文
        if not sentence_info:
            if output_format == "json":
                return {"text": text, "segments": None}
            return text

        # 根据 output_format 格式化输出
        if output_format == "json":
            return self._format_as_json(text, sentence_info)
        elif output_format == "txt":
            return self._format_as_txt(sentence_info, with_timestamp)
        elif output_format == "srt":
            return self._format_as_srt(sentence_info)

        # 默认返回 JSON
        return self._format_as_json(text, sentence_info)

    def _format_as_json(self, text: str, sentence_info: list[dict]) -> dict:
        """
        返回完整的结构化数据。
        这是你的"数字资产"，包含所有原始信息以便后续处理。
        """
        segments = [
            {
                "speaker": f"Speaker {info.get('spk', 0)}",
                "text": info.get("text", ""),
                "start": self._ms_to_seconds(info.get("start", 0)),
                "end": self._ms_to_seconds(info.get("end", 0)),
            }
            for info in sentence_info
        ]
        duration = max(
            (
                segment["end"]
                for segment in segments
                if isinstance(segment.get("end"), int | float)
            ),
            default=0.0,
        )
        return {"text": text, "segments": segments, "duration": duration}

    @staticmethod
    def _ms_to_seconds(value: object) -> float:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return round(float(value) / 1000.0, 3)
        return 0.0

    def _format_as_txt(self, sentence_info: list[dict], with_timestamp: bool) -> str:
        """
        生成人类易读的访谈文本。
        适合用于 RAG 知识库或 LLM 处理。

        格式样例:
        - 纯净模式: [Speaker 0]: 大家好...
        - 带时间戳: [02:15] [Speaker 0]: 大家好...
        """
        lines = []

        for info in sentence_info:
            spk = info.get("spk")
            text = info.get("text", "")
            start_ms = info.get("start", 0)

            # 格式化说话人标签
            spk_tag = f"[Speaker {spk}]" if spk is not None else "[Unknown]"

            # 格式化时间戳 (仅当用户需要时)
            time_tag = ""
            if with_timestamp and start_ms is not None:
                # 毫秒 -> MM:SS
                seconds = int(start_ms / 1000)
                m, s = divmod(seconds, 60)
                time_tag = f"[{m:02d}:{s:02d}] "

            # 组合一行
            line = f"{time_tag}{spk_tag}: {text}"
            lines.append(line)

        return "\n".join(lines)

    def _format_as_srt(self, sentence_info: list[dict]) -> str:
        """
        生成标准 SRT 字幕格式。

        格式:
        1
        00:00:05,000 --> 00:00:20,000
        [Speaker 0]: so what is some of the questions？
        """
        lines = []

        for idx, info in enumerate(sentence_info, start=1):
            spk = info.get("spk", 0)
            text = info.get("text", "")
            start_ms = info.get("start", 0)
            end_ms = info.get("end", 0)

            # 毫秒转 SRT 时间格式 (HH:MM:SS,mmm)
            start_srt = self._ms_to_srt_time(start_ms)
            end_srt = self._ms_to_srt_time(end_ms)

            # SRT 格式
            lines.append(str(idx))
            lines.append(f"{start_srt} --> {end_srt}")
            lines.append(f"[Speaker {spk}]: {text}")
            lines.append("")  # 空行分隔

        return "\n".join(lines)

    def _ms_to_srt_time(self, ms: int) -> str:
        """将毫秒转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
        if ms < 0:
            ms = 0
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def release(self) -> None:
        """
        释放显存资源。
        用于热更新模型或服务关闭时清理资源。
        """
        if self.model:
            print(f"♻️ Releasing model '{self.model_id}'...")
            del self.model
            self.model = None

            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

            gc.collect()
            print("✅ Model released and memory cleared.")
