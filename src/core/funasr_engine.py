import gc
import importlib
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

import torch
from funasr import AutoModel

from src.core.base_engine import EngineCapabilities

_log = logging.getLogger(__name__)


# Monkey-patch for FunASR bug: ts_prediction_lfr6_standard returns fewer token
# timestamps than tokens when CIF must fire twice within one frame. Its CIF loop
# records at most one fire per frame and carries leftover weight to the next
# frame, so fires are lost only as weight still pending after the final frame.
# Renormalisation cannot help when the weight sum is already correct (seen on a
# ~120 ms VAD segment holding only "嗯", alphas [0.03, 0, 0, 0, 0.24, 1.72]).
# FunASR >= 1.4.19 sentence_postprocess then raises "time_stamp is missing a
# retained token timestamp" and the whole request fails.
#
# The patch returns upstream output unchanged whenever sentence_postprocess would
# accept it. Otherwise it gives each missing token one upsampled frame at the
# segment end, where the lost fires belong. Only the last upstream span can
# change (truncated, or shared from its start); fills never start before the
# segment start, so a merged span that started earlier is clamped to it.
_TsResult = tuple[str, list[list[int]]]
_TsPrediction = Callable[..., _TsResult]
_LFR6_FRAME_MS = 10.0 * 6  # one LFR6 encoder frame; matches upstream TIME_RATE
# Tokens sentence_postprocess drops; trailing ones need no timestamp.
_UNTIMED_TOKENS = frozenset({"<s>", "</s>", "<unk>", "<OOV>"})
_REQUIRED_TS_PARAMS = frozenset({"us_alphas", "char_list", "vad_offset", "upsample_rate"})


def _fill_missing_timestamps(
    timestamps: list[list[int]],
    missing: int,
    segment_start: int,
    segment_end: int,
    frame_ms: float,
) -> list[list[int]]:
    """Append `missing` spans in the last frames of the segment.

    Each missing token gets one upsampled frame at the segment end. The last
    timed span, if it overlaps those frames, is truncated when it keeps at least
    one frame; otherwise its token shares the end frames from its own start,
    clamped to the segment start. Every other upstream span is returned unchanged.
    """
    kept = [list(pair) for pair in timestamps]
    share_count = missing
    share_start = int(segment_end - missing * frame_ms)
    if kept and kept[-1][1] > share_start:
        if share_start - kept[-1][0] >= frame_ms:
            kept[-1][1] = share_start
        else:
            share_start = kept.pop()[0]
            share_count += 1
    share_start = max(share_start, segment_start)
    step = (segment_end - share_start) / share_count
    shared = [
        [int(share_start + i * step), int(share_start + (i + 1) * step)]
        for i in range(share_count)
    ]
    shared[-1][1] = segment_end
    return kept + shared


def _build_ts_prediction_patch(upstream: _TsPrediction) -> _TsPrediction:
    signature = inspect.signature(upstream)

    def patched(*args: object, **kwargs: object) -> _TsResult:
        # Forward every argument unchanged. If the arguments cannot be read by
        # name, return the upstream result as is; the fill only repairs results
        # sentence_postprocess would reject.
        res_txt, timestamps = upstream(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return res_txt, timestamps
        bound.apply_defaults()
        tokens = [
            t.decode("utf-8") if isinstance(t, bytes) else str(t)
            for t in bound.arguments["char_list"]
        ]
        needed = max(
            (i + 1 for i, t in enumerate(tokens) if t not in _UNTIMED_TOKENS), default=0
        )
        missing = needed - len(timestamps)
        if missing <= 0:
            return res_txt, timestamps

        vad_offset = float(bound.arguments["vad_offset"] or 0.0)
        frame_ms = _LFR6_FRAME_MS / int(bound.arguments["upsample_rate"])
        segment_start = int(vad_offset)
        segment_end = int(vad_offset + int(bound.arguments["us_alphas"].shape[-1]) * frame_ms)
        filled = _fill_missing_timestamps(
            timestamps, missing, segment_start, segment_end, frame_ms
        )
        _log.warning(
            "ts_prediction patch: filled %d/%d token timestamps ending at %d ms "
            "(CIF fired fewer times than tokens).",
            missing,
            needed,
            segment_end,
        )
        return res_txt, filled

    return patched


def _install_ts_prediction_patch() -> _TsPrediction | None:
    """Patch the SEACO call site, or warn and return None if FunASR changed.

    seaco_paraformer uses `from funasr.utils.timestamp_tools import
    ts_prediction_lfr6_standard`, so the call-site namespace is patched. SEACO is
    the only registered Paraformer alias (model_registry.py). Ad-hoc `iic/` models
    that load other Paraformer classes, and the source module, stay unpatched.
    """
    try:
        timestamp_tools = importlib.import_module("funasr.utils.timestamp_tools")
        seaco = importlib.import_module("funasr.models.seaco_paraformer.model")
    except Exception as exc:  # noqa: BLE001 - FunASR itself tolerates any submodule import error
        _skip_ts_patch(f"FunASR timestamp modules failed to import ({exc!r})")
        return None
    upstream = getattr(timestamp_tools, "ts_prediction_lfr6_standard", None)
    binding = getattr(seaco, "ts_prediction_lfr6_standard", None)
    if upstream is None or binding is None:
        _skip_ts_patch("ts_prediction_lfr6_standard moved out of its FunASR modules")
        return None
    if binding is not upstream and getattr(binding, "__module__", None) != __name__:
        _skip_ts_patch("seaco_paraformer no longer binds the timestamp_tools function")
        return None
    try:
        parameters = set(inspect.signature(upstream).parameters)
    except (TypeError, ValueError):
        _skip_ts_patch("ts_prediction_lfr6_standard has no readable signature")
        return None
    absent = _REQUIRED_TS_PARAMS - parameters
    if absent:
        _skip_ts_patch(f"ts_prediction_lfr6_standard lacks parameters {sorted(absent)}")
        return None
    patched = _build_ts_prediction_patch(upstream)
    setattr(seaco, "ts_prediction_lfr6_standard", patched)  # noqa: B010 - dynamic module
    return patched


def _skip_ts_patch(reason: str) -> None:
    _log.warning(
        "%s; the single-frame CIF timestamp patch was NOT applied. FunASR may have "
        "been updated — verify the patch is still needed.",
        reason,
    )


_patched_ts_prediction_lfr6_standard = _install_ts_prediction_patch()

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
        ms = max(ms, 0)
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
