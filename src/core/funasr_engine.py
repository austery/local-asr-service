import torch
import time
import gc
from funasr import AutoModel
from typing import Optional, Union, List, Dict

# 推荐使用的 Paraformer 模型 ID (支持时间戳，必须用于说话人分离)
# SEACO-Paraformer 是目前阿里最成熟的串联模型，中文识别 SOTA
DEFAULT_MODEL_ID = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"


class FunASREngine:
    """
    FunASR/Paraformer 推理引擎封装类。
    负责模型的生命周期管理（加载、推理、资源释放）。
    实现 ASREngine Protocol。
    
    SPEC-007: 支持说话人分离 (Speaker Diarization) 和多格式输出。
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        # 自动检测 M4 Pro (MPS) 环境
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        print(f"⚙️ Engine initialized. Target device: {self.device}")

    def load(self):
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
        print("   (If this is the first run, it will download the model automatically. Please wait.)")
        
        try:
            start_time = time.time()
            
            # SPEC-007: Paraformer + VAD + Punc + Cam++ 组合
            # 这是实现说话人分离的完整管道
            self.model = AutoModel(
                model=self.model_id,
                vad_model="fsmn-vad",  # 语音活动检测，用于切分长音频
                vad_kwargs={"max_single_segment_time": 30000},  # 30秒切片优化
                punc_model="ct-punc",  # 标点符号模型
                spk_model="cam++",     # 声纹识别模型（说话人分离）
                device=self.device,
                disable_update=True,   # 禁止每次都去 check update，加快启动速度
                log_level="ERROR"      # 减少刷屏日志
            )
            
            duration = time.time() - start_time
            print(f"✅ Model loaded successfully in {duration:.2f}s")
            
            # 简单的 Warmup (预热)，防止第一次推理卡顿
            self._warmup()
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise e

    def _warmup(self):
        """执行一次空推理，让 MPS 图编译完成"""
        print("🔥 Warming up model...")
        try:
            # 实际 FunASR 在加载时内部会有初始化
            pass 
        except Exception:
            pass

    def transcribe_file(
        self, 
        file_path: str, 
        language: str = "auto",
        output_format: str = "json",  # 选项: 'json', 'txt', 'srt'
        with_timestamp: bool = False,  # txt/srt 中是否包含时间戳
        **kwargs
    ) -> Union[Dict, str, List[Dict]]:
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

        # Paraformer 主要针对中文优化，但也支持部分语言
        # 注意: Paraformer 的 language 参数与 SenseVoice 不同
        
        # 调用 FunASR (SPEC-007: 启用说话人分离后会返回 sentence_info)
        res = self.model.generate(
            input=file_path,
            cache={},
            use_itn=use_itn,       # 逆文本标准化 (一百 -> 100)
            batch_size_s=60,       # 批处理大小 (60秒音频切片)
            merge_vad=True,        # 自动合并短句
            merge_length_s=15
        )
        
        # 解析结果
        result_data = res[0] if res else {}
        text = result_data.get("text", "")
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

    def _format_as_json(self, text: str, sentence_info: List[Dict]) -> Dict:
        """
        返回完整的结构化数据。
        这是你的"数字资产"，包含所有原始信息以便后续处理。
        """
        segments = [
            {
                "speaker": f"Speaker {info.get('spk', 0)}",
                "text": info.get("text", ""),
                "start": info.get("start", 0),  # 毫秒
                "end": info.get("end", 0)       # 毫秒
            }
            for info in sentence_info
        ]
        return {"text": text, "segments": segments}

    def _format_as_txt(self, sentence_info: List[Dict], with_timestamp: bool) -> str:
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

    def _format_as_srt(self, sentence_info: List[Dict]) -> str:
        """
        生成标准 SRT 字幕格式。
        
        格式:
        1
        00:00:05,000 --> 00:01:190,000
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

    def release(self):
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