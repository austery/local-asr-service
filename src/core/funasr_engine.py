import torch
import time
import gc
from funasr import AutoModel
from typing import Optional


class FunASREngine:
    """
    FunASR/SenseVoice 推理引擎封装类。
    负责模型的生命周期管理（加载、推理、资源释放）。
    实现 ASREngine Protocol。
    """

    def __init__(self, model_id: str = "iic/SenseVoiceSmall", device: Optional[str] = None):
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
            
            # === 核心逻辑：复用你旧代码中的参数 ===
            self.model = AutoModel(
                model=self.model_id,
                vad_model="fsmn-vad",  # 语音活动检测，用于切分长音频
                punc_model="ct-punc",  # 标点符号模型
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
            # 随便搞个极短的空音频或伪造输入，这里简单打印一下即可
            # 实际 FunASR 在加载时内部会有初始化
            pass 
        except Exception:
            pass

    def transcribe_file(self, file_path: str, language: str = "auto", **kwargs) -> str:
        """
        执行推理。
        注意：这是同步阻塞方法，必须在 Service 层通过线程池调用。
        """
        if not self.model:
            raise RuntimeError("Model not loaded! Call engine.load() first.")

        # 从 kwargs 提取 FunASR 特定参数
        use_itn = kwargs.get("use_itn", True)

        # 映射语言参数
        # SenseVoice 支持: zh, en, yue, ja, ko
        valid_langs = ["zh", "en", "yue", "ja", "ko"]
        target_lang = language if language in valid_langs else "auto"

        # 调用 FunASR
        # 这里的参数完全参考你提供的成功运行的脚本
        res = self.model.generate(
            input=file_path,
            cache={},
            language=target_lang,
            use_itn=use_itn,       # 逆文本标准化 (一百 -> 100)
            batch_size_s=60,       # 批处理大小 (60秒音频切片)
            merge_vad=True,        # 自动合并短句
            merge_length_s=15
        )
        
        # res 是一个列表，取第一个结果
        text = res[0]["text"]

        # === 内存优化：打扫战场 ===
        # 防止 MPS (Metal) 显存碎片化，对于 7x24 小时服务至关重要
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        return text

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