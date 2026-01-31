---
specId: SPEC-103
title: 音频处理与文本清洗 (Audio & Text Processing)
status: 🚧 规划中
priority: P1
owner: User
relatedSpecs: [SPEC-102]
---

## 1. 目标
处理输入（音频解码）和输出（文本清洗），保持无状态 (Stateless)，便于单元测试。

## 2. 音频模块 (audio_utils.py)

*不需要 `AudioProcessor` 类*。

```python
def load_and_resample(file_bytes: bytes, target_sr=16000) -> np.ndarray:
    """
    输入: 文件二进制流
    输出: Float32 Numpy Array (16kHz)
    实现: 调用 ffmpeg-python pipe 模式，直接内存转换，不写磁盘。
    """
    pass
````

## 3\. 文本清洗模块 (text\_utils.py)

**不需要 `TagCleaner` 类*。SenseVoice 的输出通常包含 `<|zh|><|NEUTRAL|>...`。

```python
import re

def clean_sensevoice_output(raw_text: str, remove_emotions: bool = True) -> str:
    """
    纯函数：输入原始文本，输出清洗后的文本。
    """
    # 1. 移除语言标签 <|zh|>
    text = re.sub(r'<\|[a-z]{2}\|>', '', raw_text)
    
    # 2. 处理情感标签
    if remove_emotions:
        # 移除 <|NEUTRAL|>, <|HAPPY|> 等
        text = re.sub(r'<\|[A-Z]+\|>', '', text)
    
    # 3. 规范化空格
    return text.strip()

def extract_emotions(raw_text: str) -> list:
    """
    如果需要 metadata，单独提取情感标签。
    """
    return re.findall(r'<\|([A-Z]+)\|>', raw_text)
```

## 4\. 格式化模块 (formatters.py)

负责将 SenseVoice 的结果字典转换为 OpenAI 格式的字典。

  * **Standard**: `{"text": clean_text}`
  * **Verbose**: 计算 duration，填充 segments。

## 5\. 测试策略