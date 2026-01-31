---
specId: SPEC-103
title: 音频处理与文本清洗 (Audio & Text Processing)
status: ✅ 已实现
priority: P1
owner: User
relatedSpecs: [SPEC-102]
---

## 1. 目标
处理输入（音频临时文件管理）和输出（文本清洗），保持无状态 (Stateless)，便于单元测试。

## 2. 音频模块

**策略调整**: 
为了兼容 FunASR 和 ffmpeg 的文件读取需求，以及避免在 Python 内存中持有可能巨大的音频数据，我们采用**写盘策略**。

*   **流程**: API 接收 `UploadFile` -> Service 写入临时文件 -> Engine 读取路径处理 -> Service 删除临时文件。
*   **优势**: 内存占用低，直接利用底层库 (ffmpeg/funasr) 的文件处理能力。

## 3\. 文本清洗模块 (`src/adapters/text.py`)

SenseVoice 的输出通常包含富文本标签，需要清洗。

```python
# src/adapters/text.py
import re

def clean_sensevoice_tags(text: str, clean_tags: bool = True) -> str:
    """
    清洗 SenseVoice 输出的富文本标签。
    
    Args:
        text: 原始文本，例如 "<|zh|><|NEUTRAL|>你好"
        clean_tags: 是否执行清洗
    """
    if not clean_tags:
        return text
        
    # 1. 移除语言标签 <|zh|>
    text = re.sub(r'<\|[a-z]{2}\|>', '', text)
    
    # 2. 移除情感/事件标签 <|NEUTRAL|>, <|Speech|>
    text = re.sub(r'<\|[A-Za-z]+\|>', '', text)
    
    # 3. 规范化空格
    return text.strip()
```

## 4\. 格式化模块

负责将引擎的原始结果转换为 OpenAI 格式。
*   **FunASR**: 返回纯文本，需手动包装。
*   **MLX**: 引擎内部已返回包含 `segments` 的字典结构。

## 5\. 测试策略
参见 SPEC-004。