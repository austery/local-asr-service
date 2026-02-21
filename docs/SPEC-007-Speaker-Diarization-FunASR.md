---
specId: SPEC-007
title: 说话人分离集成 (Speaker Diarization with FunASR & Cam++)
status: ✅ 已实现 (Implemented)
priority: P1
owner: User
relatedSpecs: [SPEC-002, SPEC-005]
created: 2026-02-01
updated: 2026-02-21
---

## 1. 目标 (Goal)

在本地 ASR 服务的 FunASR 引擎中集成**说话人分离 (Speaker Diarization)** 能力。

> [!IMPORTANT]
> **关键发现**: SenseVoice 不支持时间戳预测，因此无法与 Cam++ 配合使用。
> 必须使用 **Paraformer** 模型（阿里中文识别 SOTA）来实现说话人分离。

## 2. 方案概述 (Overview)

### 2.1 核心模型
- **ASR 模型**: `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` ✅
- **VAD 模型**: `fsmn-vad` (用于语音活动检测和长语音切分)
- **声纹模型**: `cam++` (阿里开源的高性能声纹识别模型，用于提取说话人特征)
- **标点模型**: `ct-punc` (用于句式预测)

### 2.2 为什么必须用 Paraformer？

| 模型 | 时间戳支持 | 说话人分离 | 适用场景 |
|-----|-----------|-----------|---------|
| **SenseVoice** | ❌ | ❌ | 多语言快速转录 |
| **Paraformer** | ✅ | ✅ | Podcast/访谈全量抓取 |

Cam++ 需要精准的时间戳来切分音频段并识别说话人。Paraformer 的 CIF 机制原生支持时间戳预测。

### 2.3 关键优势
- **物理级分离**: 基于声纹特征而非逻辑推断，准确率高。
- **自动下载**: 通过 `AutoModel` 参数配置，无需手动维护模型权重。
- **MPS 优化**: 在 Mac M1 Max 上 RTF ~0.016（16秒处理5分钟音频）。
- **多格式输出**: 支持 JSON/TXT/SRT 格式，满足不同场景需求。

## 3. 实现要求 (Implementation Requirements)

### 3.1 FunASREngine 改造

#### 3.1.1 模型加载 (`load` 方法)

```python
DEFAULT_MODEL_ID = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

self.model = AutoModel(
    model=self.model_id,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    punc_model="ct-punc",
    spk_model="cam++",  # 声纹识别模型
    device=self.device,
    disable_update=True,
    log_level="ERROR"
)
```

#### 3.1.2 多格式输出 (`transcribe_file` 方法)

```python
def transcribe_file(
    self, 
    file_path: str, 
    language: str = "auto",
    output_format: str = "json",  # json, txt, srt
    with_timestamp: bool = False,
    **kwargs
) -> Union[Dict, str]:
```

### 3.2 输出格式 (Output Formats)

| 格式 | 用途 | 示例 |
|-----|------|-----|
| `json` | 完整数据存储 | `{"text": "...", "segments": [...]}` |
| `txt` | RAG/LLM 处理 | `[Speaker 0]: 大家好...` |
| `txt` + timestamp | 带时间引用 | `[02:15] [Speaker 0]: 大家好...` |
| `srt` | 字幕文件 | 标准 SRT 格式 |

## 4. 性能与资源 (Performance & Resources)

### 4.1 测试结果 (Mac M1 Max)

| 指标 | 结果 |
|-----|------|
| 测试音频 | 5分钟英语对话 (2位说话人) |
| 推理时间 | 16秒 |
| RTF | ~0.016 |
| 检测片段 | 125 个 |
| 说话人识别 | Speaker 0, Speaker 1 ✅ |

### 4.2 资源占用
- **显存占用**: `cam++` 模型极小，显存增加可忽略不计。
- **兼容性**: 完美支持 `torch >= 2.1` 和 MPS 后端。

## 5. 验收标准 (Acceptance Criteria)

- [x] `FunASREngine` 能够成功加载 `cam++` 模型。
- [x] 转录结果能够区分不同说话人（Speaker 0, Speaker 1...）。
- [x] 接口响应支持返回带说话人标记的结构化数据（Verbose JSON）。
- [x] 时间戳精度符合预期（毫秒级）。
- [x] 支持多格式输出（json/txt/srt）。
- [x] 支持纯净文本模式（无时间戳，适合 RAG/LLM）。

## 6. 后续优化方向 (Future Enhancements)

- **Speaker Mapping**: 将 Speaker ID 映射为具体的角色（如"医生"、"患者"）
- **Evidence-based EMR**: 在生成的病历中自动关联说话人与时间戳引用
- **API 层集成**: 通过 `response_format` 参数暴露多格式输出能力

---

## 7. 已知 Bug 修复 (Bug Fixes)

### 7.1 FunASR distribute_spk NoneType Bug (2026-02-21)

**错误现象**:
```
TypeError: '>' not supported between instances of 'float' and 'NoneType'
  File "funasr/models/campplus/utils.py", line 203, in distribute_spk
    overlap = max(min(sentence_end, spk_ed) - max(sentence_start, spk_st), 0)
```

**根因**: FunASR `campplus/utils.py` 的 `distribute_spk` 函数在处理某些短片段或无法识别说话人的音频段时，`sv_output` 中存在 `spk_st=None` 或 `spk_ed=None` 的条目，导致 float 与 NoneType 之间的大小比较失败。这是第三方库的 bug。

**修复方案**: 在 `src/core/funasr_engine.py` 模块导入时注入 monkey-patch，过滤掉无效的 None 条目：

```python
def _patched_distribute_spk(sentence_list, sd_time_list):
    # 过滤 spk_st/spk_ed 为 None 的无效条目
    valid = [(st, ed, spk) for st, ed, spk in sd_time_list if st is not None and ed is not None]
    sd_time_ms = [(st * 1000, ed * 1000, spk) for st, ed, spk in valid]
    for d in sentence_list:
        ...
    return sentence_list

import funasr.models.campplus.utils as _campplus_utils
_campplus_utils.distribute_spk = _patched_distribute_spk
```

**行为影响**: 被过滤的条目对应的句子 `spk` 会 fallback 到 `0`（默认说话人），不会崩溃。其余句子说话人识别结果不受影响。

---

**参考资料**:
- [FunASR 官方文档](https://github.com/modelscope/FunASR)
- [Paraformer 论文](https://arxiv.org/abs/2206.08317)
- [Cam++ 模型](https://modelscope.cn/models/iic/speech_campplus_speaker-diarization_common)
