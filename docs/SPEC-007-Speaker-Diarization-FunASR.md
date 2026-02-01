---
specId: SPEC-007
title: 说话人分离集成 (Speaker Diarization with FunASR & Cam++)
status: 📝 草案 (Draft)
priority: P1
owner: User
relatedSpecs: [SPEC-002, SPEC-005]
created: 2026-02-01
updated: 2026-02-01
---

## 1. 目标 (Goal)

在本地 ASR 服务的 FunASR 引擎中集成**说话人分离 (Speaker Diarization)** 能力。
由于现有的 `SPEC-005` 提议的 Pyannote 方案在集成复杂度和性能上存在挑战，本项目决定采用 **SenseVoiceSmall + Cam++** 的原生 FunASR 组合，实现高性能、低延迟且高度集成的说话人识别。

## 2. 方案概述 (Overview)

### 2.1 核心模型
- **ASR 模型**: `iic/SenseVoiceSmall` (已在 `SPEC-002` 中实现)
- **VAD 模型**: `fsmn-vad` (用于语音活动检测和长语音切分)
- **声纹模型**: `cam++` (阿里开源的高性能声纹识别模型，用于提取说话人特征)
- **标点模型**: `ct-punc` (用于句式预测)

### 2.2 关键优势
- **物理级分离**: 基于声纹特征而非逻辑推断，准确率高。
- **自动下载**: 通过 `AutoModel` 参数配置，无需手动维护模型权重。
- **MPS 优化**: `cam++` 作为轻量级嵌入模型，在 Mac M4 Pro 上具有极高性能。
- **深度集成**: 无需在 Service 层做复杂的时间戳对齐，FunASR 内部完成 VAD -> Speaker -> ASR 的全流程。

## 3. 实现要求 (Implementation Requirements)

### 3.1 FunASREngine 改造

#### 3.1.1 模型加载 (`load` 方法)
我们需要在 `AutoModel` 初始化时增加 `spk_model="cam++"`，并启用 VAD 切分优化。

```python
self.model = AutoModel(
    model=self.model_id,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000}, # 负责把长语音切成小段
    punc_model="ct-punc",
    spk_model="cam++",          # <--- 核心：声纹识别模型
    device=self.device,
    disable_update=True,
    log_level="ERROR"
)
```

#### 3.1.2 转录与解析 (`transcribe_file` 方法)
启用 `spk_model` 后，`res` 的返回结构会包含 `sentence_info`。我们需要解析它：

1. 执行推理：`res = self.model.generate(...)`
2. 获取 `sentence_info = res[0].get("sentence_info", [])`
3. 遍历 `sentence_info` 提取 `spk` (Speaker ID) 和 `text`。

### 3.2 数据契约 (Data Contract)

#### EMR 推荐结构 (List[Dict])
为了支持 Evidence-based Medical Record，建议返回包含时间戳的 JSON 列表：
```json
[
  {
    "speaker": "Speaker 0",
    "start": 1230,
    "end": 4560,
    "text": "患者自述胸痛。"
  }
]
```
这样 AI 在生成病历时可以进行精确引用（例如：`"患者自述胸痛 (Reference: 00:01, Speaker 0)"`）。

## 4. 性能与资源 (Performance & Resources)

- **显存占用**: `cam++` 模型极小，显存增加可忽略不计。
- **计算延迟**: 对比纯 ASR，增加说话人分离后的延迟预计增长小于 10%。
- **兼容性**: 完美支持 `torch >= 2.1` 和 MPS 后端。

## 5. 验收标准 (Acceptance Criteria)

- [ ] `FunASREngine` 能够成功加载 `cam++` 模型。
- [ ] 转录结果能够区分不同说话人（Speaker 0, Speaker 1...）。
- [ ] 接口响应支持返回带说话人标记的结构化数据（Verbose JSON）。
- [ ] 时间戳精度符合预期（毫秒级）。

## 6. 后续优化方向 (Future Enhancements)

- **Speaker Mapping**: 将 Speaker ID 映射为具体的角色（如“医生”、“患者”），这可以通过后续的 LLM 逻辑处理实现。
- **Evidence-based EMR**: 在生成的病历中自动关联说话人与时间戳引用。

---

**参考资料**:
- FunASR 官方文档
- Cam++ 模型介绍
