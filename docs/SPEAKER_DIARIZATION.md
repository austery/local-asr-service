# 说话人分离功能说明

## 当前状态

### API 支持
✅ **API 层已完全实现**
- 支持 `response_format=verbose_json` 参数
- 返回包含 `segments` 数组（含时间戳）
- Segment 模型包含 `speaker` 字段

### 模型支持
⚠️ **当前使用的 Qwen3-ASR-1.7B-4bit 不支持说话人分离**
- `speaker` 字段返回 `null`
- 仍然提供准确的转录文本和时间戳
- 性能优秀：17分钟音频仅需2分10秒

## 测试的模型

| 模型 | 说话人分离 | 长音频支持 | 处理速度 | 推荐 |
|-----|----------|-----------|---------|-----|
| **Qwen3-ASR-1.7B-4bit** | ❌ | ✅ (>17min) | ⚡ 快 | ✅ 推荐 |

## API 使用示例

### 基础转录（仅文本）
```bash
curl -X POST http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "response_format=json" \
  -F "language=zh"
```

**返回示例：**
```json
{
  "text": "完整的转录文本...",
  "duration": 45.2,
  "model": "mlx-community/Qwen3-ASR-1.7B-4bit"
}
```

### 详细转录（含时间戳）
```bash
curl -X POST http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "response_format=verbose_json" \
  -F "language=zh"
```

**返回示例：**
```json
{
  "text": "完整的转录文本...",
  "duration": 45.2,
  "language": "zh",
  "model": "mlx-community/Qwen3-ASR-1.7B-4bit",
  "segments": [
    {
      "id": 0,
      "speaker": null,
      "start": 0.0,
      "end": 3.5,
      "text": "第一段话"
    },
    {
      "id": 1,
      "speaker": null,
      "start": 3.5,
      "end": 7.2,
      "text": "第二段话"
    }
  ]
}
```

## 如何启用说话人分离？

### 方案1: 等待模型更新
- 关注 HuggingFace 上新发布的 MLX 格式模型
- 测试是否原生支持说话人分离
- 目前已知支持的模型较少

### 方案2: 集成 pyannote-audio（推荐）
如果确实需要说话人分离功能，可以集成独立的说话人分离库。

#### 实现方案
1. 添加依赖：`pyannote-audio`
2. 在音频预处理阶段进行说话人分离
3. 将说话人信息与转录结果对齐
4. 更新 segments 中的 speaker 字段

#### 预估工作量
- 开发时间：2-3小时
- 增加依赖：pyannote-audio + torch
- 内存增加：~2GB（pyannote 模型）

#### 优势
- 专业的说话人分离效果
- 与转录流程解耦
- 可以选择是否启用

#### 劣势
- 增加处理时间（约30%）
- 需要额外的模型下载
- 增加系统复杂度

## 架构设计

当前架构已经为说话人分离预留了接口：

```
┌─────────────────┐
│   API Layer     │  ← 已支持 verbose_json 格式
│  (routes.py)    │
└────────┬────────┘
         │
┌────────▼────────┐
│  Service Layer  │  ← 透传 response_format 参数
│ (transcription) │
└────────┬────────┘
         │
┌────────▼────────┐
│  MLX Engine     │  ← 支持 format=json 参数
│ (mlx_engine.py) │  ← 合并多个分片的 segments
└────────┬────────┘
         │
┌────────▼────────┐
│  MLX Model      │  ← Qwen3-ASR (无说话人分离)
│                 │  ← 可替换为支持的模型
└─────────────────┘
```

如需集成 pyannote，在 MLX Engine 之前添加一个预处理步骤即可。

## 相关文档
- [模型切换总结](/tmp/model_switch_summary.md)
- [API 文档](../README.md)
