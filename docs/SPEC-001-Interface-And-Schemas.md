---
specId: SPEC-101
title: Local ASR API 定义与数据模型 (Interface & Schemas)
status: ✅ 已实现
priority: P0
owner: User
relatedSpecs: [SPEC-102]
---

## 1. 范围 (Scope)
本通过定义系统的“外壳”：URL 路由、Pydantic 数据模型以及完整的 OpenAPI 规范。
**原则**: 这一层不包含任何业务逻辑，只负责将 HTTP 请求转换为 Pydantic 对象，并传递给 Service 层。

## 2. 数据模型 (Type-First Schemas)

严格遵循 ADR-001 的 "Type-First" 原则。

```python
# src/api/routes.py
from pydantic import BaseModel, Field
from typing import Optional, List

class Segment(BaseModel):
    """Segment with timestamp and optional speaker info"""
    id: int
    speaker: Optional[str] = None  # Speaker ID (e.g., "SPEAKER_00")
    start: float = 0.0
    end: float = 0.0
    text: str

class TranscriptionResponse(BaseModel):
    """OpenAI Whisper API compatible response format"""
    text: str
    duration: Optional[float] = None
    language: Optional[str] = None
    model: Optional[str] = None  # 返回实际使用的模型
    raw_text: Optional[str] = Field(None, description="转录前的原始文本（带所有标签）")
    is_cleaned: Optional[bool] = Field(True, description="是否经过清理")
    segments: Optional[List[Segment]] = Field(None, description="详细分段信息（带说话人识别）")
```

## 3\. OpenAPI 规范 (The Contract)

这是前端/客户端开发的唯一事实来源 (Source of Truth)。

```yaml
openapi: 3.0.0
info:
  title: Local ASR API
  version: 1.0.0
paths:
  /v1/audio/transcriptions:
    post:
      summary: 语音转录 (ASR)
      operationId: createTranscription
      requestBody:
        content:
          multipart/form-data:
            schema:
              type: object
              required: [file]  # 唯一必须的只有文件
              properties:
                # --- 真正干活的参数 ---
                file:
                  type: string
                  format: binary
                  description: 音频文件
                language:
                  type: string
                  description: 语言代码 (zh, en, ja, ko, auto)
                response_format:
                  type: string
                  enum: [json, verbose_json] # 简化支持
                  default: json
                  description: json (仅文本) 或 verbose_json (含时间戳/说话人)
                clean_tags:  # 你的自定义参数
                  type: boolean
                  default: true
                  description: 是否清洗 <happy> 等情感标签 (FunASR 专用)
                
                # --- "吉祥物"参数 (为了兼容客户端不报错而存在) ---
                model:
                  type: string
                  default: sense-voice-small
                temperature:
                  type: number
                  description: (Ignored)
                prompt:
                  type: string
                  description: (Ignored)
      responses:
        '200':
          description: 成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TranscriptionResponse'

components:
  schemas:
    TranscriptionResponse:
      type: object
      properties:
        text:
          type: string
        duration:
          type: number
        raw_text:
          type: string
        segments:
          type: array
          items: 
            type: object
            properties:
              id: {type: integer}
              speaker: {type: string}
              start: {type: number}
              end: {type: number}
              text: {type: string}
```



## 4\. 路由层逻辑

  * **Controller**: `src/api/routes.py`
  * **行为**:
    1.  校验 Multipart Form 数据。
    2.  调用 `TranscriptionService.submit(file, params)`，传入 UploadFile 对象。
    3.  Service 负责将文件写入临时目录。
    4.  `await future` 等待结果。
    5.  处理 Service 返回的 dict 或 string 结果，构造 `TranscriptionResponse`。


