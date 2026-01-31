---
specId: SPEC-101
title: API 接口定义与数据模型 (Interface & Schemas)
status: 🚧 规划中
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
# src/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Union

class TranscriptionRequest(BaseModel):
    # 注意：File 不在 Pydantic 中直接定义，而在 FastAPI controller 参数中
    language: str = Field(default="auto", description="ISO-639-1 语言代码")
    clean_tags: bool = Field(default=True, description="是否清洗情感标签")
    response_format: str = Field(default="json", pattern="^(json|text|verbose_json)$")

class Segment(BaseModel):
    start: float
    end: float
    text: str

class TranscriptionResponse(BaseModel):
    text: str
    task: str = "transcribe"
    duration: Optional[float] = None
    segments: Optional[List[Segment]] = None
```

## 3\. OpenAPI 规范 (The Contract)

这是前端/客户端开发的唯一事实来源 (Source of Truth)。

```yaml
openapi: 3.0.0
info:
  title: Local SenseVoice API (Lean Version)
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
                  enum: [json, text, verbose_json, srt] # 加了 srt 方便字幕
                  default: json
                clean_tags:  # 你的自定义参数
                  type: boolean
                  default: true
                  description: 是否清洗 <happy> 等情感标签
                
                # --- "吉祥物"参数 (为了兼容客户端不报错而存在) ---
                model:
                  type: string
                  default: sense-voice-small
                temperature:
                  type: number
                  description: (Ignored in SenseVoice)
                prompt:
                  type: string
                  description: (Limited support)
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
      # 保持这个结构，大多数客户端只看 text
      type: object
      properties:
        text:
          type: string
        duration:
          type: number
        segments:
          type: array
          items: 
            type: object
            properties:
              start: {type: number}
              end: {type: number}
              text: {type: string}
```



## 4\. 路由层逻辑

  * **Controller**: `src/api/routes.py`
  * **行为**:
    1.  校验 Multipart Form 数据。
    2.  构造 `TranscriptionRequest` 对象。
    3.  **立即**调用 `TranscriptionService.submit()` 获取 Future。
    4.  `await future` 等待结果。
    5.  返回 `TranscriptionResponse`。


