---
specId: SPEC-004
title: 测试策略与质量保证 (Testing Strategy & QA)
status: 🚧 规划中
priority: P1
owner: User
relatedSpecs: [SPEC-101, SPEC-102, SPEC-103]
---

## 1. 目标 (Goal)
建立稳健的自动化测试体系，确保重构和功能迭代不会破坏核心功能。采用 **Pytest** 作为主要测试框架。

## 2. 测试分层 (Testing Pyramid)

### 2.1 单元测试 (Unit Tests)
*   **范围**: 独立的函数和类，不依赖外部系统（如文件系统、网络、GPU）。
*   **工具**: `pytest`, `unittest.mock`
*   **覆盖目标**:
    *   `src/adapters/text.py`: 文本清洗逻辑 (Pure Functions)。
    *   `src/core/engine.py`: 引擎加载与推理逻辑 (Mock 掉 `funasr.AutoModel`)。
    *   `src/services/transcription.py`: 队列调度逻辑 (Mock 掉 `Engine`)。

### 2.2 集成测试 (Integration Tests)
*   **范围**: API 接口层，验证组件间的协作。
*   **工具**: `fastapi.testclient.TestClient` (基于 `httpx`)
*   **覆盖目标**:
    *   `/v1/audio/transcriptions`: 验证参数解析、文件上传、Service 调用链路。
    *   **注意**: 在 CI/CD 环境中，应 Mock 掉 Engine 的实际推理，避免需要 GPU/MPS 环境。

## 3. 测试结构 (Directory Structure)

```text
tests/
├── __init__.py
├── conftest.py          # 全局 Fixtures (如 Mock Engine, TestClient)
├── unit/
│   ├── test_adapters.py # 测试文本清洗
│   ├── test_engine.py   # 测试引擎逻辑 (Mocked)
│   └── test_service.py  # 测试服务调度
└── integration/
    └── test_api.py      # 测试 API 接口
```

## 4. 关键测试用例 (Key Test Cases)

### 4.1 文本清洗 (Text Adapter)
*   Case 1: 输入 `<|zh|><|NEUTRAL|>你好` -> 输出 `你好`
*   Case 2: 输入 `None` -> 输出 `""`
*   Case 3: `clean_tags=False` -> 输出原样

### 4.2 服务调度 (Service Layer)
*   Case 1: **Backpressure**: 当队列满 (size=50) 时，提交任务应抛出 `503 Service Unavailable` (或自定义异常)。
*   Case 2: **Temp File Cleanup**: 任务完成后（无论成功失败），临时文件必须被删除。
*   Case 3: **Async Execution**: 验证 `submit` 是非阻塞的，而 `consume_loop` 是串行的。

### 4.3 引擎层 (Engine Layer)
*   Case 1: **Lazy Loading**: 验证 `load()` 被调用前，`model` 为 None。
*   Case 2: **Parameter Mapping**: 验证 `language="en"` 正确传递给底层模型。

## 5. 运行方式 (Execution)

```bash
# 运行所有测试
pytest

# 运行特定文件
pytest tests/unit/test_adapters.py

# 生成覆盖率报告 (可选)
pytest --cov=src
```
