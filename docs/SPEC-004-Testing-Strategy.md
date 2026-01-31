---
specId: SPEC-004
title: 测试策略与质量保证 (Testing Strategy & QA)
status: ✅ 已实现
priority: P1
owner: User
relatedSpecs: [SPEC-101, SPEC-102, SPEC-103]
---

## 1. 目标 (Goal)
建立稳健的自动化测试体系，确保重构和功能迭代不会破坏核心功能。采用 **Pytest** 作为主要测试框架。

## 2. 测试分层 (Testing Pyramid)

### 2.1 单元测试 (Unit Tests)
*   **范围**: 独立的函数和类，不依赖外部系统。
*   **覆盖目标**:
    *   `tests/unit/test_adapters.py`: 文本清洗逻辑 (Pure Functions)。
    *   `tests/unit/test_engine.py`: FunASR 引擎逻辑 (Mocked)。
    *   `tests/unit/test_mlx_engine.py`: MLX 引擎逻辑 (Mocked)。
    *   `tests/unit/test_config_factory.py`: 配置加载与工厂模式。
    *   `tests/unit/test_service.py`: 队列调度与文件清理。

### 2.2 集成测试 (Integration Tests)
*   **范围**: API 接口层，验证组件间的协作。
*   **工具**: `fastapi.testclient.TestClient`
*   **覆盖目标**:
    *   `tests/integration/test_api.py`: 验证参数解析、Service 调用链路、Mock Engine 的返回值处理。

### 2.3 E2E 测试 (End-to-End)
*   **范围**: 真实模型加载与推理。
*   **覆盖目标**:
    *   `tests/e2e/test_full_flow.py`: 加载真实模型（FunASR 或 MLX），处理真实音频，验证最终结果。
    *   *注意*: 运行速度较慢，通常仅在本地或 Nightly Build 中运行。

### 2.4 可靠性测试 (Reliability)
*   **覆盖目标**:
    *   `tests/reliability/test_concurrency.py`: 模拟高并发请求，验证队列背压 (Backpressure) 和系统稳定性。

## 3. 关键测试用例 (Key Test Cases)

### 3.1 文本清洗 (Text Adapter)
*   Case: 输入 `<|zh|><|NEUTRAL|>你好` -> 输出 `你好`

### 3.2 服务调度 (Service Layer)
*   Case: **Backpressure**: 当队列满时，提交任务应抛出 `503 Service Unavailable`。
*   Case: **Temp File Cleanup**: 任务完成后（无论成功失败），临时文件必须被删除。

### 3.3 引擎层 (Engine Layer)
*   Case: **Engine Factory**: 验证 `ENGINE_TYPE` 环境变量能正确创建对应引擎实例。
*   Case: **MLX Fallback**: 验证 MLX 引擎在处理长音频时的切片逻辑。

## 4. 运行方式 (Execution)

```bash
# 运行所有测试
uv run python -m pytest

# 运行特定类别
uv run python -m pytest tests/unit
```
