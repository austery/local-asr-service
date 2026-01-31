---
specId: SPEC-102
title: 业务调度与推理核心 (Service & Engine)
status: 🚧 规划中
priority: P0
owner: User
relatedSpecs: [SPEC-101, SPEC-103]
---

## 1. 目标
实现 ADR-001 中的 "Service Layer" (调度) 和 "Engine Layer" (计算)，确保 M4 Pro 上的显存安全。

## 2. 架构组件

### 2.1 Service Layer: 队列管理器
**组件**: `TranscriptionService` (单例)
**职责**: 也就是 ADR 中的 "Async Queue Manager"。它是系统的交通警察。

* **属性**:
    * `queue`: `asyncio.Queue` (Maxsize=50, 保护内存)
    * `engine`: `SenseVoiceEngine` 实例
* **方法**:
    * `submit(audio_bytes, schema) -> Future`: 生产者。如果队列满，抛出 `503` 异常。
    * `consume_loop()`: 消费者协程。**永远在后台运行**。

**消费循环逻辑 (Strict Serial)**:
```python
async def consume_loop(self):
    while True:
        job = await self.queue.get()
        # 关键点：在此处调用 Engine，确保一次只跑一个
        try:
            # run_in_threadpool 防止阻塞 EventLoop
            result = await run_in_threadpool(self.engine.transcribe, job.audio, job.params)
            job.future.set_result(result)
        except Exception as e:
            job.future.set_exception(e)
        finally:
            self.queue.task_done()
````

### 2.2 Engine Layer: 推理核心

**组件**: `SenseVoiceEngine` (单例)
**职责**: 纯粹的计算单元。不知道 HTTP，不知道 Queue。

  * **生命周期**: 在 FastAPI `lifespan` 中初始化，加载模型到 MPS。
  * **方法**:
      * `load_model()`: 加载 FunASR，预热 (Warmup)。
      * `transcribe(audio_ndarray, params) -> dict`: **同步方法** (因为是 CPU/GPU 密集型)。

## 3\. 异常处理

  * 如果 `Engine` 抛出 CUDA/MPS 错误，`Service` 层应捕获并记录 Critical Log，但不应导致进程崩溃（除非是不可恢复的错误）。