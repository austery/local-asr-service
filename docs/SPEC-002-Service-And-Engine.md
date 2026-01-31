---
specId: SPEC-102
title: 业务调度与推理核心 (Service & Engine)
status: ✅ 已实现
priority: P0
owner: User
relatedSpecs: [SPEC-101, SPEC-103]
---

## 1. 目标
实现 ADR-001 中的 "Service Layer" (调度) 和 "Core Layer" (计算)，确保 M4 Pro 上的显存安全并支持多引擎。

## 2. 架构组件

### 2.1 Service Layer: 队列管理器
**组件**: `src/services/transcription.py -> TranscriptionService` (单例)
**职责**: 系统的交通警察，管理背压 (Backpressure) 和任务串行化。

* **属性**:
    * `queue`: `asyncio.Queue` (Maxsize=50, 保护内存)
    * `engine`: `ASREngine` (通过依赖注入传入的具体引擎实例)
* **方法**:
    * `submit(file: UploadFile, params) -> Future`: 生产者。
        - 将上传的文件流写入磁盘临时文件 (`temp_{uuid}.wav`)。
        - 创建 `TranscriptionJob` 入队。
        - 如果队列满，抛出 `503` 异常。
    * `_consume_loop()`: 消费者协程。**永远在后台运行**。

**消费循环逻辑 (Strict Serial)**:
```python
async def _consume_loop(self):
    while self.is_running:
        job = await self.queue.get()
        try:
            # 关键点：在此处调用 Engine，确保一次只跑一个
            # run_in_threadpool 防止阻塞 EventLoop
            result = await run_in_threadpool(
                self.engine.transcribe_file, 
                job.temp_file_path, 
                ...
            )
            job.future.set_result(result)
        finally:
            # 必须删除临时文件
            if os.path.exists(job.temp_file_path):
                os.remove(job.temp_file_path)
            self.queue.task_done()
````

### 2.2 Core Layer: 推理核心

**组件**: `src/core/*`

#### 2.2.1 抽象接口 (Protocol)
`src/core/base_engine.py -> ASREngine`
定义所有引擎必须实现的方法：
- `load_model()`: 加载模型资源。
- `transcribe_file(file_path, language, ...)`: 执行推理。

#### 2.2.2 具体实现
- **FunASREngine** (`src/core/funasr_engine.py`):
    - 封装阿里 `modelscope` 的 `AutoModel`。
    - 运行于 PyTorch (MPS/CPU)。
    - 适合短语音、高精度中文识别。
- **MlxEngine** (`src/core/mlx_engine.py`):
    - 封装 Apple `mlx-examples/whisper` 或 `mlx-community` 模型。
    - 运行于 MLX 原生框架。
    - 支持长音频自动切片 (Chunking)、Qwen3-ASR 等大模型。

#### 2.2.3 工厂模式
`src/core/factory.py -> EngineFactory`
- 根据 `ENGINE_TYPE` 环境变量决定实例化哪个引擎。

## 3\. 异常处理

  * 如果 `Engine` 抛出 Critical Error (如 OOM)，Service 层捕获异常并记录，确保 Worker 循环不退出（除非是不可恢复的系统错误）。
  * 临时文件在 `finally` 块中强制清理。