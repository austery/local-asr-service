# **🎙️ Local SenseVoice API (Mac Silicon Optimized)**

一个专为 Apple Silicon (M-series) 芯片优化的高性能、本地化语音转录服务。  
支持**双引擎架构**：
- **FunASR 引擎**：阿里 FunASR (SenseVoiceSmall) 模型，基于 PyTorch MPS
- **MLX Audio 引擎**：Apple MLX 原生模型 (VibeVoice, Whisper, Qwen3-ASR 等)

提供兼容 OpenAI Whisper 格式的 HTTP 接口。

## **📖 项目简介**

本项目旨在解决在 Mac (M4 Pro/Max) 上运行语音识别时的痛点：**并发导致的显存爆炸 (OOM)** 和 **非标准化的脚本代码**。

我们采用 **Clean Architecture (整洁架构)**，将 API 接口、调度队列和推理引擎严格分离。

### **核心特性**

* **🚀 极速推理**: 支持 Torch MPS 和 Apple MLX 双加速后端。  
* **🔄 双引擎架构**: 通过环境变量在 FunASR 和 MLX Audio 引擎间无缝切换。
* **🛡️ 显存保护**: 内置 asyncio.Queue 生产者-消费者模型，严格串行处理任务，防止并发请求撑爆统一内存。  
* **🔌 OpenAI 兼容**: 提供与 POST /v1/audio/transcriptions 完全一致的接口，可直接对接现有的 Whisper 客户端。  
* **🧹 智能清洗**: 自动清洗 SenseVoice 输出的富文本标签（如 \<|zh|\>、\<|NEUTRAL|\>），只返回纯净文本。

## **🏗️ 系统架构 (The Architecture)**

本项目遵循分层设计原则，从外向内依次为：

1. **API Layer (外观层)**: 处理 HTTP 请求，定义 Pydantic 数据契约。  
2. **Service Layer (调度层)**: 管理异步队列，协调任务调度。  
3. **Engine Layer (核心层)**: 封装 FunASR 模型，管理 MPS 资源。  
4. **Adapters (适配层)**: 纯函数工具箱（文本清洗、音频处理）。

### **⚡️ 执行流程 (Execution Flow)**

当一个请求到达时，系统内部的流转如下：

graph TD  
    A\[Client\] \--\>|POST /transcriptions| B(API Layer / Routes)  
    B \--\>|1. 校验参数 & 写入临时文件| C{Service Queue}  
    C \--\>|2. 入队 (非阻塞)| D\[Asyncio Queue (Max 50)\]  
    B \-.-\>|3. 等待 Future 结果| A  
      
    subgraph "Background Worker (Serial)"  
    D \--\>|4. 消费者取出任务| E\[Engine Layer\]  
    E \--\>|5. MPS 推理 (SenseVoice)| F\[FunASR Model\]  
    F \--\>|6. 返回 Raw Text| E  
    E \--\>|7. 文本清洗 (Adapters)| G\[Result\]  
    end  
      
    G \--\>|8. 唤醒 Future| B  
    B \--\>|9. 返回 JSON| A

## **🛠️ 环境准备 (Installation)**

### **1\. 系统要求**

* **OS**: macOS 12.3+ (推荐 macOS 15+ 以获得最佳 MPS 性能)  
* **Python**: 3.11 (本项目严格测试于 3.11 环境)  
* **System Packages**: 需要 ffmpeg 处理音频。

brew install ffmpeg

### **2\. 安装依赖**

#### **⚡️ 方案 A: 使用 uv (推荐，极速)**

如果你安装了 [uv](https://github.com/astral-sh/uv)，这是最快的方式：

```bash
# 1. 克隆或进入项目目录
cd local-asr-service

# 2. 同步依赖（首次需要 --prerelease=allow）
uv sync --prerelease=allow

# 3. 激活环境
source .venv/bin/activate
```

#### **🐢 方案 B: 使用 Conda (传统)**

```bash
conda create -n sensevoice python=3.11  
conda activate sensevoice  
pip install -e .
```

### **3\. 配置环境变量（可选）**

复制示例配置文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件设置你的配置：

```bash
# 引擎类型
ENGINE_TYPE=funasr  # 或 mlx

# 模型 ID（可选，覆盖默认值）
# MODEL_ID=mlx-community/whisper-large-v3-turbo

# 服务配置
PORT=50070
MAX_QUEUE_SIZE=50
LOG_LEVEL=INFO
```

**注意**：如果不创建 `.env` 文件，服务会使用内置默认值。

## **🚀 启动服务**

### **方式 A: FunASR 引擎 (默认)**

使用阿里 SenseVoice 模型：

```bash
# 使用 uv 运行
uv run python -m src.main

# 或指定模型
FUNASR_MODEL_ID=iic/SenseVoiceSmall uv run python -m src.main
```

### **方式 B: MLX Audio 引擎 (推荐 M4 Pro/Max)**

使用 Apple MLX 原生模型（VibeVoice、Whisper 等）：

```bash
# 默认使用 VibeVoice-ASR-4bit
ENGINE_TYPE=mlx uv run python -m src.main

# 使用 Whisper Large V3 Turbo
ENGINE_TYPE=mlx MODEL_ID=mlx-community/whisper-large-v3-turbo uv run python -m src.main

# 使用 Qwen3-ASR
ENGINE_TYPE=mlx MODEL_ID=mlx-community/Qwen3-ASR-1.7B-8bit uv run python -m src.main
```

### **环境变量配置**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENGINE_TYPE` | `funasr` | 引擎类型: `funasr` 或 `mlx` |
| `MODEL_ID` | (引擎默认) | 覆盖任意引擎的模型 ID |
| `FUNASR_MODEL_ID` | `iic/SenseVoiceSmall` | FunASR 引擎默认模型 |
| `MLX_MODEL_ID` | `mlx-community/VibeVoice-ASR-4bit` | MLX 引擎默认模型 |
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `50070` | 服务监听端口 |
| `MAX_QUEUE_SIZE` | `50` | 最大队列深度 |

### **支持的 MLX 模型**

使用 `ENGINE_TYPE=mlx` 时，可通过 `MODEL_ID` 切换：
- `mlx-community/VibeVoice-ASR-4bit` - 微软 VibeVoice（默认，9B，支持60分钟长音频）
- `mlx-community/VibeVoice-ASR-bf16` - VibeVoice 全精度版
- `mlx-community/whisper-large-v3-turbo-asr-fp16` - OpenAI Whisper Turbo
- `mlx-community/Qwen3-ASR-1.7B-8bit` - 阿里 Qwen3-ASR
- `mlx-community/parakeet-tdt-0.6b-v2` - NVIDIA Parakeet (仅英文)

### **方式 C: Uvicorn 命令行**

如果你需要自定义 worker 数量（**警告：强烈建议保持 workers=1 以避免显存翻倍**）：

\# 在项目根目录下运行  
uvicorn src.main:app \--host 0.0.0.0 \--port 50070 \--workers 1

*首次启动时，模型会自动检查并下载（FunASR 约 500MB+，MLX 模型大小不等），请耐心等待。*

## **🧪 测试接口**

服务启动后，你可以通过 curl 或任何 API 工具进行测试。

### **1\. 健康检查**

curl http://localhost:50070/health  
# 返回: {"status": "healthy", "engine_type": "funasr", "model": "iic/SenseVoiceSmall"}

### **2\. 语音转录 (OpenAI 格式)**

#### **基本调用**

curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@/path/to/your/audio.mp3" \
  -F "language=auto" \
  -F "clean_tags=true"

**预期输出:**

{  
  "text": "你好，这是一个测试音频。",  
  "task": "transcribe",  
  "language": "zh",  
  "duration": 5.2,  
  "raw_text": "<|zh|><|NEUTRAL|>你好，这是一个测试音频。",
  "is_cleaned": true,
  "segments": null
}

#### **参数说明**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file` | File | **必填** | 音频文件 (支持 wav, mp3, m4a 等) |
| `language` | String | `auto` | 语言代码: `zh`, `en`, `ja`, `ko`, `yue`, `auto` |
| `clean_tags` | Boolean | `true` | **是否清理 SenseVoice 标签** |
| `response_format` | String | `json` | 返回格式 (当前仅支持 json) |

#### **clean_tags 参数详解**

SenseVoice 模型原始输出包含丰富的元信息标签，例如：
- **语言标签**: `<|zh|>`, `<|en|>`
- **情感标签**: `<|NEUTRAL|>`, `<|HAPPY|>`, `<|ANGRY|>`
- **事件标签**: `<|Speech|>`, `<|Applause|>`

**模式 1: clean_tags=true (默认，推荐用于生产)**

curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "clean_tags=true"

返回纯净文本，适合直接展示给用户：
```json
{
  "text": "大家好，欢迎收看本期视频。",
  "raw_text": "<|zh|><|NEUTRAL|><|Speech|>大家好，欢迎收看本期视频。",
  "is_cleaned": true
}
```

**模式 2: clean_tags=false (保留原始标签，用于分析)**

curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "clean_tags=false"

返回包含所有标签的原始输出，适合：
- 情感分析
- 语言检测验证
- 调试模型输出

```json
{
  "text": "<|zh|><|NEUTRAL|><|Speech|>大家好，欢迎收看本期视频。",
  "raw_text": "<|zh|><|NEUTRAL|><|Speech|>大家好，欢迎收看本期视频。",
  "is_cleaned": false
}
```

> **💡 提示**: 无论 `clean_tags` 设置为何值，响应中始终包含 `raw_text` 字段，保存完整的模型原始输出。

### **3\. 查看自动文档 (Swagger UI)**

浏览器访问：[http://localhost:50070/docs](https://www.google.com/search?q=http://localhost:50070/docs)

### **4\. 模型存储位置**

服务会自动下载并缓存模型到以下位置：

#### **FunASR 引擎模型**
```bash
路径: ~/.cache/modelscope/hub/models/iic/

已下载的模型示例：
├─ SenseVoiceSmall (893 MB) - 主模型
├─ punc_ct-transformer (1.1 GB) - 标点符号
└─ speech_fsmn_vad (3.9 MB) - 语音活动检测

查看命令：
ls -lh ~/.cache/modelscope/hub/models/iic/
du -sh ~/.cache/modelscope/hub/models/iic/*
```

#### **MLX Audio 引擎模型**
```bash
路径: ~/.cache/huggingface/hub/

模型大小参考：
- VibeVoice-ASR-4bit: ~4-5 GB
- Whisper-large-v3-turbo: ~1.5 GB
- Qwen3-ASR-1.7B-8bit: ~1-2 GB

查看命令：
ls -lh ~/.cache/huggingface/hub/
du -sh ~/.cache/huggingface/hub/models--mlx-community*
```

#### **清理缓存**
```bash
# 删除 FunASR 模型
rm -rf ~/.cache/modelscope/hub/models/iic/SenseVoiceSmall

# 删除所有 MLX Community 模型
rm -rf ~/.cache/huggingface/hub/models--mlx-community*

# 查看总缓存大小
du -sh ~/.cache/modelscope ~/.cache/huggingface
```

## **📂 项目结构**

.  
├── src  
│   ├── adapters          \# 纯函数工具 (Clean Code)  
│   │   └── text.py       \# 正则清洗逻辑  
│   ├── api               \# 接口层  
│   │   └── routes.py     \# 路由与 Pydantic 定义  
│   ├── core              \# 核心业务  
│   │   ├── base\_engine.py   \# 引擎抽象接口 (Protocol)  
│   │   ├── funasr\_engine.py \# FunASR/SenseVoice 实现  
│   │   ├── mlx\_engine.py    \# MLX Audio 实现  
│   │   └── factory.py       \# 引擎工厂  
│   ├── services          \# 服务调度  
│   │   └── transcription.py \# 队列与并发控制  
│   ├── config.py         \# 环境变量配置  
│   └── main.py           \# 程序入口与生命周期  
├── pyproject.toml        \# 依赖配置  
└── README.md             \# 本文档

## **🧪 运行测试 (Testing)**

本项目包含完整的单元测试和集成测试，使用 `pytest` 框架。

### **1. 运行所有测试**

```bash
uv run python -m pytest
```

### **2. 测试分层说明**

*   **Unit Tests (`tests/unit`)**:
    *   `test_adapters.py`: 测试文本清洗逻辑（纯函数）。
    *   `test_engine.py`: 测试 FunASR 引擎加载与推理（Mock 掉底层模型）。
    *   `test_mlx_engine.py`: 测试 MLX Audio 引擎（Mock 掉 mlx\_audio）。
    *   `test_config_factory.py`: 测试配置和引擎工厂。
    *   `test_service.py`: 测试异步队列调度和临时文件生命周期。
*   **Integration Tests (`tests/integration`)**:
    *   `test_api.py`: 启动 FastAPI TestClient，验证 HTTP 接口契约（Mock 掉 Engine）。
*   **E2E Tests (`tests/e2e`)**:
    *   `test_full_flow.py`: **真实模型测试**。会加载真实模型并推理（需下载模型，速度较慢）。
*   **Reliability Tests (`tests/reliability`)**:
    *   `test_concurrency.py`: 测试高并发下的队列背压 (Backpressure) 和 Worker 错误恢复能力。



## **⚠️ 注意事项**

1. **队列限制**: 默认队列深度为 50。如果请求超过 50 个，API 会立即返回 503 Service Busy。  
2. **单例模式**: 由于 M 芯片统一内存特性，我们严格限制模型只加载一次。请勿开启多进程 (workers \> 1\) 模式运行，否则会导致显存成倍消耗。  
3. **临时文件**: 上传的音频会暂存到磁盘以便 ffmpeg 处理，处理完成后会自动删除。