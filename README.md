# **🎙️ Local ASR Service (Mac Silicon Optimized)**

一个专为 Apple Silicon (M-series) 芯片优化的高性能、本地化语音转录服务。
支持**双引擎架构**：
- **FunASR 引擎**：支持 **Paraformer** (SOTA 中文识别) + **Cam++** 声纹模型，实现 **说话人分离 (Diarization)**。也可切换至 **SenseVoice** 模型用于极速纯转录。
- **MLX Audio 引擎**：Apple MLX 原生模型 (Qwen3-ASR, Whisper 等)

提供兼容 OpenAI Whisper 格式的 HTTP 接口，官方端口 **50070**。

## **📖 项目简介**

本项目旨在解决在 Mac (M1/M2/M3/M4 Max/Pro) 上运行语音识别时的痛点：**并发导致的显存爆炸 (OOM)** 和 **非标准化的脚本代码**。

我们采用 **Clean Architecture (整洁架构)**，将 API 接口、调度队列和推理引擎严格分离。

### **核心特性**

* **🚀 极速推理**: 支持 Torch MPS 和 Apple MLX 双加速后端。
* **🔄 双引擎架构**: 通过环境变量在 FunASR 和 MLX Audio 引擎间无缝切换。
* **✂️ 智能音频切片**: MLX 引擎支持超长音频自动切片（静音检测 + 重叠策略），无需手动预处理。
* **🛡️ 显存保护**: 内置 asyncio.Queue 生产者-消费者模型，严格串行处理任务，防止并发请求撑爆统一内存。
* **👥 说话人分离 (Diarization)**: 集成 Cam++ 模型，自动识别不同说话人（Speaker 0, Speaker 1...）。
* **🔌 OpenAI 兼容**: 提供与 POST /v1/audio/transcriptions 完全一致的接口，支持 `response_format` 参数（`verbose_json`/`text`/`srt`）。
* **🎯 能力声明与校验**: 引擎能力自动检测，API 层校验不兼容请求并返回清晰 400 错误（如 SenseVoice + SRT）。
* **📊 性能基准测试**: 内置 benchmark 脚本，自动测量 RTF、延迟、throughput。

## **🏗️ 系统架构 (The Architecture)**

本项目遵循分层设计原则，从外向内依次为：

1. **API Layer (外观层)**: 处理 HTTP 请求，定义 Pydantic 数据契约。  
2. **Service Layer (调度层)**: 管理异步队列，协调任务调度。  
3. **Engine Layer (核心层)**: 封装 FunASR 模型，管理 MPS 资源。  
4. **Adapters (适配层)**: 纯函数工具箱（文本清洗、音频处理）。

### **⚡️ 执行流程 (Execution Flow)**

当一个请求到达时，系统内部的流转如下：

graph TD
    A[Client] -->|POST /transcriptions| B(API Layer / Routes)
    B -->|1. 校验参数 & 写入临时文件| C{Service Queue}
    C -->|2. 入队 (非阻塞)| D[Asyncio Queue (Max 50)]
    B -.->|3. 等待 Future 结果| A

    subgraph "Background Worker (Serial)"
    D -->|4. 消费者取出任务| E[Engine Layer]
    E -->|5. MPS 推理 (Paraformer/Qwen3-ASR)| F[FunASR / MLX Model]
    F -->|6. 返回 Raw Text| E
    E -->|7. 格式化输出 (Adapters)| G[Result]
    end

    G -->|8. 唤醒 Future| B
    B -->|9. 返回 JSON/Text/SRT| A

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
conda create -n local-asr python=3.11
conda activate local-asr
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

# 模型 ID（可选，覆盖默认值。若需说话人分离，建议用 Paraformer）
# FUNASR_MODEL_ID=iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

# 服务配置
PORT=50070
MAX_QUEUE_SIZE=50
LOG_LEVEL=INFO
```

**注意**：如果不创建 `.env` 文件，服务会使用内置默认值。

## **📊 FunASR 模型对比: Paraformer vs SenseVoice**

本项目的 FunASR 引擎支持两种主要模型，各有优势：

| 维度 | **SEACO-Paraformer** (默认) | **SenseVoiceSmall** |
|------|--------------------------|---------------------|
| 模型 ID | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | `iic/SenseVoiceSmall` |
| 架构 | 非自回归 encoder-decoder + CIF 预测器 | 非自回归 **encoder-only** (无 decoder) |
| 速度 (RTF) | **~0.19-0.21** (M1 Max 实测) | **~0.007** (极快，约快 30 倍) |
| 纯中文准确率 | **更好** (CER 1.95%) | 好 (CER 2.96%) |
| 中英混合 | MER 9.65% | **更好** (MER 6.71%) |
| 时间戳 | **支持** | 不支持 |
| 说话人分离 | **支持** (配合 CAM++) | 不支持 |
| 情绪识别 | 不支持 | **支持** (`<\|HAPPY\|>`, `<\|NEUTRAL\|>` 等) |
| 音频事件检测 | 不支持 | **支持** (`<\|BGM\|>`, `<\|Laughter\|>` 等) |
| 热词定制 | **支持** (SeACo bias) | 不支持 |

**实测性能 (Paraformer, M1 Max, FunASR pipeline: VAD + ASR + Punc + Speaker Diarization):**

| 音频时长 | 推理时间 | RTF |
|----------|----------|-----|
| 9s | 2.25s | 0.215 |
| 64s | 13.33s | 0.193 |
| 19s | 4.13s | 0.200 |

**如何选择：**
- **需要说话人分离**（播客、会议、多人对话）→ 用 **Paraformer** (默认)
- **需要极速纯转录**（语音输入、短音频、低延迟）→ 用 **SenseVoice**
- **需要情绪/事件标签**（情感分析、音频标注）→ 用 **SenseVoice**

> **注意**: SenseVoice 输出包含特殊标签 (如 `<|zh|><|NEUTRAL|><|Speech|>`)，本项目内置 `clean_sensevoice_tags()` 自动清洗。

## **🚀 启动服务**

### **方式 A: FunASR 引擎 (默认)**

```bash
# 使用 Paraformer（默认，支持说话人分离）
uv run python -m src.main

# 切换到 SenseVoice（极速纯转录，无说话人分离）
FUNASR_MODEL_ID=iic/SenseVoiceSmall uv run python -m src.main
```

### **方式 B: MLX Audio 引擎**

使用 Apple MLX 原生模型（Qwen3-ASR、Whisper 等）：

```bash
# 默认使用 Qwen3-ASR-1.7B-4bit（推荐）
ENGINE_TYPE=mlx uv run python -m src.main

# 使用 Whisper Large V3 Turbo
ENGINE_TYPE=mlx MODEL_ID=mlx-community/whisper-large-v3-turbo uv run python -m src.main

# 使用 Qwen3-ASR 8-bit
ENGINE_TYPE=mlx MODEL_ID=mlx-community/Qwen3-ASR-1.7B-8bit uv run python -m src.main
```

### **切换模型**

模型切换通过**环境变量**实现，需要**重启服务**：

```bash
# 方法 1: 命令行直接指定（临时）
FUNASR_MODEL_ID=iic/SenseVoiceSmall uv run python -m src.main

# 方法 2: 修改 .env 文件（持久）
# 编辑 .env，修改 FUNASR_MODEL_ID 或 ENGINE_TYPE，然后重启服务

# 方法 3: 切换整个引擎
ENGINE_TYPE=mlx uv run python -m src.main
```

> **提示**: 当前不支持运行时热切换模型，需停止服务后重启。模型首次使用会自动下载。

### **环境变量配置**

**引擎与模型：**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENGINE_TYPE` | `funasr` | 引擎类型: `funasr` 或 `mlx` |
| `MODEL_ID` | (引擎默认) | 覆盖任意引擎的模型 ID（优先级最高） |
| `FUNASR_MODEL_ID` | `iic/speech_seaco_paraformer...` | FunASR 默认模型 (Paraformer, 支持说话人分离) |
| `MLX_MODEL_ID` | `mlx-community/Qwen3-ASR-1.7B-4bit` | MLX 引擎默认模型 |

**服务与安全：**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `50070` | 服务监听端口 |
| `MAX_QUEUE_SIZE` | `50` | 最大并发队列深度 |
| `MAX_UPLOAD_SIZE_MB` | `200` | 上传文件大小限制（MB） |
| `ALLOWED_ORIGINS` | `http://localhost,http://127.0.0.1` | CORS 允许的源 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

**音频处理配置（仅 MLX 引擎）：**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_AUDIO_DURATION_MINUTES` | `50` | 最大音频时长（超过自动切片） |
| `SILENCE_THRESHOLD_SEC` | `0.5` | 静音检测最小时长（秒） |
| `SILENCE_NOISE_DB` | `-30dB` | 静音噪音阈值 |
| `AUDIO_SAMPLE_RATE` | `16000` | 音频采样率（Hz） |
| `AUDIO_BITRATE` | `64k` | 音频比特率 |
| `CHUNK_OVERLAP_SECONDS` | `15` | 切片重叠时长（秒，fallback 策略） |

### **支持的 MLX 模型**

使用 `ENGINE_TYPE=mlx` 时，可通过 `MODEL_ID` 切换：
- `mlx-community/whisper-large-v3-turbo-asr-fp16` - OpenAI Whisper Turbo
- `mlx-community/Qwen3-ASR-1.7B-8bit` - 阿里 Qwen3-ASR
- `mlx-community/parakeet-tdt-0.6b-v2` - NVIDIA Parakeet (仅英文)

**⏱️ 长音频处理：** MLX 引擎支持自动音频切片，默认限制 50 分钟。超过限制时：
1. **优先策略**：在静音点智能切片（避免断词）
2. **Fallback 策略**：固定时长 + 15秒重叠切片
3. 切片后自动合并转录结果

### **方式 C: Uvicorn 命令行**

如果你需要自定义 worker 数量（**警告：强烈建议保持 workers=1 以避免显存翻倍**）：

\# 在项目根目录下运行  
uvicorn src.main:app \--host 0.0.0.0 \--port 50070 \--workers 1

*首次启动时，模型会自动检查并下载（FunASR 约 500MB+，MLX 模型大小不等），请耐心等待。*

## **🧪 测试接口**

服务启动后，你可以通过 curl 或任何 API 工具进行测试。

### **1. 健康检查**

```bash
curl http://localhost:50070/health
# 返回: {"status": "healthy", "engine_type": "funasr", "model": "iic/speech_seaco_paraformer..."}
```

### **2. 查询当前模型和能力**

```bash
curl http://localhost:50070/v1/models/current | jq
```

**返回示例：**
```json
{
  "engine_type": "funasr",
  "model_id": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
  "capabilities": {
    "timestamp": true,
    "diarization": true,
    "emotion_tags": false,
    "language_detect": true
  },
  "queue_size": 0,
  "max_queue_size": 50
}
```

### **3. 转录接口**

#### **3.1 JSON 格式 (默认，OpenAI 兼容)**

```bash
curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg"
```

**返回 (JSON，OpenAI verbose_json 兼容):**
```json
{
  "text": "[Speaker 0]: 大家好...\n[Speaker 1]: 好的...",
  "duration": 5.2,
  "language": "zh",
  "model": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
  "segments": [
    {"id": 0, "speaker": "Speaker 0", "start": 50, "end": 1200, "text": "大家好..."},
    {"id": 1, "speaker": "Speaker 1", "start": 1200, "end": 2500, "text": "好的..."}
  ]
}
```

#### **3.2 纯文本格式 (适合 RAG/LLM)**

```bash
curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg" \
  -F "output_format=txt"
```

**返回 (Plain Text):**
```text
[Speaker 0]: 大家好，今天我们来聊聊...
[Speaker 1]: 好的，那我们开始吧。
```

#### **3.3 带时间戳的文本**

```bash
curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg" \
  -F "output_format=txt" \
  -F "with_timestamp=true"
```

**返回:**
```text
[00:00] [Speaker 0]: 大家好，今天我们来聊聊...
[00:05] [Speaker 1]: 好的，那我们开始吧。
```

#### **3.4 SRT 字幕格式**

```bash
curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg" \
  -F "output_format=srt"
```

**返回 (SRT 字幕):**
```srt
1
00:00:00,050 --> 00:00:01,200
[Speaker 0]: 大家好，今天我们来聊聊...

2
00:00:01,200 --> 00:00:02,500
[Speaker 1]: 好的，那我们开始吧。
```

#### **3.5 OpenAI 兼容参数**

使用 `response_format` 参数（OpenAI API 标准）：

```bash
# verbose_json → json
curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg" \
  -F "response_format=verbose_json"

# text → txt
curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg" \
  -F "response_format=text"

# vtt → srt
curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg" \
  -F "response_format=vtt"
```

#### **参数说明**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file` | File | **必填** | 音频文件 (wav, mp3, m4a, flac, ogg 等) |
| `output_format` | String | `json` | 输出格式: `json`, `txt`, `srt` |
| `response_format` | String | `None` | OpenAI 兼容别名: `verbose_json`, `text`, `srt`, `vtt` |
| `with_timestamp` | Boolean | `false` | txt 格式下是否包含行首时间戳 `[MM:SS]` |
| `language` | String | `auto` | 语言代码: `zh`, `en`, `auto` |
| `model` | String | (信息性) | 模型 ID（服务端配置优先，此参数仅用于 API 兼容） |

> **💡 提示**:
> 1. 默认输出格式 (`json`) 返回 OpenAI `verbose_json` 兼容的 JSON 响应（含 segments）。
> 2. `response_format` 优先级高于 `output_format`（前者覆盖后者）。
> 3. `output_format=srt` 和 `with_timestamp=true` 仅在使用 Paraformer 等支持 timestamp 的模型时有效。
> 4. 请求不兼容的格式（如 SenseVoice + SRT）会返回 400 错误并说明原因。

### **4. 查看自动文档 (Swagger UI)**

浏览器访问：http://localhost:50070/docs

### **5. 性能基准测试**

使用内置 benchmark 脚本测量转录性能：

```bash
# 使用默认 fixture (tests/fixtures/two_speakers_60s.wav)
uv run python benchmarks/run.py

# 测试指定文件
uv run python benchmarks/run.py --file path/to/audio.wav

# 测试所有样本并保存结果
uv run python benchmarks/run.py --all --save

# 测试不同输出格式
uv run python benchmarks/run.py --format txt
```

**基准结果 (M1 Max, Paraformer, 60s English audio):**
- Elapsed: 7.85s, RTF: 0.13, 7.6x realtime
- 21 segments detected with speaker diarization

### **6. 模型存储位置**

服务会自动下载并缓存模型到以下位置：

#### **FunASR 引擎模型**
```bash
路径: ~/.cache/modelscope/hub/models/iic/

Paraformer 完整管道所需模型：
├─ speech_seaco_paraformer_large (ASR 主模型，支持时间戳/说话人)
├─ speech_fsmn_vad (语音活动检测 VAD)
├─ punc_ct-transformer-cn-en (标点符号)
└─ speech_campplus_sv_zh-cn (CAM++ 声纹，说话人分离)

备选模型（如有下载）：
├─ SenseVoiceSmall (极速纯转录，不支持说话人分离)

查看命令：
ls -lh ~/.cache/modelscope/hub/models/iic/
du -sh ~/.cache/modelscope/hub/models/iic/*
```

#### **MLX Audio 引擎模型**
```bash
路径: ~/.cache/huggingface/hub/

模型大小参考：
- Qwen3-ASR-1.7B-4bit: ~2 GB
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

```
.
├── src/
│   ├── adapters/            # 纯函数工具 (Clean Code)
│   │   ├── text.py          # SenseVoice 标签清洗
│   │   └── audio_chunking.py # 音频切片（静音检测 + 重叠策略）
│   ├── api/                 # 接口层
│   │   └── routes.py        # 路由、Pydantic 模型、能力校验
│   ├── core/                # 核心业务
│   │   ├── base_engine.py   # 引擎抽象接口 (Protocol) + EngineCapabilities
│   │   ├── funasr_engine.py # FunASR (Paraformer/SenseVoice) 实现
│   │   ├── mlx_engine.py    # MLX Audio 实现 (Qwen3-ASR, Whisper)
│   │   └── factory.py       # 引擎工厂
│   ├── services/            # 服务调度
│   │   └── transcription.py # 异步队列与串行执行
│   ├── config.py            # 环境变量配置
│   └── main.py              # 程序入口与生命周期
├── benchmarks/
│   ├── run.py               # 性能基准测试脚本
│   ├── samples/             # 测试音频样本（gitignored）
│   └── results/             # 基准测试结果 JSON（gitignored）
├── tests/
│   ├── unit/                # 单元测试 (Mocked)
│   ├── integration/         # API 集成测试
│   ├── e2e/                 # 端到端测试 (真实模型)
│   ├── reliability/         # 并发与背压测试
│   └── fixtures/            # 测试音频 fixture（gitignored）
├── docs/                    # 设计文档与 SPEC
├── pyproject.toml           # 依赖配置 (uv)
└── README.md                # 本文档
```

## **🧪 运行测试 (Testing)**

本项目包含完整的单元测试和集成测试，使用 `pytest` 框架。

### **1. 运行所有测试**

```bash
uv run python -m pytest
```

### **2. 测试分层说明**

*   **Unit Tests (`tests/unit/`)** — 85 tests total:
    *   `test_adapters.py`: SenseVoice 标签清洗逻辑
    *   `test_engine.py`: FunASR 引擎能力声明、加载、推理（Mock 模型）
    *   `test_mlx_engine.py`: MLX Audio 引擎能力声明（Mock mlx_audio）
    *   `test_audio_chunking.py`: 音频切片、静音检测、SRT 格式、wave 模块优化
    *   `test_config_factory.py`: 配置和引擎工厂
    *   `test_service.py`: 异步队列调度和临时文件生命周期
    *   `test_security.py`: 安全相关单元测试
*   **Integration Tests (`tests/integration/`)**:
    *   `test_api.py`: FastAPI TestClient，验证 HTTP 接口契约、能力校验、OpenAI 兼容性（Mock Engine）
    *   `test_security_integration.py`: CORS、请求追踪、安全头
*   **E2E Tests (`tests/e2e/`)**:
    *   `test_full_flow.py`: **真实模型测试**（需下载模型，速度较慢）
*   **Reliability Tests (`tests/reliability/`)**:
    *   `test_concurrency.py`: 高并发队列背压和 Worker 错误恢复

### **3. 代码质量检查**

```bash
# 类型检查 (mypy strict mode)
uv run mypy src/

# 代码风格检查 (ruff)
uv run ruff check src/

# 代码格式化 (ruff)
uv run ruff format src/
```



## **⚠️ 注意事项**

1. **队列限制**: 默认队列深度为 50。如果请求超过 50 个，API 会立即返回 503 Service Busy。
2. **单例模式**: 由于 M 芯片统一内存特性，我们严格限制模型只加载一次。请勿开启多进程 (workers > 1) 模式运行，否则会导致显存成倍消耗。
3. **临时文件**: 上传的音频会暂存到磁盘以便 ffmpeg 处理，处理完成后会自动删除。
4. **模型切换**: 目前需要停止服务、修改环境变量、重新启动。不支持运行时热切换。
5. **端口历史**: 官方端口为 `50070`。早期使用 WhisperKit 默认的 `50060`，迁移时 +10 以区分。