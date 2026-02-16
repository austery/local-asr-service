---
title: Local ASR Service Roadmap
status: 🟢 Active
created: 2026-02-16
lastUpdated: 2026-02-16
context: 基于项目分析和市场调研，制定的改进路线图。
---

## 项目定位

本地 ASR 后台服务，运行在 Apple Silicon (M1 Max) 上，服务于两个核心场景：

1. **本地语音输入** — 短音频 (<30s)，低延迟要求
2. **PureSubs 长音频转写** — YouTube 视频/音频 (30min-2hr)，精度+速度要求

### 市场差异化优势

本项目在开源 ASR 服务中占据独特交叉点：同时支持 **MLX 原生加速 + FunASR 说话人分离 + OpenAI 兼容 REST API + 中文优化 (Paraformer)**。目前没有其他开源项目覆盖这个组合。

参照项目：[Speaches](https://github.com/speaches-ai/speaches)、[whisper.cpp server](https://github.com/ggml-org/whisper.cpp)、[LocalAI](https://github.com/mudler/LocalAI)、[FunASR Server](https://github.com/modelscope/FunASR)

---

## P0-URGENT：测试基础设施修复 & 代码健壮性

> 目标：让测试能跑、代码无潜在崩溃风险。
> 背景：项目从 [local-senseVoice](https://github.com/austery/local-senseVoice) 复制迁移而来，部分基础设施未迁移完整。

### P0-U1: 修复测试基础设施 ✅ 已完成

**问题**: 所有 10 个测试文件全部报 `ModuleNotFoundError: No module named 'src'`，测试**从未真正执行过**。

**根因**:
- `src/` 目录下缺少 `__init__.py` 文件（Python 不识别为包）
- `pyproject.toml` 缺少包发现配置（uv/pip 不知道如何安装 `src` 模块）
- 项目缺少 `conftest.py`（无共享 fixture）

**修复**:
- 添加 `src/__init__.py`、`src/core/__init__.py`、`src/api/__init__.py`、`src/services/__init__.py`、`src/adapters/__init__.py`
- 在 `pyproject.toml` 中添加 `[tool.setuptools.packages.find]` 或使用 uv 的包发现
- 添加 `tests/conftest.py`
- 添加 `[tool.pytest.ini_options]` 配置

### P0-U2: 修复 MLX 引擎 asyncio.run() 陷阱 ✅ 已完成

**问题**: `mlx_engine.py:94` 中使用 `asyncio.run()` 创建新事件循环，但该方法被 `run_in_threadpool` 调用时，主线程已有运行中的事件循环。在某些条件下会抛出 `RuntimeError: This event loop is already running`。

**修复**: 将 `audio_chunking.py` 所有方法从 async 改为同步（它内部全是同步的 `subprocess.run` 调用，没有真正的 async I/O）。MLX 引擎直接同步调用 `process_audio()`，消除 `asyncio.run()`。

### P0-U3: 文件大小校验内存问题 ✅ 已完成

**问题**: `routes.py:79` 用 `await file.read()` 读取全部内容到内存来检查大小。200MB 音频会导致双倍内存峰值。

**修复**: 改用 `file.file.seek(0, 2)` + `file.file.tell()` 获取文件大小，避免读取内容到内存。

### P0-U4: 补充缺失测试 ✅ 已完成

**问题**: `audio_chunking.py` 是项目中最复杂的模块（~490行），但完全没有测试。SRT 格式也未测试。

**修复**:
- 新增 `tests/unit/test_audio_chunking.py`（11 个测试）：归一化跳过/执行、短音频直通、静音检测解析、切分点对齐、SRT 时间格式、SRT 输出格式
- 总测试数从 0 → 67 全部通过

### P0-U5: 清理空操作和误导代码 ✅ 已完成

**修复**:
- 删除空的 `_warmup()` 方法（FunASR 加载时内部已自行初始化）
- 修正 SRT docstring 示例时间戳 (`00:01:190,000` → `00:00:20,000`)
- `_consume_loop` 添加 `stop_worker()` 方法，通过 None 哨兵优雅退出
- `main.py` shutdown 时调用 `stop_worker()` 等待消费者退出后再释放引擎

---

## P0：稳定性 & 健壮性

> 目标：让服务在日常使用中零意外，无效请求明确拒绝而非静默降级。

### P0-1: 引擎能力声明 (Engine Capabilities Protocol) ✅ 已完成

**问题**: 存在隐式约束 — SenseVoice 不支持 timestamp，因此不能做 diarization；但如果用户请求 `output_format=srt` + SenseVoice 引擎，当前行为是静默降级而非明确报错。

**实现**:
- `EngineCapabilities` frozen dataclass：`timestamp`、`diarization`、`emotion_tags`、`language_detect`
- 前缀匹配解析模型能力（FunASR + MLX 各自维护能力表）
- API 层校验：不兼容的格式/参数组合返回 `400 Bad Request`
- 新增 `response_format` 参数作为 OpenAI 兼容别名（`verbose_json` → `json`，`text` → `txt`，`vtt` → `srt`）
- SenseVoice 启动不再加载 `spk_model="cam++"` — 解决 `KeyError: 'timestamp'` 崩溃
- 总测试数 67 → 85（新增 18 个能力校验 + API 测试）

---

### P0-2: 短音频快速路径 (Short Audio Fast Path) ✅ 已完成

**问题**: 本地语音输入通常 <30s、固定格式 (16kHz mono WAV)。当前每次请求都走完整的 FFmpeg 转码 + 分片逻辑，对短音频来说是不必要的开销。

**实现**:
- WAV 格式检测改用 Python `wave` 模块（零 subprocess 开销），替代原来的 `ffprobe` 子进程
- 16kHz mono WAV 输入跳过 FFmpeg 归一化 + 直接通过 `wave` 获取时长（无 `ffprobe` 调用）
- 短音频 (<50min) 自动跳过切片逻辑（已有）
- FunASR 引擎路径本身不经过 AudioChunkingService（无 FFmpeg 开销）
- 更新了测试使用真实 WAV 文件替代 mock ffprobe 响应

---

### P0-3: 端口文档对齐 ✅ 已完成

**背景**: 早期使用 WhisperKit 默认端口 `50060`，后来迁移到 `50070`（+10 以区分）。

**实现**: README 已在多处记录官方端口 `50070`，包括端口演变历史说明。PureSubs 侧的 `LOCAL_WHISPER_URL` 更新需在该项目单独处理。

---

## P1：可观测性 & 开发体验

> 目标：新模型来了能快速评估，运行时状态一目了然。

### P1-1: 模型状态端点 (GET /v1/models/current) ✅ 已完成

**问题**: 当前无法在运行时确认加载了哪个模型、引擎类型、支持哪些能力。

**实现** (与 P0-1 一同完成):
- `GET /v1/models/current` 返回 `engine_type`、`model_id`、`capabilities` dict、`queue_size`、`max_queue_size`
- 响应示例：
  ```json
  {
    "engine_type": "funasr",
    "model_id": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "capabilities": {"timestamp": true, "diarization": true, "emotion_tags": false, "language_detect": true},
    "queue_size": 0,
    "max_queue_size": 50
  }
  ```

---

### P1-2: Benchmark 自动化脚本

**问题**: 每次试新模型都是手动记录性能数据（如 SPEC-007 中的 benchmark），缺乏标准化对比。

**方案**:
- 创建 `benchmarks/run.py` 脚本
- 标准测试集：准备 3 个音频文件（短/中/长，中英文混合）
- 输出指标：RTF (Real-Time Factor)、首字延迟、内存峰值、WER (如有参考文本)
- 结果输出为 JSON + 终端表格，方便对比

**涉及文件**:
- `benchmarks/run.py` — 新建
- `benchmarks/samples/` — 标准测试音频（gitignore 大文件，README 说明获取方式）
- `benchmarks/results/` — 历史结果存档

**使用方式**:
```bash
uv run python benchmarks/run.py --engine funasr --model-id "iic/..."
uv run python benchmarks/run.py --engine mlx --model-id "mlx-community/Qwen3-ASR-1.7B-4bit"
```

---

### P1-3: 代码质量工具链 (ruff + mypy)

**问题**: `pyproject.toml` 中缺少 ruff 和 mypy 配置，与用户编码规范不一致。

**方案**:
- 添加 `[tool.mypy]` strict 配置
- 添加 `[tool.ruff]` 配置（target Python 3.11）
- 修复现有代码中可能出现的类型错误
- 可选：添加 pre-commit hook

---

## P2：新模型评估（按需）

> 不急于集成，发现有价值的新模型时再评估。

### 待评估模型

| 模型 | 类型 | 关注点 | 状态 |
|------|------|--------|------|
| [FireRedASR](https://github.com/FireRedTeam/FireRedASR) | 火山引擎，2026.02 | 中文方言 SOTA，需确认是否可本地下载运行 | 🔍 待调研 |
| Voxtral Mini 3B | Mistral AI，Apache 2.0 | 内置分离+摘要，3B 可本地跑 | 🔍 待调研 |
| Parakeet TDT v3 | NVIDIA | 即将支持中文，ONNX 格式极快 | ⏳ 等待发布 |

### 评估标准

新模型必须满足以下条件才考虑集成：
1. **开源可下载** — 可以在本地无网络运行
2. **Apple Silicon 可用** — 支持 MPS/MLX/CPU，无 CUDA 硬依赖
3. **中文精度** — 不低于当前 Paraformer 水平
4. **速度** — 1 小时音频转录 < 3 分钟（RTF < 0.05）
5. **可集成** — 能实现 `ASREngine` Protocol

---

## 实施顺序

```
P0-1 引擎能力声明 ✅
  │
  ├──→ P0-2 短音频快速路径 ✅
  │
  └──→ P1-1 模型状态端点 ✅（与 P0-1 一同完成）
          │
          └──→ P1-2 Benchmark 脚本

P0-3 端口文档对齐 ✅（README 已记录）
P1-3 ruff + mypy
```

**剩余工作量排序**（从小到大）：
1. P1-3 ruff + mypy — 配置 + 修复类型错误
2. P1-2 Benchmark 脚本 — 新建脚本 + 测试音频准备
