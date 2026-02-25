---
specId: SPEC-108
title: 按请求动态切换模型 (Dynamic Model Switching)
status: ✅ 已实现
priority: P1
owner: User
relatedSpecs: [SPEC-101, SPEC-102, SPEC-103]
created: 2026-02-25
---

## 1. 背景与动机

### 问题

当前 ASR 服务在启动时通过 `ENGINE_TYPE` 和 `MODEL_ID` 环境变量固定加载一个模型，该模型在整个进程生命周期内常驻内存。

| 引擎/模型 | 估算内存占用 |
|-----------|-------------|
| FunASR Paraformer Large | ~6–8 GB (MPS) |
| Qwen3-ASR-1.7B-8bit | ~3–4 GB (MLX) |
| Qwen3-ASR-1.7B-4bit | ~1.5–2 GB (MLX) |
| Whisper Large v3 Turbo | ~3–4 GB (MLX) |
| Parakeet TDT 0.6B | ~1–2 GB (MLX) |

### 核心使用场景（驱动本 Spec 的真实需求）

本功能的需求来自 PureSubs 项目对 YouTube 内容的批量转录。客户端（PureSubs）在发起请求前，已通过 **Channel 级别元数据**知道内容属于哪类场景，可以主动声明所需模型：

| 场景 | 典型内容 | 所需能力 | 最优模型 | 内存代价 |
|------|---------|---------|---------|---------|
| **单人独白** | 技术讲座、个人 vlog、知识类频道 | 仅转录，无需分词 | `qwen3-asr-mini` (4bit) | ~1.5 GB |
| **多人对谈** | Lex Friedman、Paolo Cremer 等 podcast | 转录 + 说话人分离 | `paraformer` (FunASR) | ~6–8 GB |
| **本地语音输入** | 实时语音命令 (<30s) | 低延迟，无需分词 | `qwen3-asr-mini` (4bit) | ~1.5 GB |

**决策时机**：模型选择在**请求发起前**由客户端确定，不是运行时推测。服务端只需忠实执行客户端声明的模型。

### 换模等待策略（已决策）

换模耗时（10–60s）相对于实际转录耗时（1小时视频 ≈ 8分钟）占比极小（约 6%）。
服务采用**透明等待**策略：请求在队列中等待换模完成后直接执行，客户端无需实现重试逻辑。
不返回 503，因为客户端主动声明了模型，等待是预期行为。

每次都加载最重的模型是浪费。

### 目标

允许调用方在每个转写请求中通过 `model` 字段声明所需模型。服务根据此字段**动态热换**模型，使当前内存中只保留正在使用的模型。

---

## 2. 设计约束

1. **单 Worker 串行化**：当前已有单消费者队列保证同一时间只执行一个推理任务，热换模型在两次推理之间进行是安全的。
2. **OpenAI 兼容**：`model` 字段已存在于 OpenAI Whisper API 规范中（通常设为 `"whisper-1"`），本功能复用该字段而不增加新字段，保持向后兼容。
3. **内存优先**：热换发生时必须先完整释放旧引擎资源（`engine.release()`），再加载新引擎，确保两模型不同时驻留内存。
4. **不改变 ASREngine Protocol**：`base_engine.py` 的接口不变，扩展点在 `TranscriptionService` 层。

---

## 3. 模型注册表 (Model Registry)

引入 `src/core/model_registry.py`，维护模型别名到完整配置的映射。

### 3.1 数据结构

```python
@dataclass(frozen=True)
class ModelSpec:
    """某个具名模型的完整规格"""
    model_id: str           # HuggingFace / ModelScope 路径
    engine_type: EngineType # "funasr" | "mlx"
    alias: str              # 用户友好短名（也作为字典 key）
    description: str        # 人类可读描述
    capabilities: EngineCapabilities  # 预声明能力（用于预校验）
```

### 3.2 内置模型表

| alias | engine_type | model_id | timestamp | diarization | emotion_tags | 适用场景 |
|-------|-------------|----------|-----------|-------------|--------------|---------|
| `paraformer` | `funasr` | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | ✅ | ✅ | ❌ | 中文 + 说话人分离（默认 FunASR）|
| `sensevoice-small` | `funasr` | `iic/SenseVoiceSmall` | ❌ | ❌ | ✅ | **最快模型（80-85x）**；速度优先/情感标签场景；转录质量较差，不推荐常规使用 |
| `qwen3-asr-mini` | `mlx` | `mlx-community/Qwen3-ASR-1.7B-4bit` | ✅ | ❌ | ❌ | 短音频，低延迟（默认 MLX）|
| `qwen3-asr` | `mlx` | `mlx-community/Qwen3-ASR-1.7B-8bit` | ✅ | ❌ | ❌ | 中等精度 |
| `parakeet` | `mlx` | `mlx-community/parakeet-tdt-0.6b-v2` | ✅ | ❌ | ❌ | 英文专用极速 |

> **完整 HuggingFace/ModelScope 路径也可直接传入**（注册表兜底逻辑：先查 alias，查不到则尝试将其作为 `model_id` 推断 engine_type）。

### 3.3 Engine Type 推断规则（完整路径 fallback）

当 `model` 字段不在预定义 alias 列表中时：
- 包含 `mlx-community/` 前缀 → `mlx`
- 包含 `iic/` 或 `funasr` 关键词 → `funasr`
- 其他 → 使用当前运行引擎类型（不切换引擎，只切换模型 ID）

---

## 4. API 层变更

### 4.1 请求参数

`POST /v1/audio/transcriptions` 的 `model` 字段**从被忽略变为实际生效**。

```
model: string (optional)
```

- 缺省 / `null` / `"whisper-1"` → 沿用服务启动时的当前模型（**向后兼容**，零 breaking change）
- 合法 alias 或完整路径 → 触发动态换模逻辑

### 4.2 前置校验（API 层，换模之前）

1. 如果 `model` 命中注册表，读取 `ModelSpec.capabilities` 进行能力预校验（如 `diarization` 请求传给 parakeet 则 400）。
2. 如果 `model` 是未知字符串且无法推断 engine_type，返回 `400 Bad Request`：
   ```json
   { "error": "Unknown model: 'foo-bar'. Use GET /v1/models to list available models." }
   ```

---

## 5. Service 层变更

### 5.1 TranscriptionService 扩展

在 `src/services/transcription.py` 的 `_consume_loop` 中，推理执行前增加换模检查：

```
消费循环（伪代码）:

  job = await queue.get()

  if job.requested_model_spec != current_model_spec:
      # Step 1: 释放当前引擎资源
      current_engine.release()

      # Step 2: 创建新引擎（factory）
      new_engine = EngineFactory.create(job.requested_model_spec)
      new_engine.load()

      # Step 3: 原子替换（单线程安全，无需锁）
      current_engine = new_engine
      current_model_spec = job.requested_model_spec

  result = await run_in_threadpool(current_engine.transcribe_file, ...)
```

**关键点**：
- 换模发生在 `_consume_loop` 内部，即上一个 job 已完成之后、下一个 job 推理之前。串行队列保证此窗口内没有并发推理。
- `current_engine.release()` 必须成功（无论成功与否都继续，但要 log）。
- 换模失败（`load()` 抛出异常）时，job 以 `500` 失败，但服务需保持可用状态（不崩溃）。

### 5.2 状态持久化

`TranscriptionService` 新增字段：
```python
_current_model_spec: ModelSpec   # 当前已加载模型的规格
```

### 5.3 错误处理

| 错误场景 | 行为 |
|---------|------|
| 请求模型不在注册表中 | API 层 400，不入队 |
| `engine.release()` 超时或异常 | 记录 WARNING，继续尝试加载新模型 |
| `engine.load()` 失败（模型未下载等） | 该 job 返回 500；**尝试恢复：重新加载上一个已知可用的模型** |
| 恢复失败 | 返回 503，标记 service 为 `DEGRADED` 状态 |

---

## 6. 新增 API 端点

### GET /v1/models

返回服务支持的所有模型列表（注册表内容），供客户端枚举可用模型。

```json
{
  "models": [
    {
      "alias": "qwen3-asr-mini",
      "model_id": "mlx-community/Qwen3-ASR-1.7B-4bit",
      "engine_type": "mlx",
      "description": "短音频，低延迟（默认 MLX）",
      "capabilities": {
        "timestamp": true,
        "diarization": false,
        "emotion_tags": false,
        "language_detect": true
      }
    }
  ],
  "current": "qwen3-asr-mini"
}
```

---

## 7. GET /v1/models/current 变更

现有端点增加 `model_alias` 字段（若当前模型在注册表中）：

```json
{
  "engine_type": "mlx",
  "model_id": "mlx-community/Qwen3-ASR-1.7B-4bit",
  "model_alias": "qwen3-asr-mini",
  "capabilities": { ... },
  "queue_size": 0,
  "max_queue_size": 50
}
```

---

## 8. 受影响文件

| 文件 | 变更类型 |
|------|---------|
| `src/core/model_registry.py` | 新增 |
| `src/core/factory.py` | 修改：接受 `ModelSpec` 作为输入 |
| `src/services/transcription.py` | 修改：换模逻辑 + `_current_model_spec` |
| `src/api/routes.py` | 修改：`model` 字段校验 + 新增 `/v1/models` 路由 |
| `src/config.py` | 轻微修改：默认模型 spec 初始化 |
| `tests/unit/test_model_registry.py` | 新增 |
| `tests/unit/test_dynamic_switching.py` | 新增 |
| `tests/integration/test_model_api.py` | 新增 |

---

## 9. 测试策略

> **原则**（来自 *Unit Testing: Principles, Practices, and Patterns*）：
> - 测试**可观测行为**（输出、状态、对外部系统的调用），而非实现细节。
> - Mock 只用于**非托管依赖**（第三方库、真实模型）；自有类优先用真实对象。
> - 每个测试命名描述「场景 + 预期」，遵循 `test_should_<expected>_when_<condition>` 格式。

---

### 9.1 `tests/unit/test_model_registry.py`（纯函数，无 Mock）

| ID | 测试名 | 验证的可观测行为 |
|----|--------|----------------|
| MR-1 | `test_should_return_spec_when_alias_is_known` | `lookup("paraformer")` 返回正确的 `model_id` 和 `engine_type` |
| MR-2 | `test_should_infer_mlx_engine_when_full_path_has_mlx_community_prefix` | `lookup("mlx-community/foo")` → `engine_type="mlx"` |
| MR-3 | `test_should_infer_funasr_engine_when_full_path_has_iic_prefix` | `lookup("iic/foo")` → `engine_type="funasr"` |
| MR-4 | `test_should_raise_when_alias_is_completely_unknown` | `lookup("not-a-model")` 抛出 `ValueError` |
| MR-5 | `test_should_declare_diarization_capability_for_paraformer` | `paraformer` spec 的 `capabilities.diarization == True` |
| MR-6 | `test_should_declare_no_diarization_for_parakeet` | `parakeet` spec 的 `capabilities.diarization == False` |

```python
# 示例 AAA 风格
def test_should_return_spec_when_alias_is_known():
    # Arrange: registry 是纯函数，无需 setup
    # Act
    spec = ModelRegistry.lookup("paraformer")
    # Assert
    assert spec.engine_type == "funasr"
    assert "paraformer" in spec.model_id
```

---

### 9.2 `tests/unit/test_dynamic_switching.py`（Solitary，Mock Engine）

重点：验证 `TranscriptionService` 编排层的可观测结果，Mock 掉真实引擎。

| ID | 测试名 | 验证的可观测行为 |
|----|--------|----------------|
| DS-1 | `test_should_return_result_when_same_model_requested_twice` | 连续两个相同 alias 请求，job 都成功；`load()` 仅被调用一次（不重复换模）|
| DS-2 | `test_should_return_result_after_switching_to_different_model` | 第一个 job 用 `qwen3-asr-mini`，第二个 job 用 `paraformer`，两者都返回正确结果 |
| DS-3 | `test_should_release_old_engine_before_loading_new_one` | 换模时 `old_engine.release()` 在 `new_engine.load()` 之前被调用（验证调用顺序，因为这是安全约束）|
| DS-4 | `test_should_fail_job_when_new_model_load_fails` | `load()` 抛出异常时，对应 job 以异常失败，而不是静默挂起 |
| DS-5 | `test_should_process_next_job_after_failed_switch` | 换模失败后，服务仍可接受新请求（不崩溃）|
| DS-6 | `test_should_clean_temp_file_even_when_switch_fails` | 换模失败时，上传的临时文件依然被删除 |

> **注意**：DS-3 是本规则中极少数允许验证调用顺序的例外，因为 `release-before-load` 是明确的内存安全约束，而非实现细节。

---

### 9.3 `tests/integration/test_model_api.py`（API 层，Mock Engine）

使用 `fastapi.testclient.TestClient`，通过 HTTP 验证 API 契约。

| ID | 测试名 | 验证的可观测行为 |
|----|--------|----------------|
| MA-1 | `test_should_return_model_list_on_get_models` | `GET /v1/models` 返回包含所有内置 alias 的数组，结构符合 Section 6 定义 |
| MA-2 | `test_should_include_current_model_in_get_models_response` | 响应的 `current` 字段与服务启动时加载的模型 alias 一致 |
| MA-3 | `test_should_succeed_when_valid_alias_provided` | `POST /v1/audio/transcriptions` 带 `model=qwen3-asr-mini` 返回 200 |
| MA-4 | `test_should_return_400_when_unknown_model_provided` | `model=not-a-real-model` 返回 400，body 含提示信息 |
| MA-5 | `test_should_use_current_model_when_model_field_omitted` | 不传 `model` 字段，行为与传 `model=<current_alias>` 相同（向后兼容）|
| MA-6 | `test_should_return_400_when_diarization_requested_with_non_diarization_model` | `model=parakeet` + `diarization=true` → 400（能力预校验）|
| MA-7 | `test_should_return_400_when_whisper_1_provided_with_no_matching_alias` | `model=whisper-1`（OpenAI 默认值）→ 服务降级为当前模型，不报错 |

---

### 9.4 E2E Tests（可选，需真实模型下载）

不在本 SPEC 强制范围内。在 `tests/e2e/test_full_flow.py` 补充：
- 连续发两个请求分别指定 `paraformer` 和 `qwen3-asr-mini`，验证切换后转录结果格式正确。

---

## 10. 实现顺序

```
Step 1: 实现 ModelRegistry（src/core/model_registry.py）
  └── 编写 unit tests

Step 2: 扩展 EngineFactory（接受 ModelSpec）

Step 3: 扩展 TranscriptionService（换模逻辑）
  └── 编写 unit tests（mock engine）

Step 4: API 层变更（model 字段校验 + /v1/models 端点）
  └── 编写 integration tests

Step 5: 更新 CLAUDE.md + ROADMAP.md
```

---

## 11. 设计决策记录

| # | 问题 | 决策 | 理由 |
|---|------|------|------|
| Q1 | 换模时请求等待还是返回 503？ | ✅ **透明等待** | 客户端主动声明模型，等待是预期行为；换模耗时相对转录任务可忽略不计 |
| Q2 | `model` 缺省时 GET /v1/models/current 返回什么？ | ✅ **返回当前已加载模型的 alias（若在注册表中）** | 便于客户端调试，了解服务当前状态 |
| Q3 | 是否支持跨引擎切换（MLX ↔ FunASR）？ | ✅ **支持** | `paraformer`（FunASR）和 `qwen3-asr-mini`（MLX）是核心场景的两端，必须支持跨引擎切换 |

| Q4 | 模型注册表应该用 Python 还是 YAML/JSON 配置文件？ | ✅ **保留 Python，定义迁移触发条件** | 见下方详细说明 |

### Q4 详述：Python Registry vs 外部配置文件

**用户提问**：模型都配置在 Python 文件里，是否应该改用 YAML？

**现状分析**：`model_registry.py` 本质上已经是一个配置文件，只是用 Python 语法描述。当前没有任何运行时逻辑，只有纯数据声明（`ModelSpec` dataclass 列表）。

**保留 Python 的理由**：

| 维度 | Python registry | YAML / JSON |
|------|----------------|-------------|
| 类型安全 | ✅ mypy 全量检查 | ❌ 需要手写 schema 验证 |
| 工作量 | ✅ 5 行加一个模型 | ❌ 需要额外加载/验证/映射层 |
| IDE 支持 | ✅ 自动补全 + 跳转 | ❌ 无 |
| 动态加载 | ❌ 需要重启 | ✅ 可热加载 |
| 非开发者编辑 | ❌ 需要懂 Python | ✅ 任何人可编辑 |
| 跨服务共享 | ❌ 绑定代码仓库 | ✅ 可放 S3 / 配置中心 |

**当前项目不需要 YAML 的核心原因**：
1. 这是单人个人项目，模型配置的修改者就是开发者本身
2. 模型列表变化频率极低（加一个新模型 ≈ 每几个月一次）
3. 当前已有 `reload_registry()` 的空间，但重启服务本来就是换模时的合理操作
4. 引入 YAML 需要额外验证层，而类型安全的价值在当前上下文中超过配置灵活性

**触发迁移的条件**（满足任意一项时再做）：

```
触发迁移 YAML/外部配置的条件（任意一项）：
├── [ ] 需要让非技术用户（非开发者）维护模型列表
├── [ ] 服务发展为多租户，不同用户需要不同的模型白名单
├── [ ] 部署到多个环境（dev / staging / prod），需要每环境不同配置
├── [ ] 模型配置需要跨服务共享（如同时有 ASR 服务和 translation 服务共用注册表）
└── [ ] 运营需求：线上添加/下线模型不能接受服务重启的停机窗口
```

**结论**：Python registry 是当前阶段的最优解。过早迁移到 YAML 是 YAGNI（You Aren't Gonna Need It）反模式的典型案例。上述触发条件出现时，再用 1-2 小时迁移即可，不需要提前设计。

---

### 未来扩展方向（不在本 Spec 范围内）

- **CPU engine**（faster-whisper / CTranslate2）：用于 Linux VPS 部署，届时开独立 SPEC。
- **模型预热队列**：预测下一个 job 所需模型，提前在后台加载以消除换模延迟（P3 优化）。
- **MLX 长音频分块**：Parakeet 在 > 5 分钟音频上触发 Metal OOM（known issue），需降低 MLX 引擎的分块时长阈值。

---

## 12. 实测基准数据（2026-02-25）

> 使用 `benchmarks/run.py --compare` two-pass 模式，消除热切换开销后的纯推理性能。
> M1 Max，单进程，服务启动于 `ENGINE_TYPE=funasr`（paraformer 为第一个加载的模型）。

### 12.1 测试音频

| 文件 | 时长 | 语言 | 说明 |
|------|------|------|------|
| `tests/fixtures/two_speakers_60s.wav` | 60s | 中英混合 | 关于 Claude Code 的对话片段，2人 |
| `test1.wav` | 28.9 min | 英文 | 公司技术会议，多人 |
| `chatwithJustin.wav` | 23.2 min | **中英夹杂** | 父子日常对话，2人 |

### 12.2 纯推理速度（inference only，已排除热切换）

| 模型 | 60s 混合 | 29min 英文 | 23min 中英 | 规律 |
|------|---------|-----------|-----------|------|
| **sensevoice-small** | 58.9x | **79.1x** | **83.3x** | FunASR 流式批处理，长音频反而更快 |
| **paraformer** | 49.4x | **65.3x** | **64.9x** | 同上，且含 diarization |
| qwen3-asr (8-bit) | 26.2x | 15.3x | 13.0x | MLX 自回归，中文重场景退化明显 |
| qwen3-asr-mini (4-bit) | 36.3x | 21.5x | 9.3x | 4-bit 对中文优化无明显收益 |
| parakeet | **121.7x** | ❌ OOM | ❌ OOM | 英文短片极速；> 5min 长文件 Metal OOM |

**关键发现**：
- FunASR 引擎（paraformer / sensevoice-small）在长音频下性能**不退化**，因为内部有流式分块逻辑
- MLX 模型的自回归生成在**中文**场景下比英文慢 2–3 倍（中文字符 token 密度更高）
- qwen3-asr-mini（4-bit）在中文长音频下比 qwen3-asr（8-bit）**更慢**，量化对中文帮助有限

### 12.3 转录质量（文本准确度）

以同一段中英夹杂音频为例（内容：讨论加拿大大学选择，含人名 "Brook" 和地名 "Waterloo"）：

| 模型 | 英文人名识别 | 中英切换 | 整体评价 |
|------|------------|---------|---------|
| **qwen3-asr** | "Brook，" ✅ | 自然，正确加空格 | 最佳综合质量 |
| **paraformer** | "broke" ❌（误作英文单词）| 可用，有少量噪声 | 唯一支持 diarization |
| qwen3-asr-mini | "Brook" ✅，但有幻觉（"华尔街"代替"滑铁卢"）| 基本可用 | 速度优先时的折中 |
| sensevoice-small | "bro lock?" ❌ | 词语边界错乱 | 不适合转录质量要求 |

### 12.4 最终选模决策

| 使用场景 | 推荐模型 | 理由 |
|---------|---------|------|
| **多人对话（中文/中英混）** | `paraformer` | 唯一有 diarization，长文件速度好，默认启动模型 |
| **单人英文长视频**（YouTube 博主） | `qwen3-asr` | 英文专有名词识别最准确（"Claude code" vs "cloud code"）|
| **多人英文播客/会议** | `paraformer` | diarization + 英文基本可用，速度好 |
| **中英混合单人**（质量优先）| `qwen3-asr` | 最佳格式化，正确处理双语边界 |
| **语音输入短片段**（< 30s）| `qwen3-asr-mini` | 短片段下速度差异可忽略，4-bit 够用 |
| **极速批处理**（质量要求低）| `sensevoice-small` | 最快（83x realtime），但转录质量较差 |
| **英文短片**（< 5min）| `parakeet` | 最快（121x），英文专用；**注意 OOM 限制** |

### 12.5 客户端选模建议（PureSubs 集成）

```
channel.is_multi_speaker AND language == "zh"  →  model=paraformer
channel.is_multi_speaker AND language == "en"  →  model=paraformer
channel.is_single_speaker AND language == "en" →  model=qwen3-asr
channel.is_single_speaker AND language == "zh-en-mixed" → model=qwen3-asr
voice_input (< 30s)                            →  model=qwen3-asr-mini
```

> **服务默认启动模型**：`paraformer`（`ENGINE_TYPE=funasr`），覆盖最多使用场景。
> 只有客户端明确声明 `model=qwen3-asr` 或 `model=qwen3-asr-mini` 时才触发热切换。
