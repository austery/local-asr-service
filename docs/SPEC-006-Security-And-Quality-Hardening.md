---
- specId: SPEC-006
- title: 安全与质量加固 (Security & Quality Hardening)
- status: ✅ 已实现
- priority: P1
- owner: User
- relatedSpecs: [ADR-001, SPEC-101, SPEC-102, SPEC-103, SPEC-004]
- created: 2026-01-31
- updated: 2026-02-16
---

## 1. 目标 (Goal)

在不破坏单进程/单消费者约束的前提下，补齐本地服务的最小安全边界与可回归测试，降低误用与故障排查成本。

## 2. 范围 (Scope)

- API 输入与错误处理安全边界
- 资源限制与配置项
- 关键路径测试补齐
- 不引入外部依赖的"轻量安全"措施

## 3. 非目标 (Non-Goals)

- 不做复杂的鉴权/多租户/公网暴露安全体系
- 不改变现有 Clean Architecture 分层
- 不引入额外的外部依赖（如 JWT 库、认证框架等）

## 4. 安全与稳健性要求 (Requirements)

### 4.1 上传大小限制

**目标**: 防止超大文件导致内存溢出或磁盘空间耗尽

**实现要求**:
- 新增环境变量 `MAX_UPLOAD_SIZE_MB`（默认值：200 MB）
- 在 FastAPI 路由层使用 `File(..., max_length=...)` 或自定义验证
- 当上传文件大小超限时：
  - 立即返回 **HTTP 413 Payload Too Large**
  - 响应体示例：`{"error": "File size exceeds maximum allowed (200 MB)"}`
  - 超限文件必须被删除，不留临时文件残留

**配置示例**:
```env
MAX_UPLOAD_SIZE_MB=200
```

### 4.2 文件类型校验

**目标**: 防止非音频文件上传导致推理失败或安全风险

**实现要求**:
- 定义音频 MIME 类型白名单：
  - `audio/wav`, `audio/x-wav`
  - `audio/mpeg`, `audio/mp3`
  - `audio/mp4`, `audio/x-m4a`
  - `audio/flac`
  - `audio/ogg`
  - `audio/webm`
- 仅根据 `content_type` 进行初步校验（不强制 ffprobe 验证以保持轻量）
- 当文件类型不在白名单内时：
  - 返回 **HTTP 415 Unsupported Media Type**
  - 响应体示例：`{"error": "Unsupported file type. Only audio files are allowed."}`
- 不信任客户端提供的 `filename` 扩展名

**配置方式**:
硬编码白名单（不需要额外环境变量）

### 4.3 错误信息最小化

**目标**: 防止服务端内部异常细节泄露给客户端

**实现要求**:
- 客户端错误响应（4xx/5xx）不包含：
  - Python 堆栈跟踪
  - 文件系统内部路径
  - 模型加载/推理的详细错误
- 服务端日志记录完整异常信息（包括 `request_id` 关联）
- 保留现有 HTTP 状态码语义：
  - 400: 参数错误
  - 413: 文件过大
  - 415: 文件类型不支持
  - 503: 队列已满

**响应示例**:
```json
{
  "error": "Internal server error occurred. Please check server logs for details.",
  "request_id": "req_abc123"
}
```

### 4.4 CORS 与访问范围

**目标**: 默认收敛跨域访问，仅允许本地调用

**实现要求**:
- 新增环境变量 `ALLOWED_ORIGINS`（默认值：`http://localhost,http://127.0.0.1`）
- 使用 FastAPI 的 `CORSMiddleware` 实现
- 允许用户显式配置 `ALLOWED_ORIGINS=*` 放开限制
- 默认行为：仅允许本地前端访问

**配置示例**:
```env
# 默认（仅本地）
ALLOWED_ORIGINS=http://localhost,http://127.0.0.1

# 放开所有源（仅在受信任网络环境使用）
ALLOWED_ORIGINS=*
```

### 4.5 资源与并发安全

**目标**: 确保单进程约束与队列背压策略不被误配置

**实现要求**:
- 明确保留 `workers=1`（在启动脚本、文档、README 中强调）
- `MAX_QUEUE_SIZE` 默认值保持 50
- 在启动时打印警告日志，提醒用户不要随意增加 worker 数
- 确保任何异常路径（如推理失败、ffmpeg 错误）都能正确清理临时文件

**日志示例**:
```
⚠️ Running with workers=1 (REQUIRED for Mac Silicon to prevent OOM)
```

## 5. 可观测性要求 (Observability)

### 5.1 请求追踪

**实现要求**:
- 为每个请求生成唯一 `request_id`（使用 `uuid.uuid4()`）
- 在请求日志、错误日志、响应头中包含 `X-Request-ID`
- 日志格式示例：
  ```
  [INFO] [req_abc123] Received transcription request: file=audio.wav, language=auto
  ```

### 5.2 性能指标

**实现要求**:
- 记录关键耗时指标：
  - **排队时间**: 从请求接收到 worker 开始处理的时间
  - **推理时间**: ASR 引擎实际推理的时间
  - **总耗时**: 从请求接收到响应返回的时间
- 日志示例：
  ```
  [INFO] [req_abc123] Transcription completed: queue_time=1.2s, inference_time=3.5s, total_time=4.7s
  ```

## 6. 测试策略补充 (Testing Additions)

### 6.1 单元测试 (Unit Tests)

新增测试用例（放置在 `tests/unit/`）：

- **`test_upload_size_limit`**: 验证超限文件返回 413
- **`test_file_type_validation`**: 验证非音频文件返回 415
- **`test_error_message_sanitization`**: 验证错误响应不包含堆栈信息
- **`test_request_id_generation`**: 验证每个请求都有唯一 request_id

### 6.2 集成测试 (Integration Tests)

新增测试用例（放置在 `tests/integration/`）：

- **`test_cors_default_local_only`**: 验证默认 CORS 仅允许本地访问
- **`test_cors_wildcard_override`**: 验证 `ALLOWED_ORIGINS=*` 时允许任意源
- **`test_queue_full_503`**: 验证队列满时返回 503（已存在，确认不破坏）
- **`test_temp_file_cleanup_on_error`**: 验证推理失败时临时文件被清理

### 6.3 可靠性测试 (Reliability Tests)

新增测试用例（放置在 `tests/reliability/`）：

- **`test_large_file_cleanup`**: 验证超大文件上传失败后无临时文件残留
- **`test_concurrent_requests_resource_cleanup`**: 验证并发请求失败时资源清理正确

## 7. 验收标准 (Acceptance Criteria)

- [x] 任何超限上传均返回 413，且无残留临时文件
- [x] 非音频文件返回 415（含文件名扩展名 fallback 校验）
- [x] 生产错误响应不包含堆栈与内部路径
- [x] 默认 CORS 仅允许 localhost/127.0.0.1
- [x] 每个请求有唯一 request_id 并记录关键耗时
- [x] 新增测试通过且不影响现有测试（85 tests）
- [x] README 更新新增环境变量说明

## 8. 实施计划 (Implementation Plan)

### Phase 1: 配置与中间件 (Config & Middleware) ✅
- [x] 新增 `MAX_UPLOAD_SIZE_MB`、`ALLOWED_ORIGINS` 环境变量到 `src/config.py`
- [x] 在 `src/main.py` 添加 CORS 中间件
- [x] 添加请求日志中间件（生成 request_id）

### Phase 2: API 层校验 (API Validation) ✅
- [x] 在 `src/api/routes.py` 添加文件大小校验（`file.file.seek/tell` 零内存读取）
- [x] 添加 MIME 类型校验 + 文件扩展名 fallback
- [x] 错误响应标准化处理

### Phase 3: 可观测性增强 (Observability) ✅
- [x] 在 `src/services/transcription.py` 添加耗时统计
- [x] 日志输出格式化（包含 request_id 与耗时）

### Phase 4: 测试补充 (Testing) ✅
- [x] 编写单元测试 (`tests/unit/test_security.py`)
- [x] 编写集成测试 (`tests/integration/test_security_integration.py`)
- [x] 编写可靠性测试 (`tests/reliability/test_concurrency.py`)

### Phase 5: 文档更新 (Documentation) ✅
- [x] 更新 README.md 环境变量说明
- [x] 更新 `.env.example`

## 9. 兼容性与取舍 (Compatibility & Trade-offs)

### 兼容性保证
- 所有新增配置项均有合理默认值，现有部署无需修改即可运行
- 不改变现有 API 接口签名
- 不引入新的外部依赖

### 取舍说明
- **不做强制 ffprobe 校验**: 保持轻量，仅通过 MIME 类型初步过滤
- **不做复杂鉴权**: 本地服务定位，安全措施以"最小防误用"为主
- **允许配置放宽限制**: 通过环境变量允许用户根据实际场景调整（如 `ALLOWED_ORIGINS=*`）

## 10. 风险与缓解 (Risks & Mitigation)

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 默认 CORS 限制过严导致合法使用受阻 | 中 | 通过文档明确说明如何放宽限制 |
| 文件大小限制不符合用户实际需求 | 低 | 默认 200MB 覆盖大部分场景，可通过环境变量调整 |
| 新增校验逻辑影响性能 | 低 | 校验逻辑轻量（仅 MIME/大小检查），影响可忽略 |

## 11. 后续优化方向 (Future Enhancements)

以下功能不在本 SPEC 范围内，但可作为后续迭代方向：

- **ffprobe 深度校验**: 通过 ffprobe 验证文件确实为有效音频（需权衡性能）
- **速率限制 (Rate Limiting)**: 防止单一客户端恶意占用资源
- **API Key 认证**: 提供可选的轻量认证机制
- **结构化日志输出**: 支持 JSON 格式日志便于分析

---

**审阅说明**:
请审阅本 SPEC 是否符合预期。确认后我将按以下步骤实施：
1. 实现配置与中间件
2. 添加 API 层校验
3. 增强可观测性
4. 补充测试
5. 更新文档

如有调整需求，请告知具体章节。
