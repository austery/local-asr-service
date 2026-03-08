import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from src.api.routes import router as api_router

# 引入配置和工厂
from src.config import (
    ALLOWED_ORIGINS,
    ENGINE_TYPE,
    HOST,
    LOG_LEVEL,
    MAX_QUEUE_SIZE,
    MODEL_IDLE_TIMEOUT_SEC,
    PORT,
    get_model_id,
)
from src.core.model_registry import lookup
from src.services.transcription import TranscriptionService

# === 基础日志配置 ===
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("local_asr.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    生命周期管理器 (The System Lifecycle)
    FastAPI 启动前执行 yield 前的代码，关闭后执行 yield 后的代码。
    """
    logger.info("🌱 System starting up...")
    logger.info(f"📋 Engine type: {ENGINE_TYPE}")
    logger.info(f"📋 Model ID: {get_model_id()}")
    logger.warning("⚠️  Running with workers=1 (REQUIRED for Mac Silicon to prevent OOM)")

    # 1. 解析启动模型的 ModelSpec（用于 dynamic switching 的基准）
    startup_model_id = get_model_id()
    try:
        initial_spec = lookup(startup_model_id)
    except ValueError:
        initial_spec = None
        logger.warning(f"⚠️  Startup model '{startup_model_id}' not in registry; model tracking disabled.")

    # 2. 初始化服务（Worker subprocess spawns lazily on first request）
    service = TranscriptionService(
        engine_type=ENGINE_TYPE,
        model_id=startup_model_id,
        max_queue_size=MAX_QUEUE_SIZE,
        initial_model_spec=initial_spec,
        idle_timeout=MODEL_IDLE_TIMEOUT_SEC,
    )

    await service.start_worker()
    logger.info(f"💤 Idle timeout: {MODEL_IDLE_TIMEOUT_SEC}s (0 = disabled)")

    # 3. 依赖注入（engine_type/model_id 保留供 health check 和降级路径使用）
    app.state.service = service
    app.state.engine_type = ENGINE_TYPE
    app.state.model_id = startup_model_id

    logger.info("✅ System ready! Worker subprocess spawns on first transcription request.")

    yield  # --- 服务运行中 ---

    logger.info("🛑 System shutting down...")
    if hasattr(app.state, "service"):
        await app.state.service.stop_worker()


# === 初始化 FastAPI ===
app = FastAPI(
    title="Local ASR Service",
    version="1.0.0",
    lifespan=lifespan,  # 挂载生命周期
)

# 解析 CORS origins
cors_origins = ALLOWED_ORIGINS.split(",") if ALLOWED_ORIGINS != "*" else ["*"]
logger.info(f"🔒 CORS allowed origins: {cors_origins}")

# CORS 中间件（默认仅本地）
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志中间件（生成 request_id 并记录耗时）
@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Response:
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()
    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    response: Response = await call_next(request)

    duration = time.time() - start_time
    logger.info(f"[{request_id}] Completed in {duration:.2f}s - Status: {response.status_code}")

    response.headers["X-Request-ID"] = request_id
    return response


# 注册路由
app.include_router(api_router)


# 简单的健康检查
@app.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "engine_type": app.state.engine_type if hasattr(app.state, "engine_type") else "unknown",
        "model": app.state.model_id if hasattr(app.state, "model_id") else "unknown",
    }


if __name__ == "__main__":
    # 开发模式启动
    uvicorn.run(app, host=HOST, port=PORT)
