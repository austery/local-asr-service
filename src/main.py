from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 引入配置和工厂
from src.config import ENGINE_TYPE, get_model_id, HOST, PORT, MAX_QUEUE_SIZE, LOG_LEVEL
from src.core.factory import create_engine
from src.services.transcription import TranscriptionService
from src.api.routes import router as api_router

# === 基础日志配置 ===
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("sensevoice.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    生命周期管理器 (The System Lifecycle)
    FastAPI 启动前执行 yield 前的代码，关闭后执行 yield 后的代码。
    """
    logger.info("🌱 System starting up...")
    logger.info(f"📋 Engine type: {ENGINE_TYPE}")
    logger.info(f"📋 Model ID: {get_model_id()}")
    
    # 1. 使用工厂创建引擎
    engine = create_engine()
    engine.load()
    
    # 2. 初始化并启动服务
    service = TranscriptionService(engine=engine, max_queue_size=MAX_QUEUE_SIZE)
    
    # 3. 启动后台消费者
    await service.start_worker()
    
    # 4. 依赖注入
    app.state.service = service
    app.state.engine_type = ENGINE_TYPE
    app.state.model_id = get_model_id()
    
    logger.info("✅ System ready! Listening for requests...")
    
    yield  # --- 服务运行中 ---
    
    logger.info("🛑 System shutting down...")
    if hasattr(app.state, "service"):
        app.state.service.engine.release()

# === 初始化 FastAPI ===
app = FastAPI(
    title="Local SenseVoice API",
    version="1.0.0",
    lifespan=lifespan  # 挂载生命周期
)

# 允许跨域 (方便前端调用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router)

# 简单的健康检查
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine_type": app.state.engine_type if hasattr(app.state, "engine_type") else "unknown",
        "model": app.state.model_id if hasattr(app.state, "model_id") else "unknown"
    }

if __name__ == "__main__":
    # 开发模式启动
    uvicorn.run(app, host=HOST, port=PORT)