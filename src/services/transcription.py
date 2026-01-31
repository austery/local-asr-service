import asyncio
import shutil
import os
import uuid
import time
import logging
import tempfile
from dataclasses import dataclass
from typing import Dict, Any
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

# 引入抽象接口和适配器
from src.core.base_engine import ASREngine
from src.adapters.text import clean_sensevoice_tags

# 定义一个简单的任务对象，用于在队列中传递
@dataclass
class TranscriptionJob:
    uid: str
    temp_dir: str  # 任务专属临时目录
    temp_file_path: str # 原始文件路径
    params: Dict[str, Any]
    future: asyncio.Future
    received_at: float

class TranscriptionService:
    """
    转录服务调度器。
    职责：
    1. 管理异步队列 (Async Queue)
    2. 协调 Engine 进行串行推理
    3. 管理临时文件的生命周期
    """

    def __init__(self, engine: ASREngine, max_queue_size: int = 50):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        # 核心设计：使用 asyncio.Queue 实现背压 (Backpressure)
        # 如果队列满 50 个，前端会直接收到 503 错误，保护系统不崩溃
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.is_running = False
        self.logger.info(f"🚦 Service initialized. Queue size: {max_queue_size}")

    async def start_worker(self):
        """启动后台消费者循环 (在 main.py 的 lifespan 中调用)"""
        self.is_running = True
        asyncio.create_task(self._consume_loop())
        self.logger.info("👷 Background worker started.")

    async def submit(self, file: UploadFile, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        提交任务接口 (供 API 层调用)。
        这个方法是非阻塞的：它只是把任务扔进队列，然后等待结果。
        """
        # 1. 检查队列是否已满 (快速失败)
        if self.queue.full():
            raise RuntimeError("Service busy: Queue is full.")

        # 2. "临时文件之舞" (The Temp File Dance)
        # 为每个请求创建一个独立的临时目录，方便统一清理
        temp_dir = tempfile.mkdtemp(prefix="asr_task_")
        
        try:
            file_ext = os.path.splitext(file.filename)[1] or ".wav"
            # 文件名使用 original 以便区分，但实际上只要在目录下就行
            temp_filename = f"original{file_ext}"
            temp_path = os.path.join(temp_dir, temp_filename)

            # 将上传的文件流写入磁盘
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 3. 创建任务对象
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            
            job = TranscriptionJob(
                uid=uuid.uuid4().hex[:8],
                temp_dir=temp_dir,
                temp_file_path=temp_path,
                params=params,
                future=future,
                received_at=time.time()
            )

            # 4. 入队
            await self.queue.put(job)
            
            # 5. 等待处理结果 (Await the future)
            # 这里的 await 会挂起当前请求，直到后台 worker 完成处理
            result = await future
            return result

        except Exception as e:
            # 如果在入队前就失败了，确保清理临时目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise e

    async def _consume_loop(self):
        """
        消费者循环 (Strict Serial Execution)。
        这是保护 M4 Pro 显存的关键。
        """
        while self.is_running:
            # 从队列获取任务
            job: TranscriptionJob = await self.queue.get()
            
            try:
                # === 核心推理逻辑 ===
                # 根据 response_format 参数决定引擎返回格式
                format_param = job.params.get("response_format", "txt")
                
                # run_in_threadpool 是为了把同步的 Engine 代码放到线程池里跑
                # 防止阻塞 asyncio 的事件循环
                result_data = await run_in_threadpool(
                    self.engine.transcribe_file,
                    file_path=job.temp_file_path,
                    language=job.params.get("language", "auto"),
                    use_itn=True,
                    format=format_param  # 传递 format 参数给 MLX 引擎
                )

                # 处理返回值（可能是字符串或字典）
                if isinstance(result_data, dict):
                    # MLX 引擎返回了 JSON 格式（包含 segments）
                    raw_text = result_data.get("text", "")
                    # 调用适配器清洗文本（仅清理文本内容，不影响 segments）
                    clean_tags = job.params.get("clean_tags", True)
                    cleaned_text = clean_sensevoice_tags(raw_text, clean_tags=clean_tags)
                    
                    # 构造结果
                    process_time = time.time() - job.received_at
                    result = {
                        "text": cleaned_text,
                        "duration": process_time,
                        "raw_text": raw_text,
                        "is_cleaned": clean_tags,
                        "segments": result_data.get("segments")  # 透传 segments
                    }
                else:
                    # 文本格式返回（FunASR 或 MLX txt 格式）
                    raw_text = result_data
                    # 调用适配器清洗文本
                    clean_tags = job.params.get("clean_tags", True)
                    cleaned_text = clean_sensevoice_tags(raw_text, clean_tags=clean_tags)
                    
                    # 构造结果
                    process_time = time.time() - job.received_at
                    result = {
                        "text": cleaned_text,
                        "duration": process_time,
                        "raw_text": raw_text,
                        "is_cleaned": clean_tags
                    }
                
                # 唤醒等待的 API 请求
                if not job.future.done():
                    job.future.set_result(result)

            except Exception as e:
                self.logger.exception(f"❌ Job {job.uid} failed")
                if not job.future.done():
                    job.future.set_exception(e)
            
            finally:
                # === 打扫战场 ===
                # 无论成功失败，必须删除临时目录
                # 这会连带删除原始文件、归一化文件、切片文件等所有中间产物
                if os.path.exists(job.temp_dir):
                    shutil.rmtree(job.temp_dir, ignore_errors=True)
                
                # 标记队列任务完成
                self.queue.task_done()