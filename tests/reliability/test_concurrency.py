import pytest
import asyncio
import httpx
from src.services.transcription import TranscriptionService
from src.core.engine import SenseVoiceEngine
from unittest.mock import MagicMock

@pytest.mark.asyncio
class TestReliability:
    """
    可靠性测试：专注于并发、压力和错误恢复
    """

    async def test_queue_backpressure(self):
        """
        测试高并发下的背压机制 (Backpressure)。
        当队列满 (50) 时，第 51 个请求应该被拒绝。
        """
        # 1. Setup Service with Mock Engine
        mock_engine = MagicMock()
        # 让推理稍微慢一点 (0.1s)，模拟真实负载，这样队列才会积压
        mock_engine.transcribe_file.side_effect = lambda *args, **kwargs: "ok"
        
        # 设定一个小一点的队列，方便测试 (比如 size=5)
        service = TranscriptionService(engine=mock_engine, max_queue_size=5)
        
        # 我们不启动 worker，这样任务只进不出，必然会满
        # service.start_worker() 
        
        # 2. 填满队列
        for i in range(5):
            await service.queue.put(f"job_{i}")
            
        assert service.queue.full()
        
        # 3. 尝试提交第 6 个任务 -> 应该报错
        # 模拟 UploadFile
        mock_file = MagicMock()
        mock_file.filename = "test.wav"
        
        try:
            with pytest.raises(RuntimeError, match="Queue is full"):
                await service.submit(mock_file, {})
        finally:
            # 清理：把队列里的东西拿出来，防止报错
            while not service.queue.empty():
                service.queue.get_nowait()
                service.queue.task_done()

    async def test_worker_recovery(self):
        """
        测试 Worker 在遇到致命错误后的恢复能力。
        (虽然我们在 Unit Test 测过，这里模拟更复杂的连续失败场景)
        """
        mock_engine = MagicMock()
        service = TranscriptionService(engine=mock_engine, max_queue_size=10)
        
        # 模拟前 3 次失败，第 4 次成功
        mock_engine.transcribe_file.side_effect = [
            ValueError("Fail 1"),
            RuntimeError("Fail 2"),
            Exception("Fail 3"),
            "Success"
        ]
        
        service.is_running = True
        worker_task = asyncio.create_task(service._consume_loop())
        
        try:
            # 提交 4 个任务
            futures = []
            for i in range(4):
                # 我们需要 mock submit 内部逻辑，或者直接往 queue 里塞 job
                # 为了简单，我们直接构造 Job 塞进去
                from src.services.transcription import TranscriptionJob
                import time
                
                fut = asyncio.get_running_loop().create_future()
                job = TranscriptionJob(
                    uid=f"job_{i}",
                    temp_file_path=f"dummy_{i}.wav", # 文件不存在没关系，mock engine 不会读
                    params={},
                    future=fut,
                    received_at=time.time()
                )
                # Mock os.remove 防止报错
                await service.queue.put(job)

                futures.append(fut)
            
            # 等待所有任务完成（无论成功失败）
            results = []
            for fut in futures:
                try:
                    res = await fut
                    results.append("OK")
                except Exception:
                    results.append("ERR")
            
            # 验证结果：前3个失败，第4个成功
            assert results == ["ERR", "ERR", "ERR", "OK"]
            
            # 验证 Worker 依然健在
            assert not worker_task.done()
            
        finally:
            service.is_running = False
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
