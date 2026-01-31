import pytest
import asyncio
import os
from unittest.mock import MagicMock, AsyncMock
from io import BytesIO
from fastapi import UploadFile
from src.services.transcription import TranscriptionService

# 使用 pytest-asyncio 处理异步测试
@pytest.mark.asyncio
class TestTranscriptionService:
    
    @pytest.fixture
    def mock_engine(self):
        """Mock SenseVoiceEngine"""
        engine = MagicMock()
        # transcribe_file 是同步方法，但在 service 中被 run_in_threadpool 调用
        # 我们只需要 mock 它的返回值
        engine.transcribe_file.return_value = "Mocked Transcription"
        return engine

    @pytest.fixture
    def service(self, mock_engine):
        """初始化 Service，队列设小一点方便测试"""
        svc = TranscriptionService(engine=mock_engine, max_queue_size=2)
        return svc

    @pytest.fixture
    def mock_upload_file(self):
        """Mock FastAPI UploadFile"""
        file_content = b"fake audio content"
        file_obj = BytesIO(file_content)
        return UploadFile(file=file_obj, filename="test.wav")

    async def test_submit_success(self, service, mock_upload_file):
        """测试正常提交和处理流程"""
        # 1. 启动 Worker (后台运行)
        service.is_running = True
        worker_task = asyncio.create_task(service._consume_loop())
        
        try:
            # 2. 提交任务
            params = {"language": "zh"}
            result = await service.submit(mock_upload_file, params)
            
            # 3. 验证结果
            assert result["text"] == "Mocked Transcription"
            assert "duration" in result
            assert "raw_text" in result  # 新增：验证 raw_text 字段
            assert "is_cleaned" in result  # 新增：验证 is_cleaned 字段
            assert result["is_cleaned"] is True  # 默认应该清理
            
            # 4. 验证 Engine 调用
            service.engine.transcribe_file.assert_called_once()
            
        finally:
            # 5. 清理 Worker
            service.is_running = False
            # 发送一个空任务或者直接 cancel，这里直接 cancel 比较简单
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    async def test_submit_with_clean_tags_false(self, service, mock_upload_file):
        """测试 clean_tags=false 的情况"""
        service.is_running = True
        worker_task = asyncio.create_task(service._consume_loop())
        
        try:
            # 提交任务，明确设置 clean_tags=False
            params = {"language": "zh", "clean_tags": False}
            result = await service.submit(mock_upload_file, params)
            
            # 验证结果
            assert result["text"] == "Mocked Transcription"  # Mock 返回的是已清理的文本
            assert result["is_cleaned"] is False  # 应该标记为未清理
            assert result["raw_text"] == "Mocked Transcription"
            
        finally:
            service.is_running = False
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass


    async def test_queue_full(self, service, mock_upload_file):
        """测试队列满时的拒绝策略"""
        # 1. 填满队列 (max_size=2)
        # 我们不启动 worker，所以任务会堆积
        await service.queue.put("job1")
        await service.queue.put("job2")
        
        # 2. 尝试提交第三个任务
        with pytest.raises(RuntimeError, match="Queue is full"):
            await service.submit(mock_upload_file, {})

    async def test_temp_file_lifecycle(self, service, mock_upload_file):
        """测试临时文件的创建与删除"""
        # 1. 启动 Worker
        service.is_running = True
        worker_task = asyncio.create_task(service._consume_loop())
        
        # 2. 提交任务
        # 我们需要拦截 engine 调用来检查文件是否存在
        original_transcribe = service.engine.transcribe_file
        
        captured_path = None
        def side_effect(file_path, **kwargs):
            nonlocal captured_path
            captured_path = file_path
            # 此时文件应该存在
            assert os.path.exists(file_path)
            return "text"
            
        service.engine.transcribe_file.side_effect = side_effect
        
        try:
            await service.submit(mock_upload_file, {})
            
            # 3. 任务完成后，文件应该不存在了
            assert captured_path is not None
            assert not os.path.exists(captured_path)
            
        finally:
            service.is_running = False
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    async def test_worker_error_handling(self, service, mock_upload_file):
        """测试 Worker 遇到异常时的行为"""
        service.is_running = True
        worker_task = asyncio.create_task(service._consume_loop())
        
        # 让 Engine 抛出异常
        service.engine.transcribe_file.side_effect = ValueError("Model Error")
        
        try:
            # submit 应该抛出这个异常
            with pytest.raises(ValueError, match="Model Error"):
                await service.submit(mock_upload_file, {})
                
            # Worker 应该还活着 (没有 crash)
            assert not worker_task.done()
            
        finally:
            service.is_running = False
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
