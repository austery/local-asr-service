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
        """Mock FunASR Engine"""
        engine = MagicMock()
        # transcribe_file 是同步方法，但在 service 中被 run_in_threadpool 调用
        # 返回 FunASR 格式的结构化数据
        engine.transcribe_file.return_value = {
            "text": "Mocked Transcription",
            "segments": [
                {"speaker": "Speaker 0", "text": "Mocked", "start": 0, "end": 500},
                {"speaker": "Speaker 0", "text": "Transcription", "start": 500, "end": 1000}
            ]
        }
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
            params = {"language": "zh", "output_format": "json"}
            result = await service.submit(mock_upload_file, params)
            
            # 3. 验证结果 (FunASR 返回格式)
            assert result["text"] == "Mocked Transcription"
            assert "duration" in result
            assert "segments" in result
            
            # 4. 验证 Engine 调用
            service.engine.transcribe_file.assert_called_once()
            
        finally:
            # 5. 清理 Worker
            service.is_running = False
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    async def test_submit_txt_format(self, service, mock_upload_file):
        """测试 txt 格式输出"""
        # Mock Engine 返回 txt 格式字符串
        service.engine.transcribe_file.return_value = "[Speaker 0]: Mocked Transcription"
        
        service.is_running = True
        worker_task = asyncio.create_task(service._consume_loop())
        
        try:
            params = {"language": "zh", "output_format": "txt"}
            result = await service.submit(mock_upload_file, params)
            
            # txt 格式直接返回字符串
            assert result == "[Speaker 0]: Mocked Transcription"
            
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
            return {"text": "test", "segments": None}
            
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
