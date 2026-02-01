"""
Unit tests for security features (SPEC-006).
Tests file size limits, MIME type validation, and error message sanitization.
Updated for SPEC-007 API changes (removed clean_tags, added output_format).
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock
from fastapi import HTTPException
from io import BytesIO


class TestFileSizeLimit:
    """测试文件大小限制"""
    
    @pytest.mark.asyncio
    async def test_file_size_within_limit(self):
        """测试正常大小文件可以上传"""
        from src.api.routes import create_transcription
        
        # Mock 小文件 (1MB)
        content = b"a" * (1024 * 1024)  # 1 MB
        file = MagicMock()
        file.filename = "test.wav"
        file.file = BytesIO(content)
        file.read = AsyncMock(return_value=content)
        file.seek = AsyncMock()
        type(file).content_type = PropertyMock(return_value="audio/wav")
        
        # Mock request
        request = MagicMock()
        request.state.request_id = "test-request-id"
        request.app.state.service.submit = AsyncMock(return_value={
            "text": "test", 
            "duration": 1.0,
            "segments": None
        })
        request.app.state.model_id = "test-model"
        
        # Execute - 不应抛出异常
        result = await create_transcription(
            request=request,
            file=file,
            model="test",
            language="auto",
            output_format="json",
            with_timestamp=False
        )
        
        assert result.text == "test"
    
    @pytest.mark.asyncio
    async def test_file_size_exceeds_limit(self):
        """测试超大文件返回 413"""
        from src.api.routes import create_transcription
        
        # Mock 超大文件 (201MB, 超过默认 200MB 限制)
        with patch("src.api.routes.MAX_UPLOAD_SIZE_MB", 200):
            content = b"a" * (201 * 1024 * 1024)  # 201 MB
            file = MagicMock()
            file.filename = "large.wav"
            file.file = BytesIO(content)
            file.read = AsyncMock(return_value=content)
            file.seek = AsyncMock()
            type(file).content_type = PropertyMock(return_value="audio/wav")
            
            # Mock request
            request = MagicMock()
            request.state.request_id = "test-request-id"
            
            # Execute - 应抛出 413 HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await create_transcription(
                    request=request,
                    file=file,
                    model="test",
                    language="auto",
                    output_format="json",
                    with_timestamp=False
                )
            
            assert exc_info.value.status_code == 413
            assert "File size exceeds" in exc_info.value.detail


class TestMIMETypeValidation:
    """测试 MIME 类型校验"""
    
    @pytest.mark.asyncio
    async def test_valid_audio_mime_types(self):
        """测试所有支持的音频 MIME 类型"""
        from src.api.routes import create_transcription, ALLOWED_AUDIO_TYPES
        
        for mime_type in ALLOWED_AUDIO_TYPES:
            content = b"fake audio data"
            file = MagicMock()
            file.filename = "test.wav"
            file.file = BytesIO(content)
            file.read = AsyncMock(return_value=content)
            file.seek = AsyncMock()
            type(file).content_type = PropertyMock(return_value=mime_type)
            
            # Mock request
            request = MagicMock()
            request.state.request_id = "test-request-id"
            request.app.state.service.submit = AsyncMock(return_value={
                "text": "test", 
                "duration": 1.0,
                "segments": None
            })
            request.app.state.model_id = "test-model"
            
            # Execute - 不应抛出异常
            result = await create_transcription(
                request=request,
                file=file,
                model="test",
                language="auto",
                output_format="json",
                with_timestamp=False
            )
            
            assert result.text == "test"
    
    @pytest.mark.asyncio
    async def test_invalid_mime_type_returns_415(self):
        """测试非音频文件返回 415"""
        from src.api.routes import create_transcription
        
        # 测试不同的非音频类型
        invalid_types = [
            "image/png",
            "video/mp4",
            "application/pdf",
            "text/plain",
            "application/octet-stream"
        ]
        
        for mime_type in invalid_types:
            content = b"fake data"
            file = MagicMock()
            file.filename = "test.png"
            file.file = BytesIO(content)
            file.read = AsyncMock(return_value=content)
            file.seek = AsyncMock()
            type(file).content_type = PropertyMock(return_value=mime_type)
            
            # Mock request
            request = MagicMock()
            request.state.request_id = "test-request-id"
            
            # Execute - 应抛出 415 HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await create_transcription(
                    request=request,
                    file=file,
                    model="test",
                    language="auto",
                    output_format="json",
                    with_timestamp=False
                )
            
            assert exc_info.value.status_code == 415, f"Failed for {mime_type}"
            assert "Unsupported file type" in exc_info.value.detail


class TestErrorMessageSanitization:
    """测试错误信息最小化"""
    
    @pytest.mark.asyncio
    async def test_runtime_error_does_not_leak_details(self):
        """测试 RuntimeError 不泄露内部细节"""
        from src.api.routes import create_transcription
        
        content = b"fake audio data"
        file = MagicMock()
        file.filename = "test.wav"
        file.file = BytesIO(content)
        file.read = AsyncMock(return_value=content)
        file.seek = AsyncMock()
        type(file).content_type = PropertyMock(return_value="audio/wav")
        
        # Mock request
        request = MagicMock()
        request.state.request_id = "test-request-id"
        
        # Mock service 抛出 RuntimeError（包含敏感信息）
        sensitive_msg = "Internal error: /Users/admin/.cache/model.bin failed to load"
        request.app.state.service.submit = AsyncMock(
            side_effect=RuntimeError(sensitive_msg)
        )
        
        # Execute
        with pytest.raises(HTTPException) as exc_info:
            await create_transcription(
                request=request,
                file=file,
                model="test",
                language="auto",
                output_format="json",
                with_timestamp=False
            )
        
        # 验证：错误信息不包含敏感内容
        assert exc_info.value.status_code == 500
        assert sensitive_msg not in exc_info.value.detail
        assert "Internal server error" in exc_info.value.detail
        assert "test-request-id" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_generic_exception_does_not_leak_stack_trace(self):
        """测试通用异常不泄露堆栈信息"""
        from src.api.routes import create_transcription
        
        content = b"fake audio data"
        file = MagicMock()
        file.filename = "test.wav"
        file.file = BytesIO(content)
        file.read = AsyncMock(return_value=content)
        file.seek = AsyncMock()
        type(file).content_type = PropertyMock(return_value="audio/wav")
        
        # Mock request
        request = MagicMock()
        request.state.request_id = "test-request-id"
        
        # Mock service 抛出通用异常（包含堆栈）
        request.app.state.service.submit = AsyncMock(
            side_effect=Exception("KeyError: 'model' in file /src/engine.py line 42")
        )
        
        # Execute
        with pytest.raises(HTTPException) as exc_info:
            await create_transcription(
                request=request,
                file=file,
                model="test",
                language="auto",
                output_format="json",
                with_timestamp=False
            )
        
        # 验证：错误信息不包含堆栈信息
        assert exc_info.value.status_code == 500
        assert "/src/engine.py" not in exc_info.value.detail
        assert "Internal server error" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_queue_full_error_preserves_specific_message(self):
        """测试队列满时仍返回明确的 503 错误"""
        from src.api.routes import create_transcription
        
        content = b"fake audio data"
        file = MagicMock()
        file.filename = "test.wav"
        file.file = BytesIO(content)
        file.read = AsyncMock(return_value=content)
        file.seek = AsyncMock()
        type(file).content_type = PropertyMock(return_value="audio/wav")
        
        # Mock request
        request = MagicMock()
        request.state.request_id = "test-request-id"
        
        # Mock service 抛出队列满错误
        request.app.state.service.submit = AsyncMock(
            side_effect=RuntimeError("Service busy: Queue is full.")
        )
        
        # Execute
        with pytest.raises(HTTPException) as exc_info:
            await create_transcription(
                request=request,
                file=file,
                model="test",
                language="auto",
                output_format="json",
                with_timestamp=False
            )
        
        # 验证：503 状态码与明确的队列满消息
        assert exc_info.value.status_code == 503
        assert "Queue Full" in exc_info.value.detail


class TestRequestIDGeneration:
    """测试请求 ID 生成与传递"""
    
    @pytest.mark.asyncio
    async def test_request_id_generated_and_passed_to_service(self):
        """测试 request_id 被生成并传递给服务"""
        from src.api.routes import create_transcription
        
        content = b"fake audio data"
        file = MagicMock()
        file.filename = "test.wav"
        file.file = BytesIO(content)
        file.read = AsyncMock(return_value=content)
        file.seek = AsyncMock()
        type(file).content_type = PropertyMock(return_value="audio/wav")
        
        # Mock request
        request = MagicMock()
        expected_request_id = "unique-request-id-12345"
        request.state.request_id = expected_request_id
        
        # Mock service
        submit_mock = AsyncMock(return_value={
            "text": "test", 
            "duration": 1.0,
            "segments": None
        })
        request.app.state.service.submit = submit_mock
        request.app.state.model_id = "test-model"
        
        # Execute
        await create_transcription(
            request=request,
            file=file,
            model="test",
            language="auto",
            output_format="json",
            with_timestamp=False
        )
        
        # 验证：submit 被调用且传入了正确的 request_id
        submit_mock.assert_called_once()
        call_args = submit_mock.call_args
        assert call_args.kwargs["request_id"] == expected_request_id
