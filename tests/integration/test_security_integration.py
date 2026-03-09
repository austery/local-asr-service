"""
Integration tests for security features (SPEC-006).
Tests CORS configuration, file cleanup on errors, and end-to-end security flow.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
from io import BytesIO
from src.core.base_engine import EngineCapabilities


def _make_mock_service(submit_result: object = None) -> MagicMock:
    from src.services.transcription import TranscriptionService

    if submit_result is None:
        submit_result = {"text": "Test transcription", "segments": None, "duration": 1.0}
    service = MagicMock(spec=TranscriptionService)
    type(service).capabilities = PropertyMock(
        return_value=EngineCapabilities(timestamp=True, diarization=True, language_detect=True)
    )
    service.current_model_spec = None
    service.submit = AsyncMock(return_value=submit_result)
    service.start_worker = AsyncMock()
    service.stop_worker = AsyncMock()
    type(service).queue_size = PropertyMock(return_value=0)
    type(service).max_queue_size = PropertyMock(return_value=50)
    return service


def create_test_app_with_cors(allowed_origins: str):
    """创建测试应用（指定 CORS 配置）"""
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from src.api.routes import router as api_router
    import uuid

    mock_service = _make_mock_service()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await mock_service.start_worker()
        app.state.service = mock_service
        app.state.engine_type = "funasr"
        app.state.model_id = "test-model"
        yield
        await mock_service.stop_worker()

    app = FastAPI(lifespan=lifespan)

    # 配置 CORS
    cors_origins = allowed_origins.split(",") if allowed_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 请求日志中间件
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(api_router)

    return app


class TestCORSConfiguration:
    """测试 CORS 配置"""
    
    def test_cors_default_local_only(self):
        """测试默认 CORS 仅允许本地访问"""
        app = create_test_app_with_cors("http://localhost,http://127.0.0.1")
        client = TestClient(app)
        
        # 模拟来自允许源的请求
        response = client.options(
            "/v1/audio/transcriptions",
            headers={
                "Origin": "http://localhost",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        # 验证：允许访问
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "http://localhost"
    
    def test_cors_blocks_external_origin_by_default(self):
        """测试默认 CORS 阻止外部源访问"""
        app = create_test_app_with_cors("http://localhost,http://127.0.0.1")
        client = TestClient(app)
        
        # 模拟来自外部源的请求
        response = client.options(
            "/v1/audio/transcriptions",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        # 验证：CORS 头不存在或不允许
        # (注意：FastAPI 的 CORSMiddleware 会返回 400 对于不允许的源)
        assert "access-control-allow-origin" not in response.headers or \
               response.headers.get("access-control-allow-origin") != "https://evil.com"
    
    def test_cors_wildcard_allows_all_origins(self):
        """测试 CORS 通配符允许所有源"""
        app = create_test_app_with_cors("*")
        with TestClient(app) as client:
            # 模拟来自任意源的请求
            response = client.options(
                "/v1/audio/transcriptions",
                headers={
                    "Origin": "https://any-origin.com",
                    "Access-Control-Request-Method": "POST"
                }
            )

            # 验证：允许访问
            assert response.status_code == 200
            assert "access-control-allow-origin" in response.headers
            # 注意：当 allow_credentials=True 时，Starlette 的 CORSMiddleware
            # 会回显具体的 Origin 而非 "*"（CORS 规范要求）
            assert response.headers["access-control-allow-origin"] in (
                "*", "https://any-origin.com"
            )


class TestFileCleanupOnError:
    """测试错误时的文件清理"""

    @pytest.mark.asyncio
    async def test_temp_file_cleanup_on_validation_error(self):
        """验证 submit() 在 worker spawn 失败时不留下临时目录。

        With the subprocess architecture, cleanup always happens in the
        `except BaseException` block inside submit(). We simulate a spawn
        failure by patching _spawn_worker to raise, which triggers the
        cleanup path without starting a real subprocess.
        """
        import os
        import tempfile
        from src.services.transcription import TranscriptionService
        from fastapi import UploadFile

        service = TranscriptionService(
            engine_type="mlx", model_id="test-model", max_queue_size=10
        )
        await service.start_worker()

        # Record baseline temp dirs
        initial_temp_dirs = set(os.listdir(tempfile.gettempdir()))

        file = UploadFile(filename="test.wav", file=BytesIO(b"fake audio"))

        # Make worker spawn fail — triggers the except BaseException cleanup path
        with patch.object(
            service, "_spawn_worker", side_effect=RuntimeError("simulated spawn failure")
        ):
            try:
                await service.submit(file, {"language": "auto"}, request_id="test-id")
            except Exception:
                pass

        # No asr_task_ temp dirs should remain
        final_temp_dirs = set(os.listdir(tempfile.gettempdir()))
        new_dirs = final_temp_dirs - initial_temp_dirs
        asr_temp_dirs = [d for d in new_dirs if d.startswith("asr_task_")]

        assert len(asr_temp_dirs) == 0, f"Temp directories not cleaned: {asr_temp_dirs}"


class TestRequestTracking:
    """测试请求追踪"""
    
    def test_request_id_in_response_header(self):
        """测试响应头包含 X-Request-ID"""
        app = create_test_app_with_cors("*")
        with TestClient(app) as client:
            # 创建测试文件
            files = {"file": ("test.wav", BytesIO(b"fake audio"), "audio/wav")}
            data = {
                "model": None,
                "language": "auto",
                "response_format": "json",
                "clean_tags": "true"
            }

            response = client.post("/v1/audio/transcriptions", files=files, data=data)

            # 验证：响应头包含 X-Request-ID
            assert response.status_code == 200
            assert "X-Request-ID" in response.headers
            assert len(response.headers["X-Request-ID"]) > 0


class TestEndToEndSecurityFlow:
    """端到端安全流程测试"""
    
    def test_secure_request_lifecycle(self):
        """测试安全的完整请求生命周期"""
        app = create_test_app_with_cors("http://localhost")
        with TestClient(app) as client:
            # 创建测试文件（合法大小、合法类型）
            files = {"file": ("test.wav", BytesIO(b"fake audio"), "audio/wav")}
            data = {
                "model": None,
                "language": "auto",
                "response_format": "json",
                "clean_tags": "true"
            }

            # 发送请求
            response = client.post(
                "/v1/audio/transcriptions",
                files=files,
                data=data,
                headers={"Origin": "http://localhost"}
            )

            # 验证：请求成功
            assert response.status_code == 200

            # 验证：包含必要的安全头
            assert "X-Request-ID" in response.headers
            assert "access-control-allow-origin" in response.headers

            # 验证：响应不泄露内部信息
            response_data = response.json()
            assert "text" in response_data
            assert "duration" in response_data
    
    def test_blocked_by_file_size_limit(self):
        """测试文件大小限制阻止请求"""
        with patch("src.api.routes.MAX_UPLOAD_SIZE_MB", 1):  # 设置为 1 MB
            app = create_test_app_with_cors("*")
            client = TestClient(app)
            
            # 创建超大文件 (2 MB)
            large_content = b"a" * (2 * 1024 * 1024)
            files = {"file": ("large.wav", BytesIO(large_content), "audio/wav")}
            data = {
                "model": None,
                "language": "auto",
                "response_format": "json",
                "clean_tags": "true"
            }
            
            # 发送请求
            response = client.post("/v1/audio/transcriptions", files=files, data=data)
            
            # 验证：返回 413
            assert response.status_code == 413
            assert "File size exceeds" in response.json()["detail"]
    
    def test_blocked_by_mime_type_validation(self):
        """测试 MIME 类型校验阻止请求"""
        app = create_test_app_with_cors("*")
        client = TestClient(app)
        
        # 创建非音频文件
        files = {"file": ("test.png", BytesIO(b"fake image"), "image/png")}
        data = {
            "model": None,
            "language": "auto",
            "response_format": "json",
            "clean_tags": "true"
        }
        
        # 发送请求
        response = client.post("/v1/audio/transcriptions", files=files, data=data)
        
        # 验证：返回 415
        assert response.status_code == 415
        assert "Unsupported file type" in response.json()["detail"]
