"""Unit tests for TranscriptionService core behaviors (SPEC-009).

Tests: success result delivery, txt format, queue full, temp file cleanup, error handling.
Uses injected mock worker infrastructure (no real subprocess spawned).
"""
import asyncio
import multiprocessing
import os
import tempfile as _tempfile
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi import UploadFile

from src.core.model_registry import lookup
from src.services.transcription import TranscriptionService


@pytest.fixture
def funasr_spec():
    return lookup("paraformer")


def _make_upload() -> UploadFile:
    return UploadFile(file=BytesIO(b"fake audio content"), filename="test.wav")


def _setup_service(spec, max_queue_size: int = 2) -> TranscriptionService:
    """Create a service with injected mock worker — no subprocess spawned."""
    svc = TranscriptionService(
        engine_type=spec.engine_type,
        model_id=spec.model_id,
        max_queue_size=max_queue_size,
        initial_model_spec=spec,
        idle_timeout=0,
    )
    svc.is_running = True
    mock_proc = MagicMock()
    mock_proc.is_alive.return_value = True
    svc._worker = mock_proc
    svc._job_queue = multiprocessing.Queue()
    svc._result_queue = multiprocessing.Queue()
    return svc


async def _stop_service(svc: TranscriptionService) -> None:
    svc.is_running = False
    if svc._result_reader_task and not svc._result_reader_task.done():
        svc._result_reader_task.cancel()
        try:
            await svc._result_reader_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
class TestTranscriptionService:

    async def test_submit_success(self, funasr_spec):
        """RESULT message from worker resolves the submit() future with correct data."""
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())
        expected = {"text": "Mocked Transcription", "segments": [], "duration": 1.0}

        async def deliver() -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("RESULT", "req-1", expected))

        asyncio.create_task(deliver())
        try:
            result = await asyncio.wait_for(
                svc.submit(_make_upload(), {"language": "zh", "output_format": "json"}, request_id="req-1"),
                timeout=5.0,
            )
        finally:
            await _stop_service(svc)

        assert result["text"] == "Mocked Transcription"
        assert "segments" in result

    async def test_submit_txt_format(self, funasr_spec):
        """Plain-text result (str) is returned as-is from the worker."""
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())

        async def deliver() -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("RESULT", "req-2", "[Speaker 0]: Mocked Transcription"))

        asyncio.create_task(deliver())
        try:
            result = await asyncio.wait_for(
                svc.submit(_make_upload(), {"output_format": "txt"}, request_id="req-2"),
                timeout=5.0,
            )
        finally:
            await _stop_service(svc)

        assert result == "[Speaker 0]: Mocked Transcription"

    async def test_queue_full(self, funasr_spec):
        """submit() raises RuntimeError immediately when pending dict is at capacity."""
        svc = _setup_service(funasr_spec, max_queue_size=2)
        loop = asyncio.get_running_loop()
        svc._pending["x"] = loop.create_future()
        svc._pending["y"] = loop.create_future()

        with pytest.raises(RuntimeError, match="Queue is full"):
            await svc.submit(_make_upload(), {})

    async def test_temp_file_lifecycle(self, funasr_spec):
        """Temp directory is created before the job and deleted after result arrives."""
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())

        created_dirs: list[str] = []
        original_mkdtemp = _tempfile.mkdtemp

        def capture_mkdtemp(*args: object, **kwargs: object) -> str:
            path = original_mkdtemp(*args, **kwargs)
            created_dirs.append(path)
            return path

        async def deliver() -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("RESULT", "req-3", {"text": "ok", "segments": None, "duration": 0.5}))

        asyncio.create_task(deliver())
        try:
            with patch("src.services.transcription.tempfile.mkdtemp", side_effect=capture_mkdtemp):
                await asyncio.wait_for(
                    svc.submit(_make_upload(), {}, request_id="req-3"),
                    timeout=5.0,
                )
        finally:
            await _stop_service(svc)

        assert len(created_dirs) == 1, "Expected exactly one temp dir to be created"
        assert not os.path.exists(created_dirs[0]), "Temp dir must be deleted after job completes"

    async def test_worker_error_handling(self, funasr_spec):
        """ERROR message from worker raises RuntimeError in submit()."""
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())

        async def deliver() -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("ERROR", "req-4", "Model Error"))

        asyncio.create_task(deliver())
        try:
            with pytest.raises(RuntimeError, match="Model Error"):
                await asyncio.wait_for(
                    svc.submit(_make_upload(), {}, request_id="req-4"),
                    timeout=5.0,
                )
        finally:
            await _stop_service(svc)
