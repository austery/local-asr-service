"""
Unit tests for TranscriptionService subprocess management (SPEC-009 v2).
"""
import asyncio
import queue as _stdlib_queue
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile

from src.core.model_registry import lookup
from src.services.transcription import TranscriptionService


def _make_upload() -> UploadFile:
    return UploadFile(file=BytesIO(b"audio"), filename="test.wav")


@pytest.fixture
def funasr_spec():
    return lookup("paraformer")


@pytest.mark.asyncio
class TestTranscriptionServiceSubprocess:

    async def test_model_loaded_false_when_no_worker(self, funasr_spec):
        """model_loaded is False when no worker process exists."""
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
        )
        assert service.model_loaded is False

    async def test_model_loaded_true_when_worker_alive(self, funasr_spec):
        """model_loaded is True when worker process is running."""
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
        )
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        service._worker = mock_proc
        assert service.model_loaded is True

    async def test_capabilities_from_model_spec(self, funasr_spec):
        """capabilities returns current_model_spec.capabilities without an engine."""
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
        )
        assert service.capabilities == funasr_spec.capabilities

    async def test_submit_resolves_future_via_result_queue(self, funasr_spec):
        """submit() resolves when RESULT arrives on result_queue."""
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
            idle_timeout=0,
        )
        service.is_running = True

        # Inject mock worker infrastructure
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        service._worker = mock_proc

        import multiprocessing
        service._job_queue = multiprocessing.Queue()
        service._result_queue = multiprocessing.Queue()

        # Mock _spawn_worker so submit() doesn't actually spawn a process
        async def _fake_spawn(model_spec=None):
            pass

        expected_result = {"text": "hello", "segments": None, "duration": 1.0}

        async def _deliver_result():
            await asyncio.sleep(0.05)
            service._result_queue.put(("RESULT", "req-1", expected_result))

        service._result_reader_task = asyncio.create_task(service._result_reader_loop())
        asyncio.create_task(_deliver_result())

        with patch.object(service, "_spawn_worker", side_effect=_fake_spawn):
            result = await service.submit(_make_upload(), {"language": "auto", "output_format": "json"}, request_id="req-1")

        assert result == expected_result
        await service.stop_worker()

    async def test_submit_raises_on_worker_error(self, funasr_spec):
        """submit() raises RuntimeError when ERROR arrives on result_queue."""
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
            idle_timeout=0,
        )
        service.is_running = True

        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        service._worker = mock_proc

        import multiprocessing
        service._job_queue = multiprocessing.Queue()
        service._result_queue = multiprocessing.Queue()

        async def _fake_spawn(model_spec=None):
            pass

        async def _deliver_error():
            await asyncio.sleep(0.05)
            service._result_queue.put(("ERROR", "req-err", "GPU exploded"))

        service._result_reader_task = asyncio.create_task(service._result_reader_loop())
        asyncio.create_task(_deliver_error())

        with patch.object(service, "_spawn_worker", side_effect=_fake_spawn):
            with pytest.raises(RuntimeError, match="GPU exploded"):
                await service.submit(_make_upload(), {"language": "auto"}, request_id="req-err")

        await service.stop_worker()

    async def test_idle_exit_clears_worker_reference(self, funasr_spec):
        """IDLE_EXIT message causes worker reference to be cleared."""
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
        )
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        service._worker = mock_proc
        service.is_running = True

        import multiprocessing
        result_q = multiprocessing.Queue()
        result_q.put(("IDLE_EXIT", None))
        service._result_queue = result_q

        task = asyncio.create_task(service._result_reader_loop())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert service._worker is None

    async def test_switch_worker_called_for_different_model(self, funasr_spec):
        """submit() calls _switch_worker when model_spec differs from current."""
        mlx_spec = lookup("qwen3-asr")
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
            idle_timeout=0,
        )
        service.is_running = True

        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        service._worker = mock_proc

        import multiprocessing
        service._job_queue = multiprocessing.Queue()
        service._result_queue = multiprocessing.Queue()

        async def _fake_switch(spec):
            service._current_model_spec = spec

        async def _deliver_result():
            await asyncio.sleep(0.05)
            service._result_queue.put(("RESULT", "req-switch", {"text": "ok", "segments": None, "duration": 1.0}))

        service._result_reader_task = asyncio.create_task(service._result_reader_loop())
        asyncio.create_task(_deliver_result())

        with patch.object(service, "_switch_worker", side_effect=_fake_switch) as mock_switch:
            await service.submit(_make_upload(), {"language": "auto", "output_format": "json"}, request_id="req-switch", model_spec=mlx_spec)
            mock_switch.assert_called_once_with(mlx_spec)

        await service.stop_worker()

    async def test_queue_full_raises(self, funasr_spec):
        """submit() raises when pending queue is at max capacity."""
        service = TranscriptionService(
            engine_type="funasr",
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
            max_queue_size=2,
        )
        # Fill pending manually
        loop = asyncio.get_running_loop()
        service._pending["a"] = loop.create_future()
        service._pending["b"] = loop.create_future()

        with pytest.raises(RuntimeError, match="Queue is full"):
            await service.submit(_make_upload(), {}, request_id="c")


