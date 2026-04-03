import asyncio
import multiprocessing
from io import BytesIO
from unittest.mock import MagicMock, patch

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
class TestReliability:
    """
    可靠性测试：专注于并发、压力和错误恢复
    """

    async def test_queue_backpressure(self, funasr_spec):
        """
        测试高并发下的背压机制 (Backpressure)。
        当队列满 (50) 时，第 51 个请求应该被拒绝。
        """
        service = TranscriptionService(
            engine_type=funasr_spec.engine_type,
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
            max_queue_size=5,
            idle_timeout=0,
        )

        # 新架构使用 _pending 追踪 in-flight 请求数量，而非 asyncio.Queue。
        loop = asyncio.get_running_loop()
        for i in range(5):
            service._pending[f"job_{i}"] = loop.create_future()

        with pytest.raises(RuntimeError, match="Queue is full"):
            await service.submit(_make_upload(), {}, request_id="job_overflow")

    async def test_worker_recovery(self, funasr_spec):
        """
        测试 Worker 在遇到致命错误后的恢复能力。
        (虽然我们在 Unit Test 测过，这里模拟更复杂的连续失败场景)
        """
        service = TranscriptionService(
            engine_type=funasr_spec.engine_type,
            model_id=funasr_spec.model_id,
            initial_model_spec=funasr_spec,
            max_queue_size=10,
            idle_timeout=0,
        )
        service.is_running = True
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        service._worker = mock_proc
        service._job_queue = multiprocessing.Queue()
        service._result_queue = multiprocessing.Queue()

        async def _deliver_results() -> None:
            await asyncio.sleep(0.05)
            service._result_queue.put(("ERROR", "req-1", "Fail 1"))
            await asyncio.sleep(0.05)
            service._result_queue.put((
                "RESULT",
                "req-2",
                {"text": "Success", "segments": None, "duration": 1.0},
            ))

        async def _fake_spawn(model_spec=None) -> None:
            return None

        service._result_reader_task = asyncio.create_task(service._result_reader_loop())
        asyncio.create_task(_deliver_results())

        try:
            with patch.object(service, "_spawn_worker", side_effect=_fake_spawn):
                with pytest.raises(RuntimeError, match="Fail 1"):
                    await asyncio.wait_for(
                        service.submit(_make_upload(), {}, request_id="req-1"),
                        timeout=5.0,
                    )

                result = await asyncio.wait_for(
                    service.submit(_make_upload(), {}, request_id="req-2"),
                    timeout=5.0,
                )
                assert service.model_loaded is True
        finally:
            await service.stop_worker()

        assert result["text"] == "Success"
