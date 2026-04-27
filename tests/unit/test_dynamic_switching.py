"""Unit tests for dynamic model switching in TranscriptionService (SPEC-108).

Cases preserved (adapted to subprocess architecture):
  DS-1: Switch triggered when different model_spec submitted
  DS-2: No switch when same model_spec submitted
  DS-3: Result after switch is from the correct (new) model
  DS-6: Temp directory cleaned up even when switch fails

Release/load ordering tests removed — those invariants now live inside the
worker subprocess and are covered by test_worker.py.
"""
import asyncio
import multiprocessing
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile

from src.core.model_registry import lookup
from src.services.transcription import TranscriptionService


@pytest.fixture
def mlx_spec():
    return lookup("qwen3-asr")


@pytest.fixture
def funasr_spec():
    return lookup("paraformer")


def _make_upload() -> UploadFile:
    return UploadFile(file=BytesIO(b"fake audio"), filename="test.wav")


def _setup_service(spec) -> TranscriptionService:
    """Create a service with injected mock worker — no subprocess spawned."""
    svc = TranscriptionService(
        engine_type=spec.engine_type,
        model_id=spec.model_id,
        max_queue_size=5,
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
class TestSameModelRequests:
    # DS-2: consecutive same-model requests must not re-trigger _switch_worker
    async def test_should_return_result_when_same_model_requested_twice(
        self, funasr_spec
    ) -> None:
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())

        async def deliver(uid: str, result: object) -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("RESULT", uid, result))

        with patch.object(svc, "_switch_worker", new_callable=AsyncMock) as mock_switch:
            asyncio.create_task(deliver("req-1", {"text": "hello", "segments": None, "duration": 1.0}))
            r1 = await asyncio.wait_for(
                svc.submit(_make_upload(), {}, request_id="req-1", model_spec=funasr_spec),
                timeout=5.0,
            )
            asyncio.create_task(deliver("req-2", {"text": "world", "segments": None, "duration": 1.0}))
            r2 = await asyncio.wait_for(
                svc.submit(_make_upload(), {}, request_id="req-2", model_spec=funasr_spec),
                timeout=5.0,
            )

        await _stop_service(svc)

        mock_switch.assert_not_called()
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)


@pytest.mark.asyncio
class TestModelSwitching:
    # DS-1: _switch_worker must be called when a different model_spec is requested
    async def test_switch_triggered_for_different_model(
        self, funasr_spec, mlx_spec
    ) -> None:
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())

        async def fake_switch(spec: object) -> None:
            svc._current_model_spec = spec  # type: ignore[assignment]

        async def deliver(uid: str, result: object) -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("RESULT", uid, result))

        with patch.object(svc, "_switch_worker", side_effect=fake_switch) as mock_switch:
            asyncio.create_task(deliver("req-1", {"text": "hello", "segments": None, "duration": 1.0}))
            await asyncio.wait_for(
                svc.submit(_make_upload(), {}, request_id="req-1", model_spec=mlx_spec),
                timeout=5.0,
            )
            mock_switch.assert_called_once_with(mlx_spec)

        await _stop_service(svc)

    # DS-3: result returned after switch must come from the new model
    async def test_result_after_switch_is_correct(
        self, funasr_spec, mlx_spec
    ) -> None:
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())
        expected = {"text": "switched result", "segments": None, "duration": 2.0}

        async def fake_switch(spec: object) -> None:
            svc._current_model_spec = spec  # type: ignore[assignment]

        async def deliver(uid: str, result: object) -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("RESULT", uid, result))

        with patch.object(svc, "_switch_worker", side_effect=fake_switch):
            asyncio.create_task(deliver("req-1", expected))
            result = await asyncio.wait_for(
                svc.submit(_make_upload(), {}, request_id="req-1", model_spec=mlx_spec),
                timeout=5.0,
            )

        await _stop_service(svc)

        assert result == expected
        assert svc.current_model_spec == mlx_spec, (
            "_current_model_spec must be updated atomically after a successful switch"
        )

    # DS-6: temp directory is cleaned up even when _switch_worker raises
    async def test_temp_file_cleaned_up_when_switch_fails(
        self, funasr_spec, mlx_spec
    ) -> None:
        svc = _setup_service(funasr_spec)

        async def failing_switch(spec: object) -> None:
            raise RuntimeError("switch failed")

        with patch.object(svc, "_switch_worker", side_effect=failing_switch):
            with pytest.raises(RuntimeError, match="switch failed"):
                await svc.submit(_make_upload(), {}, request_id="req-1", model_spec=mlx_spec)

        assert len(svc._temp_dirs) == 0, "All temp dirs must be cleaned up after a failed switch"
        assert "req-1" not in svc._pending, "Pending future must be removed after failure"

    # DS-5: Service stays operational after a failed model switch
    async def test_service_recovers_after_failed_switch(
        self, funasr_spec, mlx_spec
    ) -> None:
        """DS-5: Service stays operational after a failed model switch."""
        svc = _setup_service(funasr_spec)
        svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())

        call_count = {"n": 0}

        async def sometimes_failing_switch(spec: object) -> None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient switch error")
            svc._current_model_spec = spec  # type: ignore[assignment]

        async def deliver(uid: str, result: object) -> None:
            await asyncio.sleep(0.05)
            svc._result_queue.put(("RESULT", uid, result))

        with patch.object(svc, "_switch_worker", side_effect=sometimes_failing_switch):
            # First request with new model_spec fails
            with pytest.raises(RuntimeError, match="transient switch error"):
                await svc.submit(_make_upload(), {}, request_id="req-fail", model_spec=mlx_spec)

            # Second request succeeds — service has not wedged
            asyncio.create_task(deliver("req-ok", {"text": "recovered", "segments": None, "duration": 1.0}))
            result = await asyncio.wait_for(
                svc.submit(_make_upload(), {}, request_id="req-ok", model_spec=None),
                timeout=5.0,
            )
            assert result["text"] == "recovered"  # type: ignore[index]

        await _stop_service(svc)
