"""
Unit tests for idle model offload in TranscriptionService (SPEC-009, cases IO-1..IO-8).

Uses mock engines — the observable behavior under test is the job result and
the sequence of release/load calls triggered by idle timeouts.
"""

import asyncio
from collections.abc import Callable
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi import UploadFile

from src.core.model_registry import ModelSpec, lookup
from src.services.transcription import TranscriptionService


@pytest.fixture
def mlx_spec():
    return lookup("qwen3-asr")


@pytest.fixture
def funasr_spec():
    return lookup("paraformer")


def _make_engine(return_value: object = None) -> MagicMock:
    engine = MagicMock()
    engine.transcribe_file.return_value = return_value or {"text": "hello", "segments": None}
    engine.release.return_value = None
    engine.load.return_value = None
    return engine


def _make_upload_file() -> UploadFile:
    return UploadFile(file=BytesIO(b"fake audio"), filename="test.wav")


async def _wait_until(condition: "Callable[[], bool]", timeout: float = 5.0, interval: float = 0.05) -> None:
    """Poll until condition() is True or timeout is exceeded."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(interval)
    pytest.fail(f"Condition not met within {timeout}s")


async def _run_job(
    service: TranscriptionService,
    model_spec: ModelSpec | None = None,
    upload: UploadFile | None = None,
) -> str | dict[str, object]:
    """Helper: start worker, submit one job, stop worker, return result."""
    service.is_running = True
    worker = asyncio.create_task(service._consume_loop())
    try:
        return await service.submit(
            upload or _make_upload_file(),
            params={"language": "auto", "output_format": "json"},
            request_id="test",
            model_spec=model_spec,
        )
    finally:
        await service.stop_worker()
        await asyncio.wait_for(worker, timeout=5.0)


@pytest.mark.asyncio
class TestIdleOffload:
    # IO-1: Model is released after idle timeout expires
    async def test_should_release_model_after_idle_timeout(self, mlx_spec) -> None:
        engine = _make_engine()
        service = TranscriptionService(
            engine=engine,
            max_queue_size=5,
            initial_model_spec=mlx_spec,
            idle_timeout=1,  # 1 second for fast test
        )
        assert service.model_loaded is True

        service.is_running = True
        worker = asyncio.create_task(service._consume_loop())

        # Poll until offload completes instead of sleeping a fixed duration
        await _wait_until(lambda: not service.model_loaded)

        engine.release.assert_called_once()
        assert service.model_loaded is False

        # Clean up
        await service.stop_worker()
        await asyncio.wait_for(worker, timeout=5.0)

    # IO-2: Model is re-loaded on next request after offload
    async def test_should_reload_model_on_next_request_after_offload(
        self, mlx_spec
    ) -> None:
        engine = _make_engine()
        service = TranscriptionService(
            engine=engine,
            max_queue_size=5,
            initial_model_spec=mlx_spec,
            idle_timeout=1,
        )

        service.is_running = True
        worker = asyncio.create_task(service._consume_loop())

        # Poll until offload completes
        await _wait_until(lambda: not service.model_loaded)
        engine.release.assert_called_once()

        # Now submit a job — should reload model first, then transcribe
        result = await service.submit(
            _make_upload_file(),
            params={"language": "auto", "output_format": "json"},
            request_id="reload-test",
        )

        # Model should be reloaded
        engine.load.assert_called_once()
        assert service.model_loaded is True

        # Job should succeed
        assert isinstance(result, dict)
        assert result["text"] == "hello"

        await service.stop_worker()
        await asyncio.wait_for(worker, timeout=5.0)

    # IO-3: Idle offload disabled when timeout = 0
    async def test_should_not_offload_when_timeout_is_zero(self, mlx_spec) -> None:
        engine = _make_engine()
        service = TranscriptionService(
            engine=engine,
            max_queue_size=5,
            initial_model_spec=mlx_spec,
            idle_timeout=0,  # Disabled
        )

        service.is_running = True
        worker = asyncio.create_task(service._consume_loop())

        # Wait a bit — model should NOT be released
        await asyncio.sleep(0.5)

        engine.release.assert_not_called()
        assert service.model_loaded is True

        await service.stop_worker()
        await asyncio.wait_for(worker, timeout=5.0)

    # IO-4: Idle timer resets after each successful transcription
    async def test_should_reset_idle_timer_after_transcription(self, mlx_spec) -> None:
        engine = _make_engine()
        service = TranscriptionService(
            engine=engine,
            max_queue_size=5,
            initial_model_spec=mlx_spec,
            idle_timeout=2,  # 2 seconds
        )

        service.is_running = True
        worker = asyncio.create_task(service._consume_loop())

        try:
            # Submit a job at t=0 — resets the idle timer
            await service.submit(
                _make_upload_file(),
                params={"language": "auto", "output_format": "json"},
                request_id="timer-reset-1",
            )

            # Wait 1s (less than 2s timeout) — submit another job
            await asyncio.sleep(1.0)
            await service.submit(
                _make_upload_file(),
                params={"language": "auto", "output_format": "json"},
                request_id="timer-reset-2",
            )

            # Wait 1s more (total 2s since last job, but timer was reset)
            await asyncio.sleep(1.0)

            # Model should still be loaded (timer reset after each job)
            assert service.model_loaded is True
            engine.release.assert_not_called()

        finally:
            await service.stop_worker()
            await asyncio.wait_for(worker, timeout=5.0)

    # IO-5: Model switch after offload works correctly
    async def test_should_handle_model_switch_after_offload(
        self, mlx_spec, funasr_spec
    ) -> None:
        engine = _make_engine()
        service = TranscriptionService(
            engine=engine,
            max_queue_size=5,
            initial_model_spec=mlx_spec,
            idle_timeout=1,
        )

        service.is_running = True
        worker = asyncio.create_task(service._consume_loop())

        # Wait for idle timeout to offload
        await _wait_until(lambda: not service.model_loaded)
        assert service.model_loaded is False

        # Now submit a job with a DIFFERENT model — triggers _switch_model
        new_engine = _make_engine({"text": "switched after offload", "segments": None})

        with patch(
            "src.services.transcription.create_engine_for_spec",
            return_value=new_engine,
        ):
            result = await service.submit(
                _make_upload_file(),
                params={"language": "auto", "output_format": "json"},
                request_id="switch-after-offload",
                model_spec=funasr_spec,
            )

        # Switch should have happened, model is loaded
        assert service.model_loaded is True
        assert service.current_model_spec == funasr_spec
        assert isinstance(result, dict)
        assert result["text"] == "switched after offload"
        # release() was called twice: once by idle offload, once by _switch_model (no-op on
        # already-offloaded engine, but the call is still made per SPEC-108 §5 invariant).
        assert engine.release.call_count == 2

        await service.stop_worker()
        await asyncio.wait_for(worker, timeout=5.0)

    # IO-6: release() failure marks model as unloaded (forces reload on next job)
    async def test_should_mark_model_unloaded_when_release_fails(self, mlx_spec) -> None:
        engine = _make_engine()
        engine.release.side_effect = RuntimeError("GPU flush failed")
        service = TranscriptionService(
            engine=engine,
            max_queue_size=5,
            initial_model_spec=mlx_spec,
            idle_timeout=1,
        )

        service.is_running = True
        worker = asyncio.create_task(service._consume_loop())

        # Wait for timeout + release attempt (which will fail)
        await _wait_until(lambda: engine.release.called)

        # Despite release() failing, model_loaded must be False so the next
        # job goes through the reload path rather than calling transcribe on
        # a potentially broken engine.
        assert service.model_loaded is False

        await service.stop_worker()
        await asyncio.wait_for(worker, timeout=5.0)

    # IO-7: load() failure during reload raises to caller, model stays unloaded
    async def test_should_raise_and_keep_model_unloaded_when_reload_fails(
        self, mlx_spec
    ) -> None:
        engine = _make_engine()
        service = TranscriptionService(
            engine=engine,
            max_queue_size=5,
            initial_model_spec=mlx_spec,
            idle_timeout=1,
        )

        service.is_running = True
        worker = asyncio.create_task(service._consume_loop())

        # Wait for idle offload to complete first
        await _wait_until(lambda: not service.model_loaded)

        # Now make load fail for the reload attempt
        engine.load.side_effect = RuntimeError("OOM on reload")

        # Submit a job — reload will fail, job should raise RuntimeError
        with pytest.raises(RuntimeError, match="Model reload failed"):
            await service.submit(
                _make_upload_file(),
                params={"language": "auto", "output_format": "json"},
                request_id="reload-fail",
            )

        # State: still unloaded so next job can retry reload
        assert service.model_loaded is False

        await service.stop_worker()
        await asyncio.wait_for(worker, timeout=5.0)

    # IO-8: Consecutive idle timeouts do NOT double-release (_model_loaded guard)
    async def test_should_not_double_release_on_consecutive_timeouts(
        self, mlx_spec
    ) -> None:
        engine = _make_engine()
        service = TranscriptionService(
            engine=engine,
            max_queue_size=5,
            initial_model_spec=mlx_spec,
            idle_timeout=1,
        )

        service.is_running = True
        worker = asyncio.create_task(service._consume_loop())

        # Wait for first offload, then wait for a second timeout cycle to pass
        await _wait_until(lambda: not service.model_loaded)
        await asyncio.sleep(1.5)  # Let a second timeout fire

        # release() must have been called exactly once despite two timeout cycles
        assert engine.release.call_count == 1
        assert service.model_loaded is False

        await service.stop_worker()
        await asyncio.wait_for(worker, timeout=5.0)
