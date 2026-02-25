"""
Unit tests for dynamic model switching in TranscriptionService (SPEC-108, cases DS-1..DS-6).

Uses mock engines — the observable behavior under test is the job result and
the sequence of release/load calls (justified: release-before-load is a memory-safety contract).
"""

import asyncio
import os
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import UploadFile

from src.core.model_registry import lookup
from src.services.transcription import TranscriptionService


@pytest.fixture
def mlx_spec():
    return lookup("qwen3-asr-mini")


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


@pytest.fixture
def initial_engine(mlx_spec):
    return _make_engine()


@pytest.fixture
def service(initial_engine, mlx_spec):
    return TranscriptionService(
        engine=initial_engine,
        max_queue_size=5,
        initial_model_spec=mlx_spec,
    )


async def _run_job(service: TranscriptionService, model_spec=None, upload=None) -> object:
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
class TestSameModelRequests:
    # DS-1: consecutive same-model requests don't re-trigger switch
    async def test_should_return_result_when_same_model_requested_twice(
        self, service, mlx_spec, initial_engine
    ) -> None:
        service.is_running = True
        worker = asyncio.create_task(service._consume_loop())

        try:
            r1 = await service.submit(
                _make_upload_file(), {"language": "auto", "output_format": "json"},
                model_spec=mlx_spec,
            )
            r2 = await service.submit(
                _make_upload_file(), {"language": "auto", "output_format": "json"},
                model_spec=mlx_spec,
            )
        finally:
            await service.stop_worker()
            await asyncio.wait_for(worker, timeout=5.0)

        # Both jobs succeed
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)
        # Engine was never released (no switch triggered)
        initial_engine.release.assert_not_called()


@pytest.mark.asyncio
class TestModelSwitching:
    # DS-2: result is correct after switching to a different model;
    #       current_model_spec reflects the new engine atomically.
    async def test_should_return_result_after_switching_to_different_model(
        self, service, funasr_spec
    ) -> None:
        new_engine = _make_engine({"text": "switched result", "segments": None})

        with patch(
            "src.services.transcription.create_engine_for_spec", return_value=new_engine
        ):
            result = await _run_job(service, model_spec=funasr_spec)

        assert isinstance(result, dict)
        assert result["text"] == "switched result"
        assert service.current_model_spec == funasr_spec, (
            "_current_model_spec must be atomically updated to the new spec after a successful switch"
        )

    # DS-3: release() is called before load() — memory safety contract
    async def test_should_release_old_engine_before_loading_new_one(
        self, service, initial_engine, funasr_spec
    ) -> None:
        new_engine = _make_engine()
        call_order: list[str] = []

        initial_engine.release.side_effect = lambda: call_order.append("release")
        new_engine.load.side_effect = lambda: call_order.append("load")

        with patch(
            "src.services.transcription.create_engine_for_spec", return_value=new_engine
        ):
            await _run_job(service, model_spec=funasr_spec)

        assert call_order == ["release", "load"], (
            "release() must precede load() to prevent dual-model memory peak on M-series"
        )

    # DS-4: when load() fails, job gets an exception
    async def test_should_fail_job_when_new_model_load_fails(
        self, service, funasr_spec
    ) -> None:
        bad_engine = _make_engine()
        bad_engine.load.side_effect = RuntimeError("model download failed")

        # Restore old engine load (for recovery attempt)
        old_engine = service.engine
        old_engine.load.return_value = None

        with patch(
            "src.services.transcription.create_engine_for_spec", return_value=bad_engine
        ):
            with pytest.raises(RuntimeError, match="model download failed"):
                await _run_job(service, model_spec=funasr_spec)

    # DS-5: service processes next job after a failed switch
    async def test_should_process_next_job_after_failed_switch(
        self, service, funasr_spec, mlx_spec, initial_engine
    ) -> None:
        bad_engine = _make_engine()
        bad_engine.load.side_effect = RuntimeError("load error")
        initial_engine.load.return_value = None  # recovery succeeds

        service.is_running = True
        worker = asyncio.create_task(service._consume_loop())

        try:
            # First job: switch fails
            with patch(
                "src.services.transcription.create_engine_for_spec", return_value=bad_engine
            ):
                with pytest.raises(RuntimeError):
                    await service.submit(
                        _make_upload_file(),
                        {"language": "auto", "output_format": "json"},
                        model_spec=funasr_spec,
                    )

            # Second job: no switch requested — should succeed using recovered engine
            result = await service.submit(
                _make_upload_file(),
                {"language": "auto", "output_format": "json"},
                model_spec=None,
            )
            assert isinstance(result, dict)
        finally:
            await service.stop_worker()
            await asyncio.wait_for(worker, timeout=5.0)

    # DS-7: when both load and restore fail, service is marked degraded
    async def test_should_mark_service_degraded_when_load_and_restore_both_fail(
        self, service, funasr_spec, initial_engine
    ) -> None:
        bad_engine = _make_engine()
        bad_engine.load.side_effect = RuntimeError("load error")
        initial_engine.load.side_effect = RuntimeError("restore error")  # recovery also fails

        with patch(
            "src.services.transcription.create_engine_for_spec", return_value=bad_engine
        ):
            with pytest.raises(RuntimeError, match="Engine unrecoverable"):
                await _run_job(service, model_spec=funasr_spec)

        assert service._engine_degraded is True, (
            "Service must be marked degraded so subsequent jobs fail fast "
            "instead of silently crashing against an unloaded engine"
        )

    # DS-8: degraded service rejects subsequent jobs with a clear error
    async def test_should_reject_subsequent_jobs_when_service_is_degraded(
        self, service, initial_engine
    ) -> None:
        service._engine_degraded = True

        with pytest.raises(RuntimeError, match="degraded state"):
            await _run_job(service, model_spec=None)

    # DS-6: temp file is cleaned up even when switch fails
    async def test_should_clean_temp_file_even_when_switch_fails(
        self, service, funasr_spec, initial_engine
    ) -> None:
        bad_engine = _make_engine()
        bad_engine.load.side_effect = RuntimeError("load error")
        initial_engine.load.return_value = None  # recovery succeeds

        rmtree_calls: list[str] = []

        original_rmtree = __import__("shutil").rmtree

        def spy_rmtree(path: str, **kwargs: object) -> None:
            rmtree_calls.append(str(path))
            original_rmtree(path, **kwargs)

        with patch("src.services.transcription.create_engine_for_spec", return_value=bad_engine):
            with patch("src.services.transcription.shutil.rmtree", side_effect=spy_rmtree):
                try:
                    await _run_job(service, model_spec=funasr_spec)
                except RuntimeError:
                    pass  # expected — job fails because switch fails

        # shutil.rmtree must have been called to clean up the temp dir
        assert len(rmtree_calls) >= 1, "Temp directory was not cleaned up after failed switch"
