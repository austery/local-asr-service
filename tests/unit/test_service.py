"""Unit tests for TranscriptionService core behaviors (SPEC-009).

Tests: success result delivery, txt format, queue full, temp file cleanup, error handling.
Uses injected mock worker infrastructure (no real subprocess spawned).
"""
import asyncio
import multiprocessing
import os
import tempfile as _tempfile
from contextlib import suppress
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile

from src.core.diarization_port import SpeakerTurn
from src.core.model_registry import lookup
from src.core.pipeline_registry import lookup_profile
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
        with suppress(asyncio.CancelledError):
            await svc._result_reader_task


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

    async def test_submit_should_reject_model_spec_and_pipeline_profile_together(
        self,
        funasr_spec,
    ) -> None:
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")

        with pytest.raises(
            ValueError,
            match="submit\\(\\) accepts either model_spec or pipeline_profile, not both",
        ):
            await svc.submit(
                _make_upload(),
                {"language": "zh", "output_format": "json"},
                request_id="req-invalid",
                model_spec=funasr_spec,
                pipeline_profile=profile,
            )

    async def test_submit_should_release_standard_submission_gate_when_cancelled(
        self,
        funasr_spec,
    ) -> None:
        svc = _setup_service(funasr_spec)
        job_entered = asyncio.Event()
        exit_started = asyncio.Event()
        block_job = asyncio.Event()
        original_exit = svc._exit_standard_submission

        async def blocking_job(*_args: object, **_kwargs: object) -> object:
            job_entered.set()
            await block_job.wait()
            return {"text": "unreachable"}

        async def wrapped_exit() -> None:
            exit_started.set()
            await original_exit()

        with (
            patch.object(svc, "_submit_worker_job", new=AsyncMock(side_effect=blocking_job)),
            patch.object(svc, "_exit_standard_submission", new=AsyncMock(side_effect=wrapped_exit)),
        ):
            task = asyncio.create_task(svc.submit(_make_upload(), {}, request_id="req-cancel-standard"))
            await job_entered.wait()
            assert svc._active_standard_submissions == 1

            await svc._submission_gate.acquire()
            task.cancel()
            block_job.set()
            await exit_started.wait()
            task.cancel()
            svc._submission_gate.release()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)

        for _ in range(50):
            if svc._active_standard_submissions == 0:
                break
            await asyncio.sleep(0.01)

        assert svc._active_standard_submissions == 0

    async def test_submit_should_release_pipeline_submission_gate_when_cancelled(
        self,
        funasr_spec,
    ) -> None:
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")
        pipeline_entered = asyncio.Event()
        exit_started = asyncio.Event()
        block_pipeline = asyncio.Event()
        original_exit = svc._exit_pipeline_submission

        async def blocking_pipeline(*_args: object, **_kwargs: object) -> object:
            pipeline_entered.set()
            await block_pipeline.wait()
            return {"text": "unreachable"}

        async def wrapped_exit() -> None:
            exit_started.set()
            await original_exit()

        with (
            patch.object(svc, "_run_decoupled_pipeline", new=AsyncMock(side_effect=blocking_pipeline)),
            patch.object(svc, "_exit_pipeline_submission", new=AsyncMock(side_effect=wrapped_exit)),
        ):
            task = asyncio.create_task(
                svc.submit(
                    _make_upload(),
                    {"language": "zh", "output_format": "json"},
                    request_id="req-cancel-pipeline",
                    pipeline_profile=profile,
                )
            )
            await pipeline_entered.wait()
            assert svc._pipeline_active is True

            await svc._submission_gate.acquire()
            task.cancel()
            block_pipeline.set()
            await exit_started.wait()
            task.cancel()
            svc._submission_gate.release()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)

        for _ in range(50):
            if not svc._pipeline_active:
                break
            await asyncio.sleep(0.01)

        assert svc._pipeline_active is False

    async def test_submit_should_release_pipeline_submission_gate_once(
        self,
        funasr_spec,
    ) -> None:
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")
        expected = {"text": "pipeline result", "segments": None, "duration": 1.0}
        original_exit = svc._exit_pipeline_submission

        async def wrapped_exit() -> None:
            await original_exit()

        with (
            patch.object(
                svc,
                "_run_decoupled_pipeline",
                new=AsyncMock(return_value=expected),
            ),
            patch.object(
                svc,
                "_exit_pipeline_submission",
                new=AsyncMock(side_effect=wrapped_exit),
            ) as mock_exit,
        ):
            result = await svc.submit(
                _make_upload(),
                {"language": "zh", "output_format": "json"},
                request_id="req-pipeline-release-once",
                pipeline_profile=profile,
            )

        assert result == expected
        mock_exit.assert_awaited_once()

    async def test_release_submission_gate_should_finish_cleanup_before_second_cancellation(
        self,
        funasr_spec,
    ) -> None:
        svc = _setup_service(funasr_spec)
        svc._active_standard_submissions = 1
        release_entered = asyncio.Event()
        original_exit = svc._exit_standard_submission

        async def wrapped_exit() -> None:
            release_entered.set()
            await original_exit()

        await svc._submission_gate.acquire()
        task = asyncio.create_task(svc._release_submission_gate(wrapped_exit()))
        await release_entered.wait()

        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)

        assert task.done() is False

        svc._submission_gate.release()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        assert svc._active_standard_submissions == 0

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

    async def test_run_decoupled_pipeline_aligns_speakers(self, funasr_spec):
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")
        transcript_result = {
            "text": "hello world",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 1.0, "end": 2.0, "text": "world"},
            ],
            "duration": 2.0,
        }
        speaker_turns = [
            SpeakerTurn(speaker="Speaker 1", start=0.0, end=1.2),
            SpeakerTurn(speaker="Speaker 2", start=1.2, end=2.0),
        ]

        with (
            patch.object(
                svc,
                "_transcribe_with_alias",
                new=AsyncMock(return_value=transcript_result),
            ) as mock_transcribe,
            patch.object(
                svc,
                "_diarize_with_alias",
                new=AsyncMock(return_value=speaker_turns),
            ) as mock_diarize,
        ):
            result = await svc._run_decoupled_pipeline(
                "/fake/audio.wav",
                {"language": "zh", "output_format": "json"},
                "req-pipeline",
                profile,
            )

        assert result["text"] == "hello world"
        assert result["segments"] == [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "Speaker 1"},
            {"start": 1.0, "end": 2.0, "text": "world", "speaker": "Speaker 2"},
        ]
        mock_transcribe.assert_awaited_once_with(
            "/fake/audio.wav",
            {"language": "zh", "output_format": "json"},
            "req-pipeline",
            profile.transcription_alias,
        )
        mock_diarize.assert_awaited_once_with(
            "/fake/audio.wav",
            "req-pipeline",
            profile.diarization_alias,
        )

    async def test_run_decoupled_pipeline_restores_previous_resident_model(
        self,
        funasr_spec,
    ) -> None:
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")
        firered_spec = lookup(profile.transcription_alias)
        sortformer_spec = lookup(profile.diarization_alias)
        transcript_result = {
            "text": "hello world",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "duration": 1.0,
        }
        speaker_turns = [SpeakerTurn(speaker="Speaker 1", start=0.0, end=1.0)]

        async def fake_transcribe(*_args: object) -> dict[str, object]:
            svc._current_model_spec = firered_spec
            return transcript_result

        async def fake_diarize(*_args: object) -> list[SpeakerTurn]:
            svc._current_model_spec = sortformer_spec
            return speaker_turns

        async def fake_switch(spec) -> None:
            svc._current_model_spec = spec

        with (
            patch.object(
                svc,
                "_transcribe_with_alias",
                new=AsyncMock(side_effect=fake_transcribe),
            ),
            patch.object(
                svc,
                "_diarize_with_alias",
                new=AsyncMock(side_effect=fake_diarize),
            ),
            patch.object(
                svc,
                "_switch_worker",
                new=AsyncMock(side_effect=fake_switch),
            ) as mock_switch,
        ):
            await svc._run_decoupled_pipeline(
                "/fake/audio.wav",
                {"language": "zh", "output_format": "json"},
                "req-pipeline",
                profile,
            )

        assert svc.current_model_spec == funasr_spec
        mock_switch.assert_awaited_once_with(funasr_spec)

    async def test_run_decoupled_pipeline_should_finish_restore_before_repeated_cancellation(
        self,
        funasr_spec,
    ) -> None:
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")
        firered_spec = lookup(profile.transcription_alias)
        sortformer_spec = lookup(profile.diarization_alias)
        transcript_result = {
            "text": "hello world",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "duration": 1.0,
        }
        speaker_turns = [SpeakerTurn(speaker="Speaker 1", start=0.0, end=1.0)]
        restore_entered = asyncio.Event()
        original_restore = svc._restore_resident_model

        async def fake_transcribe(*_args: object) -> dict[str, object]:
            svc._current_model_spec = firered_spec
            return transcript_result

        async def fake_diarize(*_args: object) -> list[SpeakerTurn]:
            svc._current_model_spec = sortformer_spec
            return speaker_turns

        async def wrapped_restore(previous_spec) -> None:
            restore_entered.set()
            await original_restore(previous_spec)

        async def fake_switch(spec) -> None:
            svc._current_model_spec = spec

        with (
            patch.object(svc, "_transcribe_with_alias", new=AsyncMock(side_effect=fake_transcribe)),
            patch.object(svc, "_diarize_with_alias", new=AsyncMock(side_effect=fake_diarize)),
            patch.object(svc, "_restore_resident_model", new=AsyncMock(side_effect=wrapped_restore)),
            patch.object(svc, "_switch_worker", new=AsyncMock(side_effect=fake_switch)),
            patch.object(svc, "_spawn_lock", new=asyncio.Lock()),
        ):
            await svc._spawn_lock.acquire()
            task = asyncio.create_task(
                svc._run_decoupled_pipeline(
                    "/fake/audio.wav",
                    {"language": "zh", "output_format": "json"},
                    "req-pipeline",
                    profile,
                )
            )
            await restore_entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)

            assert task.done() is False

            svc._spawn_lock.release()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)

        assert svc.current_model_spec == funasr_spec

    async def test_submit_should_not_be_cancelled_by_pipeline_restore(
        self,
        funasr_spec,
    ) -> None:
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")
        firered_spec = lookup(profile.transcription_alias)
        sortformer_spec = lookup(profile.diarization_alias)
        transcript_result = {
            "text": "hello world",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "duration": 1.0,
        }
        speaker_turns = [SpeakerTurn(speaker="Speaker 1", start=0.0, end=1.0)]
        restore_started = asyncio.Event()
        submit_registered = asyncio.Event()

        job_queue = MagicMock()
        job_queue.put_nowait.side_effect = lambda _job: submit_registered.set()
        result_queue = MagicMock()
        svc._job_queue = job_queue
        svc._result_queue = result_queue

        async def fake_transcribe(*_args: object) -> dict[str, object]:
            svc._current_model_spec = firered_spec
            return transcript_result

        async def fake_diarize(*_args: object) -> list[SpeakerTurn]:
            svc._current_model_spec = sortformer_spec
            return speaker_turns

        async def fake_shutdown() -> None:
            restore_started.set()
            with suppress(TimeoutError):
                await asyncio.wait_for(submit_registered.wait(), timeout=0.1)
            for uid, fut in list(svc._pending.items()):
                if not fut.done():
                    fut.set_exception(RuntimeError("Worker terminated (model switch or shutdown)"))
            svc._pending.clear()
            svc._temp_dirs.clear()
            svc._worker = None

        async def fake_spawn(model_spec=None) -> None:
            svc._current_model_spec = model_spec
            mock_proc = MagicMock()
            mock_proc.is_alive.return_value = True
            svc._worker = mock_proc
            svc._job_queue = job_queue
            svc._result_queue = result_queue

        with (
            patch.object(
                svc,
                "_transcribe_with_alias",
                new=AsyncMock(side_effect=fake_transcribe),
            ),
            patch.object(
                svc,
                "_diarize_with_alias",
                new=AsyncMock(side_effect=fake_diarize),
            ),
            patch.object(svc, "_shutdown_worker", new=AsyncMock(side_effect=fake_shutdown)),
            patch.object(svc, "_spawn_worker", new=AsyncMock(side_effect=fake_spawn)),
        ):
            pipeline_task = asyncio.create_task(
                svc.submit(
                    _make_upload(),
                    {"language": "zh", "output_format": "json"},
                    "req-pipeline",
                    pipeline_profile=profile,
                )
            )
            await restore_started.wait()

            submit_task = asyncio.create_task(
                svc.submit(_make_upload(), {}, request_id="req-next")
            )

            async def deliver_after_registration() -> None:
                for _ in range(100):
                    await asyncio.sleep(0.01)
                    if "req-next" in svc._pending:
                        svc._resolve_future(
                            "req-next",
                            result={"text": "after restore", "segments": None, "duration": 1.0},
                        )
                        return

            await asyncio.gather(
                pipeline_task,
                deliver_after_registration(),
            )
            submit_result = await asyncio.wait_for(submit_task, timeout=1.0)

        assert submit_result == {
            "text": "after restore",
            "segments": None,
            "duration": 1.0,
        }
        assert svc.current_model_spec == funasr_spec

    async def test_run_decoupled_pipeline_propagates_transcription_failures(
        self,
        funasr_spec,
    ) -> None:
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")

        with (
            patch.object(
                svc,
                "_transcribe_with_alias",
                new=AsyncMock(side_effect=RuntimeError("transcription boom")),
            ),
            patch.object(
                svc,
                "_diarize_with_alias",
                new=AsyncMock(),
            ) as mock_diarize,
            pytest.raises(RuntimeError, match="transcription boom"),
        ):
            await svc._run_decoupled_pipeline(
                "/fake/audio.wav",
                {"language": "zh", "output_format": "json"},
                "req-pipeline",
                profile,
            )

        mock_diarize.assert_not_awaited()

    async def test_run_decoupled_pipeline_returns_transcript_when_diarization_fails(
        self,
        funasr_spec,
        caplog: pytest.LogCaptureFixture,
    ):
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")
        transcript_result = {
            "text": "hello world",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "duration": 1.0,
        }

        with (
            patch.object(
                svc,
                "_transcribe_with_alias",
                new=AsyncMock(return_value=transcript_result),
            ),
            patch.object(
                svc,
                "_diarize_with_alias",
                new=AsyncMock(side_effect=RuntimeError("diarization boom")),
            ),
            caplog.at_level("WARNING"),
        ):
            result = await svc._run_decoupled_pipeline(
                "/fake/audio.wav",
                {"language": "zh", "output_format": "json"},
                "req-pipeline",
                profile,
            )

        assert result == transcript_result
        assert "diarization failed" in caplog.text.lower()

    async def test_run_decoupled_pipeline_propagates_worker_termination_errors(
        self,
        funasr_spec,
    ) -> None:
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")
        transcript_result = {
            "text": "hello world",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "duration": 1.0,
        }
        sortformer_spec = lookup(profile.diarization_alias)

        async def fake_diarize(*_args: object) -> list[SpeakerTurn]:
            svc._current_model_spec = sortformer_spec
            raise RuntimeError("Worker terminated (model switch or shutdown)")

        async def fake_switch(spec) -> None:
            svc._current_model_spec = spec

        with (
            patch.object(
                svc,
                "_transcribe_with_alias",
                new=AsyncMock(return_value=transcript_result),
            ),
            patch.object(
                svc,
                "_diarize_with_alias",
                new=AsyncMock(side_effect=fake_diarize),
            ),
            patch.object(
                svc,
                "_switch_worker",
                new=AsyncMock(side_effect=fake_switch),
            ) as mock_switch,
            pytest.raises(RuntimeError, match="Worker terminated"),
        ):
            await svc._run_decoupled_pipeline(
                "/fake/audio.wav",
                {"language": "zh", "output_format": "json"},
                "req-pipeline",
                profile,
            )

        assert svc.current_model_spec == funasr_spec
        mock_switch.assert_awaited_once_with(funasr_spec)

    @pytest.mark.parametrize(
        ("error_message",),
        [
            ("Worker failed to start within 120s timeout.",),
            ("Worker failed to load model: sortformer missing weights",),
            ("Worker sent unexpected startup message: ('BOOM',)",),
        ],
    )
    async def test_run_decoupled_pipeline_propagates_worker_startup_failures(
        self,
        funasr_spec,
        error_message: str,
    ) -> None:
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")
        transcript_result = {
            "text": "hello world",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "duration": 1.0,
        }

        with (
            patch.object(
                svc,
                "_transcribe_with_alias",
                new=AsyncMock(return_value=transcript_result),
            ),
            patch.object(
                svc,
                "_diarize_with_alias",
                new=AsyncMock(side_effect=RuntimeError(error_message)),
            ),
            pytest.raises(RuntimeError, match="Worker"),
        ):
            await svc._run_decoupled_pipeline(
                "/fake/audio.wav",
                {"language": "zh", "output_format": "json"},
                "req-pipeline",
                profile,
            )

    async def test_run_decoupled_pipeline_returns_transcript_when_alignment_fails(
        self,
        funasr_spec,
        caplog: pytest.LogCaptureFixture,
    ):
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")
        transcript_result = {
            "text": "hello world",
            "segments": [{"start": True, "end": 1.0, "text": "hello"}],
            "duration": 1.0,
        }
        speaker_turns = [SpeakerTurn(speaker="Speaker 1", start=0.0, end=1.0)]

        with (
            patch.object(
                svc,
                "_transcribe_with_alias",
                new=AsyncMock(return_value=transcript_result),
            ),
            patch.object(
                svc,
                "_diarize_with_alias",
                new=AsyncMock(return_value=speaker_turns),
            ),
            caplog.at_level("WARNING"),
        ):
            result = await svc._run_decoupled_pipeline(
                "/fake/audio.wav",
                {"language": "zh", "output_format": "json"},
                "req-pipeline",
                profile,
            )

        assert result == transcript_result
        assert "alignment failed" in caplog.text.lower()

    async def test_run_decoupled_pipeline_returns_transcript_when_segments_are_missing(
        self,
        funasr_spec,
    ) -> None:
        svc = _setup_service(funasr_spec)
        profile = lookup_profile("firered-sortformer")
        transcript_result = {
            "text": "hello world",
            "segments": None,
            "duration": 1.0,
        }

        with (
            patch.object(
                svc,
                "_transcribe_with_alias",
                new=AsyncMock(return_value=transcript_result),
            ),
            patch.object(
                svc,
                "_diarize_with_alias",
                new=AsyncMock(return_value=[SpeakerTurn(speaker="Speaker 1", start=0.0, end=1.0)]),
            ),
        ):
            result = await svc._run_decoupled_pipeline(
                "/fake/audio.wav",
                {"language": "zh", "output_format": "json"},
                "req-pipeline",
                profile,
            )

        assert result == transcript_result
