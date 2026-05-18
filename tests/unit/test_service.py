"""Unit tests for TranscriptionService core behaviors (SPEC-009).

Tests: success result delivery, txt format, queue full, temp file cleanup, error handling.
Uses injected mock worker infrastructure (no real subprocess spawned).
"""
import asyncio
import logging
import multiprocessing
import os
import queue as _stdlib_queue
import shutil
import tempfile as _tempfile
from contextlib import suppress
from dataclasses import replace
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile

from src.adapters.pipeline_chunking import ChunkWindow
from src.core.alignment_port import AlignedWord
from src.core.diarization_port import SpeakerTurn
from src.core.model_registry import lookup
from src.core.pipeline_registry import PipelineProfile
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


def _legacy_segment_pipeline_profile(requestable: bool = False) -> PipelineProfile:
    """Use the pre-forced-alignment path for tests that exercise segment fallback behavior."""
    from src.core.pipeline_registry import lookup_profile

    return replace(
        lookup_profile("qwen3-sortformer"),
        alignment_alias=None,
        requestable=requestable,
    )


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


@pytest.mark.asyncio
async def test_decoupled_pipeline_should_align_speakers_and_restore_previous_model(funasr_spec):
    svc = _setup_service(funasr_spec)
    profile = _legacy_segment_pipeline_profile(requestable=True)

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        assert alias == "qwen3-asr"
        assert pipeline_reserved is True
        return {"text": "hello world", "segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        assert alias == "sortformer-diar"
        assert pipeline_reserved is True
        return [SpeakerTurn(speaker="Speaker A", start=0.0, end=1.0)]

    svc._transcribe_with_alias = fake_transcribe
    svc._diarize_with_alias = fake_diarize
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    result = await svc._run_decoupled_pipeline(
        "audio.wav",
        {"output_format": "json"},
        "req-pipeline",
        profile,
    )

    assert result["segments"][0]["speaker"] == "Speaker A"
    svc._restore_resident_model.assert_awaited_once_with(funasr_spec)


@pytest.mark.asyncio
async def test_decoupled_pipeline_should_align_words_to_speaker_turns(funasr_spec):
    from src.core.pipeline_registry import lookup_profile

    svc = _setup_service(funasr_spec)
    profile = replace(lookup_profile("qwen3-sortformer"), requestable=True)

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        assert alias == "qwen3-asr"
        assert pipeline_reserved is True
        return {"text": "hello world", "segments": [{"text": "hello world", "start": 0.0, "end": 10.0}]}

    async def fake_align(temp_file_path, text, language, request_id, alias, pipeline_reserved=False):
        assert alias == "qwen3-forced-aligner"
        assert text == "hello world"
        assert language == "English"
        assert pipeline_reserved is True
        return [
            AlignedWord(text="hello", start=0.0, end=0.5),
            AlignedWord(text="world", start=0.6, end=1.0),
        ]

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        assert alias == "sortformer-diar"
        assert pipeline_reserved is True
        return [
            SpeakerTurn(speaker="Speaker A", start=0.0, end=0.55),
            SpeakerTurn(speaker="Speaker B", start=0.55, end=1.1),
        ]

    svc._transcribe_with_alias = fake_transcribe
    svc._align_with_alias = fake_align
    svc._diarize_with_alias = fake_diarize
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    result = await svc._run_decoupled_pipeline(
        "audio.wav",
        {"output_format": "json", "language": "en"},
        "req-pipeline",
        profile,
    )

    assert result["segments"] == [
        {"id": 0, "speaker": "Speaker A", "start": 0.0, "end": 0.5, "text": "hello"},
        {"id": 1, "speaker": "Speaker B", "start": 0.6, "end": 1.0, "text": "world"},
    ]
    svc._restore_resident_model.assert_awaited_once_with(funasr_spec)


@pytest.mark.asyncio
async def test_chunked_alignment_should_apply_offsets_and_drop_overlap(funasr_spec):
    svc = _setup_service(funasr_spec)
    windows = [
        ChunkWindow(index=0, start=0.0, end=900.0, emit_start=0.0, emit_end=885.0),
        ChunkWindow(index=1, start=870.0, end=1800.0, emit_start=885.0, emit_end=1800.0),
    ]

    async def fake_align(temp_file_path, text, language, request_id, alias, pipeline_reserved=False):
        if temp_file_path.endswith("chunk_000.wav"):
            return [AlignedWord(text="first", start=10.0, end=11.0)]
        return [
            AlignedWord(text="overlap", start=1.0, end=2.0),
            AlignedWord(text="second", start=20.0, end=21.0),
        ]

    svc._align_with_alias = fake_align

    result = await svc._align_chunks_with_alias(
        chunk_paths=["/tmp/chunk_000.wav", "/tmp/chunk_001.wav"],
        chunk_texts=["first", "overlap second"],
        windows=windows,
        language="English",
        request_id="req",
        alias="qwen3-forced-aligner",
        pipeline_reserved=True,
    )

    assert result == [
        AlignedWord(text="first", start=10.0, end=11.0),
        AlignedWord(text="second", start=890.0, end=891.0),
    ]


def test_long_form_pipeline_should_extract_chunk_files_before_alignment(funasr_spec, tmp_path):
    svc = _setup_service(funasr_spec)
    windows = [
        ChunkWindow(index=0, start=0.0, end=300.0, emit_start=0.0, emit_end=285.0),
        ChunkWindow(index=1, start=270.0, end=600.0, emit_start=285.0, emit_end=600.0),
    ]
    extracted: list[str] = []
    fake_chunker = MagicMock()

    def fake_extract(source_path, output_path, window):
        assert source_path == "audio.wav"
        extracted.append(output_path)
        return output_path

    fake_chunker.extract_pipeline_chunk.side_effect = fake_extract
    svc._audio_chunker = fake_chunker

    paths = svc._extract_pipeline_chunks("audio.wav", str(tmp_path), windows)

    assert paths == [
        str(tmp_path / "pipeline_chunk_000.wav"),
        str(tmp_path / "pipeline_chunk_001.wav"),
    ]
    assert extracted == paths


@pytest.mark.asyncio
async def test_long_form_pipeline_should_transcribe_each_chunk_with_its_own_audio(funasr_spec):
    svc = _setup_service(funasr_spec)
    windows = [
        ChunkWindow(index=0, start=0.0, end=300.0, emit_start=0.0, emit_end=285.0),
        ChunkWindow(index=1, start=270.0, end=600.0, emit_start=285.0, emit_end=600.0),
    ]

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        assert params["output_format"] == "json"
        assert alias == "qwen3-asr"
        assert pipeline_reserved is True
        if temp_file_path.endswith("chunk_000.wav"):
            return {"text": "chunk zero", "segments": None, "duration": 300.0, "language": "en"}
        return {"text": "chunk one", "segments": None, "duration": 330.0, "language": "en"}

    svc._transcribe_with_alias = fake_transcribe

    result = await svc._transcribe_chunks_with_alias(
        chunk_paths=["/tmp/chunk_000.wav", "/tmp/chunk_001.wav"],
        windows=windows,
        params={"output_format": "json", "language": "en"},
        request_id="req",
        alias="qwen3-asr",
        pipeline_reserved=True,
    )

    assert result == ["chunk zero", "chunk one"]


@pytest.mark.asyncio
async def test_long_form_pipeline_should_extract_transcribe_and_align_real_chunks(funasr_spec, tmp_path):
    from src.core.pipeline_registry import lookup_profile

    svc = _setup_service(funasr_spec)
    profile = replace(lookup_profile("qwen3-sortformer"), requestable=True)
    transcript = {"text": "full text", "segments": None, "language": "en"}
    extracted_windows: list[ChunkWindow] = []
    fake_chunker = MagicMock()
    fake_chunker.get_audio_duration.return_value = 600.0
    svc._audio_chunker = fake_chunker

    def fake_extract_chunks(temp_file_path, temp_dir, windows):
        assert temp_file_path == "audio.wav"
        assert temp_dir
        extracted_windows.extend(windows)
        paths = [os.path.join(temp_dir, f"chunk_{window.index:03d}.wav") for window in windows]
        for path in paths:
            with open(path, "wb") as handle:
                handle.write(b"chunk")
        return paths

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        assert temp_file_path == "audio.wav"
        return transcript

    async def fake_transcribe_chunks(**kwargs):
        return [f"text {index}" for index, _path in enumerate(kwargs["chunk_paths"])]

    async def fake_align_chunks(**kwargs):
        assert kwargs["chunk_texts"] == ["text 0", "text 1", "text 2"]
        return [AlignedWord(text="hello", start=10.0, end=10.5)]

    async def fake_diarize_chunks(**kwargs):
        assert len(kwargs["chunk_paths"]) == 3
        assert all(os.path.exists(path) for path in kwargs["chunk_paths"])
        return [SpeakerTurn(speaker="Speaker 0", start=0.0, end=600.0)]

    svc._extract_pipeline_chunks = fake_extract_chunks
    svc._transcribe_with_alias = fake_transcribe
    svc._align_with_alias = AsyncMock(side_effect=AssertionError("must not use single-file align"))
    svc._transcribe_chunks_with_alias = AsyncMock(side_effect=fake_transcribe_chunks)
    svc._align_chunks_with_alias = AsyncMock(side_effect=fake_align_chunks)
    svc._diarize_with_alias = AsyncMock(side_effect=AssertionError("must not use single-file diarize"))
    svc._diarize_chunks_with_alias = AsyncMock(side_effect=fake_diarize_chunks)
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    result = await svc._run_decoupled_pipeline("audio.wav", {"output_format": "json"}, "req", profile)

    assert extracted_windows
    assert svc._transcribe_chunks_with_alias.await_count == 1
    assert svc._align_chunks_with_alias.await_count == 1
    assert svc._diarize_chunks_with_alias.await_count == 1
    assert result["segments"] == [
        {"id": 0, "speaker": "Speaker 0", "start": 10.0, "end": 10.5, "text": "hello"}
    ]


@pytest.mark.asyncio
async def test_chunked_diarization_should_offset_turns_and_drop_overlap(funasr_spec):
    svc = _setup_service(funasr_spec)
    windows = [
        ChunkWindow(index=0, start=0.0, end=300.0, emit_start=0.0, emit_end=285.0),
        ChunkWindow(index=1, start=270.0, end=600.0, emit_start=285.0, emit_end=600.0),
    ]

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        if temp_file_path.endswith("chunk_000.wav"):
            return [SpeakerTurn(speaker="Speaker 0", start=10.0, end=20.0)]
        return [
            SpeakerTurn(speaker="Speaker 0", start=1.0, end=2.0),
            SpeakerTurn(speaker="Speaker 1", start=20.0, end=30.0),
        ]

    svc._diarize_with_alias = fake_diarize

    result = await svc._diarize_chunks_with_alias(
        chunk_paths=["/tmp/chunk_000.wav", "/tmp/chunk_001.wav"],
        windows=windows,
        request_id="req",
        alias="sortformer-diar",
        pipeline_reserved=True,
    )

    assert result == [
        SpeakerTurn(speaker="Speaker 0", start=10.0, end=20.0),
        SpeakerTurn(speaker="Speaker 1", start=290.0, end=300.0),
    ]


@pytest.mark.asyncio
async def test_chunked_diarization_reconcile_should_keep_speaker_identity_across_chunks(funasr_spec):
    svc = _setup_service(funasr_spec)
    windows = [
        ChunkWindow(index=0, start=0.0, end=300.0, emit_start=0.0, emit_end=285.0),
        ChunkWindow(index=1, start=270.0, end=600.0, emit_start=285.0, emit_end=600.0),
    ]

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        if temp_file_path.endswith("chunk_000.wav"):
            return [
                SpeakerTurn(speaker="Speaker 1", start=270.0, end=277.5),
                SpeakerTurn(speaker="Speaker 0", start=277.5, end=285.0),
            ]
        return [
            SpeakerTurn(speaker="Speaker 0", start=0.0, end=7.5),
            SpeakerTurn(speaker="Speaker 1", start=7.5, end=15.0),
            SpeakerTurn(speaker="Speaker 1", start=15.0, end=35.0),
        ]

    svc._diarize_with_alias = fake_diarize

    result = await svc._diarize_chunks_with_alias(
        chunk_paths=["/tmp/chunk_000.wav", "/tmp/chunk_001.wav"],
        windows=windows,
        request_id="req",
        alias="sortformer-diar",
        pipeline_reserved=True,
    )

    assert result == [
        SpeakerTurn(speaker="Speaker 1", start=270.0, end=277.5),
        SpeakerTurn(speaker="Speaker 0", start=277.5, end=285.0),
        SpeakerTurn(speaker="Speaker 0", start=285.0, end=305.0),
    ]


@pytest.mark.asyncio
async def test_pipeline_should_fail_loudly_when_alignment_quality_gate_fails(funasr_spec, tmp_path):
    from src.core.pipeline_registry import lookup_profile

    svc = _setup_service(funasr_spec)
    profile = replace(lookup_profile("qwen3-sortformer"), requestable=True)
    collapsed = [AlignedWord(text=f"w{i}", start=245.04, end=245.04) for i in range(12)]

    def fake_extract_chunks(temp_file_path, temp_dir, windows):
        paths = [os.path.join(temp_dir, f"chunk_{window.index:03d}.wav") for window in windows]
        for path in paths:
            with open(path, "wb") as handle:
                handle.write(b"chunk")
        return paths

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        return {
            "text": " ".join(word.text for word in collapsed),
            "segments": None,
            "duration": 600.0,
            "language": "en",
        }

    async def fake_transcribe_chunks(**kwargs):
        return [" ".join(word.text for word in collapsed)]

    async def fake_align_chunks(**kwargs):
        return collapsed

    svc._extract_pipeline_chunks = fake_extract_chunks
    svc._transcribe_with_alias = fake_transcribe
    svc._transcribe_chunks_with_alias = AsyncMock(side_effect=fake_transcribe_chunks)
    svc._align_chunks_with_alias = AsyncMock(side_effect=fake_align_chunks)
    svc._diarize_chunks_with_alias = AsyncMock(
        side_effect=AssertionError("must not diarize failed alignment")
    )
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    with pytest.raises(ValueError, match="alignment quality gate failed"):
        await svc._run_decoupled_pipeline("audio.wav", {"output_format": "json"}, "req", profile)


@pytest.mark.asyncio
async def test_decoupled_pipeline_alignment_failure_returns_transcript_and_restores_model(funasr_spec):
    svc = _setup_service(funasr_spec)
    profile = _legacy_segment_pipeline_profile(requestable=True)
    transcript = {"text": "hello world", "segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        return transcript

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        return [SpeakerTurn(speaker="Speaker A", start=0.0, end=1.0)]

    svc._transcribe_with_alias = fake_transcribe
    svc._diarize_with_alias = fake_diarize
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    with patch("src.services.transcription.align_speakers", side_effect=ValueError("bad interval")):
        result = await svc._run_decoupled_pipeline(
            "audio.wav",
            {"output_format": "json"},
            "req-pipeline",
            profile,
        )

    assert result == transcript
    svc._restore_resident_model.assert_awaited_once_with(funasr_spec)


@pytest.mark.asyncio
async def test_decoupled_pipeline_diarization_failure_returns_transcript_and_restores_model(funasr_spec):
    svc = _setup_service(funasr_spec)
    profile = _legacy_segment_pipeline_profile()
    transcript = {"text": "hello world", "segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        return transcript

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        raise ValueError("diarization failed")

    svc._transcribe_with_alias = fake_transcribe
    svc._diarize_with_alias = fake_diarize
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    result = await svc._run_decoupled_pipeline(
        "audio.wav",
        {"output_format": "json"},
        "req-pipeline",
        profile,
    )

    assert result == transcript
    svc._restore_resident_model.assert_awaited_once_with(funasr_spec)


@pytest.mark.asyncio
async def test_decoupled_pipeline_diarization_not_implemented_is_propagated(funasr_spec):
    svc = _setup_service(funasr_spec)
    profile = _legacy_segment_pipeline_profile()

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        return {"text": "hello world", "segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        raise NotImplementedError("dedicated diarization job kind required")

    svc._transcribe_with_alias = fake_transcribe
    svc._diarize_with_alias = fake_diarize
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    with pytest.raises(NotImplementedError, match="dedicated diarization job kind required"):
        await svc._run_decoupled_pipeline(
            "audio.wav",
            {"output_format": "json"},
            "req-pipeline",
            profile,
        )

    svc._restore_resident_model.assert_awaited_once_with(funasr_spec)


@pytest.mark.asyncio
async def test_decoupled_pipeline_unknown_diarizer_alias_is_propagated(funasr_spec):
    svc = _setup_service(funasr_spec)
    profile = _legacy_segment_pipeline_profile()

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        return {"text": "hello world", "segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        raise RuntimeError("'Unknown diarization alias: \\'sortformer-typo\\''")

    svc._transcribe_with_alias = fake_transcribe
    svc._diarize_with_alias = fake_diarize
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    with pytest.raises(RuntimeError, match="Unknown diarization alias"):
        await svc._run_decoupled_pipeline(
            "audio.wav",
            {"output_format": "json"},
            "req-pipeline",
            profile,
        )

    svc._restore_resident_model.assert_awaited_once_with(funasr_spec)


@pytest.mark.asyncio
async def test_decoupled_pipeline_segment_coercion_failure_returns_transcript_and_restores_model(funasr_spec):
    svc = _setup_service(funasr_spec)
    profile = _legacy_segment_pipeline_profile()
    transcript = {"text": "hello world", "segments": ["bad segment"]}

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        return transcript

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        return [SpeakerTurn(speaker="Speaker A", start=0.0, end=1.0)]

    svc._transcribe_with_alias = fake_transcribe
    svc._diarize_with_alias = fake_diarize
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    result = await svc._run_decoupled_pipeline(
        "audio.wav",
        {"output_format": "json"},
        "req-pipeline",
        profile,
    )

    assert result == transcript
    svc._restore_resident_model.assert_awaited_once_with(funasr_spec)


@pytest.mark.asyncio
async def test_decoupled_pipeline_empty_segments_logs_warning_and_returns_transcript(funasr_spec, caplog):
    svc = _setup_service(funasr_spec)
    profile = _legacy_segment_pipeline_profile()
    transcript = {"text": "hello world", "segments": []}

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        return transcript

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        return [SpeakerTurn(speaker="Speaker A", start=0.0, end=1.0)]

    svc._transcribe_with_alias = fake_transcribe
    svc._diarize_with_alias = fake_diarize
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    with caplog.at_level(logging.WARNING, logger="src.services.transcription"):
        result = await svc._run_decoupled_pipeline(
            "audio.wav",
            {"output_format": "json"},
            "req-pipeline",
            profile,
        )

    assert result == transcript
    assert "no transcript segments to align" in caplog.text
    svc._restore_resident_model.assert_awaited_once_with(funasr_spec)


@pytest.mark.asyncio
async def test_diarize_with_alias_should_submit_diarization_job(funasr_spec):
    svc = _setup_service(funasr_spec)
    svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())

    async def deliver() -> None:
        await asyncio.sleep(0.05)
        svc._result_queue.put(
            ("RESULT", "req-pipeline:diarize", [SpeakerTurn(speaker="Speaker 0", start=0.0, end=1.0)])
        )

    asyncio.create_task(deliver())
    try:
        result = await svc._diarize_with_alias("audio.wav", "req-pipeline", "sortformer-diar")
        job = svc._job_queue.get(timeout=1.0)
    finally:
        await _stop_service(svc)

    assert result[0].speaker == "Speaker 0"
    assert job.job_kind == "diarize"
    assert job.requested_diarizer_alias == "sortformer-diar"


@pytest.mark.asyncio
async def test_decoupled_pipeline_restore_failure_is_propagated(funasr_spec):
    from src.core.pipeline_registry import lookup_profile

    svc = _setup_service(funasr_spec)
    profile = lookup_profile("qwen3-sortformer")

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        return "transcript"

    svc._transcribe_with_alias = fake_transcribe
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock(side_effect=RuntimeError("restore failed"))

    with pytest.raises(RuntimeError, match="restore failed"):
        await asyncio.wait_for(
            svc._run_decoupled_pipeline(
                "audio.wav",
                {"output_format": "txt"},
                "req-pipeline",
                profile,
            ),
            timeout=1.0,
        )


@pytest.mark.asyncio
async def test_pipeline_should_hold_worker_reservation_until_restore(funasr_spec):
    from src.core.pipeline_registry import lookup_profile

    svc = _setup_service(funasr_spec)
    profile = replace(lookup_profile("qwen3-sortformer"), requestable=True)
    observed_current_specs: list[str | None] = []

    async def fake_submit_worker_job(**kwargs):
        observed_current_specs.append(svc.current_model_spec.alias if svc.current_model_spec else None)
        if kwargs["request_id"].endswith(":transcribe"):
            return {"text": "hello", "segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}
        if kwargs["request_id"].endswith(":align"):
            return [AlignedWord(text="hello", start=0.0, end=1.0)]
        return [SpeakerTurn(speaker="Speaker 0", start=0.0, end=1.0)]

    svc._submit_worker_job = AsyncMock(side_effect=fake_submit_worker_job)
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))

    result = await svc._run_decoupled_pipeline("audio.wav", {"output_format": "json"}, "req", profile)

    assert result["segments"][0]["speaker"] == "Speaker 0"
    assert observed_current_specs == ["qwen3-asr", "qwen3-asr", "qwen3-asr"]
    assert svc.current_model_spec == funasr_spec


@pytest.mark.asyncio
async def test_pipeline_waits_for_existing_pending_work_before_switching_models(funasr_spec):
    svc = _setup_service(funasr_spec)
    profile = _legacy_segment_pipeline_profile(requestable=True)
    loop = asyncio.get_running_loop()
    unrelated_future: asyncio.Future[object] = loop.create_future()
    svc._pending["other-request"] = unrelated_future

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        return {"text": "hello", "segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        assert pipeline_reserved is True
        return [SpeakerTurn(speaker="Speaker 0", start=0.0, end=1.0)]

    svc._transcribe_with_alias = fake_transcribe
    svc._diarize_with_alias = fake_diarize
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    task = asyncio.create_task(
        svc._run_decoupled_pipeline("audio.wav", {"output_format": "json"}, "req-pipeline", profile)
    )
    try:
        await asyncio.sleep(0.05)
        svc._switch_worker.assert_not_awaited()
        assert not task.done()
        assert not unrelated_future.cancelled()

        svc._resolve_future("other-request", result={"text": "finished elsewhere"})
        result = await asyncio.wait_for(task, timeout=1.0)
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    assert unrelated_future.done()
    assert not unrelated_future.cancelled()
    assert result["segments"][0]["speaker"] == "Speaker 0"
    svc._switch_worker.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_pending_drain_times_out_instead_of_waiting_forever(funasr_spec):
    svc = _setup_service(funasr_spec)
    loop = asyncio.get_running_loop()
    svc._pending["stuck-request"] = loop.create_future()

    with pytest.raises(RuntimeError, match="Timed out waiting for pending worker jobs"):
        await svc._wait_for_pending_work_to_drain(timeout_seconds=0.01)

    assert "stuck-request" in svc._pending


@pytest.mark.asyncio
async def test_pipeline_temp_dir_cleanup_should_run_off_event_loop(funasr_spec):
    svc = _setup_service(funasr_spec)

    with patch("src.services.transcription.asyncio.to_thread", new_callable=AsyncMock) as to_thread:
        await svc._remove_pipeline_temp_dir("/tmp/asr_pipeline_chunks")

    to_thread.assert_awaited_once_with(
        shutil.rmtree,
        "/tmp/asr_pipeline_chunks",
        ignore_errors=True,
    )


@pytest.mark.asyncio
async def test_restore_resident_model_raises_when_jobs_are_still_pending(funasr_spec):
    svc = _setup_service(funasr_spec)
    qwen_spec = lookup("qwen3-asr")
    svc._current_model_spec = qwen_spec
    loop = asyncio.get_running_loop()
    svc._pending["other-request"] = loop.create_future()
    svc._switch_worker = AsyncMock()

    with pytest.raises(RuntimeError, match="worker jobs are still pending"):
        await svc._restore_resident_model(funasr_spec)

    svc._switch_worker.assert_not_awaited()
    assert svc._current_model_spec == qwen_spec


@pytest.mark.asyncio
async def test_submit_pipeline_rejects_non_requestable_profile(funasr_spec):
    svc = _setup_service(funasr_spec)
    profile = PipelineProfile(
        alias="test-pipeline",
        transcription_alias="qwen3-asr",
        alignment_alias=None,
        diarization_alias="sortformer-diar",
        description="test",
        capabilities=funasr_spec.capabilities,
        requestable=False,
    )

    with pytest.raises(RuntimeError, match="not enabled"):
        await svc.submit_pipeline(
            _make_upload(),
            {"output_format": "json"},
            request_id="req-pipeline",
            profile=profile,
        )


@pytest.mark.asyncio
async def test_submit_pipeline_cleans_composite_request_state_on_error(funasr_spec):
    from src.core.pipeline_registry import lookup_profile

    svc = _setup_service(funasr_spec)
    profile = replace(lookup_profile("qwen3-sortformer"), requestable=True)
    request_id = "req-pipeline"
    loop = asyncio.get_running_loop()

    async def fake_run_pipeline(temp_file_path, params, request_id, profile):
        svc._pending[f"{request_id}:transcribe"] = loop.create_future()
        svc._pending[f"{request_id}:align"] = loop.create_future()
        svc._pending[f"{request_id}:diarize"] = loop.create_future()
        raise RuntimeError("pipeline failed")

    svc._run_decoupled_pipeline = fake_run_pipeline

    with pytest.raises(RuntimeError, match="pipeline failed"):
        await svc.submit_pipeline(
            _make_upload(),
            {"output_format": "json"},
            request_id=request_id,
            profile=profile,
        )

    assert f"{request_id}:transcribe" not in svc._pending
    assert f"{request_id}:align" not in svc._pending
    assert f"{request_id}:diarize" not in svc._pending


@pytest.mark.asyncio
async def test_submit_worker_job_enforces_queue_limit_for_internal_callers(funasr_spec):
    svc = _setup_service(funasr_spec, max_queue_size=1)
    loop = asyncio.get_running_loop()
    svc._pending["existing"] = loop.create_future()

    with pytest.raises(RuntimeError, match="Queue is full"):
        await asyncio.wait_for(
            svc._submit_worker_job(
                temp_file_path="audio.wav",
                params={},
                request_id="req-internal",
                model_spec=funasr_spec,
            ),
            timeout=0.2,
        )


@pytest.mark.asyncio
async def test_submit_worker_job_cleans_pending_when_wait_is_cancelled(funasr_spec):
    svc = _setup_service(funasr_spec)
    request_id = "req-cancelled"

    task = asyncio.create_task(
        svc._submit_worker_job(
            temp_file_path="audio.wav",
            params={},
            request_id=request_id,
            model_spec=funasr_spec,
        )
    )
    try:
        for _ in range(20):
            if request_id in svc._pending:
                break
            await asyncio.sleep(0.01)
        assert request_id in svc._pending

        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        assert request_id not in svc._pending
    finally:
        if not task.done():
            task.cancel()
        await _stop_service(svc)


@pytest.mark.asyncio
async def test_spawn_worker_cancels_existing_result_reader_before_startup_handshake(funasr_spec):
    class FakeQueue:
        def get(self):
            return ("READY", None)

        def get_nowait(self):
            raise _stdlib_queue.Empty

        def close(self) -> None:
            return None

        def join_thread(self) -> None:
            return None

        def put(self, item, timeout=None) -> None:
            return None

        def put_nowait(self, item) -> None:
            return None

    svc = TranscriptionService(
        engine_type=funasr_spec.engine_type,
        model_id=funasr_spec.model_id,
        max_queue_size=2,
        initial_model_spec=funasr_spec,
        idle_timeout=0,
    )
    svc.is_running = True
    old_reader = asyncio.create_task(asyncio.sleep(60))
    svc._result_reader_task = old_reader
    mock_process = MagicMock()
    mock_process.is_alive.return_value = True

    try:
        with (
            patch("src.services.transcription.multiprocessing.Process", return_value=mock_process),
            patch("src.services.transcription.multiprocessing.Queue", side_effect=[FakeQueue(), FakeQueue()]),
        ):
            await svc._spawn_worker(funasr_spec)

        assert old_reader.cancelled()
        assert svc._result_reader_task is not old_reader
        assert svc._result_reader_task is not None
    finally:
        await _stop_service(svc)
