from __future__ import annotations

import asyncio
import threading
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import UploadFile

from src.core.model_registry import lookup
from src.services.transcription import TranscriptionService


class FakeAppleSpeechEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, bool]] = []

    def transcribe_file(
        self,
        file_path: str,
        language: str = "auto",
        output_format: str = "json",
        with_timestamp: bool = False,
        **_kwargs: object,
    ) -> dict[str, object]:
        self.calls.append((file_path, language, output_format, with_timestamp))
        assert Path(file_path).exists()
        return {
            "text": "apple result",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "apple result",
                    "speaker": None,
                }
            ],
            "duration": 1.0,
            "language": "en-US",
        }


class ConcurrentAppleSpeechEngine(FakeAppleSpeechEngine):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def transcribe_file(
        self,
        file_path: str,
        language: str = "auto",
        output_format: str = "json",
        with_timestamp: bool = False,
        **_kwargs: object,
    ) -> dict[str, object]:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.1)
            return super().transcribe_file(file_path, language, output_format, with_timestamp)
        finally:
            with self._lock:
                self.active -= 1


def _upload() -> UploadFile:
    return UploadFile(filename="audio.wav", file=BytesIO(b"RIFF\x00\x00\x00\x00WAVEfmt "))


@pytest.mark.asyncio
async def test_submit_apple_speech_bypasses_multiprocessing_worker() -> None:
    service = TranscriptionService(engine_type="funasr", model_id="iic/default")
    fake_engine = FakeAppleSpeechEngine()

    with (
        patch.object(service, "_get_apple_speech_engine", return_value=fake_engine),
        patch.object(
            service,
            "_submit_worker_job",
            new=AsyncMock(return_value={"text": "worker result", "segments": None}),
        ) as worker_submit,
    ):
        result = await service.submit(
            _upload(),
            {"language": "en", "output_format": "json", "with_timestamp": False},
            request_id="req-1",
            model_spec=lookup("apple-speech"),
        )

    assert isinstance(result, dict)
    assert result["text"] == "apple result"
    assert result["language"] == "en-US"
    assert fake_engine.calls[0][1:] == ("en", "json", False)
    captured_path = Path(fake_engine.calls[0][0])
    assert not captured_path.exists()
    assert not captured_path.parent.exists()
    worker_submit.assert_not_called()
    assert service.queue_size == 0


@pytest.mark.asyncio
async def test_submit_apple_speech_cleans_temp_dir_on_error() -> None:
    service = TranscriptionService(engine_type="funasr", model_id="iic/default")
    captured_path: Path | None = None

    class FailingEngine(FakeAppleSpeechEngine):
        def transcribe_file(
            self,
            file_path: str,
            language: str = "auto",
            output_format: str = "json",
            with_timestamp: bool = False,
            **_kwargs: object,
        ) -> dict[str, object]:
            nonlocal captured_path
            captured_path = Path(file_path)
            assert captured_path.exists()
            raise RuntimeError("apple worker failed")

    with (
        patch.object(service, "_get_apple_speech_engine", return_value=FailingEngine()),
        patch.object(
            service,
            "_submit_worker_job",
            new=AsyncMock(return_value={"text": "worker result", "segments": None}),
        ),
    ):
        with pytest.raises(RuntimeError, match="apple worker failed"):
            await service.submit(
                _upload(),
                {"language": "en", "output_format": "json", "with_timestamp": False},
                request_id="req-2",
                model_spec=lookup("apple-speech"),
            )

    assert captured_path is not None
    assert not captured_path.exists()
    assert service.queue_size == 0


@pytest.mark.asyncio
async def test_submit_apple_speech_limits_sidecar_concurrency() -> None:
    service = TranscriptionService(engine_type="funasr", model_id="iic/default")
    fake_engine = ConcurrentAppleSpeechEngine()

    with patch.object(service, "_get_apple_speech_engine", return_value=fake_engine):
        await asyncio.gather(
            service.submit(
                _upload(),
                {"language": "en", "output_format": "json", "with_timestamp": False},
                request_id="req-3",
                model_spec=lookup("apple-speech"),
            ),
            service.submit(
                _upload(),
                {"language": "en", "output_format": "json", "with_timestamp": False},
                request_id="req-4",
                model_spec=lookup("apple-speech"),
            ),
        )

    assert fake_engine.max_active == 1


@pytest.mark.asyncio
async def test_submit_apple_speech_rejects_when_combined_queue_is_full() -> None:
    service = TranscriptionService(engine_type="funasr", model_id="iic/default", max_queue_size=1)
    service._sidecar_pending.add("busy-request")

    with pytest.raises(RuntimeError, match="Queue is full"):
        await service.submit(
            _upload(),
            {"language": "en", "output_format": "json", "with_timestamp": False},
            request_id="req-queue-full",
            model_spec=lookup("apple-speech"),
        )


@pytest.mark.asyncio
async def test_submit_apple_speech_loads_cached_engine_before_transcription() -> None:
    service = TranscriptionService(engine_type="funasr", model_id="iic/default")

    class LoadingEngine(FakeAppleSpeechEngine):
        def __init__(self) -> None:
            super().__init__()
            self.loaded = False

        def load(self) -> None:
            self.loaded = True

        def transcribe_file(
            self,
            file_path: str,
            language: str = "auto",
            output_format: str = "json",
            with_timestamp: bool = False,
            **_kwargs: object,
        ) -> dict[str, object]:
            assert self.loaded
            return super().transcribe_file(file_path, language, output_format, with_timestamp)

    fake_engine = LoadingEngine()
    with patch("src.services.transcription.AppleSpeechEngine.from_config", return_value=fake_engine):
        await service.submit(
            _upload(),
            {"language": "en", "output_format": "json", "with_timestamp": False},
            request_id="req-5",
            model_spec=lookup("apple-speech"),
        )

    assert fake_engine.loaded is True
