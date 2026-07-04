from __future__ import annotations

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
