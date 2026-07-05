from pathlib import Path

import pytest

from src.core.apple_speech_engine import AppleSpeechEngine
from src.core.apple_speech_port import (
    AppleSpeechModule,
    AppleSpeechWorkerResponseError,
    TranscriptionMetadata,
    TranscriptionResult,
    TranscriptionSegment,
    WorkerCapabilities,
    WorkerModules,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, AppleSpeechModule, bool, bool]] = []

    def transcribe(
        self,
        input_path: Path,
        locale: str,
        module: AppleSpeechModule,
        audio_time_ranges: bool = True,
        include_volatile: bool = False,
    ) -> TranscriptionResult:
        self.calls.append((input_path, locale, module, audio_time_ranges, include_volatile))
        return TranscriptionResult(
            job_id=None,
            engine="apple-speech",
            module=module,
            locale=locale,
            text="hello world",
            segments=[
                TranscriptionSegment(
                    id=0,
                    start=0.0,
                    end=1.25,
                    text="hello world",
                    is_final=True,
                    confidence=None,
                    speaker=None,
                )
            ],
            metadata=TranscriptionMetadata(
                local=True,
                apple_api=True,
                volatile_included=False,
                timing_granularity="segment",
                asset_managed_by_system=True,
                duration_ms=1250,
            ),
        )

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            runtime="apple-speech",
            platform="macOS",
            os_version="26.5",
            supported=True,
            supported_locales=["en-US", "zh-CN"],
            modules=WorkerModules(
                speech_transcriber=True,
                dictation_transcriber=True,
                speech_detector=True,
            ),
            notes=[],
        )


def test_transcribe_file_returns_service_response_shape() -> None:
    client = FakeClient()
    engine = AppleSpeechEngine(client=client, module="speechTranscriber")

    result = engine.transcribe_file("/tmp/audio.wav", language="zh-CN", output_format="json")

    assert result == {
        "text": "hello world",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.25,
                "text": "hello world",
                "speaker": None,
            }
        ],
        "duration": 1.25,
        "language": "zh-CN",
    }
    assert client.calls == [(Path("/tmp/audio.wav"), "zh-CN", "speechTranscriber", True, False)]


def test_transcribe_file_maps_short_language_codes_to_apple_locales() -> None:
    client = FakeClient()
    engine = AppleSpeechEngine(client=client, module="speechTranscriber")

    engine.transcribe_file("/tmp/audio.wav", language="zh", output_format="json")
    engine.transcribe_file("/tmp/audio.wav", language="en", output_format="json")

    assert [call[1] for call in client.calls] == ["zh-CN", "en-US"]


def test_transcribe_file_requires_language_argument() -> None:
    client = FakeClient()
    engine = AppleSpeechEngine(client=client, module="speechTranscriber")

    with pytest.raises(TypeError, match="language"):
        engine.transcribe_file("/tmp/audio.wav", output_format="json")

    assert client.calls == []


@pytest.mark.parametrize("language", ["auto", "AUTO", "   "])
def test_transcribe_file_rejects_implicit_language_for_apple_speech(language: str) -> None:
    client = FakeClient()
    engine = AppleSpeechEngine(client=client, module="speechTranscriber")

    with pytest.raises(ValueError, match="requires an explicit language"):
        engine.transcribe_file("/tmp/audio.wav", language=language, output_format="json")

    assert client.calls == []


def test_transcribe_file_never_synthesizes_speaker_labels() -> None:
    client = FakeClient()
    engine = AppleSpeechEngine(client=client, module="speechTranscriber")

    result = engine.transcribe_file("/tmp/audio.wav", language="en-US", output_format="json")

    assert isinstance(result, dict)
    segments = result["segments"]
    assert isinstance(segments, list)
    assert segments[0]["speaker"] is None


def test_transcribe_file_returns_plain_text_for_txt_output() -> None:
    client = FakeClient()
    engine = AppleSpeechEngine(client=client, module="speechTranscriber")

    result = engine.transcribe_file("/tmp/audio.wav", language="en-US", output_format="txt")

    assert result == "hello world"


def test_transcribe_file_formats_segments_as_srt() -> None:
    client = FakeClient()
    engine = AppleSpeechEngine(client=client, module="speechTranscriber")

    result = engine.transcribe_file("/tmp/audio.wav", language="en-US", output_format="srt")

    assert result == "1\n00:00:00,000 --> 00:00:01,250\nhello world\n"


def test_worker_errors_are_not_hidden() -> None:
    class FailingClient(FakeClient):
        def transcribe(
            self,
            input_path: Path,
            locale: str,
            module: AppleSpeechModule,
            audio_time_ranges: bool = True,
            include_volatile: bool = False,
        ) -> TranscriptionResult:
            raise AppleSpeechWorkerResponseError("unsupported locale: fr-FR")

    engine = AppleSpeechEngine(client=FailingClient(), module="speechTranscriber")

    with pytest.raises(AppleSpeechWorkerResponseError, match="unsupported locale"):
        engine.transcribe_file("/tmp/audio.wav", language="fr-FR", output_format="json")
