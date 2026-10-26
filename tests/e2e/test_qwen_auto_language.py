"""Real Qwen regression for automatic detection after a silent first chunk.

Requires Apple Silicon, cached Qwen3-ASR-1.7B-8bit weights, and the local
(two_speakers_60s.wav) fixture documented in tests/fixtures/README.md.
No model inference belongs in the unit suite; CI runs tests/unit only.
"""
import wave
from pathlib import Path

import pytest

from src.core.mlx_engine import MlxAudioEngine


@pytest.mark.e2e
def test_qwen_auto_should_transcribe_speech_after_silent_first_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "two_speakers_60s.wav"
    assert fixture.is_file(), "Provide the English fixture documented in tests/fixtures/README.md"
    recording = tmp_path / "silence1200_speech60.wav"
    with wave.open(str(fixture), "rb") as speech:
        assert (speech.getnchannels(), speech.getsampwidth(), speech.getframerate()) == (1, 2, 16000)
        with wave.open(str(recording), "wb") as output:
            output.setparams(speech.getparams())
            one_second = bytes(16000 * 2)
            for _ in range(1200):
                output.writeframesraw(one_second)
            output.writeframes(speech.readframes(speech.getnframes()))

    cwd = tmp_path / "caller"
    cwd.mkdir()
    preserved_export = cwd / "transcript.json"
    preserved_export.write_text("existing transcript", encoding="utf-8")
    monkeypatch.chdir(cwd)
    engine = MlxAudioEngine()
    try:
        engine.load()
        result = engine.transcribe_file(str(recording), output_format="json")
    finally:
        engine.release()

    assert isinstance(result, dict)
    text = result["text"]
    assert isinstance(text, str)
    assert "codex" in text.lower()
    assert "github" in text.lower()
    segments = result["segments"]
    assert isinstance(segments, list)
    assert len(segments) == 2
    assert segments[0]["text"].strip() == ""
    assert segments[1]["text"].strip()
    assert segments[1]["language"] == "English"
    assert segments[1]["start"] > 1190
    assert segments[1]["end"] == pytest.approx(1260)

    assert preserved_export.read_text(encoding="utf-8") == "existing transcript"
    assert list(cwd.iterdir()) == [preserved_export]
    assert not list(tmp_path.glob("mlx_output_*"))
