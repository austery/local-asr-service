"""Generic MLX exports belong to the request and survive no completion path."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.core.mlx_engine import MlxAudioEngine


@pytest.mark.parametrize("output_format", ["txt", "json", "srt", "vtt"])
@pytest.mark.parametrize("fail_second_chunk", [False, True])
def test_request_owned_exports_are_cleaned_without_touching_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    output_format: str, fail_second_chunk: bool,
) -> None:
    cwd = tmp_path / "working-directory"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    for suffix in ("json", "srt"):
        (cwd / f"transcript.{suffix}").write_text("existing transcript", encoding="utf-8")
    request_dir = tmp_path / "request"
    request_dir.mkdir()
    upload = request_dir / "original.wav"
    upload.write_bytes(b"request audio")
    chunks = [request_dir / f"chunk_{i}.wav" for i in range(2)]
    for chunk in chunks:
        chunk.write_bytes(b"chunk audio")
    exports: list[Path] = []

    def generate(
        *, output_path: str = "transcript", format: str = "txt", **options: object,
    ) -> SimpleNamespace:
        artifact = Path(f"{output_path}.{format}").resolve()
        artifact.write_text("generated export", encoding="utf-8")
        exports.append(artifact)
        if fail_second_chunk and len(exports) == 2:
            raise RuntimeError("export failed after writing")
        return SimpleNamespace(
            text="speech", language="English",
            segments=[{"text": "speech", "start": 0.0, "end": 1.0}],
        )

    with (
        patch("src.core.mlx_engine.AudioChunkingService") as chunking,
        patch("src.core.mlx_engine.generate_transcription", side_effect=generate),
    ):
        chunking.return_value.process_audio.return_value = [str(chunk) for chunk in chunks]
        engine = MlxAudioEngine()
        engine.model = object()
        if fail_second_chunk:
            with pytest.raises(RuntimeError, match="export failed after writing"):
                engine.transcribe_file(str(upload), output_format=output_format)
        else:
            result = engine.transcribe_file(str(upload), output_format=output_format)
            assert (result["text"] if isinstance(result, dict) else result) == "speech speech"

    assert len(exports) == 2
    assert all(request_dir in artifact.parents for artifact in exports)
    assert all(not artifact.exists() and not artifact.parent.exists() for artifact in exports)
    assert list(request_dir.iterdir()) == [upload]
    assert upload.read_bytes() == b"request audio"
    assert {p.name: p.read_text() for p in cwd.iterdir()} == {
        "transcript.json": "existing transcript", "transcript.srt": "existing transcript",
    }
