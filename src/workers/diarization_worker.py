"""One request-owned decode/diarization child; never starts a resident ASR engine."""

import argparse
import json
import subprocess
import wave
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import cast

from src.core.diarization_registry import cached_snapshot, lookup_diarizer
from src.core.diarization_result import DiarizationError, parse_payload
from src.core.mlx_nemotron_diarizer import MlxNemotronDiarizer
from src.workers.memory_guard import MemoryRuntime, allocation_guard

DECODED_BUDGET = 512 * 1024**2


def execute(audio: Path, output: Path, checkpoint: Path) -> None:
    spec = lookup_diarizer("nemotron-diar")
    if cached_snapshot(spec) != checkpoint:
        raise DiarizationError("model_unavailable", "Pinned local snapshot is unavailable", 503)
    decoded = output.parent / "decoded.wav"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(audio),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-fs",
            str(DECODED_BUDGET + 1024),
            str(decoded),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise DiarizationError("invalid_audio", "Audio decoding failed", 400)
    if decoded.stat().st_size >= DECODED_BUDGET:
        raise DiarizationError("input_limit", "Decoded audio exceeds the 512 MiB budget", 400)
    with wave.open(str(decoded), "rb") as f:
        duration = f.getnframes() / f.getframerate()
    runtime = cast(MemoryRuntime, import_module("mlx.core"))
    with allocation_guard(runtime, output.with_suffix(".memory.json"), 8 * 1024**3):
        diarizer = MlxNemotronDiarizer(checkpoint)
        try:
            diarizer.load()
            payload = parse_payload(
                {
                    "duration": duration,
                    "turns": [asdict(t) for t in diarizer.diarize_file(str(decoded))],
                }
            )
            output.write_text(json.dumps(asdict(payload)))
        finally:
            diarizer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    try:
        execute(args.audio, args.output, args.checkpoint)
    except DiarizationError as exc:
        args.output.write_text(
            json.dumps({"error": {"code": exc.code, "message": str(exc), "status": exc.status}})
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
