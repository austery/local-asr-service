import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from src.core.model_registry import lookup
from src.core.pipeline_registry import lookup_profile
from src.services.transcription import TranscriptionService


def _summarize_result(result: object) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"type": type(result).__name__}

    segments = result.get("segments")
    segment_count = len(segments) if isinstance(segments, list) else 0
    duration = result.get("duration")
    text = result.get("text")
    speakers = sorted({
        str(segment.get("speaker"))
        for segment in segments
        if isinstance(segment, dict) and segment.get("speaker") is not None
    }) if isinstance(segments, list) else []

    return {
        "type": "dict",
        "duration": duration,
        "text_length": len(text) if isinstance(text, str) else 0,
        "segment_count": segment_count,
        "speakers": speakers,
    }


async def _run_probe(audio_path: Path, output_dir: Path, language: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    qwen_spec = lookup("qwen3-asr")
    profile = replace(lookup_profile("qwen3-sortformer"), requestable=True)
    service = TranscriptionService(
        engine_type=qwen_spec.engine_type,
        model_id=qwen_spec.model_id,
        initial_model_spec=qwen_spec,
        idle_timeout=0,
    )

    await service.start_worker()
    try:
        with audio_path.open("rb") as handle:
            result = await service.submit_pipeline(
                UploadFile(file=handle, filename=audio_path.name),
                {"output_format": "json", "language": language},
                request_id="probe-qwen3-sortformer-longform",
                profile=profile,
            )

        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(_summarize_result(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    finally:
        await service.stop_worker()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real-runtime qwen3-sortformer probe.")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    asyncio.run(_run_probe(args.audio, args.output, args.language))


if __name__ == "__main__":
    main()
