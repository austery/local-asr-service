"""
ASR Benchmark Runner

Measures transcription performance against a running local-asr-service instance.
Outputs RTF (Real-Time Factor), latency, and result summary.

Usage:
    # Benchmark with default fixture
    uv run python benchmarks/run.py

    # Benchmark a specific file
    uv run python benchmarks/run.py --file path/to/audio.wav

    # Benchmark against a different server
    uv run python benchmarks/run.py --base-url http://localhost:50070

    # Run all files in benchmarks/samples/
    uv run python benchmarks/run.py --all

    # Save results to JSON
    uv run python benchmarks/run.py --save
"""

import argparse
import json
import subprocess
import sys
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:50070"
FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
SAMPLES_DIR = Path(__file__).parent / "samples"
RESULTS_DIR = Path(__file__).parent / "results"


def get_audio_duration(file_path: Path) -> float:
    """Get audio duration in seconds. Tries wave module first, falls back to ffprobe."""
    if file_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(file_path), "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except wave.Error:
            pass

    # Fallback to ffprobe for non-WAV formats
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def get_server_info(base_url: str) -> dict[str, Any]:
    """Fetch current model info from the server."""
    resp = requests.get(f"{base_url}/v1/models/current", timeout=5)
    resp.raise_for_status()
    result: dict[str, Any] = resp.json()
    return result


def run_benchmark(
    file_path: Path,
    base_url: str,
    output_format: str = "json",
) -> dict[str, Any]:
    """Run a single benchmark against the transcription endpoint."""
    duration = get_audio_duration(file_path)
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    # Determine MIME type
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/x-m4a",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }
    mime_type = mime_map.get(file_path.suffix.lower(), "audio/wav")

    print(f"  File: {file_path.name}")
    print(f"  Duration: {duration:.1f}s ({duration / 60:.1f} min), Size: {file_size_mb:.2f}MB")
    print(f"  Format: {output_format}")

    # Send request
    start_time = time.time()
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": (file_path.name, f, mime_type)},
            data={"output_format": output_format},
            timeout=600,
        )
    elapsed = time.time() - start_time

    if resp.status_code != 200:
        return {
            "file": file_path.name,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            "elapsed_seconds": elapsed,
        }

    # Calculate metrics
    rtf = elapsed / duration if duration > 0 else 0
    speed_ratio = duration / elapsed if elapsed > 0 else 0

    # Parse response
    result_data: dict[str, Any] = {}
    if output_format == "json":
        result_data = resp.json()
        text = result_data.get("text", "")
        num_segments = len(result_data.get("segments") or [])
    else:
        text = resp.text
        num_segments = 0

    text_length = len(text)
    text_preview = text[:150].replace("\n", " ")

    return {
        "file": file_path.name,
        "audio_duration_s": round(duration, 1),
        "file_size_mb": round(file_size_mb, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 4),
        "speed_ratio": round(speed_ratio, 1),
        "num_segments": num_segments,
        "text_length": text_length,
        "text_preview": text_preview,
        "output_format": output_format,
    }


def print_result(result: dict[str, Any]) -> None:
    """Print a single benchmark result."""
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    print(f"  Elapsed: {result['elapsed_s']:.2f}s")
    print(f"  RTF: {result['rtf']:.4f} ({result['speed_ratio']:.1f}x realtime)")
    print(f"  Segments: {result['num_segments']}, Text length: {result['text_length']} chars")
    print(f"  Preview: {result['text_preview'][:100]}...")


def print_summary_table(results: list[dict[str, Any]]) -> None:
    """Print a summary table of all benchmark results."""
    print("\n" + "=" * 90)
    print(f"{'File':<30} {'Duration':>8} {'Elapsed':>8} {'RTF':>8} {'Speed':>8} {'Segments':>8}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['file']:<30} {'ERROR':>8}")
            continue
        print(
            f"{r['file']:<30} {r['audio_duration_s']:>7.1f}s {r['elapsed_s']:>7.2f}s "
            f"{r['rtf']:>8.4f} {r['speed_ratio']:>7.1f}x {r['num_segments']:>8}"
        )
    print("=" * 90)


def save_results(results: list[dict[str, Any]], server_info: dict[str, Any]) -> Path:
    """Save benchmark results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    engine = server_info.get("engine_type", "unknown")
    output_path = RESULTS_DIR / f"benchmark_{engine}_{timestamp}.json"

    payload = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "server": server_info,
        "results": results,
    }

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return output_path


def collect_files(args: argparse.Namespace) -> list[Path]:
    """Collect audio files to benchmark based on CLI args."""
    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"Error: file not found: {p}")
            sys.exit(1)
        return [p]

    if args.all:
        files = []
        for d in [SAMPLES_DIR, FIXTURE_DIR]:
            if d.exists():
                files.extend(
                    sorted(
                        f
                        for f in d.iterdir()
                        if f.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
                    )
                )
        if not files:
            print(f"No audio files found in {SAMPLES_DIR} or {FIXTURE_DIR}")
            sys.exit(1)
        return files

    # Default: use fixture
    default_fixture = FIXTURE_DIR / "two_speakers_60s.wav"
    if default_fixture.exists():
        return [default_fixture]

    # Try any audio in samples/
    if SAMPLES_DIR.exists():
        files = sorted(
            f
            for f in SAMPLES_DIR.iterdir()
            if f.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
        )
        if files:
            return files[:1]

    print("No audio files found. Either:")
    print(f"  - Place files in {SAMPLES_DIR}/")
    print(f"  - Create fixture at {default_fixture}")
    print("  - Use --file <path>")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR Benchmark Runner")
    parser.add_argument("--file", type=str, help="Path to a specific audio file")
    parser.add_argument(
        "--all", action="store_true", help="Benchmark all files in samples/ and fixtures/"
    )
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Server base URL")
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "txt", "srt"],
        help="Output format to request",
    )
    parser.add_argument("--save", action="store_true", help="Save results to benchmarks/results/")
    args = parser.parse_args()

    # 1. Check server
    print(f"Connecting to {args.base_url}...")
    try:
        server_info = get_server_info(args.base_url)
    except requests.ConnectionError:
        print(f"Error: cannot connect to {args.base_url}. Is the service running?")
        sys.exit(1)

    print(f"Engine: {server_info['engine_type']}, Model: {server_info['model_id']}")
    caps = server_info.get("capabilities", {})
    cap_flags = [k for k, v in caps.items() if v]
    print(f"Capabilities: {', '.join(cap_flags) if cap_flags else 'none'}")
    print()

    # 2. Collect files
    files = collect_files(args)

    # 3. Run benchmarks
    results: list[dict[str, Any]] = []
    for i, f in enumerate(files):
        print(f"[{i + 1}/{len(files)}] Benchmarking...")
        result = run_benchmark(f, args.base_url, output_format=args.format)
        print_result(result)
        results.append(result)
        print()

    # 4. Summary
    if len(results) > 1:
        print_summary_table(results)

    # 5. Save
    if args.save:
        output_path = save_results(results, server_info)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
