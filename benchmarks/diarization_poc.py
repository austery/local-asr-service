"""Isolated offline diarization runner; no HTTP endpoint or registry changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import psutil
from numpy.typing import NDArray

from benchmarks.diarization_metrics import Turn, read_rttm, read_uem, score_turns

FloatAudio = NDArray[np.float32]


class Segment(Protocol):
    speaker: str | int
    start: float
    end: float


class Output(Protocol):
    segments: list[Segment]
    speaker_probs: object


class Runtime(Protocol):
    def generate(
        self,
        audio: FloatAudio,
        *,
        threshold: float,
        min_duration: float,
        merge_gap: float,
        verbose: bool,
    ) -> Output: ...
    def init_streaming_state(self) -> object: ...
    # The protocol intentionally mirrors the upstream streaming API.
    def feed(  # noqa: PLR0913
        self,
        audio: FloatAudio,
        state: object,
        *,
        sample_rate: int,
        final: bool,
        threshold: float,
        min_duration: float,
        merge_gap: float,
    ) -> tuple[Output, object]: ...


class MemoryRuntime(Protocol):
    def get_active_memory(self) -> int: ...
    def get_cache_memory(self) -> int: ...
    def get_peak_memory(self) -> int: ...


@contextmanager
def allocation_guard(runtime: MemoryRuntime, path: Path, budget: int) -> Iterator[None]:
    """Stop this owned child on observed MLX pressure; limits alone are advisory.

    Sampling can overshoot between checks. The parent independently preserves
    system memory and deadlines; this thread never targets another process.
    """
    stopped = threading.Event()

    def monitor() -> None:
        peak_seen = 0
        while not stopped.is_set():
            try:
                active = runtime.get_active_memory()
                cache = runtime.get_cache_memory()
                peak_seen = max(peak_seen, runtime.get_peak_memory(), active + cache)
                record = {
                    "active_bytes": active,
                    "cache_bytes": cache,
                    "peak_observed_bytes": peak_seen,
                    "budget_bytes": budget,
                    "stop_reason": None
                    if peak_seen <= budget
                    else "MLX allocation exceeded owned budget",
                }
                path.write_text(json.dumps(record, indent=2))
                if peak_seen > budget:
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
            except Exception as exc:
                # A broken monitor must fail closed rather than run unprotected.
                try:
                    path.write_text(json.dumps({"stop_reason": f"MLX monitor failed: {exc}"}))
                finally:
                    os.kill(os.getpid(), signal.SIGTERM)
                return
            stopped.wait(0.05)

    thread = threading.Thread(target=monitor, name="poc-mlx-memory", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=2)


def read_audio(path: Path) -> FloatAudio:
    with wave.open(str(path), "rb") as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, 16000):
            raise ValueError("Prepare mono 16 kHz PCM16 WAV before evaluation")
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768


def protect(identities: dict[int, float]) -> None:
    for pid, created in identities.items():
        p = psutil.Process(pid)
        if not p.is_running() or p.create_time() != created:
            raise RuntimeError("Production identity changed; stop the owned experiment")
        for child in p.children(recursive=True):
            if child.pid in identities or child.status() == psutil.STATUS_ZOMBIE:
                continue
            if "resource_tracker" not in " ".join(child.cmdline()):
                raise RuntimeError("Production inference active; stop the owned experiment")


def supervise(args: argparse.Namespace) -> None:
    if args.output.suffix != ".json":
        raise ValueError("Output must use the .json suffix")
    if any(
        args.output.with_suffix(suffix).exists()
        for suffix in (".json", ".log", ".npz", ".rttm", ".supervision.json", ".memory.json")
    ):
        raise FileExistsError("Use a fresh output path to preserve previous evidence")
    identities = {p: psutil.Process(p).create_time() for p in args.protect_pid}
    protect(identities)
    if psutil.virtual_memory().available < 20 * 1024**3:
        raise RuntimeError("Less than 20 GiB available; retain service priority")
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".log").open("w") as log:
        p = subprocess.Popen(
            [sys.executable, "-m", "benchmarks.diarization_poc", *sys.argv[1:], "--worker"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.monotonic()
        peak_rss = 0
        minimum_available = psutil.virtual_memory().available
        stop_reason: str | None = None
        try:
            while p.poll() is None:
                protect(identities)
                minimum_available = min(minimum_available, psutil.virtual_memory().available)
                if minimum_available < 20 * 1024**3:
                    raise MemoryError("Available memory fell below 20 GiB; retain service priority")
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError("Owned inference exceeded its deadline")
                try:
                    rss = psutil.Process(p.pid).memory_info().rss
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    p.wait(timeout=5)
                    break
                peak_rss = max(peak_rss, rss)
                if rss > args.memory_gib * 1024**3:
                    raise MemoryError("Owned process exceeded its memory cap")
                time.sleep(0.25)
        except BaseException as exc:
            stop_reason = str(exc)
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    if p.poll() is None:
                        raise
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise
        finally:
            args.output.with_suffix(".supervision.json").write_text(
                json.dumps(
                    {
                        "returncode": p.poll(),
                        "elapsed_s": time.monotonic() - started,
                        "sampled_peak_rss_bytes": peak_rss,
                        "minimum_available_bytes": minimum_available,
                        "stop_reason": stop_reason,
                    },
                    indent=2,
                )
            )
    protect(identities)
    if p.returncode:
        raise RuntimeError(
            f"Owned inference exited {p.returncode}; inspect {args.output.with_suffix('.log')}"
        )


def run(args: argparse.Namespace) -> None:
    mx = import_module("mlx.core")
    mx.set_memory_limit(round(args.memory_gib * 1024**3))
    loader = import_module("mlx_audio.vad").load
    started = time.perf_counter()
    audio = read_audio(args.audio)
    decode_s = time.perf_counter() - started
    duration = len(audio) / 16000
    started = time.perf_counter()
    model = cast(Runtime, loader(str(args.checkpoint), strict=True))
    load_s = time.perf_counter() - started
    mx.reset_peak_memory()
    started = time.perf_counter()
    outputs: list[Output] = []
    if args.mode == "feed":
        if args.model != "nemotron":
            raise ValueError("Incremental PCM probe is defined for Nemotron only")
        state = model.init_streaming_state()
        size = max(1, round(args.feed_seconds * 16000))
        for start in range(0, len(audio), size):
            result, state = model.feed(
                audio[start : start + size],
                state,
                sample_rate=16000,
                final=False,
                threshold=args.threshold,
                min_duration=args.min_duration,
                merge_gap=args.merge_gap,
            )
            outputs.append(result)
        result, state = model.feed(
            np.zeros(0, dtype=np.float32),
            state,
            sample_rate=16000,
            final=True,
            threshold=args.threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
        )
        outputs.append(result)
    else:
        outputs = [
            model.generate(
                audio,
                threshold=args.threshold,
                min_duration=args.min_duration,
                merge_gap=args.merge_gap,
                verbose=False,
            )
        ]
    mx.eval(*[o.speaker_probs for o in outputs])
    mx.synchronize()
    inference_s = time.perf_counter() - started
    raw_turns = [
        Turn(str(s.speaker), float(s.start), float(s.end)) for o in outputs for s in o.segments
    ]
    timeline_valid = not any(t.end > duration + 1e-5 for t in raw_turns)
    scored_turns = [
        Turn(t.speaker, t.start, min(t.end, duration)) for t in raw_turns if t.start < duration
    ]
    probabilities = np.concatenate([np.asarray(o.speaker_probs) for o in outputs], axis=0)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite speaker activity")
    reset_check: dict[str, object] | None = None
    if args.reset_check:
        probe_started = time.perf_counter()
        quiet = model.generate(
            np.zeros(16000, dtype=np.float32),
            threshold=args.threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
            verbose=False,
        )
        repeated = model.generate(
            audio,
            threshold=args.threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
            verbose=False,
        )
        mx.eval(quiet.speaker_probs, repeated.speaker_probs)
        mx.synchronize()
        repeated_probs = np.asarray(repeated.speaker_probs)
        same_shape = repeated_probs.shape == probabilities.shape
        repeated_turns = [
            Turn(str(t.speaker), float(t.start), float(t.end)) for t in repeated.segments
        ]
        reset_check = {
            "elapsed_s": time.perf_counter() - probe_started,
            "intervening_silence_speakers": len({t.speaker for t in quiet.segments}),
            "same_shape": same_shape,
            "max_probability_difference": float(np.max(np.abs(probabilities - repeated_probs)))
            if same_shape and probabilities.size
            else None,
            "identical_turns": raw_turns == repeated_turns,
        }
    scores = []
    if args.reference:
        reference = read_rttm(args.reference, args.recording_id, duration, args.reference_offset)
        regions = (
            read_uem(args.uem, args.recording_id, duration, args.reference_offset)
            if args.uem
            else None
        )
        scores = [
            asdict(score_turns(reference, scored_turns, duration, c, regions)) for c in (0.0, 0.25)
        ]
    record = {
        "reset_check": reset_check,
        "status": "ok" if timeline_valid else "invalid_timeline",
        "timeline_valid": timeline_valid,
        "score_scope": "Intervals cropped to the recording/UEM for diagnostic scoring; invalid native bounds remain a failed output gate",
        "uem": str(args.uem) if args.uem else None,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "audio": str(args.audio),
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "duration_s": duration,
        "decode_s": decode_s,
        "load_s": load_s,
        "inference_s": inference_s,
        "rtf_inference": inference_s / duration if duration else None,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "versions": {p: version(p) for p in ("mlx-audio", "mlx", "numpy")},
        "options": {
            "mode": args.mode,
            "threshold": args.threshold,
            "min_duration": args.min_duration,
            "merge_gap": args.merge_gap,
            "feed_seconds": args.feed_seconds,
        },
        "raw_turns": [asdict(t) for t in raw_turns],
        "speakers": len({t.speaker for t in raw_turns}),
        "probability_shape": list(probabilities.shape),
        "scores": scores,
        "reference": str(args.reference) if args.reference else None,
        "reference_offset_s": args.reference_offset,
        "cpwer": None,
        "cpwer_reason": "No transcript-attribution integration in this PoC",
    }
    np.savez_compressed(args.output.with_suffix(".npz"), probabilities=probabilities)
    args.output.write_text(json.dumps(record, indent=2))
    args.output.with_suffix(".rttm").write_text(
        "".join(
            f"SPEAKER {args.audio.stem} 1 {t.start:.6f} {t.end - t.start:.6f} <NA> <NA> {t.speaker} <NA> <NA>\n"
            for t in raw_turns
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("nemotron", "sortformer"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--uem", type=Path)
    p.add_argument("--recording-id", default="")
    p.add_argument("--reference-offset", type=float, default=0)
    p.add_argument("--mode", choices=("whole", "feed"), default="whole")
    p.add_argument("--feed-seconds", type=float, default=1.0)
    p.add_argument("--threshold", type=float)
    p.add_argument("--min-duration", type=float)
    p.add_argument("--merge-gap", type=float)
    p.add_argument("--protect-pid", type=int, action="append", required=True)
    p.add_argument("--memory-gib", type=float, default=8)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument(
        "--reset-check",
        action="store_true",
        help="Probe A/silence/A on the same loaded model; timing recorded separately",
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    defaults = (0.5, 0.0, 0.0) if args.model == "nemotron" else (0.35, 0.2, 0.3)
    for name, default in zip(("threshold", "min_duration", "merge_gap"), defaults, strict=True):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if not all(
        math.isfinite(value)
        for value in (
            args.feed_seconds,
            args.timeout,
            args.memory_gib,
            args.threshold,
            args.min_duration,
            args.merge_gap,
            args.reference_offset,
        )
    ):
        p.error("Numeric options must be finite")
    if args.reference_offset < 0:
        p.error("Reference offset must be nonnegative")
    if args.feed_seconds <= 0 or args.timeout <= 0 or args.memory_gib <= 0:
        p.error("Chunk duration, timeout and memory cap must be positive")
    if not 0 <= args.threshold <= 1 or min(args.min_duration, args.merge_gap) < 0:
        p.error("Invalid postprocessing parameters")
    if args.output.suffix != ".json":
        p.error("Output must use the .json suffix")
    if args.reset_check and args.mode != "whole":
        p.error("State-reset probe uses whole-file generation")
    if args.reference and not args.recording_id:
        p.error("A reference recording ID is required for scoring")
    if args.uem and not args.reference:
        p.error("UEM requires a reference")
    if args.worker:
        try:
            runtime = cast(MemoryRuntime, import_module("mlx.core"))
            with allocation_guard(
                runtime, args.output.with_suffix(".memory.json"), round(args.memory_gib * 1024**3)
            ):
                run(args)
        except Exception as exc:
            args.output.write_text(
                json.dumps(
                    {
                        "status": "error",
                        "model": args.model,
                        "audio": str(args.audio),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
            raise
    else:
        supervise(args)


if __name__ == "__main__":
    main()
