"""SPEC-014 Phase 3 Apple Speech evaluation harness."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import requests
from requests import Response

JsonObject = dict[str, object]
SegmentFieldState = Literal["missing", "null", "empty", "non_empty", "invalid"]
DEFAULT_BASE_URL = "http://localhost:50700"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass(frozen=True)
class SegmentSummary:
    field_state: SegmentFieldState
    count: int
    monotonic: bool | None
    missing_timing_count: int
    zero_or_negative_duration_count: int

    def to_json(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class JsonResponseSummary:
    model: str | None
    language: str | None
    duration: float | None
    text_length: int
    text_preview: str
    segments: SegmentSummary

    def to_json(self) -> JsonObject:
        data = asdict(self)
        data["segments"] = self.segments.to_json()
        return data


@dataclass(frozen=True)
class SrtSummary:
    cue_count: int
    valid_format: bool
    monotonic: bool | None

    def to_json(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class ProbeResult:
    model: str
    language: str
    file_name: str
    audio_duration_s: float
    file_size_mb: float
    elapsed_s: float
    rtf: float
    speed_ratio: float
    status_code: int
    json_summary: JsonResponseSummary | None
    srt_summary: SrtSummary | None
    peak_rss_mb: float | None
    error: str | None

    def to_json(self) -> JsonObject:
        return {
            "model": self.model,
            "language": self.language,
            "file_name": self.file_name,
            "audio_duration_s": self.audio_duration_s,
            "file_size_mb": self.file_size_mb,
            "elapsed_s": self.elapsed_s,
            "rtf": self.rtf,
            "speed_ratio": self.speed_ratio,
            "status_code": self.status_code,
            "json_summary": self.json_summary.to_json() if self.json_summary else None,
            "srt_summary": self.srt_summary.to_json() if self.srt_summary else None,
            "peak_rss_mb": self.peak_rss_mb,
            "error": self.error,
        }


@dataclass(frozen=True)
class ProbeIdentity:
    model: str
    language: str
    file_name: str
    audio_duration_s: float
    file_size_mb: float


@dataclass(frozen=True)
class ProbeOutcome:
    elapsed_s: float
    status_code: int
    json_summary: JsonResponseSummary | None
    srt_summary: SrtSummary | None
    peak_rss_mb: float | None
    error: str | None


@dataclass(frozen=True)
class ProbeRequest:
    file_path: Path
    audio_duration_s: float
    base_url: str
    model: str
    language: str
    server_pid: int | None
    timeout_seconds: float


@dataclass(frozen=True)
class ProcessRssRow:
    pid: int
    ppid: int
    rss_kb: int


class PeakRssSampler:
    """Poll one process tree RSS while a probe request is in flight."""

    def __init__(self, pid: int | None, interval_seconds: float = 0.25) -> None:
        self._pid = pid
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_mb: float | None = None

    def __enter__(self) -> PeakRssSampler:
        if self._pid is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def peak_rss_mb(self) -> float | None:
        return self._peak_rss_mb

    def _poll(self) -> None:
        if self._pid is None:
            return
        while not self._stop_event.is_set():
            rss_mb = sample_process_tree_rss_mb(self._pid)
            if rss_mb is not None and (
                self._peak_rss_mb is None or rss_mb > self._peak_rss_mb
            ):
                self._peak_rss_mb = rss_mb
            self._stop_event.wait(self._interval_seconds)


_SRT_TIME_PATTERN = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3}) --> (?P<end>\d{2}:\d{2}:\d{2},\d{3})$"
)


def build_request_data(model: str, language: str, output_format: str) -> dict[str, str]:
    """Build the multipart form fields for one explicit model/language probe."""
    return {
        "model": model,
        "language": language,
        "output_format": output_format,
    }


def analyze_segments(payload: dict[str, object]) -> SegmentSummary:
    """Summarize the response `segments` field without collapsing null and empty list."""
    if "segments" not in payload:
        return SegmentSummary("missing", 0, None, 0, 0)

    segments_obj = payload["segments"]
    if segments_obj is None:
        return SegmentSummary("null", 0, None, 0, 0)
    if not isinstance(segments_obj, list):
        return SegmentSummary("invalid", 0, None, 0, 0)
    if not segments_obj:
        return SegmentSummary("empty", 0, None, 0, 0)

    count = 0
    missing_timing_count = 0
    zero_or_negative_duration_count = 0
    monotonic = True
    previous_end: float | None = None

    for item in segments_obj:
        if not isinstance(item, dict):
            missing_timing_count += 1
            monotonic = False
            continue
        count += 1
        start = _number_or_none(item.get("start"))
        end = _number_or_none(item.get("end"))
        if start is None or end is None:
            missing_timing_count += 1
            monotonic = False
            continue
        if end <= start:
            zero_or_negative_duration_count += 1
        if previous_end is not None and start < previous_end:
            monotonic = False
        previous_end = end

    if count == 0 and missing_timing_count > 0:
        return SegmentSummary(
            field_state="invalid",
            count=0,
            monotonic=False,
            missing_timing_count=missing_timing_count,
            zero_or_negative_duration_count=0,
        )

    return SegmentSummary(
        field_state="non_empty",
        count=count,
        monotonic=monotonic,
        missing_timing_count=missing_timing_count,
        zero_or_negative_duration_count=zero_or_negative_duration_count,
    )


def summarize_json_response(payload: dict[str, object]) -> JsonResponseSummary:
    """Extract stable Phase 3 evidence fields from a JSON transcription response."""
    text_obj = payload.get("text")
    text = text_obj if isinstance(text_obj, str) else ""
    model_obj = payload.get("model")
    language_obj = payload.get("language")
    return JsonResponseSummary(
        model=model_obj if isinstance(model_obj, str) else None,
        language=language_obj if isinstance(language_obj, str) else None,
        duration=_number_or_none(payload.get("duration")),
        text_length=len(text),
        text_preview=_preview_text(text),
        segments=analyze_segments(payload),
    )


def summarize_srt_text(text: str) -> SrtSummary:
    """Summarize whether SRT output has parseable, monotonic cue ranges."""
    cue_ranges: list[tuple[float, float]] = []
    lines = [line.strip() for line in text.splitlines()]
    for line in lines:
        match = _SRT_TIME_PATTERN.match(line)
        if match is None:
            continue
        cue_ranges.append(
            (
                _parse_srt_time_seconds(match.group("start")),
                _parse_srt_time_seconds(match.group("end")),
            )
        )

    if not cue_ranges:
        return SrtSummary(cue_count=0, valid_format=False, monotonic=None)

    monotonic = True
    previous_end: float | None = None
    for start, end in cue_ranges:
        if end <= start:
            monotonic = False
        if previous_end is not None and start < previous_end:
            monotonic = False
        previous_end = end

    return SrtSummary(cue_count=len(cue_ranges), valid_format=True, monotonic=monotonic)


def build_probe_result(identity: ProbeIdentity, outcome: ProbeOutcome) -> ProbeResult:
    """Create one normalized Phase 3 probe result."""
    return ProbeResult(
        model=identity.model,
        language=identity.language,
        file_name=identity.file_name,
        audio_duration_s=round(identity.audio_duration_s, 3),
        file_size_mb=round(identity.file_size_mb, 3),
        elapsed_s=round(outcome.elapsed_s, 3),
        rtf=(
            round(outcome.elapsed_s / identity.audio_duration_s, 4)
            if identity.audio_duration_s > 0
            else 0.0
        ),
        speed_ratio=(
            round(identity.audio_duration_s / outcome.elapsed_s, 3)
            if outcome.elapsed_s > 0
            else 0.0
        ),
        status_code=outcome.status_code,
        json_summary=outcome.json_summary,
        srt_summary=outcome.srt_summary,
        peak_rss_mb=outcome.peak_rss_mb,
        error=outcome.error,
    )


def parse_ps_rss_kb(stdout: str) -> int | None:
    """Parse `ps -o rss=` output into RSS KiB."""
    stripped = stdout.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def parse_ps_process_table(stdout: str) -> list[ProcessRssRow]:
    """Parse `ps -axo pid=,ppid=,rss=` output."""
    rows: list[ProcessRssRow] = []
    for raw_line in stdout.splitlines():
        parts = raw_line.split()
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss_kb = int(parts[2])
        except ValueError:
            continue
        rows.append(ProcessRssRow(pid=pid, ppid=ppid, rss_kb=rss_kb))
    return rows


def process_tree_rss_mb_from_table(root_pid: int, rows: list[ProcessRssRow]) -> float | None:
    """Sum RSS for a root process and all descendants from a parsed process table."""
    rows_by_pid = {row.pid: row for row in rows}
    if root_pid not in rows_by_pid:
        return None

    children_by_parent: dict[int, list[int]] = {}
    for row in rows:
        children_by_parent.setdefault(row.ppid, []).append(row.pid)

    to_visit = [root_pid]
    visited: set[int] = set()
    total_rss_kb = 0
    while to_visit:
        pid = to_visit.pop()
        if pid in visited:
            continue
        visited.add(pid)
        current_row = rows_by_pid.get(pid)
        if current_row is not None:
            total_rss_kb += current_row.rss_kb
        to_visit.extend(children_by_parent.get(pid, []))

    return round(total_rss_kb / 1024, 3)


def sample_process_rss_mb(pid: int) -> float | None:
    """Sample one process RSS via POSIX `ps`; returns MiB."""
    if pid <= 0:
        return None
    completed = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    rss_kb = parse_ps_rss_kb(completed.stdout)
    if rss_kb is None:
        return None
    return round(rss_kb / 1024, 3)


def sample_process_tree_rss_mb(pid: int) -> float | None:
    """Sample a process plus descendants RSS via POSIX `ps`; returns MiB."""
    if pid <= 0:
        return None
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return process_tree_rss_mb_from_table(pid, parse_ps_process_table(completed.stdout))


def run_json_probe(request: ProbeRequest) -> ProbeResult:
    """Run one JSON probe against the OpenAI-compatible transcription endpoint."""
    mime_type = _mime_type_for_path(request.file_path)
    post_data = build_request_data(request.model, request.language, "json")
    identity = _identity_from_request(request)
    started = time.time()
    with PeakRssSampler(request.server_pid) as sampler, request.file_path.open("rb") as audio_file:
        response = requests.post(
            f"{request.base_url}/v1/audio/transcriptions",
            files={"file": (request.file_path.name, audio_file, mime_type)},
            data=post_data,
            timeout=request.timeout_seconds,
        )
    elapsed = time.time() - started

    if response.status_code != 200:
        return build_probe_result(
            identity,
            ProbeOutcome(
                elapsed_s=elapsed,
                status_code=response.status_code,
                json_summary=None,
                srt_summary=None,
                peak_rss_mb=sampler.peak_rss_mb,
                error=response.text[:500],
            ),
        )

    payload_obj, decode_error = decode_json_response(response)
    if decode_error is not None or payload_obj is None:
        return build_probe_result(
            identity,
            ProbeOutcome(
                elapsed_s=elapsed,
                status_code=response.status_code,
                json_summary=None,
                srt_summary=None,
                peak_rss_mb=sampler.peak_rss_mb,
                error=decode_error or "JSON response body is not an object",
            ),
        )

    return build_probe_result(
        identity,
        ProbeOutcome(
            elapsed_s=elapsed,
            status_code=response.status_code,
            json_summary=summarize_json_response(payload_obj),
            srt_summary=None,
            peak_rss_mb=sampler.peak_rss_mb,
            error=None,
        ),
    )


def decode_json_response(response: Response) -> tuple[JsonObject | None, str | None]:
    """Decode a requests response body as a JSON object for result summarization."""
    return decode_json_object(response.text)


def decode_json_object(text: str) -> tuple[JsonObject | None, str | None]:
    """Decode text as a JSON object, returning a structured error instead of raising."""
    try:
        decoded: object = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"Response body is not valid JSON: {exc.msg}"
    if not isinstance(decoded, dict):
        return None, "JSON response body is not an object"
    return decoded, None


def run_srt_probe(
    *,
    file_path: Path,
    base_url: str,
    model: str,
    language: str,
    timeout_seconds: float,
) -> SrtSummary:
    """Run a secondary SRT probe for one model/file/language tuple."""
    with file_path.open("rb") as audio_file:
        response = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": (file_path.name, audio_file, _mime_type_for_path(file_path))},
            data=build_request_data(model, language, "srt"),
            timeout=timeout_seconds,
        )
    if response.status_code != 200:
        return SrtSummary(cue_count=0, valid_format=False, monotonic=None)
    return summarize_srt_text(response.text)


def save_report(results: list[ProbeResult], output_dir: Path = RESULTS_DIR) -> Path:
    """Persist a Phase 3 report under benchmarks/results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"phase3_{timestamp}.json"
    payload = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "purpose": "SPEC-014 Phase 3 usability and recommendation evidence",
        "results": [result.to_json() for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return output_path


def _number_or_none(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _preview_text(text: str, limit: int = 200) -> str:
    normalized = " ".join(text.split())
    return normalized[:limit]


def _parse_srt_time_seconds(value: str) -> float:
    hours_text, minutes_text, rest = value.split(":")
    seconds_text, millis_text = rest.split(",")
    return (
        int(hours_text) * 3600
        + int(minutes_text) * 60
        + int(seconds_text)
        + int(millis_text) / 1000
    )


def _mime_type_for_path(file_path: Path) -> str:
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/x-m4a",
        ".mp4": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".webm": "audio/webm",
    }
    return mime_map.get(file_path.suffix.lower(), "audio/wav")


def _file_size_mb(file_path: Path) -> float:
    return file_path.stat().st_size / (1024 * 1024)


def _identity_from_request(request: ProbeRequest) -> ProbeIdentity:
    return ProbeIdentity(
        model=request.model,
        language=request.language,
        file_name=request.file_path.name,
        audio_duration_s=request.audio_duration_s,
        file_size_mb=_file_size_mb(request.file_path),
    )


def _probe_duration_seconds(file_path: Path) -> float:
    completed = subprocess.run(
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
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {file_path}: {completed.stderr.strip()}")
    return parse_ffprobe_duration_stdout(completed.stdout, file_path)


def parse_ffprobe_duration_stdout(stdout: str, file_path: Path) -> float:
    """Parse ffprobe duration output into seconds with actionable errors."""
    stripped = stdout.strip()
    if not stripped:
        raise RuntimeError(f"ffprobe returned empty duration for {file_path}")
    try:
        return float(stripped)
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned non-numeric duration for {file_path}: {stripped!r}"
        ) from exc


def _print_result(result: ProbeResult) -> None:
    if result.error:
        print(f"{result.model}: ERROR HTTP {result.status_code}: {result.error[:160]}")
        return
    segments_state = result.json_summary.segments.field_state if result.json_summary else "missing"
    peak = f", peak_rss={result.peak_rss_mb:.1f}MB" if result.peak_rss_mb is not None else ""
    print(
        f"{result.model}: elapsed={result.elapsed_s:.2f}s rtf={result.rtf:.4f} "
        f"speed={result.speed_ratio:.1f}x segments={segments_state}{peak}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SPEC-014 Phase 3 evaluation harness")
    parser.add_argument("--file", required=True, type=Path, help="Audio file to probe")
    parser.add_argument("--language", required=True, help="Explicit language/locale, e.g. zh-CN")
    parser.add_argument("--models", nargs="+", required=True, help="Model aliases to compare")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Service base URL")
    parser.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="Optional service PID for process-tree RSS sampling",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="Per-request timeout seconds")
    parser.add_argument("--save", action="store_true", help="Save JSON report")
    parser.add_argument("--srt-probe", action="store_true", help="Also run output_format=srt")
    args = parser.parse_args()

    file_path = args.file
    if not file_path.exists():
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    duration = _probe_duration_seconds(file_path)
    results: list[ProbeResult] = []
    for model in args.models:
        print(f"Running {model} with language={args.language} on {file_path.name}...")
        result = run_json_probe(
            ProbeRequest(
                file_path=file_path,
                audio_duration_s=duration,
                base_url=args.base_url,
                model=model,
                language=args.language,
                server_pid=args.server_pid,
                timeout_seconds=args.timeout,
            )
        )
        if args.srt_probe and result.error is None:
            srt_summary = run_srt_probe(
                file_path=file_path,
                base_url=args.base_url,
                model=model,
                language=args.language,
                timeout_seconds=args.timeout,
            )
            result = build_probe_result(
                ProbeIdentity(
                    model=result.model,
                    language=result.language,
                    file_name=result.file_name,
                    audio_duration_s=result.audio_duration_s,
                    file_size_mb=result.file_size_mb,
                ),
                ProbeOutcome(
                    elapsed_s=result.elapsed_s,
                    status_code=result.status_code,
                    json_summary=result.json_summary,
                    srt_summary=srt_summary,
                    peak_rss_mb=result.peak_rss_mb,
                    error=result.error,
                ),
            )
        _print_result(result)
        results.append(result)

    if args.save:
        output_path = save_report(results)
        print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
