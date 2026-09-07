"""Validate upstream MOSS output and normalize public text/speaker fields.

No model inference or speaker reconciliation lives here. Gap rejection is a
conservative continuous-speech contract, not a detector of every missing word.
"""

import math
import re
from dataclasses import dataclass

_SEGMENT_PATTERN = re.compile(
    r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.DOTALL
)
_MAX_GAP_SECONDS = 10.0
_ENDPOINT_TOLERANCE_SECONDS = 0.25


@dataclass(frozen=True)
class _Segment:
    start: float
    end: float
    speaker: str
    text: str

    def to_dict(self) -> dict[str, object]:
        return {"start": self.start, "end": self.end, "speaker": self.speaker, "text": self.text}


def _timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("MOSS produced a non-finite or invalid timestamp.")
    return float(value)


def _normalize_segment(value: object, duration: float) -> _Segment:
    if not isinstance(value, dict):
        raise ValueError("MOSS produced an invalid segment.")
    start, end = _timestamp(value.get("start")), _timestamp(value.get("end"))
    speaker, text = value.get("speaker_id"), value.get("text")
    if not isinstance(speaker, str) or re.fullmatch(r"S\d+", speaker) is None:
        raise ValueError("MOSS produced a segment without a valid speaker label.")
    if not isinstance(text, str) or not text.startswith(f"[{speaker}]"):
        raise ValueError("MOSS segment text and speaker label disagree.")
    text = text.removeprefix(f"[{speaker}]").strip()
    if not text or start < 0 or end <= start or end > duration + _ENDPOINT_TOLERANCE_SECONDS:
        raise ValueError("MOSS produced empty text or an out-of-bounds segment.")
    end = min(end, duration)
    if start >= end:
        raise ValueError("MOSS segment starts outside the recording.")
    return _Segment(start, end, speaker, text)


def _validate_raw_text(raw: object, segments: list[_Segment], duration: float) -> None:
    if not isinstance(raw, str):
        raise ValueError("MOSS returned no raw transcript.")
    matches = list(_SEGMENT_PATTERN.finditer(raw))
    if len(matches) != len(segments) or _SEGMENT_PATTERN.sub("", raw).strip():
        raise ValueError("MOSS output is incomplete or contains unparsed text; use a shorter recording.")
    for match, segment in zip(matches, segments, strict=True):
        expected = _Segment(
            float(match[1]), min(float(match[4]), duration), match[2], match[3].strip()
        )
        if expected != segment:
            raise ValueError("MOSS raw transcript and parsed segments disagree.")


def _validate_coverage(segments: list[_Segment], duration: float) -> None:
    covered_until = 0.0
    previous_start = 0.0
    for segment in segments:
        if segment.start < previous_start:
            raise ValueError("MOSS produced unordered timestamps.")
        if segment.start - covered_until > _MAX_GAP_SECONDS:
            raise ValueError("MOSS output has an uncovered span over 10 seconds; use a shorter recording.")
        previous_start = segment.start
        covered_until = max(covered_until, segment.end)
    if duration - covered_until > _MAX_GAP_SECONDS:
        raise ValueError("MOSS output leaves an uncovered tail over 10 seconds; use a shorter recording.")


def normalize_moss_output(result: object, duration: float, max_tokens: int) -> dict[str, object]:
    """Reject detectable truncation and return the gateway's JSON shape."""
    tokens = getattr(result, "generation_tokens", None)
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 < tokens < max_tokens:
        raise ValueError("MOSS output token budget was exhausted or token metadata is invalid.")
    raw_segments = getattr(result, "segments", None)
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("MOSS returned no speaker segments.")
    segments = [_normalize_segment(segment, duration) for segment in raw_segments]
    _validate_raw_text(getattr(result, "text", None), segments, duration)
    _validate_coverage(segments, duration)
    return {
        "text": " ".join(segment.text for segment in segments),
        "segments": [segment.to_dict() for segment in segments],
        "duration": duration,
        "language": "en",
    }


def _srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds_int, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"


def format_moss_output(result: dict[str, object], output_format: str, with_timestamp: bool) -> str:
    """Render already validated segments in the existing text/SRT contract."""
    raw_segments = result["segments"]
    assert isinstance(raw_segments, list)
    lines: list[str] = []
    for index, segment in enumerate(raw_segments, 1):
        assert isinstance(segment, dict)
        start, end = _timestamp(segment["start"]), _timestamp(segment["end"])
        text = f"[{segment['speaker']}]: {segment['text']}"
        if output_format == "srt":
            lines.append(f"{index}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n{text}\n")
        elif with_timestamp:
            lines.append(f"[{int(start // 60):02}:{int(start % 60):02}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines)
