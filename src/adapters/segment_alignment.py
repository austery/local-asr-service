import math

from src.core.diarization_port import SpeakerTurn


def _read_timestamp(segment: dict[str, object], field: str) -> float:
    if field not in segment:
        raise ValueError(f"segment missing required timestamp: {field}")
    value = segment[field]
    if isinstance(value, bool):
        raise ValueError(f"segment timestamp must be numeric: {field}")
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"segment timestamp must be numeric: {field}") from exc
    if not math.isfinite(timestamp):
        raise ValueError(f"segment timestamp must be finite: {field}")
    return timestamp


def align_speakers(
    transcript_segments: list[dict[str, object]],
    speaker_turns: list[SpeakerTurn],
) -> list[dict[str, object]]:
    aligned: list[dict[str, object]] = []
    for segment in transcript_segments:
        start = _read_timestamp(segment, "start")
        end = _read_timestamp(segment, "end")
        if end <= start:
            raise ValueError(f"Invalid segment interval: end ({end}) must be > start ({start})")
        best_speaker = "Unknown"
        best_overlap = 0.0

        for turn in speaker_turns:
            overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker

        merged = dict(segment)
        merged["speaker"] = best_speaker
        aligned.append(merged)

    return aligned
