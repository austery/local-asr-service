from dataclasses import dataclass

from src.core.alignment_port import AlignedWord
from src.core.diarization_port import SpeakerTurn


@dataclass(frozen=True)
class ChunkWindow:
    index: int
    start: float
    end: float
    emit_start: float
    emit_end: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index ({self.index}) must be non-negative")
        if self.start < 0.0:
            raise ValueError(f"start ({self.start}) must be non-negative")
        if self.emit_start < self.start:
            raise ValueError(f"emit_start ({self.emit_start}) must be >= start ({self.start})")
        if self.emit_end <= self.emit_start:
            raise ValueError(
                f"emit_end ({self.emit_end}) must be > emit_start ({self.emit_start})"
            )
        if self.end < self.emit_end:
            raise ValueError(f"end ({self.end}) must be >= emit_end ({self.emit_end})")

    @property
    def duration(self) -> float:
        return self.end - self.start


def build_chunk_plan(
    *,
    duration_seconds: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[ChunkWindow]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be non-negative")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be less than chunk_seconds")

    if duration_seconds <= chunk_seconds:
        return [
            ChunkWindow(
                index=0,
                start=0.0,
                end=duration_seconds,
                emit_start=0.0,
                emit_end=duration_seconds,
            )
        ]

    windows: list[ChunkWindow] = []
    emit_start = 0.0
    index = 0
    while emit_start < duration_seconds:
        start = max(0.0, emit_start - overlap_seconds)
        emit_end = min(duration_seconds, emit_start + chunk_seconds - overlap_seconds)
        end = min(duration_seconds, emit_end + overlap_seconds)
        windows.append(
            ChunkWindow(
                index=index,
                start=round(start, 3),
                end=round(end, 3),
                emit_start=round(emit_start, 3),
                emit_end=round(emit_end, 3),
            )
        )
        emit_start = emit_end
        index += 1

    return windows


def _midpoint(start: float, end: float) -> float:
    return (start + end) / 2


def _is_midpoint_in_emit_window(start: float, end: float, window: ChunkWindow) -> bool:
    midpoint = _midpoint(start, end)
    return window.emit_start <= midpoint < window.emit_end


def offset_words_to_global_timeline(
    words: list[AlignedWord],
    window: ChunkWindow,
) -> list[AlignedWord]:
    result: list[AlignedWord] = []
    for word in words:
        global_start = round(window.start + word.start, 3)
        global_end = round(window.start + word.end, 3)
        if _is_midpoint_in_emit_window(global_start, global_end, window):
            result.append(AlignedWord(text=word.text, start=global_start, end=global_end))
    return result


def offset_turns_to_global_timeline(
    turns: list[SpeakerTurn],
    window: ChunkWindow,
) -> list[SpeakerTurn]:
    global_turns = [
        SpeakerTurn(
            speaker=turn.speaker,
            start=round(window.start + turn.start, 3),
            end=round(window.start + turn.end, 3),
        )
        for turn in turns
    ]
    return clip_turns_to_emit_window(global_turns, window)


def clip_turns_to_emit_window(
    turns: list[SpeakerTurn],
    window: ChunkWindow,
) -> list[SpeakerTurn]:
    result: list[SpeakerTurn] = []
    for turn in turns:
        clipped_start = max(window.emit_start, turn.start)
        clipped_end = min(window.emit_end, turn.end)
        if clipped_end <= clipped_start:
            continue
        if _is_midpoint_in_emit_window(clipped_start, clipped_end, window):
            result.append(
                SpeakerTurn(
                    speaker=turn.speaker,
                    start=round(clipped_start, 3),
                    end=round(clipped_end, 3),
                )
            )
    return result


def reconcile_chunk_speaker_labels(
    *,
    existing_turns: list[SpeakerTurn],
    chunk_turns: list[SpeakerTurn],
    overlap_start: float,
    overlap_end: float,
) -> list[SpeakerTurn]:
    if overlap_end <= overlap_start or not existing_turns or not chunk_turns:
        return chunk_turns

    scores: dict[str, dict[str, float]] = {}
    for chunk_turn in chunk_turns:
        for existing_turn in existing_turns:
            segment_start = max(
                chunk_turn.start,
                existing_turn.start,
                overlap_start,
            )
            segment_end = min(
                chunk_turn.end,
                existing_turn.end,
                overlap_end,
            )
            if segment_end <= segment_start:
                continue
            per_speaker_scores = scores.setdefault(chunk_turn.speaker, {})
            per_speaker_scores[existing_turn.speaker] = per_speaker_scores.get(
                existing_turn.speaker,
                0.0,
            ) + (segment_end - segment_start)

    candidates = [
        (score, chunk_speaker, existing_speaker)
        for chunk_speaker, existing_scores in scores.items()
        for existing_speaker, score in existing_scores.items()
    ]
    candidates.sort(reverse=True, key=lambda item: item[0])

    remap: dict[str, str] = {}
    used_existing: set[str] = set()
    for _score, chunk_speaker, existing_speaker in candidates:
        if chunk_speaker in remap or existing_speaker in used_existing:
            continue
        remap[chunk_speaker] = existing_speaker
        used_existing.add(existing_speaker)

    if not remap:
        return chunk_turns

    return [
        SpeakerTurn(
            speaker=remap.get(turn.speaker, turn.speaker),
            start=turn.start,
            end=turn.end,
        )
        for turn in chunk_turns
    ]


def validate_aligned_word_quality(
    words: list[AlignedWord],
    *,
    expected_duration_seconds: float,
    tail_word_count: int = 10,
) -> None:
    if not words:
        raise ValueError("alignment quality gate failed: no aligned words")

    previous_start = -1.0
    for word in words:
        if word.start < previous_start:
            raise ValueError("alignment quality gate failed: non-monotonic aligned words")
        previous_start = word.start

    if len(words) >= tail_word_count:
        tail = words[-tail_word_count:]
        tail_positions = {(word.start, word.end) for word in tail}
        if len(tail_positions) <= 2 and expected_duration_seconds - tail[-1].end > 60.0:
            raise ValueError("alignment quality gate failed: tail timestamp collapse")
