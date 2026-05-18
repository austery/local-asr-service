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
    result: list[SpeakerTurn] = []
    for turn in turns:
        global_start = max(window.emit_start, round(window.start + turn.start, 3))
        global_end = min(window.emit_end, round(window.start + turn.end, 3))
        if global_end <= global_start:
            continue
        if _is_midpoint_in_emit_window(global_start, global_end, window):
            result.append(SpeakerTurn(speaker=turn.speaker, start=global_start, end=global_end))
    return result


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
