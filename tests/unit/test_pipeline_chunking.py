import pytest

from src.adapters.pipeline_chunking import (
    ChunkWindow,
    build_chunk_plan,
    clip_turns_to_emit_window,
    offset_turns_to_global_timeline,
    offset_words_to_global_timeline,
    reconcile_chunk_speaker_labels,
    validate_aligned_word_quality,
)
from src.core.alignment_port import AlignedWord
from src.core.diarization_port import SpeakerTurn


def test_build_chunk_plan_should_cover_full_duration_without_emit_gaps() -> None:
    plan = build_chunk_plan(
        duration_seconds=18_000.0,
        chunk_seconds=900.0,
        overlap_seconds=15.0,
    )

    assert plan[0] == ChunkWindow(index=0, start=0.0, end=900.0, emit_start=0.0, emit_end=885.0)
    assert plan[-1].end == 18_000.0
    assert plan[-1].emit_end == 18_000.0
    assert [window.index for window in plan] == list(range(len(plan)))
    assert all(plan[i].emit_end == plan[i + 1].emit_start for i in range(len(plan) - 1))
    assert all(window.start <= window.emit_start < window.emit_end <= window.end for window in plan)


def test_build_chunk_plan_should_return_one_chunk_for_short_audio() -> None:
    plan = build_chunk_plan(
        duration_seconds=60.0,
        chunk_seconds=900.0,
        overlap_seconds=15.0,
    )

    assert plan == [ChunkWindow(index=0, start=0.0, end=60.0, emit_start=0.0, emit_end=60.0)]


def test_build_chunk_plan_should_reject_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_seconds must be less than chunk_seconds"):
        build_chunk_plan(duration_seconds=120.0, chunk_seconds=60.0, overlap_seconds=60.0)


def test_chunk_window_should_reject_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="index"):
        ChunkWindow(index=-1, start=0.0, end=60.0, emit_start=0.0, emit_end=60.0)
    with pytest.raises(ValueError, match="start"):
        ChunkWindow(index=0, start=-1.0, end=60.0, emit_start=0.0, emit_end=60.0)
    with pytest.raises(ValueError, match="emit_start"):
        ChunkWindow(index=0, start=10.0, end=60.0, emit_start=5.0, emit_end=60.0)
    with pytest.raises(ValueError, match="emit_end"):
        ChunkWindow(index=0, start=0.0, end=60.0, emit_start=30.0, emit_end=30.0)
    with pytest.raises(ValueError, match="end"):
        ChunkWindow(index=0, start=0.0, end=50.0, emit_start=0.0, emit_end=60.0)


def test_offset_words_should_apply_chunk_start_and_drop_overlap_duplicates() -> None:
    window = ChunkWindow(index=1, start=885.0, end=1800.0, emit_start=900.0, emit_end=1785.0)
    words = [
        AlignedWord(text="context", start=1.0, end=2.0),
        AlignedWord(text="kept", start=20.0, end=21.0),
        AlignedWord(text="tail", start=910.0, end=911.0),
    ]

    result = offset_words_to_global_timeline(words, window)

    assert result == [AlignedWord(text="kept", start=905.0, end=906.0)]


def test_offset_turns_should_apply_chunk_start_and_clip_to_emit_window() -> None:
    window = ChunkWindow(index=1, start=885.0, end=1800.0, emit_start=900.0, emit_end=1785.0)
    turns = [
        SpeakerTurn(speaker="Speaker 0", start=0.0, end=20.0),
        SpeakerTurn(speaker="Speaker 1", start=20.0, end=40.0),
        SpeakerTurn(speaker="Speaker 2", start=900.0, end=915.0),
    ]

    result = offset_turns_to_global_timeline(turns, window)

    assert result == [
        SpeakerTurn(speaker="Speaker 0", start=900.0, end=905.0),
        SpeakerTurn(speaker="Speaker 1", start=905.0, end=925.0),
    ]


def test_clip_turns_to_emit_window_should_clip_global_turns_without_reoffset() -> None:
    window = ChunkWindow(index=1, start=270.0, end=600.0, emit_start=285.0, emit_end=600.0)
    turns = [
        SpeakerTurn(speaker="Speaker 0", start=271.0, end=272.0),
        SpeakerTurn(speaker="Speaker 1", start=280.0, end=290.0),
        SpeakerTurn(speaker="Speaker 2", start=590.0, end=605.0),
    ]

    result = clip_turns_to_emit_window(turns, window)

    assert result == [
        SpeakerTurn(speaker="Speaker 1", start=285.0, end=290.0),
        SpeakerTurn(speaker="Speaker 2", start=590.0, end=600.0),
    ]


def test_validate_aligned_word_quality_should_reject_non_monotonic_words() -> None:
    words = [
        AlignedWord(text="first", start=10.0, end=11.0),
        AlignedWord(text="second", start=9.0, end=9.5),
    ]

    with pytest.raises(ValueError, match="non-monotonic"):
        validate_aligned_word_quality(words, expected_duration_seconds=20.0)


def test_validate_aligned_word_quality_should_check_monotonic_without_duration() -> None:
    words = [
        AlignedWord(text="first", start=10.0, end=11.0),
        AlignedWord(text="second", start=9.0, end=9.5),
    ]

    with pytest.raises(ValueError, match="non-monotonic"):
        validate_aligned_word_quality(words, expected_duration_seconds=None)


def test_validate_aligned_word_quality_should_reject_tail_timestamp_collapse() -> None:
    words = [
        AlignedWord(text=f"w{i}", start=float(i), end=float(i) + 0.5)
        for i in range(20)
    ]
    words.extend(
        AlignedWord(text=f"tail{i}", start=245.04, end=245.04)
        for i in range(12)
    )

    with pytest.raises(ValueError, match="tail timestamp collapse"):
        validate_aligned_word_quality(words, expected_duration_seconds=600.0)


def test_validate_aligned_word_quality_should_accept_short_valid_alignment() -> None:
    words = [
        AlignedWord(text="hello", start=0.1, end=0.5),
        AlignedWord(text="world", start=0.6, end=1.0),
    ]

    validate_aligned_word_quality(words, expected_duration_seconds=1.2)


def test_reconcile_chunk_speaker_labels_should_remap_swapped_labels_by_overlap() -> None:
    existing = [
        SpeakerTurn(speaker="Speaker 1", start=270.0, end=277.5),
        SpeakerTurn(speaker="Speaker 0", start=277.5, end=285.0),
    ]
    chunk_turns = [
        SpeakerTurn(speaker="Speaker 0", start=270.0, end=277.5),
        SpeakerTurn(speaker="Speaker 1", start=277.5, end=285.0),
        SpeakerTurn(speaker="Speaker 1", start=285.0, end=305.0),
    ]

    result = reconcile_chunk_speaker_labels(
        existing_turns=existing,
        chunk_turns=chunk_turns,
        overlap_start=270.0,
        overlap_end=285.0,
    )

    assert result == [
        SpeakerTurn(speaker="Speaker 1", start=270.0, end=277.5),
        SpeakerTurn(speaker="Speaker 0", start=277.5, end=285.0),
        SpeakerTurn(speaker="Speaker 0", start=285.0, end=305.0),
    ]
