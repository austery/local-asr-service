import pytest

from src.adapters.segment_alignment import align_speakers
from src.core.diarization_port import SpeakerTurn


def test_should_assign_speaker_with_largest_overlap() -> None:
    transcript_segments = [
        {"text": "hello", "start": 0.0, "end": 2.0},
        {"text": "world", "start": 2.0, "end": 4.0},
    ]
    speaker_turns = [
        SpeakerTurn(speaker="Speaker A", start=0.0, end=1.0),
        SpeakerTurn(speaker="Speaker B", start=1.0, end=4.0),
    ]

    aligned = align_speakers(transcript_segments, speaker_turns)

    assert aligned[0]["speaker"] == "Speaker A"
    assert aligned[1]["speaker"] == "Speaker B"


def test_should_assign_unknown_when_no_turn_overlaps() -> None:
    aligned = align_speakers(
        [{"text": "gap", "start": 10.0, "end": 11.0}],
        [SpeakerTurn(speaker="Speaker A", start=0.0, end=1.0)],
    )

    assert aligned[0]["speaker"] == "Unknown"


def test_should_reject_missing_timestamp() -> None:
    with pytest.raises(ValueError, match="segment missing required timestamp"):
        align_speakers([{"text": "bad", "start": 0.0}], [])
