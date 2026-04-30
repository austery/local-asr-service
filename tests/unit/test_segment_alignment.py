import pytest

from src.adapters.segment_alignment import _read_timestamp, align_speakers
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


def test_should_reject_invalid_segment_interval() -> None:
    with pytest.raises(ValueError, match="end .* must be > start"):
        align_speakers([{"text": "bad", "start": 5.0, "end": 2.0}], [])


def test_should_accept_string_timestamp_values() -> None:
    assert _read_timestamp({"start": "1.5"}, "start") == 1.5


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_should_reject_non_finite_timestamp_values(value: float) -> None:
    with pytest.raises(ValueError, match="segment timestamp must be finite"):
        _read_timestamp({"start": value}, "start")
