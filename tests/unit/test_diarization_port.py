import pytest

from src.core.diarization_port import SpeakerTurn


def test_speaker_turn_is_immutable_value_object() -> None:
    turn = SpeakerTurn(speaker="Speaker 1", start=1.0, end=2.5)

    assert turn.speaker == "Speaker 1"
    assert turn.start == 1.0
    assert turn.end == 2.5


def test_speaker_turn_rejects_empty_speaker_name() -> None:
    with pytest.raises(ValueError, match="Speaker name cannot be empty"):
        SpeakerTurn(speaker="  ", start=1.0, end=2.0)


def test_speaker_turn_rejects_invalid_interval() -> None:
    with pytest.raises(ValueError, match="end .* must be > start"):
        SpeakerTurn(speaker="Speaker 1", start=5.0, end=1.0)
