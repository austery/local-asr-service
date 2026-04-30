from src.core.diarization_port import SpeakerTurn


def test_speaker_turn_is_immutable_value_object() -> None:
    turn = SpeakerTurn(speaker="Speaker 1", start=1.0, end=2.5)

    assert turn.speaker == "Speaker 1"
    assert turn.start == 1.0
    assert turn.end == 2.5
