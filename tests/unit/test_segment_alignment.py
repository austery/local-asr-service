import pytest

from src.core.diarization_port import SpeakerTurn
from src.adapters.segment_alignment import align_speakers


def test_assigns_speaker_by_largest_overlap() -> None:
    """
    验证对齐函数能正确识别最大重叠的说话人。

    场景：
    - 第一个段 [0.0-1.0s] 与说话人1 [0.0-1.2s] 有1.0s重叠
    - 第二个段 [1.0-2.0s] 与说话人2 [1.2-2.0s] 有0.8s重叠
    """
    segments = [
        {"text": "hello", "start": 0.0, "end": 1.0},
        {"text": "world", "start": 1.0, "end": 2.0},
    ]
    turns = [
        SpeakerTurn(speaker="Speaker 1", start=0.0, end=1.2),
        SpeakerTurn(speaker="Speaker 2", start=1.2, end=2.0),
    ]

    aligned = align_speakers(segments, turns)

    assert aligned[0]["speaker"] == "Speaker 1"
    assert aligned[1]["speaker"] == "Speaker 2"


def test_falls_back_to_unknown_when_no_overlap_exists() -> None:
    """
    验证当文本段与任何说话人轮次都没有时间重叠时，分配 "Unknown" 说话人。

    场景：孤立段 [10.0-11.0s] 存在说话人轮次，但都没有时间重叠
    """
    segments = [{"text": "orphan", "start": 10.0, "end": 11.0}]
    turns = [
        SpeakerTurn(speaker="Speaker 1", start=0.0, end=1.0),
        SpeakerTurn(speaker="Speaker 2", start=12.0, end=13.0),
    ]

    aligned = align_speakers(segments, turns)

    assert aligned[0]["speaker"] == "Unknown"


def test_falls_back_to_unknown_when_speaker_turns_are_empty() -> None:
    segments = [{"text": "solo", "start": 0.0, "end": 1.0}]

    aligned = align_speakers(segments, [])

    assert aligned[0]["speaker"] == "Unknown"


def test_raises_when_segment_is_missing_required_timestamps() -> None:
    segments = [{"text": "broken", "start": 0.0}]
    turns = [SpeakerTurn(speaker="Speaker 1", start=0.0, end=1.0)]

    with pytest.raises(ValueError, match="missing required timestamp"):
        align_speakers(segments, turns)


def test_raises_when_segment_timestamp_is_bool() -> None:
    segments = [{"text": "broken", "start": True, "end": 1.0}]
    turns = [SpeakerTurn(speaker="Speaker 1", start=0.0, end=1.0)]

    with pytest.raises(ValueError, match="timestamp must be numeric"):
        align_speakers(segments, turns)
