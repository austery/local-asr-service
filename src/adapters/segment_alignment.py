from src.core.diarization_port import SpeakerTurn


def align_speakers(
    transcript_segments: list[dict[str, object]],
    speaker_turns: list[SpeakerTurn],
) -> list[dict[str, object]]:
    """
    对齐转录文本段和说话人轮次，为每个文本段分配对应的说话人。

    使用重叠时长最长原则 (maximum overlap heuristic)：
    - 为每个文本段找到与其时间范围重叠最多的说话人轮次
    - 如果没有任何重叠，分配 "Unknown" 说话人

    Args:
        transcript_segments: 转录文本段列表，每个段包含 "start" 和 "end" 时间戳
        speaker_turns: 说话人轮次列表，包含每个说话人的活跃时间范围

    Returns:
        对齐后的文本段列表，每个段新增 "speaker" 字段
    """
    aligned: list[dict[str, object]] = []
    for segment in transcript_segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        best_speaker = "Unknown"
        best_overlap = -1.0

        for turn in speaker_turns:
            overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker

        merged = dict(segment)
        merged["speaker"] = best_speaker
        aligned.append(merged)

    return aligned
