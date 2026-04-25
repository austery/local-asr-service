from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpeakerTurn:
    """
    说话人轮次信息。

    Attributes:
        speaker: 说话人标识符
        start: 开始时间（秒）
        end: 结束时间（秒）
    """

    speaker: str
    start: float
    end: float


class DiarizationPort(Protocol):
    """
    说话人分离 (Diarization) 接口。

    定义实现说话人分离功能的引擎必须遵循的接口。
    使用 Protocol 实现结构化子类型 (Structural Subtyping)。
    """

    def load(self) -> None:
        """加载说话人分离模型到内存。"""
        ...

    def diarize_file(self, file_path: str) -> list[SpeakerTurn]:
        """
        执行说话人分离，识别音频文件中的说话人及其活跃时间段。

        Args:
            file_path: 音频文件路径

        Returns:
            说话人轮次列表
        """
        ...

    def release(self) -> None:
        """释放模型资源。"""
        ...
