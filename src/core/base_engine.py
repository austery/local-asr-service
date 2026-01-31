"""
ASR 引擎抽象接口定义。
使用 Protocol 实现结构化子类型 (Structural Subtyping)。
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class ASREngine(Protocol):
    """
    ASR 引擎抽象接口。
    所有引擎实现必须遵循此接口。
    """

    def load(self) -> None:
        """
        加载模型到内存/显存。
        这一步可能触发模型下载（如果本地不存在）。
        """
        ...

    def transcribe_file(self, file_path: str, language: str = "auto", **kwargs) -> str:
        """
        执行推理，返回转录文本。
        
        Args:
            file_path: 音频文件路径
            language: 语言代码 (zh, en, auto 等)
            **kwargs: 引擎特定参数
            
        Returns:
            转录文本
        """
        ...

    def release(self) -> None:
        """
        释放显存/内存资源。
        用于热更新模型或服务关闭时清理资源。
        """
        ...
