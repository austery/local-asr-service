from unittest.mock import MagicMock

import pytest

from src.core.alignment_port import AlignedWord
from src.core.mlx_qwen_forced_aligner import MlxQwenForcedAligner


class _RuntimeItem:
    def __init__(self, text: str, start_time: float, end_time: float) -> None:
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


class _RuntimeResult:
    def __init__(self, items: list[_RuntimeItem]) -> None:
        self.items = items


def test_align_file_should_require_load_first() -> None:
    aligner = MlxQwenForcedAligner()

    with pytest.raises(RuntimeError, match="load\\(\\) first"):
        aligner.align_file("sample.wav", text="hello", language="English")


def test_align_file_should_map_runtime_items_to_aligned_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MagicMock()
    runtime.generate.return_value = [
        _RuntimeItem(text="hello", start_time=0.0, end_time=0.5),
        _RuntimeItem(text="world", start_time=0.5, end_time=1.0),
    ]

    def fake_load_runtime(model_id: str) -> MagicMock:
        assert model_id == "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
        return runtime

    monkeypatch.setattr(
        "src.core.mlx_qwen_forced_aligner._load_qwen_forced_aligner_runtime",
        fake_load_runtime,
    )

    aligner = MlxQwenForcedAligner()
    aligner.load()

    words = aligner.align_file("sample.wav", text="hello world", language="en")

    assert words == [
        AlignedWord(text="hello", start=0.0, end=0.5),
        AlignedWord(text="world", start=0.5, end=1.0),
    ]
    runtime.generate.assert_called_once_with(
        "sample.wav",
        text="hello world",
        language="English",
    )


def test_align_file_should_accept_forced_align_result_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MagicMock()
    runtime.generate.return_value = _RuntimeResult(items=[
        _RuntimeItem(text="hello", start_time=0.0, end_time=0.5),
        _RuntimeItem(text="world", start_time=0.5, end_time=1.0),
    ])

    monkeypatch.setattr(
        "src.core.mlx_qwen_forced_aligner._load_qwen_forced_aligner_runtime",
        lambda _model_id: runtime,
    )

    aligner = MlxQwenForcedAligner()
    aligner.load()

    words = aligner.align_file("sample.wav", text="hello world", language="English")

    assert words == [
        AlignedWord(text="hello", start=0.0, end=0.5),
        AlignedWord(text="world", start=0.5, end=1.0),
    ]


def test_align_file_should_raise_when_runtime_item_shape_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MagicMock()
    runtime.generate.return_value = [{"text": "bad", "start": 0.0, "end": 1.0}]

    monkeypatch.setattr(
        "src.core.mlx_qwen_forced_aligner._load_qwen_forced_aligner_runtime",
        lambda _model_id: runtime,
    )

    aligner = MlxQwenForcedAligner()
    aligner.load()

    with pytest.raises(TypeError, match="text, start_time, and end_time attributes"):
        aligner.align_file("sample.wav", text="bad", language="English")
