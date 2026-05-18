from collections.abc import Iterable, Sequence
from importlib import import_module
from typing import Protocol, cast, runtime_checkable

from src.core.alignment_port import AlignedWord, AlignmentPort, RuntimeAlignmentItem


class _QwenForcedAlignerRuntime(Protocol):
    def generate(self, audio: str, *, text: str, language: str) -> object: ...


@runtime_checkable
class _RuntimeAlignmentResult(Protocol):
    items: Sequence[RuntimeAlignmentItem]


def _load_qwen_forced_aligner_runtime(model_id: str) -> _QwenForcedAlignerRuntime:
    stt_module = import_module("mlx_audio.stt")
    load = stt_module.load
    return load(model_id)


def _require_runtime_item(item: object) -> RuntimeAlignmentItem:
    if not isinstance(item, RuntimeAlignmentItem):
        raise TypeError(
            "Runtime alignment item must expose text, start_time, and end_time attributes."
        )
    return cast(RuntimeAlignmentItem, item)


def _iter_runtime_items(output: object) -> Iterable[object]:
    if isinstance(output, _RuntimeAlignmentResult):
        return output.items
    if isinstance(output, Iterable):
        return output
    raise TypeError("Runtime alignment output must be iterable or expose an items sequence.")


def _normalize_qwen_alignment_language(language: str) -> str:
    normalized = language.strip()
    if not normalized:
        return "English"

    aliases = {
        "auto": "English",
        "en": "English",
        "eng": "English",
        "english": "English",
        "zh": "Chinese",
        "cn": "Chinese",
        "zho": "Chinese",
        "chinese": "Chinese",
        "yue": "Cantonese",
        "cantonese": "Cantonese",
        "ja": "Japanese",
        "japanese": "Japanese",
        "ko": "Korean",
        "korean": "Korean",
        "de": "German",
        "german": "German",
        "es": "Spanish",
        "spanish": "Spanish",
        "fr": "French",
        "french": "French",
        "it": "Italian",
        "italian": "Italian",
        "pt": "Portuguese",
        "portuguese": "Portuguese",
        "ru": "Russian",
        "russian": "Russian",
    }
    return aliases.get(normalized.lower(), normalized)


class MlxQwenForcedAligner(AlignmentPort):
    def __init__(self, model_id: str = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit") -> None:
        self.model_id = model_id
        self._runtime: _QwenForcedAlignerRuntime | None = None

    def load(self) -> None:
        if self._runtime is None:
            self._runtime = _load_qwen_forced_aligner_runtime(self.model_id)

    def align_file(self, file_path: str, *, text: str, language: str) -> list[AlignedWord]:
        if self._runtime is None:
            raise RuntimeError("Aligner not loaded. Call load() first.")

        output = self._runtime.generate(
            file_path,
            text=text,
            language=_normalize_qwen_alignment_language(language),
        )
        return [self._to_aligned_word(item) for item in _iter_runtime_items(output)]

    def release(self) -> None:
        self._runtime = None

    def _to_aligned_word(self, item: RuntimeAlignmentItem | object) -> AlignedWord:
        runtime_item = _require_runtime_item(item)
        return AlignedWord(
            text=str(runtime_item.text),
            start=float(runtime_item.start_time),
            end=float(runtime_item.end_time),
        )
