"""Immutable request selection and completion contracts for the gateway."""

from dataclasses import dataclass
from typing import TypedDict

from src.core.base_engine import EngineCapabilities, TranscriptionInputError
from src.core.model_registry import ModelSpec


class TranscriptionResultDict(TypedDict, total=False):
    text: str
    segments: list[dict[str, object]] | None
    duration: float
    language: str


TranscriptionResult = str | TranscriptionResultDict


@dataclass(frozen=True)
class ExecutionResult:
    """Payload and identity from the same execution, independent of later switches."""

    payload: TranscriptionResult
    model: str


@dataclass(frozen=True)
class ExecutionPlan:
    """A selected model; create under the admission lock for passthrough requests."""

    spec: ModelSpec | None
    model: str
    capabilities: EngineCapabilities

    @classmethod
    def select(cls, spec: ModelSpec | None, fallback_model: str) -> "ExecutionPlan":
        return cls(
            spec=spec,
            model=spec.alias if spec is not None else fallback_model,
            capabilities=spec.capabilities if spec is not None else EngineCapabilities(),
        )

    def validate(self, params: dict[str, object]) -> None:
        language = params.get("language", "auto")
        if self.spec is not None and self.spec.engine_type == "apple-speech" and (
            not isinstance(language, str) or not language.strip() or language.strip().lower() == "auto"
        ):
            raise TranscriptionInputError(
                "model=apple-speech requires an explicit language or locale; "
                "pass 'zh', 'zh-CN', 'en', or 'en-US' instead of 'auto'."
            )
        if params.get("output_format") == "srt" and not self.capabilities.timestamp:
            raise TranscriptionInputError(
                f"SRT format requires timestamp support, but '{self.model}' "
                "does not produce timestamps. Use output_format=json or output_format=txt instead."
            )
        if params.get("with_timestamp") and not self.capabilities.timestamp:
            raise TranscriptionInputError(
                f"with_timestamp=true requires timestamp support, but '{self.model}' "
                "does not produce timestamps."
            )
