"""
Model Registry for dynamic model switching (SPEC-108).

Maps user-facing aliases to full ModelSpecs (model_id + engine_type + capabilities).
Single source of truth for all supported models.
"""

from dataclasses import dataclass
from typing import Literal

from src.core.base_engine import EngineCapabilities

EngineType = Literal["funasr", "mlx"]

# OpenAI-compat placeholder values that mean "use the server's current model"
# Empty string is also passthrough: form data serialises None as "" in some clients
_OPENAI_PASSTHROUGH_VALUES: frozenset[str] = frozenset({"whisper-1", ""})


@dataclass(frozen=True)
class ModelSpec:
    """Complete specification for a named ASR model."""

    alias: str
    model_id: str
    engine_type: EngineType
    description: str
    capabilities: EngineCapabilities


# fmt: off
_REGISTRY: dict[str, ModelSpec] = {
    spec.alias: spec
    for spec in [
        ModelSpec(
            alias="paraformer",
            model_id="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            engine_type="funasr",
            description="Mandarin + speaker diarization (FunASR). Best for multi-speaker podcasts.",
            capabilities=EngineCapabilities(timestamp=True, diarization=True, emotion_tags=False, language_detect=True),
        ),
        ModelSpec(
            alias="qwen3-asr-mini",
            model_id="mlx-community/Qwen3-ASR-1.7B-4bit",
            engine_type="mlx",
            description="Fast & light Qwen3 ASR (4-bit). Best for single-speaker, low latency.",
            capabilities=EngineCapabilities(timestamp=True, diarization=False, emotion_tags=False, language_detect=True),
        ),
        ModelSpec(
            alias="qwen3-asr",
            model_id="mlx-community/Qwen3-ASR-1.7B-8bit",
            engine_type="mlx",
            description="Qwen3 ASR (8-bit, higher accuracy).",
            capabilities=EngineCapabilities(timestamp=True, diarization=False, emotion_tags=False, language_detect=True),
        ),
        ModelSpec(
            alias="parakeet",
            model_id="mlx-community/parakeet-tdt-0.6b-v2",
            engine_type="mlx",
            description="NVIDIA Parakeet (English only, very fast). Short clips only — OOM on files > ~5 min (known issue, unfixed).",
            capabilities=EngineCapabilities(timestamp=True, diarization=False, emotion_tags=False, language_detect=False),
        ),
        ModelSpec(
            alias="sensevoice-small",
            model_id="iic/SenseVoiceSmall",
            engine_type="funasr",
            description="SenseVoice Small — fastest model (80-85x realtime). Best for: bulk speed-first processing, language detection, emotion tagging. NOT recommended for transcription quality: struggles with mixed-language and proper nouns.",
            capabilities=EngineCapabilities(timestamp=False, diarization=False, emotion_tags=True, language_detect=True),
        ),
    ]
}
# fmt: on

# Reverse index: model_id → alias (for resolving full paths back to human aliases)
_MODEL_ID_TO_ALIAS: dict[str, str] = {spec.model_id: spec.alias for spec in _REGISTRY.values()}


def lookup(model: str) -> ModelSpec:
    """
    Resolve a model string to a ModelSpec.

    Resolution order:
      1. Exact alias match         ("paraformer", "qwen3-asr-mini")
      2. Registered model_id match ("mlx-community/Qwen3-ASR-1.7B-4bit")
      3. Prefix-based engine_type inference for unknown full paths
         ("mlx-community/..." → mlx,  "iic/..." / "funasr..." → funasr)

    Raises:
        ValueError: if the string cannot be resolved to any known engine type.
    """
    # 1. Exact alias
    if model in _REGISTRY:
        return _REGISTRY[model]

    # 2. Registered model_id
    if model in _MODEL_ID_TO_ALIAS:
        return _REGISTRY[_MODEL_ID_TO_ALIAS[model]]

    # 3. Infer engine_type from path prefix for unregistered full paths
    inferred_engine: EngineType | None = None
    if model.startswith("mlx-community/"):
        inferred_engine = "mlx"
    elif model.startswith("iic/") or "funasr" in model.lower():
        inferred_engine = "funasr"

    if inferred_engine is None:
        raise ValueError(
            f"Unknown model: '{model}'. "
            f"Use GET /v1/models to see built-in models, "
            f"or pass a full path prefixed with 'mlx-community/' or 'iic/'."
        )

    # Return an ad-hoc spec; real capabilities will be resolved by the engine at load time.
    return ModelSpec(
        alias=model,
        model_id=model,
        engine_type=inferred_engine,
        description="Custom model (capabilities resolved at load time).",
        capabilities=EngineCapabilities(),
    )


def is_passthrough(model: str | None) -> bool:
    """Return True if this model value means 'use the server's current model' (no switch)."""
    return model is None or model in _OPENAI_PASSTHROUGH_VALUES


def list_all() -> list[ModelSpec]:
    """Return all built-in registered models, sorted by alias."""
    return sorted(_REGISTRY.values(), key=lambda s: s.alias)


def alias_for(model_id: str) -> str | None:
    """Return the registered alias for a given model_id, or None if not in registry."""
    return _MODEL_ID_TO_ALIAS.get(model_id)
