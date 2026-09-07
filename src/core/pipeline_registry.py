from dataclasses import dataclass

from src.core.base_engine import EngineCapabilities


@dataclass(frozen=True)
class PipelineProfile:
    alias: str
    transcription_alias: str
    alignment_alias: str | None
    diarization_alias: str
    description: str
    capabilities: EngineCapabilities
    requestable: bool = False


# No public pipeline profiles remain after retiring qwen3-sortformer.
_REGISTRY: dict[str, PipelineProfile] = {}


def lookup_profile(alias: str) -> PipelineProfile:
    try:
        return _REGISTRY[alias]
    except KeyError as exc:
        raise KeyError(f"Unknown pipeline profile: '{alias}'") from exc


def list_all_profiles() -> list[PipelineProfile]:
    return sorted(_REGISTRY.values(), key=lambda profile: profile.alias)
