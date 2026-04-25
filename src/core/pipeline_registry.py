"""
Pipeline profile registry for future decoupled execution.

Maps user-facing pipeline aliases to pairs of registered model aliases.
"""

from dataclasses import dataclass

from src.core.base_engine import EngineCapabilities


@dataclass(frozen=True)
class PipelineProfile:
    """Named decoupled pipeline profile."""

    alias: str
    transcription_alias: str
    diarization_alias: str
    description: str
    capabilities: EngineCapabilities
    requestable: bool = False


_REGISTRY: dict[str, PipelineProfile] = {
    profile.alias: profile
    for profile in [
        PipelineProfile(
            alias="firered-sortformer",
            transcription_alias="firered-asr",
            diarization_alias="sortformer-diar",
            description="decoupled FireRed + Sortformer profile",
            capabilities=EngineCapabilities(timestamp=True, diarization=True, emotion_tags=False, language_detect=True),
        )
    ]
}


def lookup_profile(alias: str) -> PipelineProfile:
    """Resolve a pipeline alias to a PipelineProfile."""
    try:
        return _REGISTRY[alias]
    except KeyError as exc:
        raise KeyError(f"Unknown pipeline profile: '{alias}'") from exc


def list_all_profiles() -> list[PipelineProfile]:
    """Return all built-in pipeline profiles, sorted by alias."""
    return sorted(_REGISTRY.values(), key=lambda profile: profile.alias)
