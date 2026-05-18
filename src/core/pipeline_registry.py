from dataclasses import dataclass

from src.core.base_engine import EngineCapabilities


@dataclass(frozen=True)
class PipelineProfile:
    alias: str
    transcription_alias: str
    diarization_alias: str
    description: str
    capabilities: EngineCapabilities
    requestable: bool = False


_REGISTRY: dict[str, PipelineProfile] = {
    "qwen3-sortformer": PipelineProfile(
        alias="qwen3-sortformer",
        transcription_alias="qwen3-asr",
        diarization_alias="sortformer-diar",
        description=(
            "Worker-backed Qwen3-ASR transcription plus Sortformer diarization "
            "pipeline for request-time multi-speaker transcription."
        ),
        capabilities=EngineCapabilities(timestamp=True, diarization=True, language_detect=True),
        requestable=True,
    )
}


def lookup_profile(alias: str) -> PipelineProfile:
    try:
        return _REGISTRY[alias]
    except KeyError as exc:
        raise KeyError(f"Unknown pipeline profile: '{alias}'") from exc


def list_all_profiles() -> list[PipelineProfile]:
    return sorted(_REGISTRY.values(), key=lambda profile: profile.alias)
