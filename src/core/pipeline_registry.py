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


_REGISTRY: dict[str, PipelineProfile] = {
    "qwen3-sortformer": PipelineProfile(
        alias="qwen3-sortformer",
        transcription_alias="qwen3-asr",
        alignment_alias="qwen3-forced-aligner",
        diarization_alias="sortformer-diar",
        description=(
            "Experimental Qwen3-ASR plus Sortformer speaker-separation pipeline. "
            "Production enablement requires forced alignment between transcript text "
            "and speaker turns."
        ),
        capabilities=EngineCapabilities(timestamp=True, diarization=True, language_detect=True),
        requestable=False,
    )
}


def lookup_profile(alias: str) -> PipelineProfile:
    try:
        return _REGISTRY[alias]
    except KeyError as exc:
        raise KeyError(f"Unknown pipeline profile: '{alias}'") from exc


def list_all_profiles() -> list[PipelineProfile]:
    return sorted(_REGISTRY.values(), key=lambda profile: profile.alias)
