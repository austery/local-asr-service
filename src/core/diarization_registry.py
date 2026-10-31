"""Independent diarization metadata; never an ASR default or passthrough."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RuntimeType = Literal["mlx"]
NEMOTRON_REVISION = "59ed2dbfc1346dcea9d423c71306a3a2499c568f"


@dataclass(frozen=True)
class DiarizationSpec:
    alias: str
    runtime: RuntimeType
    model_id: str
    description: str
    backend: Literal["sortformer", "nemotron"] = "sortformer"
    revision: str | None = None
    experimental: bool = False
    public: bool = False
    max_speakers: int = 4


_REGISTRY = {
    "sortformer-diar": DiarizationSpec(
        alias="sortformer-diar",
        runtime="mlx",
        model_id="mlx-community/diar_sortformer_4spk-v1-fp16",
        description="Sortformer diarization via mlx-audio VAD runtime on Apple MLX.",
    ),
    "nemotron-diar": DiarizationSpec(
        alias="nemotron-diar",
        runtime="mlx",
        model_id="mlx-community/Nemotron-3-Diarization",
        description="Experimental standalone speaker activity; quality not accepted. OpenMDW-1.1.",
        backend="nemotron",
        revision=NEMOTRON_REVISION,
        experimental=True,
        public=True,
        max_speakers=8,
    ),
}


def lookup_diarizer(alias: str) -> DiarizationSpec:
    try:
        return _REGISTRY[alias]
    except KeyError as exc:
        raise KeyError(f"Unknown diarization alias: '{alias}'") from exc


def public_diarizers() -> list[DiarizationSpec]:
    return [spec for spec in _REGISTRY.values() if spec.public]


def is_diarization_model(value: str) -> bool:
    return any(value in (s.alias, s.model_id) for s in _REGISTRY.values())


def cached_snapshot(spec: DiarizationSpec) -> Path | None:
    """Inspect pinned local cache entries without installing/downloading weights."""
    from huggingface_hub import try_to_load_from_cache

    if spec.revision is None:
        return None
    files = [
        try_to_load_from_cache(spec.model_id, name, revision=spec.revision)
        for name in ("config.json", "model.safetensors")
    ]
    if not all(isinstance(p, str) and Path(p).is_file() for p in files):
        return None
    paths = [Path(p) for p in files if isinstance(p, str)]
    return paths[0].parent if paths[0].parent == paths[1].parent else None
