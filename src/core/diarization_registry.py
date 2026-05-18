from dataclasses import dataclass
from typing import Literal


RuntimeType = Literal["mlx"]


@dataclass(frozen=True)
class DiarizationSpec:
    alias: str
    runtime: RuntimeType
    model_id: str
    description: str


_REGISTRY: dict[str, DiarizationSpec] = {
    "sortformer-diar": DiarizationSpec(
        alias="sortformer-diar",
        runtime="mlx",
        model_id="mlx-community/diar_sortformer_4spk-v1-fp16",
        description="Sortformer diarization via mlx-audio VAD runtime on Apple MLX.",
    )
}


def lookup_diarizer(alias: str) -> DiarizationSpec:
    try:
        return _REGISTRY[alias]
    except KeyError as exc:
        raise KeyError(f"Unknown diarization alias: '{alias}'") from exc
