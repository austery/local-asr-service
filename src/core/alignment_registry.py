from dataclasses import dataclass
from typing import Literal

RuntimeType = Literal["mlx"]


@dataclass(frozen=True)
class AlignmentSpec:
    alias: str
    runtime: RuntimeType
    model_id: str
    description: str


_REGISTRY: dict[str, AlignmentSpec] = {
    "qwen3-forced-aligner": AlignmentSpec(
        alias="qwen3-forced-aligner",
        runtime="mlx",
        model_id="mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
        description="Qwen3 forced alignment via mlx-audio STT runtime on Apple MLX.",
    )
}


def lookup_aligner(alias: str) -> AlignmentSpec:
    try:
        return _REGISTRY[alias]
    except KeyError as exc:
        raise KeyError(f"Unknown alignment alias: '{alias}'") from exc
