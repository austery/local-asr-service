import pytest

from src.core.diarization_registry import lookup_diarizer


def test_lookup_diarizer_should_resolve_builtin_sortformer_alias() -> None:
    spec = lookup_diarizer("sortformer-diar")

    assert spec.alias == "sortformer-diar"
    assert spec.runtime == "mlx"
    assert spec.model_id == "mlx-community/diar_sortformer_4spk-v1-fp16"
    assert "Sortformer" in spec.description


def test_lookup_diarizer_should_raise_for_unknown_alias() -> None:
    with pytest.raises(KeyError, match="Unknown diarization alias"):
        lookup_diarizer("not-a-real-diarizer")
