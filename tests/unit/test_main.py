import pytest

import src.main as main_module


def _resolver():
    resolver = getattr(main_module, "_resolve_startup_model_spec", None)
    assert resolver is not None
    return resolver


def test_resolve_startup_model_spec_returns_requestable_model_spec() -> None:
    spec = _resolver()("paraformer")

    assert spec is not None
    assert spec.alias == "paraformer"


def test_resolve_startup_firered_alias_is_allowed() -> None:
    """FireRed is a full ASR engine — it must be a valid startup resident model."""
    spec = _resolver()("firered-asr")

    assert spec is not None
    assert spec.alias == "firered-asr"
    assert spec.engine_type == "firered"


def test_resolve_startup_firered_model_id_is_allowed() -> None:
    """Full FireRedASR2-AED model_id must resolve without error."""
    spec = _resolver()("FireRedTeam/FireRedASR2-AED")

    assert spec is not None
    assert spec.engine_type == "firered"


def test_resolve_startup_sortformer_diar_is_rejected() -> None:
    """sortformer-diar is a diarization-only component — must not be an ASR startup model."""
    with pytest.raises(RuntimeError, match="cannot be configured as the resident model"):
        _resolver()("sortformer-diar")


def test_resolve_startup_model_spec_rejects_pipeline_profile_alias() -> None:
    with pytest.raises(RuntimeError, match="pipeline profile"):
        _resolver()("firered-sortformer")


def test_resolve_startup_model_spec_returns_none_for_unknown_model() -> None:
    assert _resolver()("unknown-model-id") is None
