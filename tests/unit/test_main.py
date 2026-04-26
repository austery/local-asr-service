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


# ---------------------------------------------------------------------------
# _effective_startup_metadata tests
# ---------------------------------------------------------------------------


def _metadata_helper():
    helper = getattr(main_module, "_effective_startup_metadata", None)
    assert helper is not None, "_effective_startup_metadata not found in src.main"
    return helper


def test_effective_startup_metadata_uses_spec_when_resolved() -> None:
    """When a spec is resolved, engine_type and model_id must come from the spec.

    This covers the cross-engine-family override case: e.g. ENGINE_TYPE=funasr
    but MODEL_ID=FireRedTeam/FireRedASR2-AED → effective engine must be 'firered'.
    """
    spec = _resolver()("firered-asr")
    assert spec is not None

    eff_engine, eff_model = _metadata_helper()(spec, "funasr", "FireRedTeam/FireRedASR2-AED")

    assert eff_engine == "firered"
    assert eff_model == spec.model_id


def test_effective_startup_metadata_spec_model_id_overrides_raw() -> None:
    """The canonical model_id from the spec must be used, not the raw alias."""
    spec = _resolver()("paraformer")
    assert spec is not None

    eff_engine, eff_model = _metadata_helper()(spec, "funasr", "paraformer")

    assert eff_engine == spec.engine_type
    assert eff_model == spec.model_id  # canonical, not alias


def test_effective_startup_metadata_fallback_when_spec_is_none() -> None:
    """When spec is None (unregistered model), fall back to configured values."""
    eff_engine, eff_model = _metadata_helper()(None, "funasr", "my-custom-model")

    assert eff_engine == "funasr"
    assert eff_model == "my-custom-model"
