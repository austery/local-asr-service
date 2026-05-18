import pytest

from src.core.pipeline_registry import list_all_profiles, lookup_profile


def test_should_resolve_qwen3_sortformer_profile() -> None:
    profile = lookup_profile("qwen3-sortformer")

    assert profile.alias == "qwen3-sortformer"
    assert profile.transcription_alias == "qwen3-asr"
    assert profile.diarization_alias == "sortformer-diar"
    assert profile.capabilities.timestamp is True
    assert profile.capabilities.diarization is True
    assert profile.requestable is True


def test_should_list_profiles_sorted_by_alias() -> None:
    aliases = [profile.alias for profile in list_all_profiles()]

    assert aliases == sorted(aliases)
    assert "qwen3-sortformer" in aliases


def test_should_raise_for_unknown_profile() -> None:
    with pytest.raises(KeyError, match="Unknown pipeline profile"):
        lookup_profile("missing-profile")
