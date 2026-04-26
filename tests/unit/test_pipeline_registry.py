import pytest

from src.core.pipeline_registry import list_all_profiles, lookup_profile


def test_should_lookup_firered_sortformer_profile() -> None:
    profile = lookup_profile("firered-sortformer")

    assert profile.alias == "firered-sortformer"
    assert profile.transcription_alias == "firered-asr"
    assert profile.diarization_alias == "sortformer-diar"
    assert profile.description == "decoupled FireRed + Sortformer profile"
    assert profile.capabilities.timestamp is True
    assert profile.capabilities.diarization is True
    assert profile.capabilities.language_detect is True


def test_should_raise_key_error_with_unknown_alias_in_message() -> None:
    with pytest.raises(KeyError, match="not-a-real-profile"):
        lookup_profile("not-a-real-profile")


def test_should_list_profiles_sorted_by_alias() -> None:
    profiles = list_all_profiles()

    aliases = [profile.alias for profile in profiles]

    assert aliases == sorted(aliases)
    assert "firered-sortformer" in aliases
