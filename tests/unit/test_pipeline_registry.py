import pytest

from src.core.pipeline_registry import list_all_profiles, lookup_profile


def test_should_reject_retired_qwen3_sortformer_profile() -> None:
    with pytest.raises(KeyError, match="Unknown pipeline profile"):
        lookup_profile("qwen3-sortformer")


def test_should_not_advertise_retired_profiles() -> None:
    assert list_all_profiles() == []


def test_should_raise_for_unknown_profile() -> None:
    with pytest.raises(KeyError, match="Unknown pipeline profile"):
        lookup_profile("missing-profile")
