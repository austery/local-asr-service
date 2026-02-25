"""
Unit tests for ModelRegistry (SPEC-108, cases MR-1..MR-6).

These are pure-function tests — no mocks needed.
"""

import pytest

from src.core.base_engine import EngineCapabilities
from src.core.model_registry import ModelSpec, alias_for, is_passthrough, list_all, lookup


class TestLookup:
    # MR-1
    def test_should_return_spec_when_alias_is_known(self) -> None:
        spec = lookup("paraformer")

        assert spec.engine_type == "funasr"
        assert "paraformer" in spec.model_id.lower()
        assert spec.alias == "paraformer"

    # MR-2
    def test_should_infer_mlx_engine_when_full_path_has_mlx_community_prefix(self) -> None:
        spec = lookup("mlx-community/some-custom-model")

        assert spec.engine_type == "mlx"
        assert spec.model_id == "mlx-community/some-custom-model"

    # MR-3
    def test_should_infer_funasr_engine_when_full_path_has_iic_prefix(self) -> None:
        spec = lookup("iic/some-custom-model")

        assert spec.engine_type == "funasr"
        assert spec.model_id == "iic/some-custom-model"

    # MR-4
    def test_should_raise_when_alias_is_completely_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            lookup("not-a-real-model")

    def test_should_resolve_registered_model_id_to_same_spec_as_alias(self) -> None:
        by_alias = lookup("qwen3-asr-mini")
        by_model_id = lookup("mlx-community/Qwen3-ASR-1.7B-4bit")

        assert by_alias == by_model_id


class TestCapabilities:
    # MR-5
    def test_should_declare_diarization_capability_for_paraformer(self) -> None:
        spec = lookup("paraformer")

        assert spec.capabilities.diarization is True
        assert spec.capabilities.timestamp is True

    # MR-6
    def test_should_declare_no_diarization_for_parakeet(self) -> None:
        spec = lookup("parakeet")

        assert spec.capabilities.diarization is False

    def test_should_declare_no_language_detect_for_parakeet(self) -> None:
        spec = lookup("parakeet")

        assert spec.capabilities.language_detect is False


class TestPassthrough:
    def test_should_return_true_for_none(self) -> None:
        assert is_passthrough(None) is True

    def test_should_return_true_for_whisper_1(self) -> None:
        assert is_passthrough("whisper-1") is True

    def test_should_return_false_for_known_alias(self) -> None:
        assert is_passthrough("paraformer") is False

    def test_should_return_false_for_full_path(self) -> None:
        assert is_passthrough("mlx-community/Qwen3-ASR-1.7B-4bit") is False

    def test_should_return_true_for_empty_string(self) -> None:
        """HTTP form data serializes Python None as '' in some clients — must be passthrough."""
        assert is_passthrough("") is True


class TestListAll:
    def test_should_return_all_builtin_models(self) -> None:
        models = list_all()

        aliases = {m.alias for m in models}
        assert "paraformer" in aliases
        assert "qwen3-asr-mini" in aliases
        assert "sensevoice-small" in aliases
        assert "parakeet" in aliases

    def test_should_return_models_sorted_by_alias(self) -> None:
        models = list_all()
        aliases = [m.alias for m in models]

        assert aliases == sorted(aliases)


class TestAliasFor:
    def test_should_return_alias_for_registered_model_id(self) -> None:
        alias = alias_for("mlx-community/Qwen3-ASR-1.7B-4bit")

        assert alias == "qwen3-asr-mini"

    def test_should_return_none_for_unknown_model_id(self) -> None:
        assert alias_for("unknown/model") is None
