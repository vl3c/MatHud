"""
Tests for the local-model registration helpers in static.ai_model.

Covers the module-level _format_display_name function and the
AIModel.register_local_models classmethod. Every expected value is derived by
tracing the implementation line by line; where a docstring and the code
disagree, the code is treated as the source of truth (disagreements are noted
in the task NOTES, not "fixed" here).
"""

from __future__ import annotations

import unittest

from static.ai_model import AIModel, _format_display_name


class TestFormatDisplayName(unittest.TestCase):
    """Test cases for the module-level _format_display_name helper."""

    def test_plain_letters_no_tag(self) -> None:
        """A bare alphabetic name is title-cased with no tag appended."""
        self.assertEqual(_format_display_name("llama"), "Llama")

    def test_hyphen_version_no_tag(self) -> None:
        """A hyphen becomes a space and a version number is left as-is."""
        self.assertEqual(_format_display_name("llama-3.1"), "Llama 3.1")

    def test_underscore_no_tag(self) -> None:
        """An underscore becomes a space in the display name."""
        self.assertEqual(_format_display_name("qwen_coder"), "Qwen Coder")

    def test_letter_digit_boundary_no_tag(self) -> None:
        """A space is inserted between a letter and a digit with no separator."""
        self.assertEqual(_format_display_name("llama3.1"), "Llama 3.1")

    def test_digit_letter_boundary_no_tag(self) -> None:
        """A space is inserted between a digit and a letter (digit first)."""
        self.assertEqual(_format_display_name("123abc"), "123 Abc")

    def test_mixed_letter_digit_boundary_no_tag(self) -> None:
        """Spaces are inserted at every letter/digit and digit/letter transition."""
        self.assertEqual(_format_display_name("a1b2"), "A 1 B 2")

    def test_decimal_point_no_space(self) -> None:
        """No space is inserted around the decimal point of a version number."""
        self.assertEqual(_format_display_name("v2.0"), "V 2.0")

    def test_with_non_latest_tag(self) -> None:
        """A non-latest tag is upper-cased and appended to the display name."""
        self.assertEqual(_format_display_name("llama3.1:8b"), "Llama 3.1 8B")

    def test_latest_tag_lower(self) -> None:
        """A lower-case 'latest' tag is not appended."""
        self.assertEqual(_format_display_name("mistral:latest"), "Mistral")

    def test_latest_tag_upper(self) -> None:
        """An upper-case 'latest' tag is not appended (case-insensitive match)."""
        self.assertEqual(_format_display_name("mistral:LATEST"), "Mistral")

    def test_latest_tag_title_case(self) -> None:
        """A title-case 'Latest' tag is not appended (case-insensitive match)."""
        self.assertEqual(_format_display_name("mistral:Latest"), "Mistral")

    def test_with_other_tag(self) -> None:
        """Any other non-latest tag is upper-cased and appended."""
        self.assertEqual(_format_display_name("mistral:8b"), "Mistral 8B")

    def test_multi_part_name_with_tag(self) -> None:
        """A multi-part name with a version and a non-latest tag is formatted."""
        self.assertEqual(_format_display_name("qwen2.5-coder:7b"), "Qwen 2.5 Coder 7B")

    def test_empty_string(self) -> None:
        """An empty model name produces an empty display name."""
        self.assertEqual(_format_display_name(""), "")

    def test_all_digits_name(self) -> None:
        """An all-digit name is returned unchanged (no title-casing)."""
        self.assertEqual(_format_display_name("12345"), "12345")


class TestRegisterLocalModels(unittest.TestCase):
    """Test cases for the AIModel.register_local_models classmethod."""

    def setUp(self) -> None:
        """Snapshot MODEL_CONFIGS so each test can restore it in place."""
        self._original_configs = dict(AIModel.MODEL_CONFIGS)

    def tearDown(self) -> None:
        """Restore MODEL_CONFIGS to its pre-test state, mutating the dict in place."""
        AIModel.MODEL_CONFIGS.clear()
        AIModel.MODEL_CONFIGS.update(self._original_configs)

    def test_empty_input_list(self) -> None:
        """An empty models_info list returns an empty list and changes nothing."""
        result = AIModel.register_local_models("ollama", [])
        self.assertEqual(result, [])
        self.assertEqual(AIModel.MODEL_CONFIGS, self._original_configs)

    def test_returns_registered_names_in_input_order(self) -> None:
        """The returned list holds the registered identifiers in the order given."""
        models_info = [
            {"name": "first-model"},
            {"name": "second-model"},
            {"name": "third-model"},
        ]
        result = AIModel.register_local_models("ollama", models_info)
        self.assertEqual(result, ["first-model", "second-model", "third-model"])

    def test_skips_entries_missing_name(self) -> None:
        """An entry without a 'name' key is skipped."""
        result = AIModel.register_local_models("ollama", [{"no-name-key": "value"}])
        self.assertEqual(result, [])
        self.assertEqual(AIModel.MODEL_CONFIGS, self._original_configs)

    def test_skips_entries_with_empty_name(self) -> None:
        """An entry with an empty 'name' value is skipped."""
        result = AIModel.register_local_models("ollama", [{"name": ""}])
        self.assertEqual(result, [])
        self.assertEqual(AIModel.MODEL_CONFIGS, self._original_configs)

    def test_stores_full_config_for_single_model(self) -> None:
        """register_local_models stores all four expected keys for each model."""
        AIModel.register_local_models("ollama", [{"name": "local-llama-3.1"}])
        self.assertIn("local-llama-3.1", AIModel.MODEL_CONFIGS)
        self.assertEqual(
            AIModel.MODEL_CONFIGS["local-llama-3.1"],
            {
                "has_vision": False,
                "is_reasoning_model": False,
                "provider": "ollama",
                "display_name": "Local Llama 3.1",
            },
        )

    def test_stored_display_name_matches_format_helper(self) -> None:
        """The stored display_name equals _format_display_name(model_name)."""
        name = "local-qwen2.5-coder"
        AIModel.register_local_models("ollama", [{"name": name}])
        stored = AIModel.MODEL_CONFIGS[name]
        self.assertEqual(stored["display_name"], _format_display_name(name))
        self.assertEqual(stored["display_name"], "Local Qwen 2.5 Coder")

    def test_stores_provider_value(self) -> None:
        """register_local_models stores the provider value that was passed in."""
        AIModel.register_local_models("my-provider", [{"name": "local-probe"}])
        self.assertEqual(AIModel.MODEL_CONFIGS["local-probe"]["provider"], "my-provider")

    def test_registers_multiple_models_leaves_originals_untouched(self) -> None:
        """Multiple local models are added while existing configs keep their values."""
        AIModel.register_local_models("ollama", [{"name": "a-local"}, {"name": "b-local"}])
        self.assertIn("a-local", AIModel.MODEL_CONFIGS)
        self.assertIn("b-local", AIModel.MODEL_CONFIGS)
        for original_id, original_cfg in self._original_configs.items():
            self.assertEqual(AIModel.MODEL_CONFIGS[original_id], original_cfg)

    def test_model_configs_is_mutated_in_place(self) -> None:
        """register_local_models writes into the existing dict object (no rebinding)."""
        before = AIModel.MODEL_CONFIGS
        AIModel.register_local_models("ollama", [{"name": "inplace-check"}])
        self.assertIs(AIModel.MODEL_CONFIGS, before)

    def test_name_none_is_stringified(self) -> None:
        """Current behaviour, likely unintended: a None name is registered as the literal 'None'."""
        result = AIModel.register_local_models("ollama", [{"name": None}])
        self.assertEqual(result, ["None"])
        self.assertIn("None", AIModel.MODEL_CONFIGS)


if __name__ == "__main__":
    unittest.main()
