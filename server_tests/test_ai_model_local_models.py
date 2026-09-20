"""
Tests for the local-model registration helpers in static.ai_model.

Covers the module-level _format_display_name function and the
AIModel.register_local_models and AIModel.unregister_local_models classmethods. Every expected value is derived by
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

    def test_version_token_single_letter_runs(self) -> None:
        """A word of single letters between digits is kept intact and upper-cased."""
        self.assertEqual(_format_display_name("a1b2"), "A1B2")

    def test_version_token_single_letter(self) -> None:
        """A version-like word with one leading letter and a decimal is kept intact."""
        self.assertEqual(_format_display_name("v2.0"), "V2.0")

    def test_two_letter_run_with_digit_is_split(self) -> None:
        """A two-letter run next to a digit is split rather than kept intact."""
        self.assertEqual(_format_display_name("ab1"), "Ab 1")

    def test_two_letter_run_before_decimal_is_split(self) -> None:
        """A two-letter run before a decimal is split rather than kept intact."""
        self.assertEqual(_format_display_name("xl2.5"), "Xl 2.5")

    def test_two_letter_acronym_run_is_split(self) -> None:
        """A two-letter acronym run is split, then upper-cased after the split."""
        self.assertEqual(_format_display_name("vl2"), "VL 2")

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

    def test_deepseek_r1_with_tag(self) -> None:
        """A single-letter suffix is kept intact and a size tag is upper-cased."""
        self.assertEqual(_format_display_name("deepseek-r1:14b"), "Deepseek R1 14B")

    def test_phi3(self) -> None:
        """A letter/digit run is split into a capitalised word and a digit token."""
        self.assertEqual(_format_display_name("phi3"), "Phi 3")

    def test_gpt_oss_acronyms(self) -> None:
        """Acronym words are fully upper-cased with a size tag appended."""
        self.assertEqual(_format_display_name("gpt-oss:20b"), "GPT OSS 20B")

    def test_gemma2_9b(self) -> None:
        """A single-letter suffix is upper-cased and the LATEST tag is omitted."""
        self.assertEqual(_format_display_name("gemma2_9b:LATEST"), "Gemma 2 9B")

    def test_qwen3_30b_a3b(self) -> None:
        """Multiple version-like words are each kept intact and upper-cased."""
        self.assertEqual(_format_display_name("qwen3-30b-a3b"), "Qwen 3 30B A3B")

    def test_mistral_7b(self) -> None:
        """A size suffix is kept intact and upper-cased."""
        self.assertEqual(_format_display_name("mistral-7b"), "Mistral 7B")

    def test_qwen2_5_vl(self) -> None:
        """The vl acronym is upper-cased alongside a version token."""
        self.assertEqual(_format_display_name("qwen2.5-vl:7b"), "Qwen 2.5 VL 7B")

    def test_llava_v1_6(self) -> None:
        """A version tag with a single leading letter is upper-cased and appended."""
        self.assertEqual(_format_display_name("llava:v1.6"), "Llava V1.6")

    def test_mixtral_8x7b(self) -> None:
        """A multi-token size tag is upper-cased and appended."""
        self.assertEqual(_format_display_name("mixtral:8x7b"), "Mixtral 8X7B")

    def test_1_5b(self) -> None:
        """A version-like word beginning with a digit is kept intact and upper-cased."""
        self.assertEqual(_format_display_name("1.5b"), "1.5B")

    def test_local_qwen2_5_coder(self) -> None:
        """A local multi-part name is title-cased with a version split."""
        self.assertEqual(_format_display_name("local-qwen2.5-coder"), "Local Qwen 2.5 Coder")

    def test_llama3_2_vision(self) -> None:
        """A versioned multi-part name with a size tag is formatted."""
        self.assertEqual(_format_display_name("llama3.2-vision:11b"), "Llama 3.2 Vision 11B")


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
        result = AIModel.register_local_models("local_agent", [])
        self.assertEqual(result, [])
        self.assertEqual(AIModel.MODEL_CONFIGS, self._original_configs)

    def test_returns_registered_names_in_input_order(self) -> None:
        """The returned list holds the registered identifiers in the order given."""
        models_info = [
            {"name": "first-model"},
            {"name": "second-model"},
            {"name": "third-model"},
        ]
        result = AIModel.register_local_models("local_agent", models_info)
        self.assertEqual(result, ["first-model", "second-model", "third-model"])

    def test_skips_entries_missing_name(self) -> None:
        """An entry without a 'name' key is skipped."""
        result = AIModel.register_local_models("local_agent", [{"no-name-key": "value"}])
        self.assertEqual(result, [])
        self.assertEqual(AIModel.MODEL_CONFIGS, self._original_configs)

    def test_skips_entries_with_empty_name(self) -> None:
        """An entry with an empty 'name' value is skipped."""
        result = AIModel.register_local_models("local_agent", [{"name": ""}])
        self.assertEqual(result, [])
        self.assertEqual(AIModel.MODEL_CONFIGS, self._original_configs)

    def test_stores_full_config_for_single_model(self) -> None:
        """register_local_models stores all four expected keys for each model."""
        AIModel.register_local_models("local_agent", [{"name": "local-llama-3.1"}])
        self.assertIn("local-llama-3.1", AIModel.MODEL_CONFIGS)
        self.assertEqual(
            AIModel.MODEL_CONFIGS["local-llama-3.1"],
            {
                "has_vision": False,
                "is_reasoning_model": False,
                "provider": "local_agent",
                "display_name": "Local Llama 3.1",
            },
        )

    def test_stored_display_name_matches_format_helper(self) -> None:
        """The stored display_name equals _format_display_name(model_name)."""
        name = "local-qwen2.5-coder"
        AIModel.register_local_models("local_agent", [{"name": name}])
        stored = AIModel.MODEL_CONFIGS[name]
        self.assertEqual(stored["display_name"], _format_display_name(name))
        self.assertEqual(stored["display_name"], "Local Qwen 2.5 Coder")

    def test_stores_provider_value(self) -> None:
        """register_local_models stores the provider value that was passed in."""
        AIModel.register_local_models("my-provider", [{"name": "local-probe"}])
        self.assertEqual(AIModel.MODEL_CONFIGS["local-probe"]["provider"], "my-provider")

    def test_registers_multiple_models_leaves_originals_untouched(self) -> None:
        """Multiple local models are added while existing configs keep their values."""
        AIModel.register_local_models("local_agent", [{"name": "a-local"}, {"name": "b-local"}])
        self.assertIn("a-local", AIModel.MODEL_CONFIGS)
        self.assertIn("b-local", AIModel.MODEL_CONFIGS)
        for original_id, original_cfg in self._original_configs.items():
            self.assertEqual(AIModel.MODEL_CONFIGS[original_id], original_cfg)

    def test_model_configs_is_mutated_in_place(self) -> None:
        """register_local_models writes into the existing dict object (no rebinding)."""
        before = AIModel.MODEL_CONFIGS
        AIModel.register_local_models("local_agent", [{"name": "inplace-check"}])
        self.assertIs(AIModel.MODEL_CONFIGS, before)

    def test_skips_none_name(self) -> None:
        """A None name is skipped instead of being registered as the literal 'None'."""
        result = AIModel.register_local_models("local_agent", [{"name": None}])
        self.assertEqual(result, [])
        self.assertNotIn("None", AIModel.MODEL_CONFIGS)
        self.assertEqual(AIModel.MODEL_CONFIGS, self._original_configs)

    def test_skips_whitespace_name(self) -> None:
        """A whitespace-only name is skipped."""
        result = AIModel.register_local_models("local_agent", [{"name": "   "}, {"name": "\t"}])
        self.assertEqual(result, [])
        self.assertEqual(AIModel.MODEL_CONFIGS, self._original_configs)


    def test_display_name_override_is_used(self) -> None:
        """A provider-supplied display_name replaces the formatted default."""
        name = r"C:\models\Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"
        AIModel.register_local_models(
            "local_agent",
            [{"name": name, "display_name": "Qwen3.8-27B-GSQ-RCO-IQ3_XXS"}],
        )
        self.assertEqual(AIModel.MODEL_CONFIGS[name]["display_name"], "Qwen3.8-27B-GSQ-RCO-IQ3_XXS")

    def test_blank_display_name_override_falls_back(self) -> None:
        """A blank display_name override falls back to the formatted default."""
        AIModel.register_local_models("local_agent", [{"name": "local-probe", "display_name": ""}])
        self.assertEqual(
            AIModel.MODEL_CONFIGS["local-probe"]["display_name"],
            _format_display_name("local-probe"),
        )


class TestUnregisterLocalModels(unittest.TestCase):
    """Test cases for the AIModel.unregister_local_models classmethod."""

    def setUp(self) -> None:
        """Snapshot MODEL_CONFIGS so each test can restore it in place."""
        self._original_configs = dict(AIModel.MODEL_CONFIGS)

    def tearDown(self) -> None:
        """Restore MODEL_CONFIGS to its pre-test state, mutating the dict in place."""
        AIModel.MODEL_CONFIGS.clear()
        AIModel.MODEL_CONFIGS.update(self._original_configs)

    def test_removes_only_matching_provider(self) -> None:
        """Models of the named provider are dropped and others are kept."""
        AIModel.register_local_models("local_agent", [{"name": "stale-local"}])
        AIModel.register_local_models("other-provider", [{"name": "kept-local"}])
        removed = AIModel.unregister_local_models("local_agent")
        self.assertEqual(removed, ["stale-local"])
        self.assertNotIn("stale-local", AIModel.MODEL_CONFIGS)
        self.assertIn("kept-local", AIModel.MODEL_CONFIGS)

    def test_leaves_static_configs_untouched(self) -> None:
        """Built-in model configs survive a local provider purge."""
        AIModel.register_local_models("local_agent", [{"name": "stale-local"}])
        AIModel.unregister_local_models("local_agent")
        self.assertEqual(AIModel.MODEL_CONFIGS, self._original_configs)

    def test_unknown_provider_removes_nothing(self) -> None:
        """A provider with no registered models is a no-op."""
        self.assertEqual(AIModel.unregister_local_models("never-registered"), [])
        self.assertEqual(AIModel.MODEL_CONFIGS, self._original_configs)

    def test_removes_every_model_of_the_provider(self) -> None:
        """All models of the provider are removed in registration order."""
        AIModel.register_local_models("local_agent", [{"name": "one"}, {"name": "two"}])
        self.assertEqual(AIModel.unregister_local_models("local_agent"), ["one", "two"])

    def test_model_configs_is_mutated_in_place(self) -> None:
        """unregister_local_models writes into the existing dict object (no rebinding)."""
        AIModel.register_local_models("local_agent", [{"name": "inplace-check"}])
        before = AIModel.MODEL_CONFIGS
        AIModel.unregister_local_models("local_agent")
        self.assertIs(AIModel.MODEL_CONFIGS, before)


if __name__ == "__main__":
    unittest.main()
