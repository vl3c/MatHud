"""
Tests for the AIModel configuration module.

Tests model configurations, factory methods, and capability flags including
the is_reasoning_model flag used for Responses API routing (OpenAI) and for
omitting the temperature parameter on adaptive-thinking Claude models.
"""

from __future__ import annotations

import unittest

from static.ai_model import (
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    PROVIDER_OPENROUTER,
    AIModel,
)


class TestAIModel(unittest.TestCase):
    """Test cases for AIModel class."""

    def test_model_initialization(self) -> None:
        """Test direct model initialization with all parameters."""
        model = AIModel(identifier="test-model", has_vision=True, is_reasoning_model=True)
        self.assertEqual(model.id, "test-model")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)

    def test_model_initialization_defaults(self) -> None:
        """Test that is_reasoning_model defaults to False."""
        model = AIModel(identifier="test-model", has_vision=False)
        self.assertFalse(model.is_reasoning_model)

    def test_from_identifier_gpt56_sol(self) -> None:
        """Test GPT-5.6 Sol is a reasoning model with vision."""
        model = AIModel.from_identifier("gpt-5.6-sol")
        self.assertEqual(model.id, "gpt-5.6-sol")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)

    def test_from_identifier_gpt55_medium_reasoning(self) -> None:
        """Test GPT-5.5 (the default) has medium reasoning effort configured."""
        model = AIModel.from_identifier("gpt-5.5")
        self.assertEqual(model.id, "gpt-5.5")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)
        self.assertEqual(model.reasoning_effort, "medium")

    def test_from_identifier_gpt52_medium_reasoning(self) -> None:
        """Test GPT-5.2 remains a reasoning model with medium reasoning effort."""
        model = AIModel.from_identifier("gpt-5.2")
        self.assertEqual(model.id, "gpt-5.2")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)
        self.assertEqual(model.reasoning_effort, "medium")

    def test_from_identifier_gpt56_luna(self) -> None:
        """Test GPT-5.6 Luna is a reasoning model with vision."""
        model = AIModel.from_identifier("gpt-5.6-luna")
        self.assertEqual(model.id, "gpt-5.6-luna")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)

    def test_from_identifier_gpt41(self) -> None:
        """Test GPT-4.1 is a standard model with vision."""
        model = AIModel.from_identifier("gpt-4.1")
        self.assertEqual(model.id, "gpt-4.1")
        self.assertTrue(model.has_vision)
        self.assertFalse(model.is_reasoning_model)

    def test_from_identifier_gpt4o_mini(self) -> None:
        """Test GPT-4o-mini is a standard model with vision."""
        model = AIModel.from_identifier("gpt-4o-mini")
        self.assertEqual(model.id, "gpt-4o-mini")
        self.assertTrue(model.has_vision)
        self.assertFalse(model.is_reasoning_model)

    def test_from_identifier_deepseek_no_vision(self) -> None:
        """Test a text-only OpenRouter model reports no vision support."""
        model = AIModel.from_identifier("deepseek/deepseek-v4-pro")
        self.assertEqual(model.id, "deepseek/deepseek-v4-pro")
        self.assertFalse(model.has_vision)
        self.assertFalse(model.is_reasoning_model)
        self.assertEqual(model.provider, PROVIDER_OPENROUTER)

    def test_from_identifier_unknown_model(self) -> None:
        """Test unknown model gets default config (no vision, not reasoning)."""
        model = AIModel.from_identifier("unknown-model-xyz")
        self.assertEqual(model.id, "unknown-model-xyz")
        self.assertFalse(model.has_vision)
        self.assertFalse(model.is_reasoning_model)

    def test_get_default_model(self) -> None:
        """Test default model is GPT-5.5 with medium reasoning."""
        model = AIModel.get_default_model()
        self.assertEqual(model.id, AIModel.DEFAULT_MODEL)
        self.assertEqual(model.id, "gpt-5.5")
        self.assertTrue(model.is_reasoning_model)
        self.assertEqual(model.reasoning_effort, "medium")

    def test_str_representation(self) -> None:
        """Test string representation returns model identifier."""
        model = AIModel.from_identifier("gpt-4.1")
        self.assertEqual(str(model), "gpt-4.1")

    def test_all_reasoning_models_identified(self) -> None:
        """Test that all OpenAI Responses-API models are flagged as reasoning models."""
        reasoning_models = ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-5.2"]
        for model_id in reasoning_models:
            model = AIModel.from_identifier(model_id)
            self.assertTrue(model.is_reasoning_model, f"{model_id} should be a reasoning model")
            self.assertEqual(model.provider, PROVIDER_OPENAI)

    def test_all_standard_models_identified(self) -> None:
        """Test that all OpenAI Chat Completions models are non-reasoning."""
        standard_models = [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o-mini",
        ]
        for model_id in standard_models:
            model = AIModel.from_identifier(model_id)
            self.assertFalse(model.is_reasoning_model, f"{model_id} should NOT be a reasoning model")
            self.assertEqual(model.provider, PROVIDER_OPENAI)

    def test_anthropic_adaptive_thinking_models_flagged_reasoning(self) -> None:
        """Adaptive-thinking Claude models are flagged is_reasoning_model so the provider
        omits the temperature parameter (which they reject with a 400)."""
        for model_id in ["claude-fable-5", "claude-opus-4-8", "claude-sonnet-5"]:
            model = AIModel.from_identifier(model_id)
            self.assertEqual(model.provider, PROVIDER_ANTHROPIC)
            self.assertTrue(model.is_reasoning_model, f"{model_id} should be flagged is_reasoning_model")
            self.assertTrue(model.has_vision)

    def test_anthropic_haiku_accepts_temperature(self) -> None:
        """Claude Haiku 4.5 still accepts temperature, so it stays non-reasoning."""
        model = AIModel.from_identifier("claude-haiku-4-5")
        self.assertEqual(model.provider, PROVIDER_ANTHROPIC)
        self.assertFalse(model.is_reasoning_model)

    def test_model_configs_completeness(self) -> None:
        """Test that MODEL_CONFIGS has both required keys for all models."""
        for model_id, config in AIModel.MODEL_CONFIGS.items():
            self.assertIn("has_vision", config, f"{model_id} missing 'has_vision' in config")
            self.assertIn("is_reasoning_model", config, f"{model_id} missing 'is_reasoning_model' in config")


if __name__ == "__main__":
    unittest.main()
