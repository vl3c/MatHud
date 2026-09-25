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

    def test_from_identifier_gpt6_sol(self) -> None:
        """Test GPT-6 Sol is a reasoning model with vision and medium effort."""
        model = AIModel.from_identifier("gpt-6-sol")
        self.assertEqual(model.id, "gpt-6-sol")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)
        self.assertEqual(model.reasoning_effort, "medium")

    def test_from_identifier_gpt6_astra_never_none_effort(self) -> None:
        """Test GPT-6 Astra keeps an effort it accepts (it rejects "none")."""
        model = AIModel.from_identifier("gpt-6-astra")
        self.assertEqual(model.id, "gpt-6-astra")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)
        self.assertEqual(model.reasoning_effort, "medium")

    def test_from_identifier_gpt6_luna_low_reasoning(self) -> None:
        """Test GPT-6 Luna is a reasoning model with low reasoning effort."""
        model = AIModel.from_identifier("gpt-6-luna")
        self.assertEqual(model.id, "gpt-6-luna")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)
        self.assertEqual(model.reasoning_effort, "low")

    def test_from_identifier_gpt56_sol(self) -> None:
        """Test GPT-5.6 Sol remains a reasoning model with medium reasoning effort."""
        model = AIModel.from_identifier("gpt-5.6-sol")
        self.assertEqual(model.id, "gpt-5.6-sol")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)
        self.assertEqual(model.reasoning_effort, "medium")

    def test_from_identifier_openrouter_no_vision(self) -> None:
        """Test a text-only OpenRouter model reports no vision support."""
        model = AIModel.from_identifier("nvidia/nemotron-3-ultra-550b-a55b:free")
        self.assertEqual(model.id, "nvidia/nemotron-3-ultra-550b-a55b:free")
        self.assertFalse(model.has_vision)
        self.assertFalse(model.is_reasoning_model)
        self.assertEqual(model.provider, PROVIDER_OPENROUTER)

    def test_from_identifier_openrouter_vision(self) -> None:
        """Test a vision-capable OpenRouter model reports vision support."""
        model = AIModel.from_identifier("google/gemini-3.8-flash")
        self.assertTrue(model.has_vision)
        self.assertFalse(model.is_reasoning_model)
        self.assertEqual(model.provider, PROVIDER_OPENROUTER)

    def test_from_identifier_unknown_model(self) -> None:
        """Test unknown model gets default config (no vision, not reasoning)."""
        model = AIModel.from_identifier("unknown-model-xyz")
        self.assertEqual(model.id, "unknown-model-xyz")
        self.assertFalse(model.has_vision)
        self.assertFalse(model.is_reasoning_model)

    def test_get_default_model(self) -> None:
        """Test default model is GPT-6 Sol with medium reasoning."""
        model = AIModel.get_default_model()
        self.assertEqual(model.id, AIModel.DEFAULT_MODEL)
        self.assertEqual(model.id, "gpt-6-sol")
        self.assertTrue(model.is_reasoning_model)
        self.assertEqual(model.reasoning_effort, "medium")

    def test_str_representation(self) -> None:
        """Test string representation returns model identifier."""
        model = AIModel.from_identifier("gpt-6-luna")
        self.assertEqual(str(model), "gpt-6-luna")

    def test_all_openai_models_are_reasoning_models(self) -> None:
        """Test that every OpenAI model is flagged as a Responses-API reasoning model."""
        openai_models = [
            model_id for model_id, config in AIModel.MODEL_CONFIGS.items() if config.get("provider") == PROVIDER_OPENAI
        ]
        self.assertEqual(openai_models, ["gpt-6-sol", "gpt-6-astra", "gpt-6-luna", "gpt-5.6-sol"])
        for model_id in openai_models:
            model = AIModel.from_identifier(model_id)
            self.assertTrue(model.is_reasoning_model, f"{model_id} should be a reasoning model")
            self.assertIn(model.reasoning_effort, ("low", "medium"), f"{model_id} needs an explicit effort")

    def test_anthropic_adaptive_thinking_models_flagged_reasoning(self) -> None:
        """Adaptive-thinking Claude models are flagged is_reasoning_model so the provider
        omits the temperature parameter (which they reject with a 400)."""
        expected_efforts = {"claude-fable-5-1": "low", "claude-opus-5-5": "medium", "claude-sonnet-5": "medium"}
        for model_id, effort in expected_efforts.items():
            model = AIModel.from_identifier(model_id)
            self.assertEqual(model.provider, PROVIDER_ANTHROPIC)
            self.assertTrue(model.is_reasoning_model, f"{model_id} should be flagged is_reasoning_model")
            self.assertTrue(model.has_vision)
            self.assertEqual(model.reasoning_effort, effort)

    def test_anthropic_haiku_accepts_temperature(self) -> None:
        """Claude Haiku 4.5 still accepts temperature, so it stays non-reasoning."""
        model = AIModel.from_identifier("claude-haiku-4-5")
        self.assertEqual(model.provider, PROVIDER_ANTHROPIC)
        self.assertFalse(model.is_reasoning_model)
        self.assertIsNone(model.reasoning_effort)

    def test_openrouter_models_are_not_reasoning_models(self) -> None:
        """OpenRouter models go through Chat Completions, never the Responses API."""
        for model_id, config in AIModel.MODEL_CONFIGS.items():
            if config.get("provider") == PROVIDER_OPENROUTER:
                self.assertFalse(config.get("is_reasoning_model"), f"{model_id} should not be a reasoning model")

    def test_model_configs_completeness(self) -> None:
        """Test that MODEL_CONFIGS has both required keys for all models."""
        for model_id, config in AIModel.MODEL_CONFIGS.items():
            self.assertIn("has_vision", config, f"{model_id} missing 'has_vision' in config")
            self.assertIn("is_reasoning_model", config, f"{model_id} missing 'is_reasoning_model' in config")


if __name__ == "__main__":
    unittest.main()
