"""
Tests for the AIModel configuration module.

Tests model configurations, factory methods, and capability flags including
the new is_reasoning_model flag for Responses API routing.
"""

from __future__ import annotations

import unittest

from static.ai_model import AIModel


class TestAIModel(unittest.TestCase):
    """Test cases for AIModel class."""

    def test_model_initialization(self) -> None:
        """Test direct model initialization with all parameters."""
        model = AIModel(
            identifier="test-model",
            has_vision=True,
            is_reasoning_model=True
        )
        self.assertEqual(model.id, "test-model")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)

    def test_model_initialization_defaults(self) -> None:
        """Test that is_reasoning_model defaults to False."""
        model = AIModel(identifier="test-model", has_vision=False)
        self.assertFalse(model.is_reasoning_model)

    def test_from_identifier_gpt5_chat_latest(self) -> None:
        """Test GPT-5-chat-latest is a reasoning model with vision."""
        model = AIModel.from_identifier("gpt-5-chat-latest")
        self.assertEqual(model.id, "gpt-5-chat-latest")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)

    def test_from_identifier_gpt52_chat_latest(self) -> None:
        """Test GPT-5.2-chat-latest is a reasoning model with vision."""
        model = AIModel.from_identifier("gpt-5.2-chat-latest")
        self.assertEqual(model.id, "gpt-5.2-chat-latest")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)

    def test_from_identifier_gpt52_medium_reasoning(self) -> None:
        """Test GPT-5.2 has medium reasoning effort configured."""
        model = AIModel.from_identifier("gpt-5.2")
        self.assertEqual(model.id, "gpt-5.2")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)
        self.assertEqual(model.reasoning_effort, "medium")

    def test_from_identifier_o3(self) -> None:
        """Test o3 is a reasoning model without vision."""
        model = AIModel.from_identifier("o3")
        self.assertEqual(model.id, "o3")
        self.assertFalse(model.has_vision)
        self.assertTrue(model.is_reasoning_model)

    def test_from_identifier_o4_mini(self) -> None:
        """Test o4-mini is a reasoning model with vision."""
        model = AIModel.from_identifier("o4-mini")
        self.assertEqual(model.id, "o4-mini")
        self.assertTrue(model.has_vision)
        self.assertTrue(model.is_reasoning_model)

    def test_from_identifier_gpt4o(self) -> None:
        """Test GPT-4o is a standard model (not reasoning) with vision."""
        model = AIModel.from_identifier("gpt-4o")
        self.assertEqual(model.id, "gpt-4o")
        self.assertTrue(model.has_vision)
        self.assertFalse(model.is_reasoning_model)

    def test_from_identifier_gpt4o_mini(self) -> None:
        """Test GPT-4o-mini is a standard model with vision."""
        model = AIModel.from_identifier("gpt-4o-mini")
        self.assertEqual(model.id, "gpt-4o-mini")
        self.assertTrue(model.has_vision)
        self.assertFalse(model.is_reasoning_model)

    def test_from_identifier_gpt41(self) -> None:
        """Test GPT-4.1 is a standard model with vision."""
        model = AIModel.from_identifier("gpt-4.1")
        self.assertEqual(model.id, "gpt-4.1")
        self.assertTrue(model.has_vision)
        self.assertFalse(model.is_reasoning_model)

    def test_from_identifier_gpt35_turbo(self) -> None:
        """Test GPT-3.5-turbo is a standard model without vision."""
        model = AIModel.from_identifier("gpt-3.5-turbo")
        self.assertEqual(model.id, "gpt-3.5-turbo")
        self.assertFalse(model.has_vision)
        self.assertFalse(model.is_reasoning_model)

    def test_from_identifier_unknown_model(self) -> None:
        """Test unknown model gets default config (no vision, not reasoning)."""
        model = AIModel.from_identifier("unknown-model-xyz")
        self.assertEqual(model.id, "unknown-model-xyz")
        self.assertFalse(model.has_vision)
        self.assertFalse(model.is_reasoning_model)

    def test_get_default_model(self) -> None:
        """Test default model is GPT-5.2 with medium reasoning."""
        model = AIModel.get_default_model()
        self.assertEqual(model.id, AIModel.DEFAULT_MODEL)
        self.assertEqual(model.id, "gpt-5.2")
        self.assertTrue(model.is_reasoning_model)
        self.assertEqual(model.reasoning_effort, "medium")

    def test_str_representation(self) -> None:
        """Test string representation returns model identifier."""
        model = AIModel.from_identifier("gpt-4o")
        self.assertEqual(str(model), "gpt-4o")

    def test_all_reasoning_models_identified(self) -> None:
        """Test that all reasoning models are correctly identified."""
        reasoning_models = ["gpt-5-chat-latest", "gpt-5.2-chat-latest", "gpt-5.2", "o3", "o4-mini"]
        for model_id in reasoning_models:
            model = AIModel.from_identifier(model_id)
            self.assertTrue(
                model.is_reasoning_model,
                f"{model_id} should be a reasoning model"
            )

    def test_all_standard_models_identified(self) -> None:
        """Test that all standard models are correctly identified as non-reasoning."""
        standard_models = [
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "gpt-4o", "gpt-4o-mini", "gpt-5-nano", "gpt-3.5-turbo"
        ]
        for model_id in standard_models:
            model = AIModel.from_identifier(model_id)
            self.assertFalse(
                model.is_reasoning_model,
                f"{model_id} should NOT be a reasoning model"
            )

    def test_model_configs_completeness(self) -> None:
        """Test that MODEL_CONFIGS has both required keys for all models."""
        for model_id, config in AIModel.MODEL_CONFIGS.items():
            self.assertIn(
                "has_vision", config,
                f"{model_id} missing 'has_vision' in config"
            )
            self.assertIn(
                "is_reasoning_model", config,
                f"{model_id} missing 'is_reasoning_model' in config"
            )


if __name__ == '__main__':
    unittest.main()

