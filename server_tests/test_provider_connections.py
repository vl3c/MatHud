"""
Provider Connection Tests

Simple integration tests that verify each provider can make basic API calls.
Uses the cheapest models from each provider to minimize costs.
Tests are run WITHOUT tools to avoid tool schema validation issues.

These tests require valid API keys to be set in the environment:
- OPENAI_API_KEY for OpenAI
- ANTHROPIC_API_KEY for Anthropic
- OPENROUTER_API_KEY for OpenRouter

Tests are skipped if the corresponding API key is not available.
Tests skip with a warning if the API returns any error (billing, rate limit, etc.).
"""

from __future__ import annotations

import os
import unittest

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


def has_anthropic_key() -> bool:
    """Check if Anthropic API key is available."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def has_openrouter_key() -> bool:
    """Check if OpenRouter API key is available."""
    return bool(os.getenv("OPENROUTER_API_KEY"))


def is_error_response(content: str) -> bool:
    """Check if the response indicates an error."""
    lower = content.lower()
    return any(phrase in lower for phrase in [
        "encountered an error",
        "credit balance",
        "billing",
        "insufficient",
        "purchase credits",
        "quota exceeded",
        "rate limit",
        "try again",
    ])


class TestOpenAIConnection(unittest.TestCase):
    """Test OpenAI provider connection."""

    @unittest.skipUnless(has_openai_key(), "OPENAI_API_KEY not set")
    def test_openai_simple_completion(self) -> None:
        """Test OpenAI connection with a simple question using gpt-4.1-nano (cheapest)."""
        from static.openai_completions_api import OpenAIChatCompletionsAPI
        from static.ai_model import AIModel

        # Use the cheapest OpenAI model, with NO tools to avoid schema issues
        model = AIModel.from_identifier("gpt-4.1-nano")
        api = OpenAIChatCompletionsAPI(model=model, max_tokens=50, tools=[])

        # Simple prompt that should get a short response
        prompt = '{"user_message": "Reply with only the word: hello"}'

        response = api.create_chat_completion(prompt)

        # Verify we got a response
        self.assertIsNotNone(response)
        content = getattr(response.message, "content", "")
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

        # Check for error responses
        if is_error_response(content):
            self.skipTest("OpenAI API returned an error response")

        # The response should contain "hello" (case insensitive)
        self.assertIn("hello", content.lower())


class TestAnthropicConnection(unittest.TestCase):
    """Test Anthropic provider connection."""

    @unittest.skipUnless(has_anthropic_key(), "ANTHROPIC_API_KEY not set")
    def test_anthropic_simple_completion(self) -> None:
        """Test Anthropic connection with a simple question using Claude Haiku (cheapest)."""
        from static.providers import discover_providers
        from static.providers.anthropic_api import AnthropicAPI
        from static.ai_model import AIModel

        # Ensure providers are discovered
        discover_providers()

        # Use the cheapest Anthropic model (Haiku), with NO tools
        model = AIModel.from_identifier("claude-haiku-4-5-20251001")
        api = AnthropicAPI(model=model, max_tokens=50, tools=[])

        # Simple prompt that should get a short response
        prompt = '{"user_message": "Reply with only the word: hello"}'

        response = api.create_chat_completion(prompt)

        # Verify we got a response
        self.assertIsNotNone(response)
        content = getattr(response.message, "content", "")
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

        # Check for error responses (skip test if error occurred)
        if is_error_response(content):
            self.skipTest("Anthropic API returned an error (likely billing/credits)")

        # The response should contain "hello" (case insensitive)
        self.assertIn("hello", content.lower())


class TestOpenRouterConnection(unittest.TestCase):
    """Test OpenRouter provider connection."""

    @unittest.skipUnless(has_openrouter_key(), "OPENROUTER_API_KEY not set")
    def test_openrouter_simple_completion(self) -> None:
        """Test OpenRouter connection with a simple question using a free model."""
        from static.providers import discover_providers
        from static.providers.openrouter_api import OpenRouterAPI
        from static.ai_model import AIModel

        # Ensure providers are discovered
        discover_providers()

        # Use a free OpenRouter model, with NO tools
        model = AIModel.from_identifier("meta-llama/llama-3.3-70b-instruct:free")
        api = OpenRouterAPI(model=model, max_tokens=50, tools=[])

        # Simple prompt that should get a short response
        prompt = '{"user_message": "Reply with only the word: hello"}'

        response = api.create_chat_completion(prompt)

        # Verify we got a response
        self.assertIsNotNone(response)
        content = getattr(response.message, "content", "")
        self.assertIsInstance(content, str)

        # Skip if empty response (API may have issues)
        if len(content) == 0:
            self.skipTest("OpenRouter API returned empty response")

        # Check for error responses
        if is_error_response(content):
            self.skipTest("OpenRouter API returned an error")

        # The response should contain "hello" (case insensitive)
        self.assertIn("hello", content.lower())


class TestProviderStreaming(unittest.TestCase):
    """Test streaming functionality for each provider."""

    @unittest.skipUnless(has_openai_key(), "OPENAI_API_KEY not set")
    def test_openai_streaming(self) -> None:
        """Test OpenAI streaming with gpt-4.1-nano."""
        from static.openai_completions_api import OpenAIChatCompletionsAPI
        from static.ai_model import AIModel

        model = AIModel.from_identifier("gpt-4.1-nano")
        api = OpenAIChatCompletionsAPI(model=model, max_tokens=50, tools=[])

        prompt = '{"user_message": "Reply with only the word: hello"}'

        tokens = []
        final_event = None

        for event in api.create_chat_completion_stream(prompt):
            if event.get("type") == "token":
                tokens.append(event.get("text", ""))
            elif event.get("type") == "final":
                final_event = event

        # Verify we got a final event
        self.assertIsNotNone(final_event)
        self.assertIn("ai_message", final_event)

        # Check for error responses
        ai_message = str(final_event.get("ai_message", ""))
        if is_error_response(ai_message):
            self.skipTest("OpenAI streaming returned an error")

        combined = "".join(tokens)
        self.assertIn("hello", combined.lower())

    @unittest.skipUnless(has_anthropic_key(), "ANTHROPIC_API_KEY not set")
    def test_anthropic_streaming(self) -> None:
        """Test Anthropic streaming with Claude Haiku."""
        from static.providers import discover_providers
        from static.providers.anthropic_api import AnthropicAPI
        from static.ai_model import AIModel

        discover_providers()

        model = AIModel.from_identifier("claude-haiku-4-5-20251001")
        api = AnthropicAPI(model=model, max_tokens=50, tools=[])

        prompt = '{"user_message": "Reply with only the word: hello"}'

        tokens = []
        final_event = None

        for event in api.create_chat_completion_stream(prompt):
            if event.get("type") == "token":
                tokens.append(event.get("text", ""))
            elif event.get("type") == "final":
                final_event = event

        # Verify we got a final event
        self.assertIsNotNone(final_event)
        self.assertIn("ai_message", final_event)

        # Check for error responses in the final message
        ai_message = str(final_event.get("ai_message", ""))
        if is_error_response(ai_message):
            self.skipTest("Anthropic streaming returned an error (likely billing/credits)")

        combined = "".join(tokens)
        self.assertIn("hello", combined.lower())

    @unittest.skipUnless(has_openrouter_key(), "OPENROUTER_API_KEY not set")
    def test_openrouter_streaming(self) -> None:
        """Test OpenRouter streaming with a free model."""
        from static.providers import discover_providers
        from static.providers.openrouter_api import OpenRouterAPI
        from static.ai_model import AIModel

        discover_providers()

        # Use a free OpenRouter model
        model = AIModel.from_identifier("meta-llama/llama-3.3-70b-instruct:free")
        api = OpenRouterAPI(model=model, max_tokens=50, tools=[])

        prompt = '{"user_message": "Reply with only the word: hello"}'

        tokens = []
        final_event = None

        for event in api.create_chat_completion_stream(prompt):
            if event.get("type") == "token":
                tokens.append(event.get("text", ""))
            elif event.get("type") == "final":
                final_event = event

        # Verify we got a final event
        self.assertIsNotNone(final_event)
        self.assertIn("ai_message", final_event)

        # Check for error responses
        ai_message = str(final_event.get("ai_message", ""))
        if is_error_response(ai_message):
            self.skipTest("OpenRouter streaming returned an error")

        # For streaming, check the accumulated message from ai_message
        # (tokens may be empty if the model doesn't support streaming well)
        combined = "".join(tokens)
        if not combined:
            combined = ai_message

        # Skip if empty response (API may have issues)
        if not combined:
            self.skipTest("OpenRouter streaming returned empty response")

        self.assertIn("hello", combined.lower())


if __name__ == "__main__":
    unittest.main()
