"""
Tests for the OpenAI Chat Completions API module.

Tests Chat Completions API implementation including streaming, tool calls,
and integration tests with actual API calls.
"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import Mock, patch, MagicMock

from static.openai_completions_api import OpenAIChatCompletionsAPI


class TestOpenAIChatCompletionsAPI(unittest.TestCase):
    """Unit tests for OpenAIChatCompletionsAPI class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.original_api_key = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = 'test-api-key'

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.original_api_key:
            os.environ['OPENAI_API_KEY'] = self.original_api_key
        else:
            os.environ.pop('OPENAI_API_KEY', None)

    @patch('static.openai_api_base.OpenAI')
    def test_initialization(self, mock_openai: Mock) -> None:
        """Test API initializes correctly."""
        api = OpenAIChatCompletionsAPI()
        self.assertIsNotNone(api.client)
        self.assertIsNotNone(api.model)

    @patch('static.openai_api_base.OpenAI')
    def test_create_assistant_message_simple(self, mock_openai: Mock) -> None:
        """Test _create_assistant_message with simple response."""
        api = OpenAIChatCompletionsAPI()
        
        response_message = SimpleNamespace(
            content="Hello, world!",
            tool_calls=None
        )
        
        result = api._create_assistant_message(response_message)
        
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"], "Hello, world!")
        self.assertNotIn("tool_calls", result)

    @patch('static.openai_api_base.OpenAI')
    def test_create_assistant_message_with_tool_calls(self, mock_openai: Mock) -> None:
        """Test _create_assistant_message with tool calls."""
        api = OpenAIChatCompletionsAPI()
        
        tool_call = SimpleNamespace(
            id="call_123",
            function=SimpleNamespace(
                name="create_point",
                arguments='{"x": 1, "y": 2}'
            )
        )
        response_message = SimpleNamespace(
            content="Creating a point...",
            tool_calls=[tool_call]
        )
        
        result = api._create_assistant_message(response_message)
        
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["id"], "call_123")
        self.assertEqual(result["tool_calls"][0]["type"], "function")
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "create_point")

    @patch('static.openai_api_base.OpenAI')
    def test_accumulate_tool_calls(self, mock_openai: Mock) -> None:
        """Test _accumulate_tool_calls accumulates streaming deltas."""
        api = OpenAIChatCompletionsAPI()
        accumulator: Dict[int, Dict[str, Any]] = {}
        
        # First delta with id and function name
        delta1 = SimpleNamespace(
            index=0,
            id="call_123",
            function=SimpleNamespace(name="create_point", arguments='{"x":')
        )
        api._accumulate_tool_calls([delta1], accumulator)
        
        # Second delta with more arguments
        delta2 = SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(name=None, arguments=' 1}')
        )
        api._accumulate_tool_calls([delta2], accumulator)
        
        self.assertEqual(accumulator[0]["id"], "call_123")
        self.assertEqual(accumulator[0]["function"]["name"], "create_point")
        self.assertEqual(accumulator[0]["function"]["arguments"], '{"x": 1}')

    @patch('static.openai_api_base.OpenAI')
    def test_normalize_tool_calls(self, mock_openai: Mock) -> None:
        """Test _normalize_tool_calls converts accumulator to list."""
        api = OpenAIChatCompletionsAPI()
        accumulator = {
            0: {"id": "call_1", "function": {"name": "func1", "arguments": "{}"}},
            1: {"id": "call_2", "function": {"name": "func2", "arguments": "{}"}}
        }
        
        result = api._normalize_tool_calls(accumulator)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "call_1")
        self.assertEqual(result[1]["id"], "call_2")

    @patch('static.openai_api_base.OpenAI')
    def test_prepare_tool_calls_for_response(self, mock_openai: Mock) -> None:
        """Test _prepare_tool_calls_for_response formats for JSON response."""
        api = OpenAIChatCompletionsAPI()
        tool_calls = [
            {"id": "call_1", "function": {"name": "create_point", "arguments": '{"x": 1, "y": 2}'}}
        ]
        
        result = api._prepare_tool_calls_for_response(tool_calls)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function_name"], "create_point")
        self.assertEqual(result[0]["arguments"], {"x": 1, "y": 2})

    @patch('static.openai_api_base.OpenAI')
    def test_prepare_tool_calls_invalid_json(self, mock_openai: Mock) -> None:
        """Test _prepare_tool_calls_for_response handles invalid JSON arguments."""
        api = OpenAIChatCompletionsAPI()
        tool_calls = [
            {"id": "call_1", "function": {"name": "test", "arguments": "invalid json"}}
        ]
        
        result = api._prepare_tool_calls_for_response(tool_calls)
        
        self.assertEqual(result[0]["function_name"], "test")
        self.assertEqual(result[0]["arguments"], {})

    @patch('static.openai_api_base.OpenAI')
    def test_create_chat_completion_success(self, mock_openai: Mock) -> None:
        """Test create_chat_completion with successful response."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create mock response
        mock_message = SimpleNamespace(
            content="Test response",
            tool_calls=None
        )
        mock_choice = SimpleNamespace(
            message=mock_message,
            finish_reason="stop"
        )
        mock_response = SimpleNamespace(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response
        
        api = OpenAIChatCompletionsAPI()
        prompt = json.dumps({"user_message": "Hello", "use_vision": False})
        result = api.create_chat_completion(prompt)
        
        self.assertEqual(result.message.content, "Test response")
        self.assertEqual(result.finish_reason, "stop")

    @patch('static.openai_api_base.OpenAI')
    def test_create_chat_completion_error(self, mock_openai: Mock) -> None:
        """Test create_chat_completion handles API errors."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        api = OpenAIChatCompletionsAPI()
        prompt = json.dumps({"user_message": "Hello", "use_vision": False})
        result = api.create_chat_completion(prompt)
        
        self.assertIn("error", result.message.content.lower())
        self.assertEqual(result.finish_reason, "error")

    @patch('static.openai_api_base.OpenAI')
    def test_create_chat_completion_stream_tokens(self, mock_openai: Mock) -> None:
        """Test create_chat_completion_stream yields token events."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create mock stream chunks
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="Hello", tool_calls=None),
                finish_reason=None
            )]),
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=" world", tool_calls=None),
                finish_reason=None
            )]),
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="!", tool_calls=None),
                finish_reason="stop"
            )]),
        ]
        mock_client.chat.completions.create.return_value = iter(chunks)
        
        api = OpenAIChatCompletionsAPI()
        prompt = json.dumps({"user_message": "Hi", "use_vision": False})
        
        events = list(api.create_chat_completion_stream(prompt))
        
        # Should have token events + final event
        token_events = [e for e in events if e.get("type") == "token"]
        final_events = [e for e in events if e.get("type") == "final"]
        
        self.assertEqual(len(token_events), 3)
        self.assertEqual(token_events[0]["text"], "Hello")
        self.assertEqual(token_events[1]["text"], " world")
        self.assertEqual(token_events[2]["text"], "!")
        
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0]["ai_message"], "Hello world!")
        self.assertEqual(final_events[0]["finish_reason"], "stop")

    @patch('static.openai_api_base.OpenAI')
    def test_create_chat_completion_stream_error(self, mock_openai: Mock) -> None:
        """Test create_chat_completion_stream handles errors."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Stream error")
        
        api = OpenAIChatCompletionsAPI()
        prompt = json.dumps({"user_message": "Hi", "use_vision": False})
        
        events = list(api.create_chat_completion_stream(prompt))
        
        final_events = [e for e in events if e.get("type") == "final"]
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0]["finish_reason"], "error")


class TestOpenAIChatCompletionsAPIIntegration(unittest.TestCase):
    """Integration tests that actually call the OpenAI API.
    
    These tests require a valid OPENAI_API_KEY environment variable.
    They are skipped if the key is not available.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Check if API key is available for integration tests."""
        cls.api_key = os.environ.get('OPENAI_API_KEY')
        if not cls.api_key or cls.api_key == 'test-api-key':
            cls.skip_integration = True
        else:
            cls.skip_integration = False

    def setUp(self) -> None:
        """Set up for integration tests."""
        if self.skip_integration:
            self.skipTest("OPENAI_API_KEY not available for integration tests")

    def test_integration_simple_completion(self) -> None:
        """Test actual API call with simple prompt (minimal tokens)."""
        api = OpenAIChatCompletionsAPI()
        # Use cheapest model, limit output tokens
        api.set_model("gpt-4o-mini")
        api.max_tokens = 10
        
        prompt = json.dumps({
            "user_message": "Say: OK",
            "use_vision": False
        })
        
        result = api.create_chat_completion(prompt)
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.message.content)

    def test_integration_stream_completion(self) -> None:
        """Test actual streaming API call (minimal tokens)."""
        api = OpenAIChatCompletionsAPI()
        api.set_model("gpt-4o-mini")
        api.max_tokens = 10
        
        prompt = json.dumps({
            "user_message": "Say: HI",
            "use_vision": False
        })
        
        events = list(api.create_chat_completion_stream(prompt))
        
        # Should have at least one token and one final event
        token_events = [e for e in events if e.get("type") == "token"]
        final_events = [e for e in events if e.get("type") == "final"]
        
        self.assertGreater(len(token_events), 0)
        self.assertEqual(len(final_events), 1)

    def test_integration_response_format(self) -> None:
        """Test that API response has correct format (minimal tokens)."""
        api = OpenAIChatCompletionsAPI()
        api.set_model("gpt-4o-mini")
        api.max_tokens = 5
        
        prompt = json.dumps({
            "user_message": "1",
            "use_vision": False
        })
        
        events = list(api.create_chat_completion_stream(prompt))
        final_event = [e for e in events if e.get("type") == "final"][0]
        
        # Verify response format
        self.assertIn("type", final_event)
        self.assertEqual(final_event["type"], "final")
        self.assertIn("ai_message", final_event)
        self.assertIn("ai_tool_calls", final_event)
        self.assertIn("finish_reason", final_event)
        self.assertIsInstance(final_event["ai_message"], str)
        self.assertIsInstance(final_event["ai_tool_calls"], list)


if __name__ == '__main__':
    unittest.main()

