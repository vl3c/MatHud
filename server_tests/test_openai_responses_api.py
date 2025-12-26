"""
Tests for the OpenAI Responses API module.

Tests Responses API implementation for reasoning models including streaming,
reasoning tokens, tool calls, and integration tests with actual API calls.
"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import Mock, patch, MagicMock

from static.openai_responses_api import OpenAIResponsesAPI


class TestOpenAIResponsesAPI(unittest.TestCase):
    """Unit tests for OpenAIResponsesAPI class."""

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
        api = OpenAIResponsesAPI()
        self.assertIsNotNone(api.client)
        self.assertIsNotNone(api.model)

    @patch('static.openai_api_base.OpenAI')
    def test_convert_messages_to_input_developer_to_system(self, mock_openai: Mock) -> None:
        """Test _convert_messages_to_input converts developer role to system."""
        api = OpenAIResponsesAPI()
        # API initializes with a developer message
        
        result = api._convert_messages_to_input()
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "system")  # developer -> system

    @patch('static.openai_api_base.OpenAI')
    def test_convert_messages_to_input_converts_tool_calls_to_text(self, mock_openai: Mock) -> None:
        """Test _convert_messages_to_input converts tool call/result pairs to text messages."""
        api = OpenAIResponsesAPI()
        api.messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "create_point", "arguments": '{"x": 5}'}}]
        })
        api.messages.append({
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Point created at (5, 0)"
        })
        
        result = api._convert_messages_to_input()
        
        # Should have developer/system message + assistant description + user results
        self.assertEqual(len(result), 3)
        
        # Find assistant message describing the call
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        self.assertEqual(len(assistant_msgs), 1)
        self.assertIn("create_point", assistant_msgs[0]["content"])
        
        # Find user message with results
        user_msgs = [m for m in result if m["role"] == "user"]
        self.assertEqual(len(user_msgs), 1)
        self.assertIn("Point created", user_msgs[0]["content"])

    @patch('static.openai_api_base.OpenAI')
    def test_convert_messages_skips_pending_tool_calls(self, mock_openai: Mock) -> None:
        """Test _convert_messages_to_input skips assistant messages with pending tool calls."""
        api = OpenAIResponsesAPI()
        api.messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]
        })
        # No tool result message - pending
        
        result = api._convert_messages_to_input()
        
        # Should only have the initial developer/system message
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "system")

    @patch('static.openai_api_base.OpenAI')
    def test_convert_tools_for_responses_api(self, mock_openai: Mock) -> None:
        """Test _convert_tools_for_responses_api flattens tool format."""
        api = OpenAIResponsesAPI()
        # Tools are in Chat Completions format (nested function)
        api.tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_point",
                    "description": "Create a point",
                    "parameters": {"type": "object", "properties": {"x": {"type": "number"}}}
                }
            }
        ]
        
        result = api._convert_tools_for_responses_api()
        
        # Should be flattened for Responses API
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "function")
        self.assertEqual(result[0]["name"], "create_point")
        self.assertEqual(result[0]["description"], "Create a point")
        self.assertIn("parameters", result[0])
        # Should NOT have nested "function" key
        self.assertNotIn("function", result[0])

    @patch('static.openai_api_base.OpenAI')
    def test_handle_function_call_delta(self, mock_openai: Mock) -> None:
        """Test _handle_function_call_delta accumulates function call data."""
        api = OpenAIResponsesAPI()
        accumulator: Dict[int, Dict[str, Any]] = {}
        
        # First event with call_id and name
        event1 = SimpleNamespace(
            output_index=0,
            call_id="call_123",
            name="create_point",
            delta='{"x":'
        )
        api._handle_function_call_delta(event1, accumulator)
        
        # Second event with more arguments
        event2 = SimpleNamespace(
            output_index=0,
            call_id=None,
            name=None,
            delta=' 5}'
        )
        api._handle_function_call_delta(event2, accumulator)
        
        self.assertEqual(accumulator[0]["id"], "call_123")
        self.assertEqual(accumulator[0]["function"]["name"], "create_point")
        self.assertEqual(accumulator[0]["function"]["arguments"], '{"x": 5}')

    @patch('static.openai_api_base.OpenAI')
    def test_extract_tool_calls(self, mock_openai: Mock) -> None:
        """Test _extract_tool_calls extracts function calls from response."""
        api = OpenAIResponsesAPI()
        accumulator: Dict[int, Dict[str, Any]] = {}
        
        # Mock response with function_call output
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_456",
                    name="create_circle",
                    arguments='{"radius": 10}'
                )
            ]
        )
        
        api._extract_tool_calls(response, accumulator)
        
        self.assertEqual(accumulator[0]["id"], "call_456")
        self.assertEqual(accumulator[0]["function"]["name"], "create_circle")
        self.assertEqual(accumulator[0]["function"]["arguments"], '{"radius": 10}')

    @patch('static.openai_api_base.OpenAI')
    def test_normalize_tool_calls(self, mock_openai: Mock) -> None:
        """Test _normalize_tool_calls converts accumulator to sorted list."""
        api = OpenAIResponsesAPI()
        accumulator = {
            1: {"id": "call_2", "function": {"name": "func2", "arguments": "{}"}},
            0: {"id": "call_1", "function": {"name": "func1", "arguments": "{}"}}
        }
        
        result = api._normalize_tool_calls(accumulator)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "call_1")  # Index 0 first
        self.assertEqual(result[1]["id"], "call_2")  # Index 1 second

    @patch('static.openai_api_base.OpenAI')
    def test_create_assistant_message(self, mock_openai: Mock) -> None:
        """Test _create_assistant_message creates correct format."""
        api = OpenAIResponsesAPI()
        
        tool_call = SimpleNamespace(
            id="call_123",
            function=SimpleNamespace(
                name="test_func",
                arguments='{"arg": "value"}'
            )
        )
        response_message = SimpleNamespace(
            content="Test content",
            tool_calls=[tool_call]
        )
        
        result = api._create_assistant_message(response_message)
        
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"], "Test content")
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["id"], "call_123")
        self.assertEqual(result["tool_calls"][0]["type"], "function")

    @patch('static.openai_api_base.OpenAI')
    def test_prepare_tool_calls_for_response(self, mock_openai: Mock) -> None:
        """Test _prepare_tool_calls_for_response formats for JSON response."""
        api = OpenAIResponsesAPI()
        tool_calls = [
            {"id": "call_1", "function": {"name": "create_point", "arguments": '{"x": 1, "y": 2}'}}
        ]
        
        result = api._prepare_tool_calls_for_response(tool_calls)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function_name"], "create_point")
        self.assertEqual(result[0]["arguments"], {"x": 1, "y": 2})

    @patch('static.openai_api_base.OpenAI')
    def test_prepare_tool_calls_invalid_json(self, mock_openai: Mock) -> None:
        """Test _prepare_tool_calls_for_response handles invalid JSON."""
        api = OpenAIResponsesAPI()
        tool_calls = [
            {"id": "call_1", "function": {"name": "test", "arguments": "not json"}}
        ]
        
        result = api._prepare_tool_calls_for_response(tool_calls)
        
        self.assertEqual(result[0]["function_name"], "test")
        self.assertEqual(result[0]["arguments"], {})

    @patch('static.openai_api_base.OpenAI')
    def test_create_response_stream_reasoning_events(self, mock_openai: Mock) -> None:
        """Test create_response_stream yields reasoning events."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create mock stream with reasoning events
        events = [
            SimpleNamespace(type="response.reasoning_text.delta", delta="Thinking..."),
            SimpleNamespace(type="response.reasoning_text.delta", delta=" about this"),
            SimpleNamespace(type="response.output_text.delta", delta="Here's the answer"),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(status="stop", output=[]))
        ]
        mock_client.responses.create.return_value = iter(events)
        
        api = OpenAIResponsesAPI()
        prompt = json.dumps({"user_message": "Test", "use_vision": False})
        
        result_events = list(api.create_response_stream(prompt))
        
        # Should have reasoning events, token events, and final
        reasoning_events = [e for e in result_events if e.get("type") == "reasoning"]
        token_events = [e for e in result_events if e.get("type") == "token"]
        final_events = [e for e in result_events if e.get("type") == "final"]
        
        self.assertEqual(len(reasoning_events), 2)
        self.assertEqual(reasoning_events[0]["text"], "Thinking...")
        self.assertEqual(reasoning_events[1]["text"], " about this")
        
        self.assertEqual(len(token_events), 1)
        self.assertEqual(token_events[0]["text"], "Here's the answer")
        
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0]["ai_message"], "Here's the answer")

    @patch('static.openai_api_base.OpenAI')
    def test_create_response_stream_error(self, mock_openai: Mock) -> None:
        """Test create_response_stream handles errors gracefully."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.responses.create.side_effect = Exception("API Error")
        
        api = OpenAIResponsesAPI()
        prompt = json.dumps({"user_message": "Test", "use_vision": False})
        
        result_events = list(api.create_response_stream(prompt))
        
        final_events = [e for e in result_events if e.get("type") == "final"]
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0]["finish_reason"], "error")

    @patch('static.openai_api_base.OpenAI')
    def test_create_response_stream_normalizes_completed_status(self, mock_openai: Mock) -> None:
        """Test that 'completed' status is normalized to 'stop' finish_reason."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="Done"),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed", output=[]))
        ]
        mock_client.responses.create.return_value = iter(events)
        
        api = OpenAIResponsesAPI()
        prompt = json.dumps({"user_message": "Test", "use_vision": False})
        
        result_events = list(api.create_response_stream(prompt))
        
        final_event = [e for e in result_events if e.get("type") == "final"][0]
        self.assertEqual(final_event["finish_reason"], "stop")

    @patch('static.openai_api_base.OpenAI')
    def test_create_response_stream_sets_tool_calls_finish_reason(self, mock_openai: Mock) -> None:
        """Test that finish_reason is 'tool_calls' when there are tool calls."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        events = [
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                output_index=0,
                call_id="call_123",
                name="create_point",
                delta='{"x": 5}'
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    status="requires_action",
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            call_id="call_123",
                            name="create_point",
                            arguments='{"x": 5}'
                        )
                    ]
                )
            )
        ]
        mock_client.responses.create.return_value = iter(events)
        
        api = OpenAIResponsesAPI()
        prompt = json.dumps({"user_message": "Create point", "use_vision": False})
        
        result_events = list(api.create_response_stream(prompt))
        
        final_event = [e for e in result_events if e.get("type") == "final"][0]
        self.assertEqual(final_event["finish_reason"], "tool_calls")
        self.assertEqual(len(final_event["ai_tool_calls"]), 1)

    @patch('static.openai_api_base.OpenAI')
    def test_create_response_stream_reasoning_placeholder_sent_once(self, mock_openai: Mock) -> None:
        """Test that reasoning placeholder is only sent once per stream."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Multiple reasoning items without summaries
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                output_index=0,
                item=SimpleNamespace(type="reasoning", summary=None)
            ),
            SimpleNamespace(
                type="response.output_item.added",
                output_index=1,
                item=SimpleNamespace(type="reasoning", summary=None)
            ),
            SimpleNamespace(
                type="response.output_item.added",
                output_index=2,
                item=SimpleNamespace(type="reasoning", summary=None)
            ),
            SimpleNamespace(type="response.output_text.delta", delta="Answer"),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed", output=[]))
        ]
        mock_client.responses.create.return_value = iter(events)
        
        api = OpenAIResponsesAPI()
        prompt = json.dumps({"user_message": "Test", "use_vision": False})
        
        result_events = list(api.create_response_stream(prompt))
        
        # Count reasoning events with placeholder text
        placeholder_events = [
            e for e in result_events 
            if e.get("type") == "reasoning" and "Reasoning in progress" in e.get("text", "")
        ]
        
        # Should only have ONE placeholder despite multiple reasoning items
        self.assertEqual(len(placeholder_events), 1)

    @patch('static.openai_api_base.OpenAI')
    def test_create_response_stream_fallback_without_reasoning_summary(self, mock_openai: Mock) -> None:
        """Test that API falls back gracefully when reasoning summary not supported."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # First call fails with reasoning.summary error, second succeeds
        def create_side_effect(*args, **kwargs):
            if kwargs.get("reasoning"):
                raise Exception("reasoning.summary is not supported")
            return iter([
                SimpleNamespace(type="response.output_text.delta", delta="Hello"),
                SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed", output=[]))
            ])
        
        mock_client.responses.create.side_effect = create_side_effect
        
        api = OpenAIResponsesAPI()
        prompt = json.dumps({"user_message": "Test", "use_vision": False})
        
        result_events = list(api.create_response_stream(prompt))
        
        # Should succeed with fallback
        final_events = [e for e in result_events if e.get("type") == "final"]
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0]["ai_message"], "Hello")
        self.assertNotEqual(final_events[0]["finish_reason"], "error")

    @patch('static.openai_api_base.OpenAI')
    def test_create_response_stream_yields_reasoning_summaries(self, mock_openai: Mock) -> None:
        """Test that reasoning summaries are properly yielded when available."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Reasoning item with summary
        summary_item = SimpleNamespace(text="I need to analyze this problem")
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                output_index=0,
                item=SimpleNamespace(type="reasoning", summary=[summary_item])
            ),
            SimpleNamespace(type="response.output_text.delta", delta="The answer is 42"),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed", output=[]))
        ]
        mock_client.responses.create.return_value = iter(events)
        
        api = OpenAIResponsesAPI()
        prompt = json.dumps({"user_message": "Test", "use_vision": False})
        
        result_events = list(api.create_response_stream(prompt))
        
        # Should have reasoning event with the summary text
        reasoning_events = [e for e in result_events if e.get("type") == "reasoning"]
        self.assertEqual(len(reasoning_events), 1)
        self.assertIn("analyze this problem", reasoning_events[0]["text"])

    @patch('static.openai_api_base.OpenAI')
    def test_create_response_stream_multiple_tool_calls(self, mock_openai: Mock) -> None:
        """Test handling multiple tool calls in a single response."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        events = [
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                output_index=0,
                call_id="call_1",
                name="create_point",
                delta='{"x": 1}'
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                output_index=1,
                call_id="call_2",
                name="create_circle",
                delta='{"r": 5}'
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    status="requires_action",
                    output=[
                        SimpleNamespace(type="function_call", call_id="call_1", name="create_point", arguments='{"x": 1}'),
                        SimpleNamespace(type="function_call", call_id="call_2", name="create_circle", arguments='{"r": 5}')
                    ]
                )
            )
        ]
        mock_client.responses.create.return_value = iter(events)
        
        api = OpenAIResponsesAPI()
        prompt = json.dumps({"user_message": "Create shapes", "use_vision": False})
        
        result_events = list(api.create_response_stream(prompt))
        
        final_event = [e for e in result_events if e.get("type") == "final"][0]
        self.assertEqual(len(final_event["ai_tool_calls"]), 2)
        self.assertEqual(final_event["ai_tool_calls"][0]["function_name"], "create_point")
        self.assertEqual(final_event["ai_tool_calls"][1]["function_name"], "create_circle")

    @patch('static.openai_api_base.OpenAI')
    def test_create_response_stream_with_tool_calls(self, mock_openai: Mock) -> None:
        """Test create_response_stream handles function calls."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create mock stream with function call events
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="Creating point..."),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                output_index=0,
                call_id="call_123",
                name="create_point",
                delta='{"x": 5, "y": 10}'
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    status="tool_calls",
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            call_id="call_123",
                            name="create_point",
                            arguments='{"x": 5, "y": 10}'
                        )
                    ]
                )
            )
        ]
        mock_client.responses.create.return_value = iter(events)
        
        api = OpenAIResponsesAPI()
        prompt = json.dumps({"user_message": "Create a point", "use_vision": False})
        
        result_events = list(api.create_response_stream(prompt))
        
        final_event = [e for e in result_events if e.get("type") == "final"][0]
        
        self.assertEqual(len(final_event["ai_tool_calls"]), 1)
        self.assertEqual(final_event["ai_tool_calls"][0]["function_name"], "create_point")
        self.assertEqual(final_event["ai_tool_calls"][0]["arguments"], {"x": 5, "y": 10})


class TestOpenAIResponsesAPIIntegration(unittest.TestCase):
    """Integration tests that actually call the OpenAI Responses API.
    
    These tests require a valid OPENAI_API_KEY environment variable
    and use reasoning models (GPT-5, o3, o4-mini).
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

    def test_integration_response_stream_format(self) -> None:
        """Test actual Responses API call returns correct format (minimal tokens)."""
        api = OpenAIResponsesAPI()
        # Use o4-mini as it's a reasoning model, limit tokens
        api.set_model("o4-mini")
        api.max_tokens = 20
        
        prompt = json.dumps({
            "user_message": "Say: OK",
            "use_vision": False
        })
        
        events = list(api.create_response_stream(prompt))
        
        # Should have final event with correct structure
        final_events = [e for e in events if e.get("type") == "final"]
        self.assertEqual(len(final_events), 1)
        
        final = final_events[0]
        self.assertIn("type", final)
        self.assertEqual(final["type"], "final")
        self.assertIn("ai_message", final)
        self.assertIn("ai_tool_calls", final)
        self.assertIn("finish_reason", final)
        self.assertIsInstance(final["ai_message"], str)
        self.assertIsInstance(final["ai_tool_calls"], list)

    def test_integration_reasoning_tokens(self) -> None:
        """Test that reasoning models stream reasoning tokens (minimal tokens)."""
        api = OpenAIResponsesAPI()
        api.set_model("o4-mini")
        api.max_tokens = 30
        
        prompt = json.dumps({
            "user_message": "2+2=?",
            "use_vision": False
        })
        
        events = list(api.create_response_stream(prompt))
        
        # Check for reasoning events (may or may not be present depending on model)
        token_events = [e for e in events if e.get("type") == "token"]
        final_events = [e for e in events if e.get("type") == "final"]
        
        # Must have final event
        self.assertEqual(len(final_events), 1)
        
        # Should have some response content
        total_content = "".join(
            [e.get("text", "") for e in token_events] +
            [final_events[0].get("ai_message", "")]
        )
        self.assertGreater(len(total_content), 0)

    def test_integration_event_types_are_valid(self) -> None:
        """Test that all streamed events have valid types (minimal tokens)."""
        api = OpenAIResponsesAPI()
        api.set_model("o4-mini")
        api.max_tokens = 10
        
        prompt = json.dumps({
            "user_message": "1",
            "use_vision": False
        })
        
        events = list(api.create_response_stream(prompt))
        
        valid_types = {"reasoning", "token", "final"}
        for event in events:
            self.assertIn("type", event)
            self.assertIn(event["type"], valid_types)


class TestOpenAIResponsesAPIModelRouting(unittest.TestCase):
    """Test that models are correctly identified for API routing."""

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
    def test_set_model_to_reasoning(self, mock_openai: Mock) -> None:
        """Test setting model to a reasoning model."""
        api = OpenAIResponsesAPI()
        api.set_model("o3")
        
        self.assertEqual(api.model.id, "o3")
        self.assertTrue(api.model.is_reasoning_model)

    @patch('static.openai_api_base.OpenAI')
    def test_set_model_to_o4_mini(self, mock_openai: Mock) -> None:
        """Test setting model to o4-mini."""
        api = OpenAIResponsesAPI()
        api.set_model("o4-mini")
        
        self.assertEqual(api.model.id, "o4-mini")
        self.assertTrue(api.model.is_reasoning_model)
        self.assertTrue(api.model.has_vision)

    @patch('static.openai_api_base.OpenAI')
    def test_set_model_to_gpt5(self, mock_openai: Mock) -> None:
        """Test setting model to GPT-5-chat-latest."""
        api = OpenAIResponsesAPI()
        api.set_model("gpt-5-chat-latest")
        
        self.assertEqual(api.model.id, "gpt-5-chat-latest")
        self.assertTrue(api.model.is_reasoning_model)
        self.assertTrue(api.model.has_vision)

    @patch('static.openai_api_base.OpenAI')
    def test_set_model_to_gpt52_chat_latest(self, mock_openai: Mock) -> None:
        """Test setting model to GPT-5.2-chat-latest."""
        api = OpenAIResponsesAPI()
        api.set_model("gpt-5.2-chat-latest")

        self.assertEqual(api.model.id, "gpt-5.2-chat-latest")
        self.assertTrue(api.model.is_reasoning_model)
        self.assertTrue(api.model.has_vision)

    @patch('static.openai_api_base.OpenAI')
    def test_set_model_to_gpt52_medium_reasoning(self, mock_openai: Mock) -> None:
        """Test setting model to GPT-5.2 with medium reasoning effort."""
        api = OpenAIResponsesAPI()
        api.set_model("gpt-5.2")

        self.assertEqual(api.model.id, "gpt-5.2")
        self.assertTrue(api.model.is_reasoning_model)
        self.assertTrue(api.model.has_vision)
        self.assertEqual(api.model.reasoning_effort, "medium")


if __name__ == '__main__':
    unittest.main()

