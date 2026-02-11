"""
Tests for the OpenAI API Base module.

Tests shared functionality used by both Chat Completions and Responses APIs.
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import Mock, patch

from static.openai_api_base import OpenAIAPIBase
from static.ai_model import AIModel


class TestOpenAIAPIBase(unittest.TestCase):
    """Test cases for OpenAIAPIBase class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Ensure OPENAI_API_KEY is set for tests
        self.original_api_key = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = 'test-api-key'
        self.original_summary_mode = os.environ.get('AI_CANVAS_SUMMARY_MODE')
        self.original_hybrid_max = os.environ.get('AI_CANVAS_HYBRID_FULL_MAX_BYTES')
        self.original_summary_telemetry = os.environ.get('AI_CANVAS_SUMMARY_TELEMETRY')
        os.environ.pop('AI_CANVAS_SUMMARY_MODE', None)
        os.environ.pop('AI_CANVAS_HYBRID_FULL_MAX_BYTES', None)
        os.environ.pop('AI_CANVAS_SUMMARY_TELEMETRY', None)

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.original_api_key:
            os.environ['OPENAI_API_KEY'] = self.original_api_key
        else:
            os.environ.pop('OPENAI_API_KEY', None)
        if self.original_summary_mode is None:
            os.environ.pop('AI_CANVAS_SUMMARY_MODE', None)
        else:
            os.environ['AI_CANVAS_SUMMARY_MODE'] = self.original_summary_mode
        if self.original_hybrid_max is None:
            os.environ.pop('AI_CANVAS_HYBRID_FULL_MAX_BYTES', None)
        else:
            os.environ['AI_CANVAS_HYBRID_FULL_MAX_BYTES'] = self.original_hybrid_max
        if self.original_summary_telemetry is None:
            os.environ.pop('AI_CANVAS_SUMMARY_TELEMETRY', None)
        else:
            os.environ['AI_CANVAS_SUMMARY_TELEMETRY'] = self.original_summary_telemetry

    @patch('static.openai_api_base.OpenAI')
    def test_initialization_default_model(self, mock_openai: Mock) -> None:
        """Test API initializes with default model."""
        api = OpenAIAPIBase()
        self.assertEqual(api.model.id, AIModel.DEFAULT_MODEL)
        self.assertEqual(api.temperature, 0.2)
        self.assertEqual(api.max_tokens, 16000)

    @patch('static.openai_api_base.OpenAI')
    def test_initialization_custom_model(self, mock_openai: Mock) -> None:
        """Test API initializes with custom model."""
        custom_model = AIModel.from_identifier("gpt-4o")
        api = OpenAIAPIBase(model=custom_model)
        self.assertEqual(api.model.id, "gpt-4o")

    @patch('static.openai_api_base.OpenAI')
    def test_initialization_messages(self, mock_openai: Mock) -> None:
        """Test API initializes with developer message."""
        api = OpenAIAPIBase()
        self.assertEqual(len(api.messages), 1)
        self.assertEqual(api.messages[0]["role"], "developer")
        self.assertIn("educational graphing calculator", api.messages[0]["content"])

    @patch('static.openai_api_base.OpenAI')
    def test_get_model(self, mock_openai: Mock) -> None:
        """Test get_model returns current model."""
        api = OpenAIAPIBase()
        model = api.get_model()
        self.assertEqual(model.id, AIModel.DEFAULT_MODEL)

    @patch('static.openai_api_base.OpenAI')
    def test_reset_conversation(self, mock_openai: Mock) -> None:
        """Test reset_conversation clears messages except developer message."""
        api = OpenAIAPIBase()
        # Add some messages
        api.messages.append({"role": "user", "content": "test"})
        api.messages.append({"role": "assistant", "content": "response"})
        self.assertEqual(len(api.messages), 3)

        # Reset
        api.reset_conversation()
        self.assertEqual(len(api.messages), 1)
        self.assertEqual(api.messages[0]["role"], "developer")

    @patch('static.openai_api_base.OpenAI')
    def test_set_model(self, mock_openai: Mock) -> None:
        """Test set_model changes the model."""
        api = OpenAIAPIBase()
        api.set_model("gpt-4o")
        self.assertEqual(api.model.id, "gpt-4o")
        self.assertFalse(api.model.is_reasoning_model)

    @patch('static.openai_api_base.OpenAI')
    def test_set_model_to_reasoning(self, mock_openai: Mock) -> None:
        """Test set_model to a reasoning model."""
        api = OpenAIAPIBase()
        api.set_model("o3")
        self.assertEqual(api.model.id, "o3")
        self.assertTrue(api.model.is_reasoning_model)

    @patch('static.openai_api_base.OpenAI')
    def test_set_model_same_model_no_change(self, mock_openai: Mock) -> None:
        """Test set_model with same model doesn't change."""
        api = OpenAIAPIBase()
        original_model = api.model
        api.set_model(AIModel.DEFAULT_MODEL)
        # Model should be the same instance (no change)
        self.assertEqual(api.model.id, original_model.id)

    @patch('static.openai_api_base.OpenAI')
    def test_create_tool_message(self, mock_openai: Mock) -> None:
        """Test _create_tool_message creates correct format."""
        api = OpenAIAPIBase()
        tool_msg = api._create_tool_message("call_123", "result content")
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "call_123")
        self.assertEqual(tool_msg["content"], "result content")

    @patch('static.openai_api_base.OpenAI')
    def test_append_tool_messages(self, mock_openai: Mock) -> None:
        """Test _append_tool_messages adds placeholder messages."""
        api = OpenAIAPIBase()
        initial_count = len(api.messages)

        # Create mock tool calls
        tool_calls = [
            Mock(id="call_1"),
            Mock(id="call_2"),
        ]
        api._append_tool_messages(tool_calls)

        self.assertEqual(len(api.messages), initial_count + 2)
        self.assertEqual(api.messages[-2]["role"], "tool")
        self.assertEqual(api.messages[-2]["tool_call_id"], "call_1")
        self.assertEqual(api.messages[-1]["tool_call_id"], "call_2")

    @patch('static.openai_api_base.OpenAI')
    def test_append_tool_messages_none(self, mock_openai: Mock) -> None:
        """Test _append_tool_messages with None does nothing."""
        api = OpenAIAPIBase()
        initial_count = len(api.messages)
        api._append_tool_messages(None)
        self.assertEqual(len(api.messages), initial_count)

    @patch('static.openai_api_base.OpenAI')
    def test_update_tool_messages_with_results(self, mock_openai: Mock) -> None:
        """Test _update_tool_messages_with_results updates last tool message."""
        api = OpenAIAPIBase()
        # Add a tool message
        api.messages.append({
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "Awaiting result..."
        })

        results = {"status": "success", "data": "test"}
        api._update_tool_messages_with_results(json.dumps(results))

        self.assertEqual(api.messages[-1]["content"], json.dumps(results))

    @patch('static.openai_api_base.OpenAI')
    def test_parse_prompt_json_valid(self, mock_openai: Mock) -> None:
        """Test _parse_prompt_json with valid JSON."""
        api = OpenAIAPIBase()
        prompt = json.dumps({"user_message": "test", "use_vision": False})
        result = api._parse_prompt_json(prompt)
        self.assertIsNotNone(result)
        self.assertEqual(result["user_message"], "test")

    @patch('static.openai_api_base.OpenAI')
    def test_parse_prompt_json_invalid(self, mock_openai: Mock) -> None:
        """Test _parse_prompt_json with invalid JSON returns None."""
        api = OpenAIAPIBase()
        result = api._parse_prompt_json("not valid json")
        self.assertIsNone(result)

    @patch('static.openai_api_base.OpenAI')
    def test_parse_prompt_json_non_dict(self, mock_openai: Mock) -> None:
        """Test _parse_prompt_json with non-dict JSON returns None."""
        api = OpenAIAPIBase()
        result = api._parse_prompt_json(json.dumps(["list", "not", "dict"]))
        self.assertIsNone(result)

    @patch('static.openai_api_base.OpenAI')
    def test_prepare_message_content_no_vision(self, mock_openai: Mock) -> None:
        """Test _prepare_message_content without vision returns original prompt."""
        api = OpenAIAPIBase()
        os.environ['AI_CANVAS_SUMMARY_MODE'] = 'off'
        prompt = json.dumps({"user_message": "test", "use_vision": False})
        result = api._prepare_message_content(prompt)
        self.assertEqual(result, prompt)

    @patch('static.openai_api_base.OpenAI')
    def test_prepare_message_content_summary_only_removes_full_canvas_state(self, mock_openai: Mock) -> None:
        api = OpenAIAPIBase()
        os.environ['AI_CANVAS_SUMMARY_MODE'] = 'summary_only'
        prompt = json.dumps(
            {
                "user_message": "test",
                "use_vision": False,
                "canvas_state": {"Points": [{"name": "A", "args": {"position": {"x": 1, "y": 2}}}]},
            }
        )
        result = api._prepare_message_content(prompt)
        parsed = json.loads(result)

        self.assertNotIn("canvas_state", parsed)
        self.assertIn("canvas_state_summary", parsed)
        summary = parsed["canvas_state_summary"]
        self.assertEqual(summary["mode"], "summary_only")
        self.assertFalse(summary["includes_full_state"])
        self.assertIn("state", summary)
        self.assertIn("metrics", summary)

    @patch('static.openai_api_base.OpenAI')
    def test_prepare_message_content_hybrid_keeps_small_full_state(self, mock_openai: Mock) -> None:
        api = OpenAIAPIBase()
        os.environ['AI_CANVAS_SUMMARY_MODE'] = 'hybrid'
        os.environ['AI_CANVAS_HYBRID_FULL_MAX_BYTES'] = '999999'
        prompt = json.dumps(
            {
                "user_message": "test",
                "use_vision": False,
                "canvas_state": {"Points": [{"name": "A", "args": {"position": {"x": 1, "y": 2}}}]},
            }
        )
        result = api._prepare_message_content(prompt)
        parsed = json.loads(result)

        self.assertIn("canvas_state", parsed)
        self.assertIn("canvas_state_summary", parsed)
        summary = parsed["canvas_state_summary"]
        self.assertTrue(summary["includes_full_state"])
        self.assertNotIn("state", summary)
        self.assertIn("metrics", summary)

    @patch('static.openai_api_base.OpenAI')
    def test_prepare_message_content_hybrid_drops_large_full_state(self, mock_openai: Mock) -> None:
        api = OpenAIAPIBase()
        os.environ['AI_CANVAS_SUMMARY_MODE'] = 'hybrid'
        os.environ['AI_CANVAS_HYBRID_FULL_MAX_BYTES'] = '10'
        prompt = json.dumps(
            {
                "user_message": "test",
                "use_vision": False,
                "canvas_state": {"Points": [{"name": "A", "args": {"position": {"x": 1, "y": 2}}}]},
            }
        )
        result = api._prepare_message_content(prompt)
        parsed = json.loads(result)

        self.assertNotIn("canvas_state", parsed)
        self.assertIn("canvas_state_summary", parsed)
        self.assertFalse(parsed["canvas_state_summary"]["includes_full_state"])
        self.assertIn("state", parsed["canvas_state_summary"])

    @patch('static.openai_api_base.OpenAI')
    def test_prepare_message_content_default_mode_is_hybrid(self, mock_openai: Mock) -> None:
        api = OpenAIAPIBase()
        os.environ['AI_CANVAS_HYBRID_FULL_MAX_BYTES'] = '10'
        prompt = json.dumps(
            {
                "user_message": "test",
                "use_vision": False,
                "canvas_state": {"Points": [{"name": "A", "args": {"position": {"x": 1, "y": 2}}}]},
            }
        )
        result = api._prepare_message_content(prompt)
        parsed = json.loads(result)

        self.assertEqual(parsed["canvas_state_summary"]["mode"], "hybrid")
        self.assertNotIn("canvas_state", parsed)

    @patch('static.openai_api_base.OpenAI')
    def test_prepare_message_content_invalid_mode_falls_back_to_hybrid(self, mock_openai: Mock) -> None:
        api = OpenAIAPIBase()
        os.environ['AI_CANVAS_SUMMARY_MODE'] = 'garbage'
        os.environ['AI_CANVAS_HYBRID_FULL_MAX_BYTES'] = '10'
        prompt = json.dumps(
            {
                "user_message": "test",
                "use_vision": False,
                "canvas_state": {"Points": [{"name": "A", "args": {"position": {"x": 1, "y": 2}}}]},
            }
        )
        result = api._prepare_message_content(prompt)
        parsed = json.loads(result)

        self.assertEqual(parsed["canvas_state_summary"]["mode"], "hybrid")
        self.assertNotIn("canvas_state", parsed)

    @patch('static.openai_api_base.OpenAI')
    def test_prepare_message_content_invalid_json(self, mock_openai: Mock) -> None:
        """Test _prepare_message_content with invalid JSON returns original."""
        api = OpenAIAPIBase()
        prompt = "plain text prompt"
        result = api._prepare_message_content(prompt)
        self.assertEqual(result, prompt)

    @patch('static.openai_api_base.OpenAI')
    @patch('static.openai_api_base._logger')
    def test_prepare_message_content_emits_telemetry_when_enabled(self, mock_logger: Mock, mock_openai: Mock) -> None:
        api = OpenAIAPIBase()
        os.environ['AI_CANVAS_SUMMARY_MODE'] = 'summary_only'
        os.environ['AI_CANVAS_SUMMARY_TELEMETRY'] = '1'
        prompt = json.dumps(
            {
                "user_message": "test",
                "use_vision": False,
                "canvas_state": {"Points": [{"name": "A", "args": {"position": {"x": 1, "y": 2}}}]},
            }
        )

        result = api._prepare_message_content(prompt)
        parsed = json.loads(result)
        self.assertIn("canvas_state_summary", parsed)
        mock_logger.info.assert_called()

        args = mock_logger.info.call_args[0]
        self.assertIn("canvas_prompt_telemetry", args[0])
        payload = json.loads(args[1])
        self.assertEqual(payload["mode"], "summary_only")
        self.assertEqual(payload["prompt_kind"], "text")
        self.assertIn("normalize_elapsed_ms", payload)
        self.assertIn("input_bytes", payload)
        self.assertIn("normalized_prompt_bytes", payload)
        self.assertIn("output_payload_bytes", payload)

    @patch('static.openai_api_base.OpenAI')
    @patch('static.openai_api_base._logger')
    def test_prepare_message_content_skips_telemetry_when_disabled(self, mock_logger: Mock, mock_openai: Mock) -> None:
        api = OpenAIAPIBase()
        os.environ['AI_CANVAS_SUMMARY_MODE'] = 'summary_only'
        os.environ['AI_CANVAS_SUMMARY_TELEMETRY'] = '0'
        prompt = json.dumps(
            {
                "user_message": "test",
                "use_vision": False,
                "canvas_state": {"Points": [{"name": "A", "args": {"position": {"x": 1, "y": 2}}}]},
            }
        )
        _ = api._prepare_message_content(prompt)

        mock_logger.info.assert_not_called()

    @patch('static.openai_api_base.OpenAI')
    @patch('static.openai_api_base._logger')
    def test_prepare_message_content_telemetry_multimodal_reports_output_payload_size(self, mock_logger: Mock, mock_openai: Mock) -> None:
        api = OpenAIAPIBase()
        os.environ['AI_CANVAS_SUMMARY_MODE'] = 'summary_only'
        os.environ['AI_CANVAS_SUMMARY_TELEMETRY'] = '1'
        prompt = json.dumps(
            {
                "user_message": "test",
                "use_vision": False,
                "attached_images": ["data:image/png;base64,AAAA"],
                "canvas_state": {"Points": [{"name": "A", "args": {"position": {"x": 1, "y": 2}}}]},
            }
        )

        result = api._prepare_message_content(prompt)
        self.assertIsInstance(result, list)
        mock_logger.info.assert_called()

        args = mock_logger.info.call_args[0]
        payload = json.loads(args[1])
        self.assertEqual(payload["prompt_kind"], "multimodal")
        self.assertIn("normalized_prompt_bytes", payload)
        self.assertIn("output_payload_bytes", payload)

    @patch('static.openai_api_base.OpenAI')
    def test_create_error_response(self, mock_openai: Mock) -> None:
        """Test _create_error_response creates proper error structure."""
        api = OpenAIAPIBase()
        error_resp = api._create_error_response()
        self.assertEqual(error_resp.message.content, "I encountered an error processing your request. Please try again.")
        self.assertEqual(error_resp.message.tool_calls, [])
        self.assertEqual(error_resp.finish_reason, "error")

    @patch('static.openai_api_base.OpenAI')
    def test_create_error_response_custom_message(self, mock_openai: Mock) -> None:
        """Test _create_error_response with custom message."""
        api = OpenAIAPIBase()
        error_resp = api._create_error_response("Custom error message")
        self.assertEqual(error_resp.message.content, "Custom error message")

    @patch('static.openai_api_base.OpenAI')
    def test_remove_canvas_state_from_user_messages(self, mock_openai: Mock) -> None:
        """Test _remove_canvas_state_from_user_messages removes state payloads."""
        api = OpenAIAPIBase()
        api.messages.append({
            "role": "user",
            "content": json.dumps({
                "canvas_state": {"shapes": []},
                "canvas_state_summary": {"state": {"Points": []}},
                "user_message": "test"
            })
        })

        api._remove_canvas_state_from_user_messages()

        content = json.loads(api.messages[-1]["content"])
        self.assertNotIn("canvas_state", content)
        self.assertNotIn("canvas_state_summary", content)
        self.assertEqual(content["user_message"], "test")

    @patch('static.openai_api_base.OpenAI')
    def test_remove_images_from_user_messages(self, mock_openai: Mock) -> None:
        """Test _remove_images_from_user_messages removes image content."""
        api = OpenAIAPIBase()
        api.messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "test message"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]
        })

        api._remove_images_from_user_messages()

        # Content should now be just the text string
        self.assertEqual(api.messages[-1]["content"], "test message")


class TestOpenAIAPIBaseInitialization(unittest.TestCase):
    """Test API key initialization scenarios."""

    def test_api_key_from_environment(self) -> None:
        """Test API key is read from environment variable."""
        os.environ['OPENAI_API_KEY'] = 'test-env-key'
        try:
            api_key = OpenAIAPIBase._initialize_api_key()
            self.assertEqual(api_key, 'test-env-key')
        finally:
            os.environ.pop('OPENAI_API_KEY', None)

    @patch('static.openai_api_base.load_dotenv')
    @patch('static.openai_api_base.os.path.exists')
    def test_api_key_missing_returns_placeholder(self, mock_exists: Mock, mock_load_dotenv: Mock) -> None:
        """Test missing API key returns placeholder instead of crashing."""
        # Mock .env file doesn't exist
        mock_exists.return_value = False
        # Remove API key from environment
        original = os.environ.pop('OPENAI_API_KEY', None)
        try:
            result = OpenAIAPIBase._initialize_api_key()
            self.assertEqual(result, "not-configured")
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original


if __name__ == '__main__':
    unittest.main()
