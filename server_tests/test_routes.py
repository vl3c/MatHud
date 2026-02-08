from __future__ import annotations

import json
import os
import unittest
from typing import Any, Optional
from unittest.mock import Mock, patch

from static.app_manager import AppManager, MatHudFlask
from static.routes import CANVAS_SNAPSHOT_PATH, save_canvas_snapshot_from_data_url
from static.openai_completions_api import OpenAIChatCompletionsAPI
from static.openai_responses_api import OpenAIResponsesAPI


class TestRoutes(unittest.TestCase):
    SAMPLE_PNG_BASE64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/w8AAwMB/aqVw0sAAAAASUVORK5CYII="
    )

    def setUp(self) -> None:
        """Set up test client before each test."""
        # Set test environment variables to disable authentication
        self.original_require_auth: Optional[str] = os.environ.get('REQUIRE_AUTH')
        os.environ['REQUIRE_AUTH'] = 'false'

        self.app: MatHudFlask = AppManager.create_app()
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
        # Ensure webdriver_manager is None at start
        self.app.webdriver_manager = None
        self._remove_canvas_snapshot()

    def tearDown(self) -> None:
        """Clean up after each test."""
        # Clean up webdriver if it exists
        if hasattr(self.app, 'webdriver_manager') and self.app.webdriver_manager:
            self.app.webdriver_manager = None
        self._remove_canvas_snapshot()

        # Restore original REQUIRE_AUTH environment variable
        if self.original_require_auth is not None:
            os.environ['REQUIRE_AUTH'] = self.original_require_auth
        else:
            os.environ.pop('REQUIRE_AUTH', None)

    def _remove_canvas_snapshot(self) -> None:
        if os.path.exists(CANVAS_SNAPSHOT_PATH):
            try:
                os.remove(CANVAS_SNAPSHOT_PATH)
            except OSError:
                pass

    @patch('static.webdriver_manager.WebDriverManager')
    def test_init_webdriver_route(self, mock_webdriver_class: Mock) -> None:
        """Test the webdriver initialization route."""
        # Create a mock instance
        mock_instance = Mock()
        mock_webdriver_class.return_value = mock_instance

        response = self.client.get('/init_webdriver')
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['message'], 'WebDriver initialization successful')
        mock_webdriver_class.assert_called_once()
        self.assertEqual(self.app.webdriver_manager, mock_instance)

    def test_index_route(self) -> None:
        """Test the index route returns HTML."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.content_type)

    def test_workspace_operations(self) -> None:
        """Test workspace CRUD operations."""
        # Test creating a workspace
        test_state = {'test': 'data'}
        response = self.client.post('/save_workspace',
                                  json={'state': test_state, 'name': 'test_workspace'})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')

        # Test listing workspaces
        response = self.client.get('/list_workspaces')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertIn('test_workspace', data['data'])

        # Test loading workspace
        response = self.client.get('/load_workspace?name=test_workspace')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['state'], test_state)

        # Test deleting workspace
        response = self.client.get('/delete_workspace?name=test_workspace')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')

        # Verify workspace is deleted
        response = self.client.get('/list_workspaces')
        data = json.loads(response.data)
        self.assertNotIn('test_workspace', data['data'])

    @patch('static.openai_completions_api.OpenAIChatCompletionsAPI.create_chat_completion')
    def test_send_message(self, mock_chat: Mock) -> None:
        """Test the send_message route."""
        # Configure mock response
        class MockMessage:
            content = "Test response"
            tool_calls = None

        class MockResponse:
            message = MockMessage()
            finish_reason = "stop"

        mock_chat.return_value = MockResponse()

        test_message = {
            'message': json.dumps({
                'user_message': 'test message',
                'use_vision': False
            }),
            'svg_state': None
        }
        response = self.client.post('/send_message', json=test_message)
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertIn('ai_message', data['data'])
        self.assertIn('ai_tool_calls', data['data'])

    def test_save_canvas_snapshot_helper(self) -> None:
        data_url = f"data:image/png;base64,{self.SAMPLE_PNG_BASE64}"
        saved = save_canvas_snapshot_from_data_url(data_url)
        self.assertTrue(saved)
        self.assertTrue(os.path.exists(CANVAS_SNAPSHOT_PATH))
        self.assertGreater(os.path.getsize(CANVAS_SNAPSHOT_PATH), 0)

    @patch('static.openai_completions_api.OpenAIChatCompletionsAPI.create_chat_completion')
    def test_send_message_uses_canvas_snapshot(self, mock_chat: Mock) -> None:
        class MockMessage:
            content = "Test response with vision"
            tool_calls = None

        class MockResponse:
            message = MockMessage()
            finish_reason = "stop"

        mock_chat.return_value = MockResponse()

        canvas_data_url = f"data:image/png;base64,{self.SAMPLE_PNG_BASE64}"
        payload = {
            'message': json.dumps({
                'user_message': 'vision request',
                'use_vision': True
            }),
            'vision_snapshot': {
                'renderer_mode': 'canvas2d',
                'canvas_image': canvas_data_url
            }
        }

        response = self.client.post('/send_message', json=payload)
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertTrue(os.path.exists(CANVAS_SNAPSHOT_PATH))
        self.assertGreater(os.path.getsize(CANVAS_SNAPSHOT_PATH), 0)
        self.assertIsNone(self.app.webdriver_manager)

    def test_new_conversation_route(self) -> None:
        """Test the new_conversation route resets the AI conversation history."""
        # Simulate existing conversation history in both APIs (standard + reasoning).
        self.app.ai_api.messages.append({"role": "user", "content": "test message"})
        self.app.ai_api.messages.append({"role": "assistant", "content": "Test response"})
        self.app.responses_api.messages.append({"role": "user", "content": "test message"})
        self.app.responses_api.messages.append({"role": "assistant", "content": "Test response"})

        # Check that the conversation history has more than the initial developer message
        self.assertGreater(len(self.app.ai_api.messages), 1)
        self.assertGreater(len(self.app.responses_api.messages), 1)

        # Now, call the new_conversation route
        response = self.client.post('/new_conversation')
        data = json.loads(response.data)

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['message'], 'New conversation started.')

        # Check that the conversation history has been reset
        self.assertEqual(len(self.app.ai_api.messages), 1)
        self.assertEqual(self.app.ai_api.messages[0]["role"], "developer")
        self.assertEqual(len(self.app.responses_api.messages), 1)
        self.assertEqual(self.app.responses_api.messages[0]["role"], "developer")

    def test_error_handling(self) -> None:
        """Test error handling in routes."""
        # Test invalid workspace name
        response = self.client.get('/load_workspace?name=nonexistent')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(data['status'], 'error')

        # Test missing workspace name
        response = self.client.get('/delete_workspace')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['status'], 'error')

        # Test missing message
        response = self.client.post('/send_message', json={'invalid': 'format'})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)  # Changed from 500 to 400 for invalid request
        self.assertEqual(data['status'], 'error')
        self.assertIn('message', data['message'].lower())  # Error message should mention 'message'


class TestAPIRouting(unittest.TestCase):
    """Test API routing between Chat Completions and Responses APIs."""

    def setUp(self) -> None:
        """Set up test client before each test."""
        self.original_require_auth: Optional[str] = os.environ.get('REQUIRE_AUTH')
        os.environ['REQUIRE_AUTH'] = 'false'

        self.app: MatHudFlask = AppManager.create_app()
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True

    def tearDown(self) -> None:
        """Clean up after each test."""
        if self.original_require_auth is not None:
            os.environ['REQUIRE_AUTH'] = self.original_require_auth
        else:
            os.environ.pop('REQUIRE_AUTH', None)

    def test_app_has_both_apis(self) -> None:
        """Test that app manager initializes both API instances."""
        self.assertIsNotNone(self.app.ai_api)
        self.assertIsNotNone(self.app.responses_api)
        self.assertIsInstance(self.app.ai_api, OpenAIChatCompletionsAPI)
        self.assertIsInstance(self.app.responses_api, OpenAIResponsesAPI)

    def test_standard_model_uses_completions_api(self) -> None:
        """Test that standard models route to Chat Completions API."""
        # Set a standard model
        self.app.ai_api.set_model("gpt-4o-mini")

        # Check model is not a reasoning model
        model = self.app.ai_api.get_model()
        self.assertFalse(model.is_reasoning_model)

    def test_reasoning_model_identified(self) -> None:
        """Test that reasoning models are correctly identified."""
        # Set a reasoning model
        self.app.ai_api.set_model("o3")
        self.app.responses_api.set_model("o3")

        # Check both APIs have the model set
        self.assertEqual(self.app.ai_api.get_model().id, "o3")
        self.assertEqual(self.app.responses_api.get_model().id, "o3")
        self.assertTrue(self.app.ai_api.get_model().is_reasoning_model)

    @patch.object(OpenAIChatCompletionsAPI, 'create_chat_completion_stream')
    def test_stream_route_uses_completions_for_standard_model(self, mock_stream: Mock) -> None:
        """Test /send_message_stream uses Chat Completions for standard models."""
        # Configure mock to return an iterator
        mock_stream.return_value = iter([
            {"type": "token", "text": "Hello"},
            {"type": "final", "ai_message": "Hello", "ai_tool_calls": [], "finish_reason": "stop"}
        ])

        test_message = {
            'message': json.dumps({
                'user_message': 'test',
                'use_vision': False,
                'ai_model': 'gpt-4o-mini'  # Standard model
            }),
            'svg_state': None
        }

        self.client.post('/send_message_stream', json=test_message)

        # Check that Chat Completions API was called (not Responses API)
        mock_stream.assert_called_once()

    @patch.object(OpenAIResponsesAPI, 'create_response_stream')
    @patch.object(OpenAIChatCompletionsAPI, 'set_model')
    def test_stream_route_uses_responses_for_reasoning_model(self, mock_set_model: Mock, mock_stream: Mock) -> None:
        """Test /send_message_stream uses Responses API for reasoning models."""
        # Configure mock to return an iterator
        mock_stream.return_value = iter([
            {"type": "reasoning", "text": "Thinking..."},
            {"type": "token", "text": "Answer"},
            {"type": "final", "ai_message": "Answer", "ai_tool_calls": [], "finish_reason": "stop"}
        ])

        # Pre-set the model to a reasoning model
        self.app.ai_api.model = self.app.ai_api.model.from_identifier("o3")

        test_message = {
            'message': json.dumps({
                'user_message': 'test',
                'use_vision': False,
                'ai_model': 'o3'  # Reasoning model
            }),
            'svg_state': None
        }

        self.client.post('/send_message_stream', json=test_message)

        # Check that Responses API was called
        mock_stream.assert_called_once()

    def test_model_selector_options(self) -> None:
        """Test that all expected models are configured."""
        from static.ai_model import AIModel

        reasoning_models = ["gpt-5-chat-latest", "gpt-5.2-chat-latest", "gpt-5.2", "o3", "o4-mini"]
        standard_models = ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "gpt-5-nano", "gpt-3.5-turbo"]

        for model_id in reasoning_models:
            model = AIModel.from_identifier(model_id)
            self.assertTrue(model.is_reasoning_model, f"{model_id} should be reasoning")

        for model_id in standard_models:
            model = AIModel.from_identifier(model_id)
            self.assertFalse(model.is_reasoning_model, f"{model_id} should not be reasoning")


class TestStreamingResponseFormat(unittest.TestCase):
    """Test streaming response format for both APIs."""

    def setUp(self) -> None:
        """Set up test client before each test."""
        self.original_require_auth: Optional[str] = os.environ.get('REQUIRE_AUTH')
        os.environ['REQUIRE_AUTH'] = 'false'

        self.app: MatHudFlask = AppManager.create_app()
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True

    def tearDown(self) -> None:
        """Clean up after each test."""
        if self.original_require_auth is not None:
            os.environ['REQUIRE_AUTH'] = self.original_require_auth
        else:
            os.environ.pop('REQUIRE_AUTH', None)

    @patch.object(OpenAIChatCompletionsAPI, 'create_chat_completion_stream')
    def test_stream_response_ndjson_format(self, mock_stream: Mock) -> None:
        """Test streaming response is in NDJSON format."""
        mock_stream.return_value = iter([
            {"type": "token", "text": "Hello"},
            {"type": "final", "ai_message": "Hello", "ai_tool_calls": [], "finish_reason": "stop"}
        ])

        test_message = {
            'message': json.dumps({
                'user_message': 'test',
                'use_vision': False,
                'ai_model': 'gpt-4o-mini'
            }),
            'svg_state': None
        }

        response = self.client.post('/send_message_stream', json=test_message)

        # Check content type
        self.assertEqual(response.content_type, 'application/x-ndjson')

        # Parse NDJSON response
        lines = response.data.decode('utf-8').strip().split('\n')
        events = [json.loads(line) for line in lines if line.strip()]

        # Should have token and final events
        self.assertGreater(len(events), 0)

        # Last event should be final
        self.assertEqual(events[-1]["type"], "final")
        self.assertIn("ai_message", events[-1])
        self.assertIn("ai_tool_calls", events[-1])
        self.assertIn("finish_reason", events[-1])

    @patch.object(OpenAIResponsesAPI, 'create_response_stream')
    @patch.object(OpenAIChatCompletionsAPI, 'set_model')
    def test_reasoning_stream_includes_reasoning_events(self, mock_set_model: Mock, mock_stream: Mock) -> None:
        """Test reasoning model stream includes reasoning events."""
        mock_stream.return_value = iter([
            {"type": "reasoning", "text": "Let me think..."},
            {"type": "token", "text": "The answer is"},
            {"type": "final", "ai_message": "The answer is", "ai_tool_calls": [], "finish_reason": "stop"}
        ])

        # Pre-set the model to a reasoning model
        self.app.ai_api.model = self.app.ai_api.model.from_identifier("o3")

        test_message = {
            'message': json.dumps({
                'user_message': 'test',
                'use_vision': False,
                'ai_model': 'o3'
            }),
            'svg_state': None
        }

        response = self.client.post('/send_message_stream', json=test_message)

        # Parse NDJSON response
        lines = response.data.decode('utf-8').strip().split('\n')
        events = [json.loads(line) for line in lines if line.strip()]

        # Should have reasoning event
        reasoning_events = [e for e in events if e.get("type") == "reasoning"]
        self.assertGreater(len(reasoning_events), 0)
        self.assertEqual(reasoning_events[0]["text"], "Let me think...")


class TestInterceptSearchTools(unittest.TestCase):
    """Test the _intercept_search_tools function for filtering tool calls."""

    def setUp(self) -> None:
        """Set up test client before each test."""
        self.original_require_auth: Optional[str] = os.environ.get('REQUIRE_AUTH')
        os.environ['REQUIRE_AUTH'] = 'false'

        self.app: MatHudFlask = AppManager.create_app()
        self.app.config['TESTING'] = True

    def tearDown(self) -> None:
        """Clean up after each test."""
        if self.original_require_auth is not None:
            os.environ['REQUIRE_AUTH'] = self.original_require_auth
        else:
            os.environ.pop('REQUIRE_AUTH', None)

    def test_no_search_tools_returns_unchanged(self) -> None:
        """When no search_tools call, tool_calls should be returned unchanged."""
        from static.routes import _intercept_search_tools

        tool_calls = [
            {'function_name': 'create_circle', 'arguments': {'x': 0, 'y': 0}},
            {'function_name': 'create_point', 'arguments': {'x': 10, 'y': 20}},
        ]

        result = _intercept_search_tools(self.app, tool_calls)

        self.assertEqual(result, tool_calls)
        self.assertEqual(len(result), 2)

    def test_empty_tool_calls_returns_empty(self) -> None:
        """Empty tool_calls should return empty list."""
        from static.routes import _intercept_search_tools

        result = _intercept_search_tools(self.app, [])

        self.assertEqual(result, [])

    @patch('static.tool_search_service.ToolSearchService')
    def test_filters_disallowed_tools(self, mock_service_class: Mock) -> None:
        """Tools not in search results should be filtered out."""
        from static.routes import _intercept_search_tools

        # Mock search_tools to return only create_circle
        mock_service = Mock()
        mock_service.search_tools.return_value = [
            {'function': {'name': 'create_circle'}},
        ]
        mock_service_class.return_value = mock_service

        tool_calls = [
            {'function_name': 'search_tools', 'arguments': {'query': 'draw circle'}},
            {'function_name': 'create_circle', 'arguments': {'x': 0, 'y': 0}},
            {'function_name': 'delete_all', 'arguments': {}},  # Not in search results
        ]

        result = _intercept_search_tools(self.app, tool_calls)

        # search_tools and create_circle should be allowed (essentials include search_tools)
        # delete_all should be filtered out
        names = [c.get('function_name') for c in result]
        self.assertIn('search_tools', names)
        self.assertIn('create_circle', names)
        self.assertNotIn('delete_all', names)

    @patch('static.tool_search_service.ToolSearchService')
    def test_essential_tools_always_allowed(self, mock_service_class: Mock) -> None:
        """Essential tools should always be allowed even if not in search results."""
        from static.routes import _intercept_search_tools

        # Mock search_tools to return only create_circle (no essentials)
        mock_service = Mock()
        mock_service.search_tools.return_value = [
            {'function': {'name': 'create_circle'}},
        ]
        mock_service_class.return_value = mock_service

        tool_calls = [
            {'function_name': 'search_tools', 'arguments': {'query': 'draw'}},
            {'function_name': 'undo', 'arguments': {}},  # Essential tool
            {'function_name': 'create_circle', 'arguments': {}},
        ]

        result = _intercept_search_tools(self.app, tool_calls)

        names = [c.get('function_name') for c in result]
        self.assertIn('undo', names)  # Essential should be allowed
        self.assertIn('create_circle', names)

    def test_empty_query_returns_unchanged(self) -> None:
        """search_tools with empty query should return tool_calls unchanged."""
        from static.routes import _intercept_search_tools

        tool_calls = [
            {'function_name': 'search_tools', 'arguments': {'query': ''}},
            {'function_name': 'create_circle', 'arguments': {}},
        ]

        result = _intercept_search_tools(self.app, tool_calls)

        # Should return unchanged because query is empty
        self.assertEqual(result, tool_calls)

    @patch('static.tool_search_service.ToolSearchService')
    def test_service_error_returns_unchanged(self, mock_service_class: Mock) -> None:
        """On ToolSearchService error, original tool_calls should be returned."""
        from static.routes import _intercept_search_tools

        mock_service = Mock()
        mock_service.search_tools.side_effect = Exception("API Error")
        mock_service_class.return_value = mock_service

        tool_calls = [
            {'function_name': 'search_tools', 'arguments': {'query': 'draw'}},
            {'function_name': 'create_circle', 'arguments': {}},
            {'function_name': 'delete_all', 'arguments': {}},
        ]

        result = _intercept_search_tools(self.app, tool_calls)

        # On error, should return original calls unchanged
        self.assertEqual(result, tool_calls)

    @patch('static.tool_search_service.ToolSearchService')
    def test_handles_json_string_arguments(self, mock_service_class: Mock) -> None:
        """Should handle arguments as JSON string (from some API responses)."""
        from static.routes import _intercept_search_tools

        mock_service = Mock()
        mock_service.search_tools.return_value = [
            {'function': {'name': 'create_point'}},
        ]
        mock_service_class.return_value = mock_service

        tool_calls: list[dict[str, Any]] = [
            {
                'function_name': 'search_tools',
                'arguments': '{"query": "point", "max_results": 5}'  # JSON string
            },
            {'function_name': 'create_point', 'arguments': {}},
        ]

        result = _intercept_search_tools(self.app, tool_calls)

        names = [c.get('function_name') for c in result]
        self.assertIn('search_tools', names)
        self.assertIn('create_point', names)

    @patch('static.tool_search_service.ToolSearchService')
    def test_handles_alternative_function_key(self, mock_service_class: Mock) -> None:
        """Should handle tool calls with 'function' key instead of 'function_name'."""
        from static.routes import _intercept_search_tools

        mock_service = Mock()
        mock_service.search_tools.return_value = [
            {'function': {'name': 'create_circle'}},
        ]
        mock_service_class.return_value = mock_service

        # Some API responses use 'function' key with nested 'name'
        tool_calls = [
            {'function': {'name': 'search_tools'}, 'arguments': {'query': 'circle'}},
            {'function': {'name': 'create_circle'}, 'arguments': {}},
            {'function': {'name': 'delete_all'}, 'arguments': {}},
        ]

        result = _intercept_search_tools(self.app, tool_calls)

        # Extract names using the same logic as the function
        names = [
            c.get('function_name') or c.get('function', {}).get('name')
            for c in result
        ]
        self.assertIn('search_tools', names)
        self.assertIn('create_circle', names)
        self.assertNotIn('delete_all', names)

    @patch('static.tool_search_service.ToolSearchService')
    def test_injects_tools_into_both_apis(self, mock_service_class: Mock) -> None:
        """Should inject tools into both ai_api and responses_api."""
        from static.routes import _intercept_search_tools

        mock_service = Mock()
        returned_tools = [{'function': {'name': 'create_circle'}}]
        mock_service.search_tools.return_value = returned_tools
        mock_service_class.return_value = mock_service

        # Spy on inject_tools
        with patch.object(self.app.ai_api, 'inject_tools') as mock_ai_inject, \
             patch.object(self.app.responses_api, 'inject_tools') as mock_resp_inject:

            tool_calls = [
                {'function_name': 'search_tools', 'arguments': {'query': 'circle'}},
            ]

            _intercept_search_tools(self.app, tool_calls)

            mock_ai_inject.assert_called_once_with(returned_tools, include_essentials=True)
            mock_resp_inject.assert_called_once_with(returned_tools, include_essentials=True)


class TestSearchToolHelpers(unittest.TestCase):
    """Helper-level tests for search tool interception/injection parsing."""

    def test_extract_search_query_and_limit_handles_json_args(self) -> None:
        from static.routes import _extract_search_query_and_limit

        call = {"function_name": "search_tools", "arguments": '{"query":"area","max_results":3}'}
        query, limit = _extract_search_query_and_limit(call)
        self.assertEqual(query, "area")
        self.assertEqual(limit, 3)

    def test_extract_search_query_and_limit_defaults_on_invalid(self) -> None:
        from static.routes import _extract_search_query_and_limit

        call = {"function_name": "search_tools", "arguments": "not-json"}
        query, limit = _extract_search_query_and_limit(call)
        self.assertEqual(query, "")
        self.assertEqual(limit, 10)

    def test_extract_injectable_tools_returns_tools_payload(self) -> None:
        from static.routes import _extract_injectable_tools

        payload = json.dumps(
            {
                "tool_call_1": {
                    "query": "circle",
                    "tools": [{"function": {"name": "create_circle"}}],
                }
            }
        )
        tools = _extract_injectable_tools(payload)
        self.assertIsNotNone(tools)
        assert tools is not None
        self.assertEqual(tools[0]["function"]["name"], "create_circle")

    def test_extract_injectable_tools_ignores_non_search_payload(self) -> None:
        from static.routes import _extract_injectable_tools

        payload = json.dumps({"tool_call_1": {"value": 123}})
        tools = _extract_injectable_tools(payload)
        self.assertIsNone(tools)


if __name__ == '__main__':
    unittest.main()
