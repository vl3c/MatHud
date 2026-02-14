from __future__ import annotations

import json
import os
import unittest
from typing import Optional
from unittest.mock import Mock, patch

from static.app_manager import AppManager, MatHudFlask
from static.openai_completions_api import OpenAIChatCompletionsAPI
from static.openai_responses_api import OpenAIResponsesAPI


class TestSearchToolWiringSmoke(unittest.TestCase):
    """Minimal route-level wiring checks for search_tools interception."""

    def setUp(self) -> None:
        self.original_require_auth: Optional[str] = os.environ.get("REQUIRE_AUTH")
        os.environ["REQUIRE_AUTH"] = "false"

        self.app: MatHudFlask = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self.original_require_auth is not None:
            os.environ["REQUIRE_AUTH"] = self.original_require_auth
        else:
            os.environ.pop("REQUIRE_AUTH", None)

    @patch("static.routes._intercept_search_tools")
    @patch.object(OpenAIChatCompletionsAPI, "create_chat_completion_stream")
    def test_streaming_path_calls_intercept(
        self,
        mock_stream: Mock,
        mock_intercept: Mock,
    ) -> None:
        filtered_calls = [{"function_name": "create_circle", "arguments": {"center_x": 0, "center_y": 0, "radius": 1}}]
        mock_intercept.return_value = filtered_calls

        mock_stream.return_value = iter(
            [
                {
                    "type": "final",
                    "ai_message": "Using tools",
                    "ai_tool_calls": [
                        {"function_name": "search_tools", "arguments": {"query": "draw a circle"}},
                        {"function_name": "create_circle", "arguments": {}},
                    ],
                    "finish_reason": "tool_calls",
                }
            ]
        )

        payload = {
            "message": json.dumps({"user_message": "draw circle", "use_vision": False, "ai_model": "gpt-4.1"}),
            "svg_state": None,
        }
        response = self.client.post("/send_message_stream", json=payload)

        lines = [json.loads(line) for line in response.data.decode("utf-8").split("\n") if line.strip()]
        finals = [entry for entry in lines if isinstance(entry, dict) and entry.get("type") == "final"]

        self.assertEqual(response.status_code, 200)
        self.assertTrue(finals)
        self.assertEqual(finals[-1].get("ai_tool_calls"), filtered_calls)
        mock_intercept.assert_called_once()

    @patch("static.routes._intercept_search_tools")
    @patch.object(OpenAIResponsesAPI, "create_response_stream")
    def test_non_stream_reasoning_path_calls_intercept(
        self,
        mock_response_stream: Mock,
        mock_intercept: Mock,
    ) -> None:
        filtered_calls = [{"function_name": "solve", "arguments": {"expression": "x^2-1=0"}}]
        mock_intercept.return_value = filtered_calls

        mock_response_stream.return_value = iter(
            [
                {
                    "type": "final",
                    "ai_message": "Reasoning complete",
                    "ai_tool_calls": [
                        {"function_name": "search_tools", "arguments": {"query": "solve equation"}},
                        {"function_name": "solve", "arguments": {"expression": "x^2-1=0"}},
                    ],
                    "finish_reason": "tool_calls",
                }
            ]
        )

        payload = {
            "message": json.dumps(
                {
                    "user_message": "solve x^2-1=0",
                    "use_vision": False,
                    "ai_model": "o3",
                }
            ),
            "svg_state": None,
        }
        response = self.client.post("/send_message", json=payload)
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["data"]["ai_tool_calls"], filtered_calls)
        mock_intercept.assert_called_once()

    @patch("static.routes._intercept_search_tools")
    @patch.object(OpenAIChatCompletionsAPI, "create_chat_completion")
    def test_non_stream_chat_path_calls_intercept(
        self,
        mock_chat_completion: Mock,
        mock_intercept: Mock,
    ) -> None:
        class MockToolFunction:
            def __init__(self, name: str, arguments: str) -> None:
                self.name = name
                self.arguments = arguments

        class MockToolCall:
            def __init__(self, name: str, arguments: str) -> None:
                self.function = MockToolFunction(name, arguments)

        class MockMessage:
            content = "I will draw that."
            tool_calls = [
                MockToolCall("search_tools", '{\"query\": \"plot x^2\"}'),
                MockToolCall("draw_function", '{\"expression\": \"x^2\"}'),
            ]

        class MockChoice:
            finish_reason = "tool_calls"
            message = MockMessage()

        filtered_calls = [{"function_name": "draw_function", "arguments": {"expression": "x^2"}}]
        mock_chat_completion.return_value = MockChoice()
        mock_intercept.return_value = filtered_calls

        payload = {
            "message": json.dumps({"user_message": "plot x^2", "use_vision": False, "ai_model": "gpt-4.1"}),
            "svg_state": None,
        }
        response = self.client.post("/send_message", json=payload)
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["data"]["ai_tool_calls"], filtered_calls)
        mock_intercept.assert_called_once()


if __name__ == "__main__":
    unittest.main()
