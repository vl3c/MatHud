"""
Tests that each parallel tool call gets its own result, for every provider path.

The client returns one result entry per tool call (tagged with the tool-call id
that the providers now include in ``ai_tool_calls``); the providers must write
each entry into the matching placeholder tool message instead of dumping all
results into the last one.
"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

from static.ai_model import AIModel
from static.app_manager import AppManager, MatHudFlask
from static.openai_api_base import OpenAIAPIBase
from static.openai_completions_api import OpenAIChatCompletionsAPI
from static.openai_responses_api import OpenAIResponsesAPI
from static.providers.anthropic_api import AnthropicAPI
from static.providers.local.local_agent_api import LocalAgentAPI
from static.providers.openrouter_api import OpenRouterAPI

PLACEHOLDER = "Awaiting result..."

TWO_CALLS: List[Dict[str, Any]] = [
    {"id": "call_a", "function": {"name": "create_point", "arguments": '{"x": 1, "y": 2, "name": "A"}'}},
    {"id": "call_b", "function": {"name": "evaluate_expression", "arguments": '{"expression": "2+3"}'}},
]

RESULT_A = {"create_point(x:1, y:2, name:A)": "Call successful!"}
RESULT_B = {"2+3": 5}


def _results_prompt(entries: Any) -> str:
    return json.dumps({"user_message": None, "tool_call_results": json.dumps(entries)})


def _per_call_entries(order: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    entries = {
        "call_a": {"tool_call_id": "call_a", "result": RESULT_A},
        "call_b": {"tool_call_id": "call_b", "result": RESULT_B},
    }
    return [entries[key] for key in (order or ["call_a", "call_b"])]


def _tool_messages(api: OpenAIAPIBase) -> Dict[str, str]:
    return {m["tool_call_id"]: m["content"] for m in api.messages if m.get("role") == "tool"}


class TestBaseToolResultMatching(unittest.TestCase):
    """Matching rules of OpenAIAPIBase._update_tool_messages_with_results."""

    def setUp(self) -> None:
        with patch("static.openai_api_base.OpenAI"):
            self.api = OpenAIAPIBase()
        self.api.messages.append({"role": "assistant", "content": "", "tool_calls": TWO_CALLS})
        self.api.messages.append({"role": "tool", "tool_call_id": "call_a", "content": PLACEHOLDER})
        self.api.messages.append({"role": "tool", "tool_call_id": "call_b", "content": PLACEHOLDER})

    def test_matches_results_by_id_regardless_of_order(self) -> None:
        self.api._update_tool_messages_with_results(json.dumps(_per_call_entries(["call_b", "call_a"])))
        self.assertEqual(_tool_messages(self.api), {"call_a": json.dumps(RESULT_A), "call_b": json.dumps(RESULT_B)})

    def test_entries_without_ids_fall_back_to_call_order(self) -> None:
        entries = [{"tool_call_id": None, "result": RESULT_A}, {"tool_call_id": None, "result": RESULT_B}]
        self.api._update_tool_messages_with_results(json.dumps(entries))
        self.assertEqual(_tool_messages(self.api), {"call_a": json.dumps(RESULT_A), "call_b": json.dumps(RESULT_B)})

    def test_missing_result_is_reported_instead_of_left_awaiting(self) -> None:
        self.api._update_tool_messages_with_results(json.dumps(_per_call_entries(["call_a"])))
        messages = _tool_messages(self.api)
        self.assertEqual(messages["call_a"], json.dumps(RESULT_A))
        self.assertTrue(messages["call_b"].startswith("Error:"))

    def test_already_answered_placeholder_is_not_overwritten(self) -> None:
        self.assertTrue(self.api.record_tool_call_result("call_b", "Error: tool 'x' is not loaded"))
        self.api._update_tool_messages_with_results(json.dumps(_per_call_entries(["call_a"])))
        self.assertEqual(
            _tool_messages(self.api),
            {"call_a": json.dumps(RESULT_A), "call_b": "Error: tool 'x' is not loaded"},
        )

    def test_legacy_dict_shape_still_accepted(self) -> None:
        legacy = {**RESULT_A, **RESULT_B}
        self.api._update_tool_messages_with_results(json.dumps(legacy))
        messages = _tool_messages(self.api)
        self.assertEqual(messages["call_b"], json.dumps(legacy))
        self.assertNotEqual(messages["call_a"], PLACEHOLDER)

    def test_older_turns_are_not_touched(self) -> None:
        self.api._update_tool_messages_with_results(json.dumps(_per_call_entries()))
        self.api.messages.append({"role": "assistant", "content": "done"})
        self.api._update_tool_messages_with_results(json.dumps(_per_call_entries(["call_b", "call_a"])))
        self.assertEqual(_tool_messages(self.api), {"call_a": json.dumps(RESULT_A), "call_b": json.dumps(RESULT_B)})


class TestProviderToolResultRoundTrip(unittest.TestCase):
    """Each provider exposes call ids to the client and routes per-call results back."""

    def _assert_each_call_answered(self, api: OpenAIAPIBase) -> None:
        self.assertEqual(_tool_messages(api), {"call_a": json.dumps(RESULT_A), "call_b": json.dumps(RESULT_B)})

    def _assert_ids_exposed(self, ai_tool_calls: List[Dict[str, Any]]) -> None:
        self.assertEqual([c["id"] for c in ai_tool_calls], ["call_a", "call_b"])
        self.assertEqual([c["function_name"] for c in ai_tool_calls], ["create_point", "evaluate_expression"])

    def test_chat_completions(self) -> None:
        with patch("static.openai_api_base.OpenAI"):
            api = OpenAIChatCompletionsAPI()
        api._finalize_stream("", TWO_CALLS)
        self._assert_ids_exposed(api._prepare_tool_calls_for_response(TWO_CALLS))
        api._prepare_messages_for_request(_results_prompt(_per_call_entries(["call_b", "call_a"])))
        self._assert_each_call_answered(api)

    def test_openrouter(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            api = OpenRouterAPI()
        api._finalize_stream("", TWO_CALLS)
        self._assert_ids_exposed(api._prepare_tool_calls_for_response(TWO_CALLS))
        api._prepare_messages_for_request(_results_prompt(_per_call_entries()))
        self._assert_each_call_answered(api)

    def test_responses_api(self) -> None:
        with patch("static.openai_api_base.OpenAI"):
            api = OpenAIResponsesAPI()
        api._finalize_stream("", TWO_CALLS)
        self._assert_ids_exposed(api._prepare_tool_calls_for_response(TWO_CALLS))
        api._prepare_messages_for_stream(_results_prompt(_per_call_entries(["call_b", "call_a"])))
        self._assert_each_call_answered(api)
        converted = api._convert_messages_to_input()
        results_text = converted[-1]["content"]
        self.assertIn(json.dumps(RESULT_A), results_text)
        self.assertIn(json.dumps(RESULT_B), results_text)

    def test_anthropic(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            api = AnthropicAPI(model=AIModel.from_identifier("claude-haiku-4-5"))
        api._finalize_anthropic_stream("", TWO_CALLS)
        self._assert_ids_exposed(api._prepare_tool_calls_for_response(TWO_CALLS))
        self.assertIsNone(api._parse_and_prepare_message(_results_prompt(_per_call_entries(["call_b", "call_a"]))))
        self._assert_each_call_answered(api)
        tool_results = api._convert_messages_to_anthropic()[-1]["content"]
        self.assertEqual(
            {block["tool_use_id"]: block["content"] for block in tool_results},
            {"call_a": json.dumps(RESULT_A), "call_b": json.dumps(RESULT_B)},
        )

    def test_local_agent(self) -> None:
        api = LocalAgentAPI(model=AIModel.from_identifier("local-model"))
        api._finalize_stream("", TWO_CALLS)
        self._assert_ids_exposed(api._prepare_tool_calls_for_response(TWO_CALLS))
        self.assertIsNone(api._parse_and_prepare_message(_results_prompt(_per_call_entries(["call_b", "call_a"]))))
        self._assert_each_call_answered(api)


class TestNonStreamingRouteExposesIds(unittest.TestCase):
    """The /send_message fallback must also send tool-call ids to the client."""

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

    @patch("static.openai_completions_api.OpenAIChatCompletionsAPI.create_chat_completion")
    def test_send_message_includes_tool_call_ids(self, mock_chat: Mock) -> None:
        tool_calls = [
            SimpleNamespace(id=tc["id"], function=SimpleNamespace(**tc["function"])) for tc in TWO_CALLS
        ]
        mock_chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=tool_calls), finish_reason="tool_calls"
        )
        # A non-reasoning OpenAI model routes through the (mocked) Chat Completions path.
        prompt = {"user_message": "hi", "use_vision": False, "ai_model": "gpt-4.1-mini"}
        payload = {"message": json.dumps(prompt), "svg_state": None}
        response = self.client.post("/send_message", json=payload)
        data = json.loads(response.data)
        self.assertEqual([c["id"] for c in data["data"]["ai_tool_calls"]], ["call_a", "call_b"])


if __name__ == "__main__":
    unittest.main()
