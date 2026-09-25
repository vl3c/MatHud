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

    def test_legacy_dict_skips_a_last_message_holding_a_dropped_call_error(self) -> None:
        self.assertTrue(self.api.record_tool_call_result("call_b", "Error: tool 'x' is not loaded"))
        self.api._update_tool_messages_with_results(json.dumps(RESULT_A))
        self.assertEqual(
            _tool_messages(self.api),
            {"call_a": json.dumps(RESULT_A), "call_b": "Error: tool 'x' is not loaded"},
        )

    def test_entry_with_unknown_id_is_not_matched_by_position(self) -> None:
        with self.assertLogs("mathud", level="WARNING"):
            self.api._update_tool_messages_with_results(
                json.dumps(
                    [{"tool_call_id": "call_zzz", "result": RESULT_A}, {"tool_call_id": "call_b", "result": RESULT_B}]
                )
            )
        messages = _tool_messages(self.api)
        self.assertTrue(messages["call_a"].startswith("Error:"))
        self.assertEqual(messages["call_b"], json.dumps(RESULT_B))

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


class TestUnrunToolCallsAreNotStored(unittest.TestCase):
    """Tool calls of a reply that did not end in tool calls (e.g. partial calls of a
    reply cut off at the token limit) never run, so no placeholder waits for them."""

    PARTIAL_CALL = SimpleNamespace(id="call_cut", function=SimpleNamespace(name="create_point", arguments='{"x": 1'))

    @staticmethod
    def _stream_chunk(content: Optional[str], tool_calls: Any, finish_reason: Optional[str]) -> SimpleNamespace:
        delta = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=None)

    def _partial_stream(self, finish_reason: str) -> List[SimpleNamespace]:
        tool_delta = SimpleNamespace(index=0, id="call_cut", function=self.PARTIAL_CALL.function)
        return [
            self._stream_chunk("Placing", None, None),
            self._stream_chunk(None, [tool_delta], finish_reason),
        ]

    def _assert_nothing_pending(self, api: OpenAIAPIBase) -> None:
        self.assertEqual(_tool_messages(api), {})
        assistant = [m for m in api.messages if m.get("role") == "assistant"]
        self.assertTrue(assistant)
        self.assertNotIn("tool_calls", assistant[-1])

    def test_chat_completions_non_streaming_length(self) -> None:
        with patch("static.openai_api_base.OpenAI"):
            api = OpenAIChatCompletionsAPI()
        message = SimpleNamespace(content="Placing", tool_calls=[self.PARTIAL_CALL])
        api.client = Mock()
        api.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="length")], usage=None
        )

        result = api.create_chat_completion(json.dumps({"user_message": "hi"}))

        self.assertEqual(result.finish_reason, "length")
        self._assert_nothing_pending(api)

    def test_chat_completions_unrun_calls_without_text_store_empty_content(self) -> None:
        """Null content is only valid alongside tool calls, so it becomes an empty string."""
        with patch("static.openai_api_base.OpenAI"):
            api = OpenAIChatCompletionsAPI()
        message = SimpleNamespace(content=None, tool_calls=[self.PARTIAL_CALL])

        stored = api._create_assistant_message(message, include_tool_calls=False)

        self.assertEqual(stored, {"role": "assistant", "content": ""})

    def test_chat_completions_streaming_length(self) -> None:
        with patch("static.openai_api_base.OpenAI"):
            api = OpenAIChatCompletionsAPI()
        api.client = Mock()
        api.client.chat.completions.create.return_value = iter(self._partial_stream("length"))

        final = list(api.create_chat_completion_stream(json.dumps({"user_message": "hi"})))[-1]

        self.assertEqual(final["finish_reason"], "length")
        self._assert_nothing_pending(api)

    def test_chat_completions_streaming_tool_calls_are_stored(self) -> None:
        with patch("static.openai_api_base.OpenAI"):
            api = OpenAIChatCompletionsAPI()
        api.client = Mock()
        api.client.chat.completions.create.return_value = iter(self._partial_stream("tool_calls"))

        list(api.create_chat_completion_stream(json.dumps({"user_message": "hi"})))

        self.assertEqual(_tool_messages(api), {"call_cut": PLACEHOLDER})

    def test_local_non_streaming_length(self) -> None:
        api = LocalAgentAPI(model=AIModel.from_identifier("local-model"))
        choice = SimpleNamespace(
            message=SimpleNamespace(content="Placing", tool_calls=[self.PARTIAL_CALL]), finish_reason="length"
        )

        result = api._process_response(choice)

        self.assertEqual(result.finish_reason, "length")
        self.assertIsNone(result.message.tool_calls)
        self._assert_nothing_pending(api)

    def test_local_streaming_length(self) -> None:
        api = LocalAgentAPI(model=AIModel.from_identifier("local-model"))
        api.client = Mock()
        api.client.chat.completions.create.return_value = iter(self._partial_stream("length"))

        final = list(api.create_chat_completion_stream(json.dumps({"user_message": "hi"})))[-1]

        self.assertEqual(final["finish_reason"], "length")
        self._assert_nothing_pending(api)


class TestCallsWithoutIds(unittest.TestCase):
    """Tool calls that arrive without ids (some local servers) still get the right results."""

    CALLS: List[Dict[str, Any]] = [
        {"id": "", "function_name": "search_tools", "arguments": {"query": "draw circle"}},
        {"id": "", "function_name": "delete_all", "arguments": {}},
        {"id": "", "function_name": "create_circle", "arguments": {"x": 0, "y": 0}},
    ]

    def setUp(self) -> None:
        self.original_require_auth: Optional[str] = os.environ.get("REQUIRE_AUTH")
        os.environ["REQUIRE_AUTH"] = "false"
        self.app: MatHudFlask = AppManager.create_app()
        self.app.config["TESTING"] = True

    def tearDown(self) -> None:
        if self.original_require_auth is not None:
            os.environ["REQUIRE_AUTH"] = self.original_require_auth
        else:
            os.environ.pop("REQUIRE_AUTH", None)

    def test_search_then_dropped_then_kept_call_without_ids(self) -> None:
        from static.routes import _intercept_search_tools

        provider = LocalAgentAPI(model=AIModel.from_identifier("local-model"))
        provider.set_tool_mode("search")
        provider.messages.append({"role": "assistant", "content": "", "tool_calls": []})
        for _ in self.CALLS:
            provider.messages.append({"role": "tool", "tool_call_id": "", "content": PLACEHOLDER})

        with patch("static.tool_search_service.ToolSearchService") as service_class:
            service_class.return_value.search_tools.return_value = [{"function": {"name": "create_circle"}}]
            with self.assertLogs("static.routes", level="WARNING"):
                kept = _intercept_search_tools(self.app, [dict(c) for c in self.CALLS], provider)
        self.assertEqual([c["function_name"] for c in kept], ["search_tools", "create_circle"])

        search_result = {"search_tools(query:draw circle)": {"count": 1}}
        circle_result = {"create_circle(x:0, y:0)": "Call successful!"}
        entries = [{"tool_call_id": "", "result": search_result}, {"tool_call_id": "", "result": circle_result}]
        provider._update_tool_messages_with_results(json.dumps(entries))

        contents = [m["content"] for m in provider.messages if m.get("role") == "tool"]
        self.assertEqual(contents[0], json.dumps(search_result))
        self.assertTrue(contents[1].startswith("Error: tool 'delete_all' is not loaded"))
        self.assertEqual(contents[2], json.dumps(circle_result))

    def test_local_stream_gives_calls_without_ids_an_id(self) -> None:
        def delta(index: int, name: str, finish_reason: Optional[str] = None) -> SimpleNamespace:
            function = SimpleNamespace(name=name, arguments="{}")
            tool_call = SimpleNamespace(index=index, id=None, function=function)
            choice = SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[tool_call]), finish_reason=finish_reason
            )
            return SimpleNamespace(choices=[choice])

        provider = LocalAgentAPI(model=AIModel.from_identifier("local-model"))
        provider.client = Mock()
        provider.client.chat.completions.create.return_value = iter([delta(0, "undo"), delta(1, "redo", "tool_calls")])
        events = list(provider.create_chat_completion_stream(json.dumps({"user_message": "hi"})))

        final = events[-1]
        self.assertEqual([c["id"] for c in final["ai_tool_calls"]], ["call_0", "call_1"])
        placeholders = [m["tool_call_id"] for m in provider.messages if m.get("role") == "tool"]
        self.assertEqual(placeholders, ["call_0", "call_1"])

    def test_local_non_streaming_response_gives_calls_without_ids_an_id(self) -> None:
        provider = LocalAgentAPI(model=AIModel.from_identifier("local-model"))
        tool_call = SimpleNamespace(id=None, function=SimpleNamespace(name="undo", arguments="{}"))
        choice = SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tool_call, tool_call]))
        response = provider._process_response(choice)
        self.assertEqual([tc.id for tc in response.message.tool_calls], ["call_0", "call_1"])


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
        tool_calls = [SimpleNamespace(id=tc["id"], function=SimpleNamespace(**tc["function"])) for tc in TWO_CALLS]
        mock_chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=tool_calls), finish_reason="tool_calls"
        )
        # A model missing from MODEL_CONFIGS routes through the (mocked) OpenAI Chat Completions path.
        prompt = {"user_message": "hi", "use_vision": False, "ai_model": "chat-completions-test-model"}
        payload = {"message": json.dumps(prompt), "svg_state": None}
        response = self.client.post("/send_message", json=payload)
        data = json.loads(response.data)
        self.assertEqual([c["id"] for c in data["data"]["ai_tool_calls"]], ["call_a", "call_b"])


if __name__ == "__main__":
    unittest.main()
