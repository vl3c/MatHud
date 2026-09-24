"""
Tests that tool calls dropped by search_tools filtering get an explicit result.

When search_tools is called alongside other tools, calls to tools the search did
not load are filtered out before the client sees them. The model must still get
an answer for each of those calls instead of a silent "Awaiting result...".
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

from static.app_manager import AppManager, MatHudFlask
from static.openai_api_base import OpenAIAPIBase
from static.providers.local.local_agent_api import LocalAgentAPI
from static.ai_model import AIModel

CALLS: List[Dict[str, Any]] = [
    {"id": "call_s", "function_name": "search_tools", "arguments": {"query": "draw circle"}},
    {"id": "call_c", "function_name": "create_circle", "arguments": {"x": 0, "y": 0}},
    {"id": "call_d", "function_name": "delete_all", "arguments": {}},
]


def _add_pending_turn(api: OpenAIAPIBase) -> None:
    api.messages.append({"role": "assistant", "content": "", "tool_calls": []})
    for call in CALLS:
        api.messages.append({"role": "tool", "tool_call_id": call["id"], "content": "Awaiting result..."})


def _tool_messages(api: OpenAIAPIBase) -> Dict[str, str]:
    return {m["tool_call_id"]: m["content"] for m in api.messages if m.get("role") == "tool"}


class TestDroppedToolCalls(unittest.TestCase):
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

    def _intercept(self, provider: Optional[OpenAIAPIBase] = None) -> List[Dict[str, Any]]:
        from static.routes import _intercept_search_tools

        with patch("static.tool_search_service.ToolSearchService") as service_class:
            service = Mock()
            service.search_tools.return_value = [{"function": {"name": "create_circle"}}]
            service_class.return_value = service
            with self.assertLogs("static.routes", level="WARNING") as logs:
                result = _intercept_search_tools(self.app, [dict(c) for c in CALLS], provider)
        self.assertTrue(any("delete_all" in line for line in logs.output))
        return result

    def test_dropped_call_gets_explicit_error_result(self) -> None:
        _add_pending_turn(self.app.ai_api)

        result = self._intercept()

        self.assertEqual([c["function_name"] for c in result], ["search_tools", "create_circle"])
        messages = _tool_messages(self.app.ai_api)
        self.assertEqual(
            messages["call_d"], "Error: tool 'delete_all' is not loaded; call search_tools first to load it."
        )
        self.assertEqual(messages["call_s"], "Awaiting result...")
        self.assertEqual(messages["call_c"], "Awaiting result...")

    def test_client_results_fill_the_kept_calls_and_keep_the_error(self) -> None:
        _add_pending_turn(self.app.ai_api)
        self._intercept()

        client_results = [
            {"tool_call_id": "call_s", "result": {"search_tools(query:draw circle)": {"count": 1}}},
            {"tool_call_id": "call_c", "result": {"create_circle(x:0, y:0)": "Call successful!"}},
        ]
        self.app.ai_api._update_tool_messages_with_results(json.dumps(client_results))

        messages = _tool_messages(self.app.ai_api)
        self.assertEqual(messages["call_c"], json.dumps({"create_circle(x:0, y:0)": "Call successful!"}))
        self.assertTrue(messages["call_d"].startswith("Error: tool 'delete_all' is not loaded"))

    def test_error_is_recorded_on_a_distinct_active_provider(self) -> None:
        provider = LocalAgentAPI(model=AIModel.from_identifier("local-model"))
        _add_pending_turn(provider)

        self._intercept(provider)

        self.assertTrue(_tool_messages(provider)["call_d"].startswith("Error: tool 'delete_all' is not loaded"))

    def test_full_tool_mode_keeps_every_call_and_the_full_tool_set(self) -> None:
        from static.routes import _intercept_search_tools

        for api in (self.app.ai_api, self.app.responses_api):
            api.set_tool_mode("full")
            _add_pending_turn(api)
        full_count = len(self.app.ai_api.tools)

        with patch("static.tool_search_service.ToolSearchService") as service_class:
            service_class.return_value.search_tools.return_value = [{"function": {"name": "create_circle"}}]
            result = _intercept_search_tools(self.app, [dict(c) for c in CALLS])

        self.assertEqual([c["function_name"] for c in result], [c["function_name"] for c in CALLS])
        self.assertEqual(len(self.app.ai_api.tools), full_count)
        self.assertEqual(_tool_messages(self.app.ai_api)["call_d"], "Awaiting result...")

    def test_full_tool_mode_ignores_search_results_from_the_client(self) -> None:
        from static.routes import _maybe_inject_search_tools

        api = self.app.ai_api
        api.set_tool_mode("full")
        full_count = len(api.tools)
        results = [
            {
                "tool_call_id": "call_s",
                "result": {"search_tools(query:circle)": {"query": "circle", "tools": [{"function": {"name": "x"}}]}},
            }
        ]
        _maybe_inject_search_tools(api, json.dumps(results))

        self.assertFalse(api.has_injected_tools())
        self.assertEqual(len(api.tools), full_count)


if __name__ == "__main__":
    unittest.main()
