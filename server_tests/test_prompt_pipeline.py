"""Mocked end-to-end prompt pipeline tests.

Verifies the full pipeline:  natural-language prompt  →  real local tool search
→  real filtering  →  correct tool calls returned to the client.

Only the OpenAI API call is mocked; ``_intercept_search_tools``,
``ToolSearchService.search_tools_local``, and the filtering helpers all run
for real with ``TOOL_SEARCH_MODE=local``.
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import Mock, patch

from static.app_manager import AppManager, MatHudFlask
from static.openai_completions_api import OpenAIChatCompletionsAPI
from static.openai_responses_api import OpenAIResponsesAPI
from static.tool_search_service import clear_search_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_payload(msg: str, model: str = "gpt-4.1") -> Dict[str, Any]:
    """Build the POST body expected by ``/send_message_stream`` and ``/send_message``."""
    return {
        "message": json.dumps(
            {"user_message": msg, "use_vision": False, "ai_model": model}
        ),
        "svg_state": None,
    }


def _search_call(query: str) -> Dict[str, Any]:
    """Build a ``search_tools`` tool-call dict."""
    return {"function_name": "search_tools", "arguments": {"query": query}}


def _tool_call(name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build an action tool-call dict."""
    return {"function_name": name, "arguments": args or {}}


def _mock_stream_final(
    message: str,
    tool_calls: List[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """Return a single-event stream matching the ``create_chat_completion_stream`` contract."""
    return iter(
        [
            {
                "type": "final",
                "ai_message": message,
                "ai_tool_calls": tool_calls,
                "finish_reason": "tool_calls",
            }
        ]
    )


def _parse_ndjson_events(response_data: bytes) -> List[Dict[str, Any]]:
    """Parse NDJSON response body into a list of event dicts."""
    return [
        json.loads(line)
        for line in response_data.decode("utf-8").split("\n")
        if line.strip()
    ]


def _get_final_event(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the last ``final`` event, if any."""
    finals = [e for e in events if isinstance(e, dict) and e.get("type") == "final"]
    return finals[-1] if finals else None


def _get_final_tool_names(events: List[Dict[str, Any]]) -> List[str]:
    """Extract function names from the final event's tool calls."""
    final = _get_final_event(events)
    if final is None:
        return []
    return [
        tc.get("function_name", "")
        for tc in final.get("ai_tool_calls", [])
        if isinstance(tc, dict)
    ]


# ---------------------------------------------------------------------------
# Streaming tests (14)
# ---------------------------------------------------------------------------


class TestPromptPipelineStream(unittest.TestCase):
    """Full pipeline tests via ``/send_message_stream``.

    Mock only the OpenAI streaming call; everything else runs for real.
    """

    def setUp(self) -> None:
        self._saved_env: Dict[str, Optional[str]] = {}
        for key in ("REQUIRE_AUTH", "TOOL_SEARCH_MODE"):
            self._saved_env[key] = os.environ.get(key)
        os.environ["REQUIRE_AUTH"] = "false"
        os.environ["TOOL_SEARCH_MODE"] = "local"

        clear_search_cache()

        self.app: MatHudFlask = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        for key, val in self._saved_env.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)
        clear_search_cache()

    # -- helpers --

    def _post_stream(
        self,
        msg: str,
        tool_calls: List[Dict[str, Any]],
        ai_message: str = "Using tools",
    ) -> List[Dict[str, Any]]:
        """POST to ``/send_message_stream`` with a mocked final event and return parsed events."""
        with patch.object(
            OpenAIChatCompletionsAPI,
            "create_chat_completion_stream",
            return_value=_mock_stream_final(ai_message, tool_calls),
        ):
            resp = self.client.post(
                "/send_message_stream",
                json=_make_payload(msg),
            )
        self.assertEqual(resp.status_code, 200)
        return _parse_ndjson_events(resp.data)

    def _assert_tool_passes(
        self,
        msg: str,
        search_query: str,
        action_name: str,
        action_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Assert that ``action_name`` survives the pipeline."""
        events = self._post_stream(
            msg,
            [_search_call(search_query), _tool_call(action_name, action_args)],
        )
        names = _get_final_tool_names(events)
        self.assertIn(action_name, names)

    # -- individual tests --

    def test_circle_creation(self) -> None:
        self._assert_tool_passes(
            "draw a circle",
            "draw circle",
            "create_circle",
            {"center_x": 0, "center_y": 0, "radius": 5},
        )

    def test_triangle_creation(self) -> None:
        self._assert_tool_passes(
            "create a triangle",
            "create triangle",
            "create_polygon",
            {"vertices": [[0, 0], [4, 0], [2, 3]]},
        )

    def test_derivative(self) -> None:
        self._assert_tool_passes(
            "find the derivative of x^2",
            "derivative",
            "derive",
            {"expression": "x^2", "variable": "x"},
        )

    def test_solve_equation(self) -> None:
        self._assert_tool_passes(
            "solve x^2 - 1 = 0",
            "solve equation",
            "solve",
            {"expression": "x^2-1=0", "variable": "x"},
        )

    def test_plot_distribution(self) -> None:
        self._assert_tool_passes(
            "plot a normal distribution",
            "plot normal distribution",
            "plot_distribution",
            {"distribution_type": "normal", "mean": 0, "std_dev": 1},
        )

    def test_descriptive_stats(self) -> None:
        self._assert_tool_passes(
            "compute descriptive statistics for [1,2,3]",
            "descriptive statistics",
            "compute_descriptive_statistics",
            {"data": [1, 2, 3]},
        )

    def test_create_graph(self) -> None:
        self._assert_tool_passes(
            "create a weighted graph",
            "create weighted graph vertices edges",
            "generate_graph",
            {"graph_name": "G1", "vertices": ["A", "B"]},
        )

    def test_undo(self) -> None:
        self._assert_tool_passes("undo last action", "undo", "undo")

    def test_save_workspace(self) -> None:
        self._assert_tool_passes(
            "save my workspace",
            "save workspace",
            "save_workspace",
            {"name": "MyProject"},
        )

    def test_rotate_object(self) -> None:
        self._assert_tool_passes(
            "rotate the triangle",
            "rotate triangle",
            "rotate_object",
            {"object_name": "t1", "angle": 45},
        )

    def test_multi_tool(self) -> None:
        """Both ``create_point`` and ``create_segment`` should pass."""
        events = self._post_stream(
            "create a point and a segment",
            [
                _search_call("create point segment"),
                _tool_call("create_point", {"x": 0, "y": 0}),
                _tool_call("create_segment", {"x1": 0, "y1": 0, "x2": 1, "y2": 1}),
            ],
        )
        names = _get_final_tool_names(events)
        self.assertIn("create_point", names)
        self.assertIn("create_segment", names)

    def test_filters_irrelevant_tool(self) -> None:
        """``analyze_graph`` should be filtered out for a circle query."""
        events = self._post_stream(
            "draw a circle",
            [
                _search_call("draw circle"),
                _tool_call("create_circle", {"center_x": 0, "center_y": 0, "radius": 5}),
                _tool_call("analyze_graph", {"graph_name": "G1", "algorithm": "bfs"}),
            ],
        )
        names = _get_final_tool_names(events)
        self.assertIn("create_circle", names)
        self.assertNotIn("analyze_graph", names)

    def test_essential_passthrough(self) -> None:
        """Essential tools pass even if not in search results."""
        events = self._post_stream(
            "get canvas state and undo",
            [
                _search_call("canvas state undo"),
                _tool_call("get_current_canvas_state"),
                _tool_call("undo"),
            ],
        )
        names = _get_final_tool_names(events)
        self.assertIn("get_current_canvas_state", names)
        self.assertIn("undo", names)

    def test_no_search_tools_passthrough(self) -> None:
        """When no ``search_tools`` call is present, all tools pass unfiltered."""
        events = self._post_stream(
            "create a point",
            [_tool_call("create_point", {"x": 5, "y": 10})],
        )
        names = _get_final_tool_names(events)
        self.assertIn("create_point", names)


# ---------------------------------------------------------------------------
# Non-streaming tests (3)
# ---------------------------------------------------------------------------


class TestPromptPipelineNonStream(unittest.TestCase):
    """Full pipeline tests via ``/send_message``.

    Mock only the OpenAI call; everything else runs for real.
    """

    def setUp(self) -> None:
        self._saved_env: Dict[str, Optional[str]] = {}
        for key in ("REQUIRE_AUTH", "TOOL_SEARCH_MODE"):
            self._saved_env[key] = os.environ.get(key)
        os.environ["REQUIRE_AUTH"] = "false"
        os.environ["TOOL_SEARCH_MODE"] = "local"

        clear_search_cache()

        self.app: MatHudFlask = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        for key, val in self._saved_env.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)
        clear_search_cache()

    # -- reasoning model (o3): uses create_response_stream, consumed via /send_message --

    @patch.object(OpenAIResponsesAPI, "create_response_stream")
    def test_reasoning_model_derivative(self, mock_stream: Mock) -> None:
        mock_stream.return_value = _mock_stream_final(
            "Taking derivative",
            [
                _search_call("derivative"),
                _tool_call("derive", {"expression": "x^3", "variable": "x"}),
            ],
        )
        resp = self.client.post("/send_message", json=_make_payload("derivative of x^3", "o3"))
        data = json.loads(resp.data)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data["status"], "success")
        tool_names = [
            tc.get("function_name", "")
            for tc in data["data"]["ai_tool_calls"]
            if isinstance(tc, dict)
        ]
        self.assertIn("derive", tool_names)

    # -- chat completions model (gpt-4.1): uses create_chat_completion --

    @patch.object(OpenAIChatCompletionsAPI, "create_chat_completion")
    def test_chat_completion_circle(self, mock_completion: Mock) -> None:
        mock_completion.return_value = self._make_mock_choice(
            "Drawing circle",
            [
                ("search_tools", json.dumps({"query": "draw circle"})),
                ("create_circle", json.dumps({"center_x": 0, "center_y": 0, "radius": 5})),
            ],
        )
        resp = self.client.post("/send_message", json=_make_payload("draw a circle", "gpt-4.1"))
        data = json.loads(resp.data)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data["status"], "success")
        tool_names = [
            tc.get("function_name", "")
            for tc in data["data"]["ai_tool_calls"]
            if isinstance(tc, dict)
        ]
        self.assertIn("create_circle", tool_names)

    @patch.object(OpenAIChatCompletionsAPI, "create_chat_completion")
    def test_chat_completion_filters_irrelevant(self, mock_completion: Mock) -> None:
        mock_completion.return_value = self._make_mock_choice(
            "Drawing circle",
            [
                ("search_tools", json.dumps({"query": "draw circle"})),
                ("create_circle", json.dumps({"center_x": 0, "center_y": 0, "radius": 5})),
                ("analyze_graph", json.dumps({"graph_name": "G1", "algorithm": "bfs"})),
            ],
        )
        resp = self.client.post("/send_message", json=_make_payload("draw a circle", "gpt-4.1"))
        data = json.loads(resp.data)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data["status"], "success")
        tool_names = [
            tc.get("function_name", "")
            for tc in data["data"]["ai_tool_calls"]
            if isinstance(tc, dict)
        ]
        self.assertIn("create_circle", tool_names)
        self.assertNotIn("analyze_graph", tool_names)

    # -- mock builder --

    @staticmethod
    def _make_mock_choice(
        content: str,
        tool_calls: List[tuple[str, str]],
    ) -> Any:
        """Build a ``SimpleNamespace`` choice matching the ``ToolCallObject`` protocol.

        Each entry in *tool_calls* is ``(function_name, arguments_json_str)``.
        """
        from types import SimpleNamespace

        mock_tool_calls = [
            SimpleNamespace(
                id=f"call_{i}",
                function=SimpleNamespace(name=name, arguments=args),
            )
            for i, (name, args) in enumerate(tool_calls)
        ]
        return SimpleNamespace(
            finish_reason="tool_calls",
            message=SimpleNamespace(content=content, tool_calls=mock_tool_calls),
        )


if __name__ == "__main__":
    unittest.main()
