"""
Tests how the non-streaming path ends a turn for each finish reason.

A reply that stopped for length or was refused carries no tool calls; it must be
shown as a final message instead of taking the tool-call branch.

AIInterface is built without __init__ and its UI and tool collaborators are
replaced by recorders, so only _process_ai_response's branching is exercised.
"""

from __future__ import annotations

import unittest
from typing import Any, Callable, List, Optional


class _Stub:
    """Accepts any method call and does nothing."""

    def __init__(self, **attrs: Any) -> None:
        for name, value in attrs.items():
            setattr(self, name, value)

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


class TestAIResponseEndings(unittest.TestCase):
    def setUp(self) -> None:
        from ai_interface import AIInterface

        self.printed: List[str] = []
        self.outcomes: List[str] = []
        self.tool_branch_calls: List[Any] = []

        ai = AIInterface.__new__(AIInterface)

        def print_ai_message(message: str, turn_metrics: Optional[Any] = None) -> None:
            self.printed.append(message)

        def finish_turn(outcome: str, turn_token: Optional[int] = None) -> None:
            self.outcomes.append(outcome)

        def get_canvas_state() -> dict:
            self.tool_branch_calls.append(True)
            return {}

        ai._chat_ui = _Stub(print_ai_message=print_ai_message)
        ai._turn_metrics_collector = _Stub(finish_turn=finish_turn)  # backs the _turn_metrics property
        ai.canvas = _Stub(get_canvas_state=get_canvas_state)
        ai._enable_send_controls = lambda: None
        ai._debug_log_ai_response = lambda *args: None
        self.ai = ai

    def test_length_without_tool_calls_is_a_final_message(self) -> None:
        self.ai._process_ai_response("cut off", [], "length")

        self.assertEqual(self.printed, ["cut off"])
        self.assertEqual(self.outcomes, ["truncated"])
        self.assertEqual(self.tool_branch_calls, [])

    def test_refusal_is_a_final_message(self) -> None:
        self.ai._process_ai_response("declined", [], "refusal")

        self.assertEqual(self.printed, ["declined"])
        self.assertEqual(self.outcomes, ["filtered"])
        self.assertEqual(self.tool_branch_calls, [])

    def test_tool_calls_finish_reason_without_calls_does_not_run_tools(self) -> None:
        self.ai._process_ai_response("", [], "tool_calls")

        self.assertEqual(self.tool_branch_calls, [])
        self.assertEqual(len(self.printed), 1)
