"""
Tests how the streaming and non-streaming paths end a turn for each finish reason.

Only a reply that ends in tool calls and carries some runs tools; a reply that
stopped for length (even with partial tool calls) or was refused must be shown as
a final message instead of taking the tool-call branch.

AIInterface is built without __init__ and its UI and tool collaborators are
replaced by recorders, so only the branching of _process_ai_response and
_on_stream_final is exercised.
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

        def finalize_stream(final_message: Optional[str] = None) -> None:
            self.printed.append(final_message or "")

        def get_canvas_state() -> dict:
            self.tool_branch_calls.append(True)
            return {}

        ai._chat_ui = _Stub(print_ai_message=print_ai_message, stream_buffer="", pending_turn_metrics=None)
        ai._turn_metrics_collector = _Stub(finish_turn=finish_turn)  # backs the _turn_metrics property
        ai.canvas = _Stub(get_canvas_state=get_canvas_state)
        ai._enable_send_controls = lambda: None
        ai._debug_log_ai_response = lambda *args: None
        ai._finalize_stream_message = finalize_stream
        ai._restore_user_message_on_error = lambda: None
        ai._last_user_message = ""
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

    def test_should_run_tools_only_for_tool_call_endings_with_calls(self) -> None:
        from ai_interface import AIInterface

        calls = [{"function_name": "create_point", "arguments": {}}]
        self.assertTrue(AIInterface._should_run_tools("tool_calls", calls))
        self.assertTrue(AIInterface._should_run_tools("function_call", calls))
        for finish_reason in ("stop", "length", "refusal", "error", "completed", "content_filter", None):
            self.assertFalse(AIInterface._should_run_tools(finish_reason, calls), finish_reason)
        self.assertFalse(AIInterface._should_run_tools("tool_calls", []))
        self.assertFalse(AIInterface._should_run_tools("tool_calls", None))

    def test_length_with_partial_tool_calls_is_a_final_message(self) -> None:
        calls = [{"function_name": "create_point", "arguments": {}}]
        self.ai._process_ai_response("cut off", calls, "length")

        self.assertEqual(self.printed, ["cut off"])
        self.assertEqual(self.outcomes, ["truncated"])
        self.assertEqual(self.tool_branch_calls, [])

    def test_text_with_tool_calls_is_shown_before_the_tools_run(self) -> None:
        calls = [{"function_name": "create_point", "arguments": {}}]
        self.ai._process_ai_response("Cut off; 1 call was not run.", calls, "tool_calls")

        self.assertEqual(self.printed, ["Cut off; 1 call was not run."])
        self.assertTrue(self.tool_branch_calls)

    def test_tool_calls_without_text_print_nothing(self) -> None:
        calls = [{"function_name": "create_point", "arguments": {}}]
        self.ai._process_ai_response("", calls, "tool_calls")

        self.assertEqual(self.printed, [])
        self.assertTrue(self.tool_branch_calls)

    def test_streamed_length_with_partial_tool_calls_is_a_final_message(self) -> None:
        event = {
            "finish_reason": "length",
            "ai_message": "cut off",
            "ai_tool_calls": [{"function_name": "create_point", "arguments": {}}],
        }
        self.ai._on_stream_final(event)

        self.assertEqual(self.printed, ["cut off"])
        self.assertEqual(self.outcomes, ["truncated"])
        self.assertEqual(self.tool_branch_calls, [])
        self.assertEqual(self.ai._chat_ui.stream_buffer, "cut off")

    def test_streamed_refusal_is_a_final_message(self) -> None:
        event = {"finish_reason": "refusal", "ai_message": "declined", "ai_tool_calls": []}
        self.ai._on_stream_final(event)

        self.assertEqual(self.printed, ["declined"])
        self.assertEqual(self.outcomes, ["filtered"])
        self.assertEqual(self.tool_branch_calls, [])
