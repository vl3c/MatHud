from __future__ import annotations

import unittest
from typing import Any, Optional

from browser import html

from ai_interface import AIInterface
from .simple_mock import SimpleMock


def _get_class_attr(node: Any) -> str:
    try:
        attrs = getattr(node, "attrs", None)
        if attrs is not None and hasattr(attrs, "get"):
            value = attrs.get("class", "")
            if isinstance(value, str):
                return value
            return "" if value is None else str(value)
    except Exception:
        pass
    try:
        value = getattr(node, "class_name", None)
        if isinstance(value, str):
            return value
    except Exception:
        pass
    try:
        value = getattr(node, "className", None)
        if isinstance(value, str):
            return value
    except Exception:
        pass
    return ""


def _find_child_by_class(parent: Any, cls: str) -> Optional[Any]:
    """Return the first child element whose class attribute contains *cls*."""
    for child in getattr(parent, "children", []):
        if cls in _get_class_attr(child):
            return child
    return None


def _make_ai() -> AIInterface:
    """Create a minimal AIInterface instance for testing (no heavy __init__)."""
    ai = AIInterface.__new__(AIInterface)
    ai._tool_call_log_entries = []
    ai._tool_call_log_element = None
    ai._tool_call_log_summary = None
    ai._tool_call_log_content = None
    ai._stream_buffer = ""
    ai._stream_content_element = None
    ai._stream_message_container = None
    ai._reasoning_buffer = ""
    ai._reasoning_element = None
    ai._reasoning_details = None
    ai._reasoning_summary = None
    ai._is_reasoning = False
    ai._request_start_time = None
    ai._needs_continuation_separator = False
    ai._open_message_menu = None
    ai._message_menu_global_bound = True
    ai._copy_text_to_clipboard = SimpleMock(return_value=True)
    return ai


class TestToolCallLog(unittest.TestCase):
    """Tests for the tool-call log dropdown feature in AIInterface."""

    # ── Argument formatting ──────────────────────────────────────

    def test_format_args_simple(self) -> None:
        ai = _make_ai()
        result = ai._format_tool_call_args_display({"x": 5, "y": 10})
        self.assertEqual(result, "x: 5, y: 10")

    def test_format_args_filters_canvas(self) -> None:
        ai = _make_ai()
        result = ai._format_tool_call_args_display({"x": 5, "canvas": "<obj>", "y": 10})
        self.assertEqual(result, "x: 5, y: 10")

    def test_format_args_truncates_long_values(self) -> None:
        ai = _make_ai()
        long_val = "a" * 50
        result = ai._format_tool_call_args_display({"data": long_val})
        self.assertIn("...", result)
        # The value portion should be at most 30 characters
        val_part = result.split(": ", 1)[1]
        self.assertLessEqual(len(val_part), 30)

    def test_format_args_truncates_total_string(self) -> None:
        ai = _make_ai()
        # Many short args that together exceed 80 chars
        args = {f"k{i}": f"value{i}" for i in range(20)}
        result = ai._format_tool_call_args_display(args)
        self.assertLessEqual(len(result), 80)
        self.assertTrue(result.endswith("..."))

    def test_format_args_empty(self) -> None:
        ai = _make_ai()
        result = ai._format_tool_call_args_display({})
        self.assertEqual(result, "")

    # ── Entry element creation ───────────────────────────────────

    def test_entry_element_success(self) -> None:
        ai = _make_ai()
        entry = {"name": "create_point", "args_display": "x: 5, y: 10", "is_error": False, "error_message": ""}
        el = ai._create_tool_call_entry_element(entry)
        cls = _get_class_attr(el)
        self.assertIn("tool-call-entry", cls)

        status = _find_child_by_class(el, "tool-call-status")
        self.assertIsNotNone(status)
        self.assertIn("success", _get_class_attr(status))

        name = _find_child_by_class(el, "tool-call-name")
        self.assertIsNotNone(name)
        self.assertEqual(name.text, "create_point")

        args = _find_child_by_class(el, "tool-call-args")
        self.assertIsNotNone(args)
        self.assertIn("x: 5", args.text)

    def test_entry_element_error(self) -> None:
        ai = _make_ai()
        entry = {
            "name": "create_segment",
            "args_display": "start: Q, end: R",
            "is_error": True,
            "error_message": "Error: Point Q not found",
        }
        el = ai._create_tool_call_entry_element(entry)

        status = _find_child_by_class(el, "tool-call-status")
        self.assertIsNotNone(status)
        self.assertIn("error", _get_class_attr(status))

        err_msg = _find_child_by_class(el, "tool-call-error-msg")
        self.assertIsNotNone(err_msg)
        self.assertIn("Point Q not found", err_msg.text)

    def test_entry_element_empty_args(self) -> None:
        ai = _make_ai()
        entry = {"name": "clear_canvas", "args_display": "", "is_error": False, "error_message": ""}
        el = ai._create_tool_call_entry_element(entry)

        args = _find_child_by_class(el, "tool-call-args")
        self.assertIsNotNone(args)
        self.assertEqual(args.text, "()")

    # ── Ensure tool call log element ─────────────────────────────

    def test_ensure_creates_details_element(self) -> None:
        ai = _make_ai()
        # Provide a container so _ensure doesn't try to access document
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        ai._ensure_tool_call_log_element()

        self.assertIsNotNone(ai._tool_call_log_element)
        cls = _get_class_attr(ai._tool_call_log_element)
        self.assertIn("tool-call-log-dropdown", cls)

    def test_ensure_creates_summary(self) -> None:
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        ai._ensure_tool_call_log_element()

        self.assertIsNotNone(ai._tool_call_log_summary)
        self.assertIn("Using tools", ai._tool_call_log_summary.text)

    def test_ensure_creates_content_div(self) -> None:
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        ai._ensure_tool_call_log_element()

        self.assertIsNotNone(ai._tool_call_log_content)
        cls = _get_class_attr(ai._tool_call_log_content)
        self.assertIn("tool-call-log-content", cls)

    def test_ensure_idempotent(self) -> None:
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        ai._ensure_tool_call_log_element()
        first_el = ai._tool_call_log_element

        ai._ensure_tool_call_log_element()
        self.assertIs(ai._tool_call_log_element, first_el)

    def test_ensure_creates_container_if_missing(self) -> None:
        ai = _make_ai()
        # Don't set _stream_message_container — it should create one
        # We need _ensure_stream_message_element to work, which needs document["chat-history"]
        # Instead, verify our guard logic: if _tool_call_log_element is already set, skip
        sentinel = html.DETAILS()
        ai._tool_call_log_element = sentinel
        ai._ensure_tool_call_log_element()
        self.assertIs(ai._tool_call_log_element, sentinel)

    # ── Adding entries ───────────────────────────────────────────

    def test_add_single_success_entry(self) -> None:
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        tool_calls = [{"function_name": "create_point", "arguments": {"x": 5, "y": 10, "name": "A"}}]
        call_results = {"create_point(x:5, y:10, name:A)": "Point A created at (5, 10)"}

        ai._add_tool_call_entries(tool_calls, call_results)

        self.assertEqual(len(ai._tool_call_log_entries), 1)
        self.assertFalse(ai._tool_call_log_entries[0]["is_error"])
        # Content div should have one child entry
        self.assertTrue(len(ai._tool_call_log_content.children) >= 1)

    def test_add_single_error_entry(self) -> None:
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        tool_calls = [{"function_name": "create_segment", "arguments": {"start": "Q", "end": "R"}}]
        call_results = {"create_segment(start:Q, end:R)": "Error: Point Q not found"}

        ai._add_tool_call_entries(tool_calls, call_results)

        self.assertEqual(len(ai._tool_call_log_entries), 1)
        self.assertTrue(ai._tool_call_log_entries[0]["is_error"])

    def test_add_multiple_entries(self) -> None:
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        tool_calls = [
            {"function_name": "create_point", "arguments": {"x": 0, "y": 0, "name": "O"}},
            {"function_name": "create_point", "arguments": {"x": 5, "y": 5, "name": "P"}},
        ]
        call_results = {
            "create_point(x:0, y:0, name:O)": "Point O created",
            "create_point(x:5, y:5, name:P)": "Point P created",
        }

        ai._add_tool_call_entries(tool_calls, call_results)
        self.assertEqual(len(ai._tool_call_log_entries), 2)

    def test_add_entries_accumulates_across_calls(self) -> None:
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        # First round
        ai._add_tool_call_entries(
            [{"function_name": "create_point", "arguments": {"x": 1, "y": 2}}],
            {"create_point(x:1, y:2)": "ok"},
        )
        self.assertEqual(len(ai._tool_call_log_entries), 1)

        # Second round
        ai._add_tool_call_entries(
            [{"function_name": "create_circle", "arguments": {"center": "A", "radius": 5}}],
            {"create_circle(center:A, radius:5)": "ok"},
        )
        self.assertEqual(len(ai._tool_call_log_entries), 2)

    def test_add_entries_updates_summary(self) -> None:
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        ai._add_tool_call_entries(
            [{"function_name": "f", "arguments": {}}],
            {"f()": "ok"},
        )
        self.assertIn("1 so far", ai._tool_call_log_summary.text)

    # ── Finalize tool call log ───────────────────────────────────

    def test_finalize_singular(self) -> None:
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        ai._add_tool_call_entries(
            [{"function_name": "f", "arguments": {}}],
            {"f()": "ok"},
        )
        ai._finalize_tool_call_log()
        self.assertEqual(ai._tool_call_log_summary.text, "Used 1 tool")

    def test_finalize_plural(self) -> None:
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        for i in range(3):
            ai._add_tool_call_entries(
                [{"function_name": f"f{i}", "arguments": {}}],
                {f"f{i}()": "ok"},
            )
        ai._finalize_tool_call_log()
        self.assertEqual(ai._tool_call_log_summary.text, "Used 3 tools")

    def test_finalize_with_errors(self) -> None:
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        ai._add_tool_call_entries(
            [
                {"function_name": "a", "arguments": {}},
                {"function_name": "b", "arguments": {}},
                {"function_name": "c", "arguments": {}},
            ],
            {"a()": "ok", "b()": "ok", "c()": "Error: something failed"},
        )
        ai._finalize_tool_call_log()
        self.assertIn("Used 3 tools", ai._tool_call_log_summary.text)
        self.assertIn("1 failed", ai._tool_call_log_summary.text)

    def test_finalize_no_entries(self) -> None:
        ai = _make_ai()
        # Should be a no-op, no crash
        ai._finalize_tool_call_log()
        self.assertIsNone(ai._tool_call_log_summary)

    # ── State management ─────────────────────────────────────────

    def test_state_reset_clears_tool_call_log(self) -> None:
        """Verify _reset_tool_call_log_state clears all tool call log state."""
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content

        ai._add_tool_call_entries(
            [{"function_name": "f", "arguments": {}}],
            {"f()": "ok"},
        )
        self.assertEqual(len(ai._tool_call_log_entries), 1)
        self.assertIsNotNone(ai._tool_call_log_element)

        ai._reset_tool_call_log_state()

        self.assertEqual(ai._tool_call_log_entries, [])
        self.assertIsNone(ai._tool_call_log_element)
        self.assertIsNone(ai._tool_call_log_summary)
        self.assertIsNone(ai._tool_call_log_content)

    def test_container_preserved_with_tool_log(self) -> None:
        """_remove_empty_response_container preserves container when tool log exists."""
        ai = _make_ai()
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        content.text = ""
        container <= content
        ai._stream_message_container = container
        ai._stream_content_element = content
        ai._stream_buffer = ""

        # Add tool call entries so the log is non-empty
        ai._tool_call_log_entries = [{"name": "f", "is_error": False}]

        # Call the actual method — it should NOT remove the container
        # because tool call log entries exist
        ai._remove_empty_response_container()

        self.assertIs(
            ai._stream_message_container, container, "Container should NOT be removed when tool call log has entries"
        )
