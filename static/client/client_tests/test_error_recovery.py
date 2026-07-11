"""Tests for the message recovery feature on AI errors.

When an AI request fails, the user's message should be restored to the
input field so they can edit and retry.
"""

from __future__ import annotations

import unittest
from typing import Any

from browser import document, html, window

from tool_call_log_manager import ToolCallLogManager
from message_menu_manager import MessageMenuManager
from chat_ui_manager import ChatUIManager


class TestErrorRecovery(unittest.TestCase):
    """Test the message recovery mechanism on AI errors."""

    def _create_ai_interface(self) -> Any:
        """Create an AIInterface instance without running __init__."""
        from ai_interface import AIInterface

        ai = AIInterface.__new__(AIInterface)
        # Initialize minimal state needed for error recovery
        ai._last_user_message = ""
        return ai

    def test_restore_user_message_on_error_populates_input(self) -> None:
        """Test that _restore_user_message_on_error restores the buffered message."""
        # Skip if chat-input doesn't exist in DOM
        if "chat-input" not in document:
            self.skipTest("chat-input element not in DOM")

        ai = self._create_ai_interface()
        chat_input = document["chat-input"]
        original_value = chat_input.value

        try:
            # Set the buffered message
            test_message = "simulate server error for retry"
            ai._last_user_message = test_message

            # Call the recovery method
            ai._restore_user_message_on_error()

            # Verify the message was restored
            self.assertEqual(chat_input.value, test_message)
            # Verify the error-flash class was added
            self.assertTrue(chat_input.classList.contains("error-flash"))
        finally:
            # Restore original state
            chat_input.value = original_value
            chat_input.classList.remove("error-flash")

    def test_restore_user_message_does_nothing_when_empty(self) -> None:
        """Test that _restore_user_message_on_error does nothing with empty buffer."""
        # Skip if chat-input doesn't exist in DOM
        if "chat-input" not in document:
            self.skipTest("chat-input element not in DOM")

        ai = self._create_ai_interface()
        chat_input = document["chat-input"]

        # Set a known value first
        original_value = chat_input.value
        test_existing = "existing content for test"
        chat_input.value = test_existing

        try:
            # Buffer is empty (default)
            ai._last_user_message = ""

            # Call the recovery method
            ai._restore_user_message_on_error()

            # Verify the input was NOT modified
            self.assertEqual(chat_input.value, test_existing)
        finally:
            # Restore original state
            chat_input.value = original_value

    def test_message_buffer_cleared_on_success(self) -> None:
        """Test that successful completion clears the message buffer."""
        ai = self._create_ai_interface()

        # Set up minimal state for _on_stream_final
        tool_call_log = ToolCallLogManager()
        ai._tool_call_log = tool_call_log
        ai._chat_ui = ChatUIManager(
            message_menu=MessageMenuManager(),
            tool_call_log=tool_call_log,
        )
        ai.is_processing = True
        ai._stop_requested = False
        ai._response_timeout_id = None

        # Mock methods
        ai._finalize_stream_message = lambda msg=None: None
        ai._enable_send_controls = lambda: None
        ai._normalize_stream_event = lambda e: e if isinstance(e, dict) else {}

        # Set the buffered message
        ai._last_user_message = "test message"

        # Simulate successful completion event
        success_event = {
            "finish_reason": "stop",
            "ai_tool_calls": [],
            "ai_message": "Response",
        }
        ai._on_stream_final(success_event)

        # Verify the buffer was cleared
        self.assertEqual(ai._last_user_message, "")

    def test_message_buffer_preserved_on_error(self) -> None:
        """Test that error completion preserves the message buffer for recovery."""
        ai = self._create_ai_interface()

        # Set up minimal state
        tool_call_log = ToolCallLogManager()
        ai._tool_call_log = tool_call_log
        ai._chat_ui = ChatUIManager(
            message_menu=MessageMenuManager(),
            tool_call_log=tool_call_log,
        )
        ai.is_processing = True
        ai._stop_requested = False
        ai._response_timeout_id = None

        # Track if restore was called
        restore_called = [False]

        def mock_restore() -> None:
            restore_called[0] = True

        # Mock methods
        ai._finalize_stream_message = lambda msg=None: None
        ai._enable_send_controls = lambda: None
        ai._normalize_stream_event = lambda e: e if isinstance(e, dict) else {}
        ai._restore_user_message_on_error = mock_restore

        # Set the buffered message
        original_message = "simulate server error for retry"
        ai._last_user_message = original_message

        # Simulate error completion event
        error_event = {
            "finish_reason": "error",
            "ai_tool_calls": [],
            "ai_message": "Error occurred",
            "error_details": "Test error triggered",
        }
        ai._on_stream_final(error_event)

        # Verify restore was called
        self.assertTrue(restore_called[0])
        # Verify buffer was NOT cleared (restore should happen before any clearing)
        self.assertEqual(ai._last_user_message, original_message)
