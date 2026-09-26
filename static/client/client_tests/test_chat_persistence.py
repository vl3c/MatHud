"""Tests for chat persistence with workspaces.

Covers the transcript (``ChatTranscript``), recording and restoring it in the
chat UI (``ChatUIManager``), the save/restore bridge (``ChatPersistenceManager``)
and how ``WorkspaceManager`` saves the chat and restores it only on user loads.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional

from browser import html

from chat_persistence_manager import ChatPersistenceManager
from chat_transcript import TRUNCATION_MARKER, ChatTranscript
from chat_ui_manager import ChatUIManager
from constants import (
    MAX_SAVED_CHAT_MESSAGE_CHARS,
    MAX_SAVED_CHAT_MESSAGES,
    MAX_SAVED_CHAT_TOOL_ENTRIES,
    MAX_SAVED_CHAT_TOTAL_CHARS,
)
from message_menu_manager import MessageMenuManager
from tool_call_log_manager import ToolCallLogManager
from workspace_manager import WorkspaceManager
from .simple_mock import get_class_attr


def _log_entry(name: str, args: str = "", is_error: bool = False) -> Dict[str, Any]:
    """A ToolCallLogManager entry as recorded during a live turn."""
    return {
        "name": name,
        "args_display": args,
        "args_full": args,
        "is_error": is_error,
        "error_message": "Error: failed" if is_error else "",
        "result_display": "ok",
        "result_full": "ok",
    }


def _saved_chat() -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "text": "draw point A", "images": 2},
            {
                "role": "assistant",
                "text": "Created **A**.",
                "tools": [
                    {"name": "create_point", "args": "x: 1, y: 2", "error": False},
                    {"name": "create_label", "args": "", "error": True},
                ],
            },
        ],
        "truncated": 0,
    }


def _children_with_class(parent: Any, class_name: str) -> List[Any]:
    return [child for child in parent.children if class_name in get_class_attr(child).split()]


class _DetachedChatUI(ChatUIManager):
    """ChatUIManager drawing into a detached element instead of the page's chat."""

    def __init__(self) -> None:
        super().__init__(message_menu=MessageMenuManager(), tool_call_log=ToolCallLogManager())
        self.history = html.DIV()

    def _chat_history_element(self) -> Any:
        return self.history

    def render_math(self, root: Optional[Any] = None) -> None:
        pass


class TestChatTranscript(unittest.TestCase):
    def test_records_user_and_assistant_messages(self) -> None:
        transcript = ChatTranscript()
        transcript.record_user("draw A", image_count=1)
        transcript.record_assistant("Done.", [_log_entry("create_point", "x: 1", False)])
        self.assertEqual(
            transcript.messages,
            [
                {"role": "user", "text": "draw A", "images": 1},
                {
                    "role": "assistant",
                    "text": "Done.",
                    "tools": [{"name": "create_point", "args": "x: 1", "error": False}],
                },
            ],
        )

    def test_skips_empty_messages(self) -> None:
        transcript = ChatTranscript()
        transcript.record_user("   ")
        transcript.record_assistant("", [])
        self.assertEqual(transcript.messages, [])

    def test_keeps_tool_only_assistant_turn(self) -> None:
        transcript = ChatTranscript()
        transcript.record_assistant("", [_log_entry("undo")])
        self.assertEqual(transcript.messages[0]["tools"][0]["name"], "undo")

    def test_compact_tool_entries_caps_and_skips_invalid(self) -> None:
        entries: List[Any] = [_log_entry(f"t{i}", "a", i % 2 == 0) for i in range(MAX_SAVED_CHAT_TOOL_ENTRIES + 4)]
        entries.insert(0, {"name": ""})
        compact = ChatTranscript.compact_tool_entries(entries)
        self.assertEqual(len(compact), MAX_SAVED_CHAT_TOOL_ENTRIES - 1)
        self.assertEqual(set(compact[0].keys()), {"name", "args", "error"})
        self.assertTrue(compact[0]["error"])

    def test_messages_returns_copies(self) -> None:
        transcript = ChatTranscript()
        transcript.record_assistant("x", [_log_entry("undo")])
        copy = transcript.messages
        copy[0]["text"] = "changed"
        copy[0]["tools"][0]["name"] = "changed"
        self.assertEqual(transcript.messages[0]["text"], "x")
        self.assertEqual(transcript.messages[0]["tools"][0]["name"], "undo")

    def test_truncates_long_message(self) -> None:
        transcript = ChatTranscript()
        transcript.record_user("z" * (MAX_SAVED_CHAT_MESSAGE_CHARS + 10))
        text = transcript.messages[0]["text"]
        self.assertEqual(len(text), MAX_SAVED_CHAT_MESSAGE_CHARS)
        self.assertTrue(text.endswith(TRUNCATION_MARKER))

    def test_drops_oldest_over_message_cap(self) -> None:
        transcript = ChatTranscript()
        for i in range(MAX_SAVED_CHAT_MESSAGES + 3):
            transcript.record_user(f"m{i}")
        self.assertEqual(len(transcript.messages), MAX_SAVED_CHAT_MESSAGES)
        self.assertEqual(transcript.messages[0]["text"], "m3")
        self.assertEqual(transcript.truncated, 3)

    def test_drops_oldest_over_total_chars_cap(self) -> None:
        transcript = ChatTranscript()
        count = MAX_SAVED_CHAT_TOTAL_CHARS // MAX_SAVED_CHAT_MESSAGE_CHARS + 2
        for i in range(count):
            transcript.record_user(str(i % 10) * MAX_SAVED_CHAT_MESSAGE_CHARS)
        total = sum(len(message["text"]) for message in transcript.messages)
        self.assertLessEqual(total, MAX_SAVED_CHAT_TOTAL_CHARS)
        self.assertEqual(transcript.truncated, count - len(transcript.messages))

    def test_state_round_trip(self) -> None:
        transcript = ChatTranscript()
        transcript.load_state(_saved_chat())
        restored = ChatTranscript()
        restored.load_state(json.loads(json.dumps(transcript.to_state())))
        self.assertEqual(restored.messages, _saved_chat()["messages"])
        self.assertEqual(restored.truncated, 0)

    def test_load_state_skips_invalid_messages(self) -> None:
        transcript = ChatTranscript()
        transcript.load_state(
            {
                "messages": [
                    {"role": "system", "text": "Canvas cleared."},
                    {"role": "user", "text": 5},
                    "junk",
                    {"role": "assistant", "text": "kept", "tools": "junk", "images": True},
                ],
                "truncated": 2,
            }
        )
        self.assertEqual(transcript.messages, [{"role": "assistant", "text": "kept"}])
        self.assertEqual(transcript.truncated, 2)

    def test_load_state_without_chat_empties_transcript(self) -> None:
        transcript = ChatTranscript()
        transcript.record_user("old")
        for value in (None, "chat", {"messages": "x"}):
            transcript.load_state(value)
            self.assertEqual(transcript.messages, [])
            self.assertEqual(transcript.truncated, 0)


class TestChatUIRecordAndRestore(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = _DetachedChatUI()

    def test_print_user_message_records_only_when_asked(self) -> None:
        self.ui.print_user_message("/save x")
        self.ui.print_user_message("draw A", images=["data:image/png;base64,AAAA"], record=True)
        self.assertEqual(self.ui.transcript.messages, [{"role": "user", "text": "draw A", "images": 1}])

    def test_print_ai_message_records_only_when_asked(self) -> None:
        self.ui.print_ai_message("Tests passed.")
        self.ui.print_ai_message("Drew A.", record=True)
        self.assertEqual(self.ui.transcript.messages, [{"role": "assistant", "text": "Drew A."}])

    def test_finalize_stream_records_text_and_tool_entries(self) -> None:
        container = html.DIV()
        content = html.DIV(Class="chat-content")
        container <= content
        self.ui._stream_message_container = container
        self.ui._stream_content_element = content
        self.ui.stream_buffer = "Drew the segment."
        log = self.ui._tool_call_log
        log.element = html.DETAILS()
        log.entries = [_log_entry("create_segment", "p1: A", False)]

        self.ui.finalize_stream()

        self.assertEqual(
            self.ui.transcript.messages,
            [
                {
                    "role": "assistant",
                    "text": "Drew the segment.",
                    "tools": [{"name": "create_segment", "args": "p1: A", "error": False}],
                }
            ],
        )

    def test_restore_transcript_renders_saved_messages(self) -> None:
        self.ui.history <= html.DIV("old message", Class="chat-message normal")

        count = self.ui.restore_transcript(_saved_chat())

        self.assertEqual(count, 2)
        messages = _children_with_class(self.ui.history, "chat-message")
        self.assertEqual(len(messages), 2)
        user_el, ai_el = messages
        self.assertIn("user", get_class_attr(_children_with_class(user_el, "chat-sender")[0]))
        self.assertEqual(len(_children_with_class(user_el, "chat-restored-note")), 1)
        self.assertIn("2 attached images", _children_with_class(user_el, "chat-restored-note")[0].text)

        tool_logs = _children_with_class(ai_el, "tool-call-log-dropdown")
        self.assertEqual(len(tool_logs), 1)
        summary = _children_with_class(tool_logs[0], "tool-call-log-summary")[0]
        self.assertEqual(summary.text, "Used 2 tools (1 failed)")
        entries = _children_with_class(
            _children_with_class(tool_logs[0], "tool-call-log-content")[0], "tool-call-entry"
        )
        self.assertEqual(len(entries), 2)
        self.assertIn("markdown", get_class_attr(_children_with_class(ai_el, "chat-content")[0]))
        self.assertEqual(self.ui.transcript.messages, _saved_chat()["messages"])

    def test_restore_transcript_notes_trimmed_messages(self) -> None:
        chat = _saved_chat()
        chat["truncated"] = 5
        self.ui.restore_transcript(chat)
        notes = _children_with_class(self.ui.history, "chat-restored-note")
        self.assertEqual(len(notes), 1)
        self.assertIn("5 earlier messages", notes[0].text)

    def test_restore_without_chat_empties_the_chat(self) -> None:
        self.ui.print_user_message("current", record=True)
        self.ui.history <= html.DIV("current", Class="chat-message normal")
        self.assertEqual(self.ui.restore_transcript(None), 0)
        self.assertEqual(len(self.ui.history.children), 0)
        self.assertEqual(self.ui.transcript.messages, [])

    def test_clear_chat_forgets_transcript(self) -> None:
        self.ui.print_user_message("q", record=True)
        self.ui.history <= html.DIV("q")
        self.ui.clear_chat()
        self.assertEqual(self.ui.transcript.messages, [])
        self.assertEqual(len(self.ui.history.children), 0)


class TestChatPersistenceManager(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = _DetachedChatUI()
        self.payloads: List[Dict[str, Any]] = []
        self.sync_ok = True

        def send(payload: Dict[str, Any]) -> bool:
            self.payloads.append(payload)
            return self.sync_ok

        self.manager = ChatPersistenceManager(self.ui, get_model_id=lambda: "gpt-test", send_restore_request=send)

    def test_export_returns_transcript_state(self) -> None:
        self.ui.print_user_message("q", record=True)
        self.assertEqual(
            self.manager.export_chat_state(), {"messages": [{"role": "user", "text": "q"}], "truncated": 0}
        )

    def test_restore_replaces_chat_and_restores_server_history(self) -> None:
        message = self.manager.restore_chat_state(_saved_chat())
        self.assertEqual(message, "Restored 2 chat messages.")
        self.assertEqual(len(self.payloads), 1)
        self.assertEqual(self.payloads[0]["ai_model"], "gpt-test")
        self.assertEqual(self.payloads[0]["chat"]["messages"], _saved_chat()["messages"])

    def test_restore_without_chat_starts_fresh(self) -> None:
        self.ui.print_user_message("current", record=True)
        message = self.manager.restore_chat_state(None)
        self.assertIn("no saved chat", message)
        self.assertEqual(self.payloads[0]["chat"], None)
        self.assertEqual(self.ui.transcript.messages, [])

    def test_restore_reports_server_failure(self) -> None:
        self.sync_ok = False
        message = self.manager.restore_chat_state(_saved_chat())
        self.assertIn("could not be restored", message)
        self.assertEqual(len(self.ui.transcript.messages), 2, "the chat is still shown")


class _FakeCanvas:
    def get_canvas_state(self) -> Dict[str, Any]:
        return {"Points": []}


class _FakeChatPersistence:
    def __init__(self) -> None:
        self.restored: List[Any] = []

    def export_chat_state(self) -> Dict[str, Any]:
        return {"messages": [{"role": "user", "text": "q"}], "truncated": 0}

    def restore_chat_state(self, chat: Any) -> str:
        self.restored.append(chat)
        return "Restored 2 chat messages."


class _FakeRequest:
    def __init__(self, response: Dict[str, Any]) -> None:
        self.text = json.dumps(response)


class TestWorkspaceManagerChat(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = WorkspaceManager(_FakeCanvas())  # type: ignore[arg-type]
        self.restored_states: List[Any] = []
        self.manager._restore_workspace_state = self.restored_states.append  # type: ignore[method-assign]
        self.chat = _FakeChatPersistence()

    def _load(self, name: Optional[str], restore_chat: bool, chat: Any) -> str:
        response = {"status": "success", "data": {"state": {"Points": []}, "chat": chat}}
        return self.manager._parse_load_workspace_response(_FakeRequest(response), name, restore_chat)

    def test_save_payload_includes_chat(self) -> None:
        self.manager.set_chat_persistence(self.chat)  # type: ignore[arg-type]
        payload = self.manager._build_save_workspace_payload("ws")
        self.assertEqual(payload["chat"], self.chat.export_chat_state())
        self.assertEqual(payload["state"], {"Points": []})

    def test_save_payload_without_chat_persistence(self) -> None:
        self.assertNotIn("chat", self.manager._build_save_workspace_payload("ws"))

    def test_user_load_restores_chat(self) -> None:
        self.manager.set_chat_persistence(self.chat)  # type: ignore[arg-type]
        result = self._load("ws", True, _saved_chat())
        self.assertEqual(result, 'Workspace "ws" loaded successfully. Restored 2 chat messages.')
        self.assertEqual(self.chat.restored, [_saved_chat()])
        self.assertEqual(len(self.restored_states), 1)

    def test_user_load_of_old_workspace_restores_empty_chat(self) -> None:
        self.manager.set_chat_persistence(self.chat)  # type: ignore[arg-type]
        self._load("old", True, None)
        self.assertEqual(self.chat.restored, [None])

    def test_ai_load_keeps_chat_and_mentions_saved_one(self) -> None:
        self.manager.set_chat_persistence(self.chat)  # type: ignore[arg-type]
        result = self._load("ws", False, _saved_chat())
        self.assertEqual(self.chat.restored, [])
        self.assertTrue(result.startswith('Workspace "ws" loaded successfully.'))
        self.assertIn("saved chat (2 messages) was not restored", result)
        self.assertIn("/load ws", result)

    def test_ai_load_without_saved_chat_has_plain_message(self) -> None:
        self.manager.set_chat_persistence(self.chat)  # type: ignore[arg-type]
        self.assertEqual(self._load(None, False, None), 'Workspace "current" loaded successfully.')

    def test_failed_load_does_not_touch_chat(self) -> None:
        self.manager.set_chat_persistence(self.chat)  # type: ignore[arg-type]
        request = _FakeRequest({"status": "error", "message": "No workspace found"})
        result = self.manager._parse_load_workspace_response(request, "missing", True)
        self.assertTrue(result.startswith("Error loading workspace"))
        self.assertEqual(self.chat.restored, [])
        self.assertEqual(self.restored_states, [])

    def test_sync_request_runs_on_complete_once(self) -> None:
        """A synchronous request completes inside send(); the load must not restore twice."""
        import workspace_manager as workspace_manager_module

        class _SyncAjax:
            def __init__(self) -> None:
                self.handlers: Dict[str, Any] = {}
                self.text = '{"status": "success", "data": {"state": {"Points": []}, "chat": null}}'

            def bind(self, event: str, handler: Any) -> None:
                self.handlers[event] = handler

            def open(self, method: str, url: str, is_async: bool) -> None:
                pass

            def set_header(self, name: str, value: str) -> None:
                pass

            def send(self, data: Any = None) -> None:
                if "complete" in self.handlers:
                    self.handlers["complete"](self)

        class _FakeAjaxModule:
            Ajax = _SyncAjax

        calls: List[Any] = []
        original_ajax = workspace_manager_module.ajax
        workspace_manager_module.ajax = _FakeAjaxModule()
        try:
            result = self.manager._execute_sync_request(
                "GET", "/load_workspace", lambda req: calls.append(req) or "ok", "Error"
            )
        finally:
            workspace_manager_module.ajax = original_ajax
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)
