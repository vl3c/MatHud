"""Chat transcript kept for saving with workspaces.

Records the user and assistant messages of the AI conversation as they are
shown in the chat, as raw (markdown) text. Slash commands, system notes and
test output are not part of the conversation and are not recorded.

The transcript is saved with a workspace (``to_state``) and restored from one
(``load_state``). Attached images are not saved; a user message only counts
them so the chat can show a placeholder. Assistant messages keep compact
tool-call entries (name, short arguments, error flag). The oldest messages are
dropped when the transcript exceeds the caps in ``constants.py``, and
``truncated`` counts them. The server applies the same caps again
(``static/workspace_chat.py``).

This module does not import ``browser``; its logic is plain Python.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from constants import (
    MAX_SAVED_CHAT_MESSAGE_CHARS,
    MAX_SAVED_CHAT_MESSAGES,
    MAX_SAVED_CHAT_TOOL_ENTRIES,
    MAX_SAVED_CHAT_TOTAL_CHARS,
)

TRUNCATION_MARKER = " [...]"
CHAT_ROLES = ("user", "assistant")


class ChatTranscript:
    """Ordered user/assistant messages of the conversation shown in the chat.

    Each message is a dict ``{"role", "text"}`` with optional ``images`` (count)
    and ``tools`` (compact entries ``{"name", "args", "error"}``).
    """

    def __init__(self) -> None:
        self._messages: List[Dict[str, Any]] = []
        self._truncated: int = 0

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Return copies of the recorded messages, oldest first."""
        return [self._copy_message(message) for message in self._messages]

    @property
    def truncated(self) -> int:
        """Return how many older messages were dropped to respect the caps."""
        return self._truncated

    def record_user(self, text: str, image_count: int = 0) -> None:
        """Record a user message sent to the AI, with the number of attached images."""
        message: Dict[str, Any] = {"role": "user", "text": text}
        if image_count > 0:
            message["images"] = image_count
        self._append(message)

    def record_assistant(self, text: str, tool_entries: Optional[List[Dict[str, Any]]] = None) -> None:
        """Record an assistant message with the tool-call log entries of its turn."""
        message: Dict[str, Any] = {"role": "assistant", "text": text}
        tools = self.compact_tool_entries(tool_entries or [])
        if tools:
            message["tools"] = tools
        self._append(message)

    def clear(self) -> None:
        """Forget every message (a new conversation)."""
        self._messages = []
        self._truncated = 0

    def to_state(self) -> Dict[str, Any]:
        """Return the JSON-ready chat saved with a workspace."""
        return {"messages": self.messages, "truncated": self._truncated}

    def load_state(self, state: Any) -> None:
        """Replace the transcript with a saved chat; anything that is not a chat empties it."""
        self.clear()
        if not isinstance(state, dict):
            return
        raw_messages = state.get("messages")
        if isinstance(raw_messages, list):
            for raw in raw_messages:
                message = self._message_from_state(raw)
                if message is not None:
                    self._messages.append(message)
        self._truncated = _non_negative_int(state.get("truncated"))
        self._enforce_caps()

    @staticmethod
    def compact_tool_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reduce tool-call log entries (see ToolCallLogManager) to name, short args and error flag."""
        compact: List[Dict[str, Any]] = []
        for entry in entries[:MAX_SAVED_CHAT_TOOL_ENTRIES]:
            name = entry.get("name") if isinstance(entry, dict) else None
            if not isinstance(name, str) or not name:
                continue
            args = entry.get("args_display", "")
            compact.append(
                {
                    "name": name,
                    "args": args if isinstance(args, str) else "",
                    "error": entry.get("is_error") is True,
                }
            )
        return compact

    # ── Private helpers ─────────────────────────────────────────

    def _append(self, message: Dict[str, Any]) -> None:
        if not self._has_content(message):
            return
        message["text"] = _truncate(message["text"], MAX_SAVED_CHAT_MESSAGE_CHARS)
        self._messages.append(message)
        self._enforce_caps()

    def _enforce_caps(self) -> None:
        while self._messages and (
            len(self._messages) > MAX_SAVED_CHAT_MESSAGES or self._total_chars() > MAX_SAVED_CHAT_TOTAL_CHARS
        ):
            self._messages.pop(0)
            self._truncated += 1

    def _total_chars(self) -> int:
        return sum(len(message["text"]) for message in self._messages)

    def _message_from_state(self, raw: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict) or raw.get("role") not in CHAT_ROLES or not isinstance(raw.get("text"), str):
            return None
        message: Dict[str, Any] = {
            "role": raw["role"],
            "text": _truncate(raw["text"], MAX_SAVED_CHAT_MESSAGE_CHARS),
        }
        images = _non_negative_int(raw.get("images"))
        if images:
            message["images"] = images
        tools = self._tools_from_state(raw.get("tools"))
        if tools:
            message["tools"] = tools
        return message if self._has_content(message) else None

    @staticmethod
    def _tools_from_state(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        tools: List[Dict[str, Any]] = []
        for item in raw[:MAX_SAVED_CHAT_TOOL_ENTRIES]:
            if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"]:
                args = item.get("args")
                tools.append(
                    {
                        "name": item["name"],
                        "args": args if isinstance(args, str) else "",
                        "error": item.get("error") is True,
                    }
                )
        return tools

    @staticmethod
    def _has_content(message: Dict[str, Any]) -> bool:
        return bool(message["text"].strip() or message.get("images") or message.get("tools"))

    @staticmethod
    def _copy_message(message: Dict[str, Any]) -> Dict[str, Any]:
        copied = dict(message)
        if "tools" in copied:
            copied["tools"] = [dict(entry) for entry in copied["tools"]]
        return copied


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(TRUNCATION_MARKER))] + TRUNCATION_MARKER


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
