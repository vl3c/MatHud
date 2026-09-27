"""
MatHud Workspace Chat Persistence

Validates the chat transcript saved with a workspace and rebuilds a compact AI
conversation history from it when the workspace is loaded.

A saved chat is a JSON object stored next to the canvas ``state`` in the
workspace record::

    {
        "messages": [
            {"role": "user", "text": "draw a circle", "images": 1},
            {"role": "assistant", "text": "Done.", "tools": [{"name": "create_circle", "args": "...", "error": false}]}
        ],
        "truncated": 0
    }

Only user and assistant messages are kept, as raw (markdown) text. Attached
images are not saved: ``images`` counts them so the chat can show a placeholder.
``tools`` holds compact tool-call entries for display. ``truncated`` counts the
older messages dropped to respect the size caps in ``static/config.py``.

The provider-side history is not saved. On load it is rebuilt from the saved
messages as plain user/assistant text turns (see ``build_restored_history``).
"""

from __future__ import annotations

from typing import List, NotRequired, Optional, Sequence, TypedDict

from static.config import (
    MAX_RESTORED_HISTORY_CHARS,
    MAX_RESTORED_HISTORY_MESSAGE_CHARS,
    MAX_RESTORED_HISTORY_MESSAGES,
    MAX_SAVED_CHAT_MESSAGE_CHARS,
    MAX_SAVED_CHAT_MESSAGES,
    MAX_SAVED_CHAT_TOOL_ARGS_CHARS,
    MAX_SAVED_CHAT_TOOL_ENTRIES,
    MAX_SAVED_CHAT_TOOL_NAME_CHARS,
    MAX_SAVED_CHAT_TOTAL_CHARS,
)

CHAT_ROLES = ("user", "assistant")
TRUNCATION_MARKER = " [...]"
OMITTED_HISTORY_NOTE = "(Earlier messages of this conversation were omitted.)"


class ChatToolEntry(TypedDict):
    name: str
    args: str
    error: bool


class ChatMessageRecord(TypedDict):
    role: str
    text: str
    images: NotRequired[int]
    tools: NotRequired[List[ChatToolEntry]]


class ChatRecord(TypedDict):
    messages: List[ChatMessageRecord]
    truncated: int


class HistoryTurn(TypedDict):
    role: str
    content: str


def sanitize_chat_record(raw: object) -> Optional[ChatRecord]:
    """Return a validated, size-capped copy of a saved chat, or None when it is not one.

    Invalid messages are skipped. When the chat is over the caps, the oldest
    messages are dropped and counted in ``truncated``.
    """
    if not isinstance(raw, dict):
        return None
    raw_messages = raw.get("messages")
    if not isinstance(raw_messages, list):
        return None
    messages = [message for message in map(_sanitize_message, raw_messages) if message is not None]
    kept = _drop_oldest_over_caps(messages)
    dropped = len(messages) - len(kept)
    return {"messages": kept, "truncated": _non_negative_int(raw.get("truncated")) + dropped}


def build_restored_history(chat: Optional[ChatRecord]) -> List[HistoryTurn]:
    """Rebuild a plain-text AI conversation history from a saved chat.

    Each saved message becomes one user or assistant text turn; consecutive turns
    of the same role are merged so roles alternate. Only the newest turns that fit
    the restored-history caps are kept. The history starts with a user turn and
    ends with an assistant turn, so the next live user message follows naturally.
    """
    if chat is None:
        return []
    turns = _merge_consecutive_roles([turn for turn in map(_history_turn, chat["messages"]) if turn["content"]])
    kept = _keep_newest_within_history_caps(turns)
    omitted = chat["truncated"] > 0 or len(kept) < len(turns)
    kept = _trim_to_user_first_assistant_last(kept)
    if omitted and kept:
        kept[0] = {"role": "user", "content": f"{OMITTED_HISTORY_NOTE}\n\n{kept[0]['content']}"}
    return kept


def _sanitize_message(raw: object) -> Optional[ChatMessageRecord]:
    if not isinstance(raw, dict):
        return None
    role = raw.get("role")
    text = raw.get("text")
    if role not in CHAT_ROLES or not isinstance(text, str):
        return None
    message: ChatMessageRecord = {"role": str(role), "text": _truncate(text, MAX_SAVED_CHAT_MESSAGE_CHARS)}
    images = _non_negative_int(raw.get("images"))
    if images:
        message["images"] = images
    tools = _sanitize_tool_entries(raw.get("tools"))
    if tools:
        message["tools"] = tools
    if not message["text"].strip() and not images and not tools:
        return None
    return message


def _sanitize_tool_entries(raw: object) -> List[ChatToolEntry]:
    if not isinstance(raw, list):
        return []
    entries: List[ChatToolEntry] = []
    for item in raw[:MAX_SAVED_CHAT_TOOL_ENTRIES]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
            continue
        args = item.get("args")
        entries.append(
            {
                "name": _truncate(item["name"], MAX_SAVED_CHAT_TOOL_NAME_CHARS),
                "args": _truncate(args, MAX_SAVED_CHAT_TOOL_ARGS_CHARS) if isinstance(args, str) else "",
                "error": item.get("error") is True,
            }
        )
    return entries


def _drop_oldest_over_caps(messages: List[ChatMessageRecord]) -> List[ChatMessageRecord]:
    kept = messages[-MAX_SAVED_CHAT_MESSAGES:]
    total = sum(len(message["text"]) for message in kept)
    start = 0
    while total > MAX_SAVED_CHAT_TOTAL_CHARS and start < len(kept):
        total -= len(kept[start]["text"])
        start += 1
    return kept[start:]


def _history_turn(message: ChatMessageRecord) -> HistoryTurn:
    text = message["text"].strip()
    if message["role"] == "user":
        return {"role": "user", "content": _with_image_note(text, message.get("images", 0))}
    return {"role": "assistant", "content": text or _tools_only_note(message.get("tools", []))}


def _with_image_note(text: str, images: int) -> str:
    if not images:
        return text
    noun = "image was" if images == 1 else "images were"
    note = f"({images} attached {noun} not kept in the saved conversation.)"
    return f"{text}\n{note}" if text else note


def _tools_only_note(tools: Sequence[ChatToolEntry]) -> str:
    if not tools:
        return ""
    names = ", ".join(dict.fromkeys(entry["name"] for entry in tools))
    return f"(Used tools: {names}.)"


def _merge_consecutive_roles(turns: List[HistoryTurn]) -> List[HistoryTurn]:
    merged: List[HistoryTurn] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1] = {"role": turn["role"], "content": f"{merged[-1]['content']}\n\n{turn['content']}"}
        else:
            merged.append({"role": turn["role"], "content": turn["content"]})
    return merged


def _keep_newest_within_history_caps(turns: List[HistoryTurn]) -> List[HistoryTurn]:
    kept: List[HistoryTurn] = []
    total = 0
    for turn in reversed(turns[-MAX_RESTORED_HISTORY_MESSAGES:]):
        content = _truncate(turn["content"], MAX_RESTORED_HISTORY_MESSAGE_CHARS)
        if total + len(content) > MAX_RESTORED_HISTORY_CHARS:
            break
        total += len(content)
        kept.append({"role": turn["role"], "content": content})
    kept.reverse()
    return kept


def _trim_to_user_first_assistant_last(turns: List[HistoryTurn]) -> List[HistoryTurn]:
    start = 0
    while start < len(turns) and turns[start]["role"] != "user":
        start += 1
    end = len(turns)
    while end > start and turns[end - 1]["role"] != "assistant":
        end -= 1
    return turns[start:end]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(TRUNCATION_MARKER))] + TRUNCATION_MARKER


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
