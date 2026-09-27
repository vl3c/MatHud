"""Chat persistence for workspaces.

Connects the chat to workspace save and load:

- ``export_chat_state`` returns the transcript saved with a workspace.
- ``restore_chat_state`` replaces the chat with a saved transcript and asks the
  server (``POST /restore_conversation``) to rebuild the AI conversation history
  from it, so the assistant can continue the restored conversation.

``WorkspaceManager`` calls ``restore_chat_state`` only for loads the user asks
for (``/load``). A ``load_workspace`` tool call from the AI in the middle of a
turn restores the canvas only; replacing the chat and the server history then
would break the running turn.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from browser import ajax

if TYPE_CHECKING:
    from chat_ui_manager import ChatUIManager

RestoreRequestSender = Callable[[Dict[str, Any]], bool]


class ChatPersistenceManager:
    """Saves the chat transcript with workspaces and restores it on load.

    Attributes:
        _chat_ui: Chat UI that owns the transcript and the chat DOM.
        _get_model_id: Returns the selected AI model, so the server restores the
            history of the provider the user continues with.
        _send_restore_request: Sends the restore payload; returns True on success.
    """

    def __init__(
        self,
        chat_ui: "ChatUIManager",
        get_model_id: Callable[[], str],
        send_restore_request: Optional[RestoreRequestSender] = None,
    ) -> None:
        self._chat_ui = chat_ui
        self._get_model_id = get_model_id
        self._send_restore_request: RestoreRequestSender = send_restore_request or self._post_restore_conversation

    def export_chat_state(self) -> Dict[str, Any]:
        """Return the chat transcript to save with a workspace."""
        return self._chat_ui.transcript.to_state()

    def restore_chat_state(self, chat_state: Any) -> str:
        """Replace the chat and the AI conversation with a saved chat.

        Args:
            chat_state: The saved chat, or ``None`` for a workspace saved without
                one (the chat and the conversation then start empty).

        Returns:
            A short sentence for the load result message.
        """
        count = self._chat_ui.restore_transcript(chat_state)
        synced = self._send_restore_request(self._build_restore_payload())
        return self._describe_restore(count, synced)

    def _build_restore_payload(self) -> Dict[str, Any]:
        transcript = self._chat_ui.transcript
        has_chat = bool(transcript.messages) or transcript.truncated > 0
        return {
            "chat": transcript.to_state() if has_chat else None,
            "ai_model": self._safe_model_id(),
        }

    def _safe_model_id(self) -> Optional[str]:
        try:
            model_id = self._get_model_id()
        except Exception:
            return None
        return model_id if isinstance(model_id, str) and model_id else None

    @staticmethod
    def _describe_restore(count: int, synced: bool) -> str:
        if count == 0:
            summary = "The workspace has no saved chat; the conversation starts fresh."
        else:
            noun = "message" if count == 1 else "messages"
            summary = f"Restored {count} chat {noun}."
        if not synced:
            summary += " The AI conversation history could not be restored."
        return summary

    @staticmethod
    def _post_restore_conversation(payload: Dict[str, Any]) -> bool:
        """POST the payload synchronously, like the other workspace requests."""
        try:
            req: Any = ajax.Ajax()
            req.open("POST", "/restore_conversation", False)
            req.set_header("Content-Type", "application/json")
            req.send(json.dumps(payload))
            return bool(req.status == 200)
        except Exception as e:
            print(f"Error restoring the AI conversation: {e}")
            return False
