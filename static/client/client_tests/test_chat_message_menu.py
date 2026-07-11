from __future__ import annotations

import unittest
from typing import Any, Optional

from browser import html, window

from message_menu_manager import MessageMenuManager
from .simple_mock import SimpleMock, get_class_attr as _get_class_attr


class TestChatMessageMenu(unittest.TestCase):
    def test_copy_message_text_uses_raw_source(self) -> None:
        # Create a MessageMenuManager instance without TTS callbacks.
        mgr = MessageMenuManager()

        copy_mock = SimpleMock(return_value=True)
        mgr.copy_to_clipboard = copy_mock

        container = html.DIV()
        raw_text = "Hello \\(x^2\\)"
        mgr.set_raw_text(container, raw_text)
        mgr.attach(container)

        menu_button: Optional[Any] = None
        menu: Optional[Any] = None
        for child in getattr(container, "children", []):
            cls = _get_class_attr(child)
            if cls == "chat-message-menu-button":
                menu_button = child
            elif cls == "chat-message-menu":
                menu = child

        self.assertIsNotNone(menu_button)
        self.assertIsNotNone(menu)

        # Show the menu (prefer actual click dispatch; fall back to forcing visibility).
        try:
            menu_button.click()
        except Exception:
            try:
                menu.style.display = "block"
            except Exception:
                pass

        copy_item: Optional[Any] = None
        for child in getattr(menu, "children", []):
            if _get_class_attr(child) == "chat-message-menu-item":
                copy_item = child
                break

        self.assertIsNotNone(copy_item)

        # Click copy item.
        try:
            copy_item.click()
        except Exception:
            try:
                evt = window.MouseEvent.new("click")
                copy_item.dispatchEvent(evt)
            except Exception:
                pass

        copy_mock.assert_called_once_with(raw_text)
