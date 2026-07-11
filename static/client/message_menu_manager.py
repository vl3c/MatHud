"""Message menu manager for the AI interface.

Manages the per-message context menu (copy, TTS, etc.) that appears on
chat messages. Handles global click-to-close behaviour, clipboard
operations, and menu item construction.

Extracted from ``AIInterface`` to reduce god-class complexity while
preserving the identical public behaviour.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from browser import document, html, window


class MessageMenuManager:
    """Manages per-message context menus in the chat interface.

    Attributes:
        _open_menu: The currently visible menu DOM element, or ``None``.
        _global_bound: Whether the document-level click handler has been
            registered (to avoid duplicate bindings).
    """

    def __init__(
        self,
        on_read_aloud: Optional[Callable[[str, Any], None]] = None,
        on_tts_settings: Optional[Callable[[], None]] = None,
    ) -> None:
        self._open_menu: Optional[Any] = None
        self._global_bound: bool = False
        self._on_read_aloud = on_read_aloud
        self._on_tts_settings = on_tts_settings

    # ── Raw text storage ────────────────────────────────────────

    def set_raw_text(self, container: Any, text: str) -> None:
        """Attach raw message source text to a container for later actions (copy, etc.)."""
        try:
            setattr(container, "_raw_message_text", text)
        except Exception:
            pass

    def get_raw_text(self, container: Any) -> str:
        """Return the stored raw message text from the container, or empty string if missing."""
        try:
            value = getattr(container, "_raw_message_text", "")
            if isinstance(value, str):
                return value
            return str(value)
        except Exception:
            return ""

    # ── Clipboard ───────────────────────────────────────────────

    def copy_to_clipboard(self, text: str) -> bool:
        """Copy text to clipboard using the modern API with a fallback for older contexts."""
        if text is None:
            text = ""
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                text = ""

        # Prefer navigator.clipboard when available (may require secure context).
        try:
            navigator = getattr(window, "navigator", None)
            clipboard = getattr(navigator, "clipboard", None) if navigator is not None else None
            write_text = getattr(clipboard, "writeText", None) if clipboard is not None else None
            if callable(write_text):
                write_text(text)
                return True
        except Exception:
            pass

        # Fallback: temporary textarea + execCommand('copy')
        try:
            textarea = html.TEXTAREA()
            textarea.value = text
            textarea.attrs["readonly"] = "readonly"
            textarea.style.position = "fixed"
            textarea.style.left = "0"
            textarea.style.top = "0"
            textarea.style.opacity = "0"

            # Append to DOM, select content, copy, then remove.
            document <= textarea
            try:
                textarea.focus()
            except Exception:
                pass
            try:
                textarea.select()
            except Exception:
                pass
            try:
                textarea.setSelectionRange(0, len(text))
            except Exception:
                pass

            copied = False
            try:
                copied = bool(window.document.execCommand("copy"))
            except Exception:
                try:
                    copied = bool(document.execCommand("copy"))
                except Exception:
                    copied = False

            try:
                textarea.remove()
            except Exception:
                pass

            return copied
        except Exception:
            return False

    # ── Global handlers ─────────────────────────────────────────

    def bind_global_handlers(self) -> None:
        """Bind global document handlers needed for message menus (close on outside click)."""
        if self._global_bound:
            return
        try:
            document.bind("click", self._on_document_click)
            self._global_bound = True
        except Exception:
            self._global_bound = False

    def _on_document_click(self, _event: Any) -> None:
        """Close any open message menu when clicking outside of it."""
        try:
            if self._open_menu is not None:
                self._hide(self._open_menu)
        except Exception:
            self._open_menu = None

    # ── Show / hide / toggle ────────────────────────────────────

    def _hide(self, menu: Any) -> None:
        try:
            menu.style.display = "none"
        except Exception:
            pass
        if self._open_menu is menu:
            self._open_menu = None

    def _show(self, menu: Any) -> None:
        try:
            if self._open_menu is not None and self._open_menu is not menu:
                self._hide(self._open_menu)
        except Exception:
            self._open_menu = None

        try:
            menu.style.display = "block"
        except Exception:
            pass
        self._open_menu = menu

    def _toggle(self, menu: Any) -> None:
        try:
            current_display = getattr(menu.style, "display", "")
        except Exception:
            current_display = ""

        if current_display == "none" or not current_display:
            self._show(menu)
        else:
            self._hide(menu)

    # ── Attach menu to a message container ──────────────────────

    def attach(self, container: Any, is_ai_message: bool = False) -> None:
        """Attach the per-message '...' menu to the message container (idempotent).

        Args:
            container: The DOM element to attach the menu to
            is_ai_message: Whether this is an AI message (enables TTS option)
        """
        try:
            if bool(getattr(container, "_has_message_menu", False)):
                return
            setattr(container, "_has_message_menu", True)
        except Exception:
            # If we cannot track state on the element, continue anyway.
            pass

        self.bind_global_handlers()

        menu_button = html.BUTTON("...", Class="chat-message-menu-button")
        try:
            menu_button.attrs["type"] = "button"
            menu_button.attrs["title"] = "Message options"
            menu_button.attrs["aria-label"] = "Message options"
        except Exception:
            pass

        menu = html.DIV(Class="chat-message-menu")
        try:
            menu.style.display = "none"
        except Exception:
            pass

        copy_item = html.BUTTON("Copy message text", Class="chat-message-menu-item")
        try:
            copy_item.attrs["type"] = "button"
        except Exception:
            pass

        def _stop_propagation(ev: Any) -> None:
            try:
                ev.stopPropagation()
            except Exception:
                pass

        def _on_menu_button_click(ev: Any) -> None:
            _stop_propagation(ev)
            self._toggle(menu)

        def _on_menu_click(ev: Any) -> None:
            _stop_propagation(ev)

        def _on_copy_click(ev: Any) -> None:
            _stop_propagation(ev)
            raw_text = self.get_raw_text(container)
            self.copy_to_clipboard(raw_text)
            self._hide(menu)

        try:
            menu_button.bind("click", _on_menu_button_click)
            menu.bind("click", _on_menu_click)
            copy_item.bind("click", _on_copy_click)
        except Exception:
            pass

        menu <= copy_item

        # Add TTS options for AI messages when callbacks are provided
        if is_ai_message and self._on_read_aloud is not None:
            read_aloud_item = html.BUTTON("Read aloud", Class="chat-message-menu-item tts-read-aloud")
            try:
                read_aloud_item.attrs["type"] = "button"
            except Exception:
                pass

            def _on_read_aloud_click(ev: Any) -> None:
                _stop_propagation(ev)
                self._hide(menu)
                raw_text = self.get_raw_text(container)
                if self._on_read_aloud is not None:
                    self._on_read_aloud(raw_text, read_aloud_item)

            try:
                read_aloud_item.bind("click", _on_read_aloud_click)
            except Exception:
                pass

            menu <= read_aloud_item

            # TTS settings option
            if self._on_tts_settings is not None:
                tts_settings_item = html.BUTTON("TTS settings...", Class="chat-message-menu-item")
                try:
                    tts_settings_item.attrs["type"] = "button"
                except Exception:
                    pass

                def _on_tts_settings_click(ev: Any) -> None:
                    _stop_propagation(ev)
                    self._hide(menu)
                    if self._on_tts_settings is not None:
                        self._on_tts_settings()

                try:
                    tts_settings_item.bind("click", _on_tts_settings_click)
                except Exception:
                    pass

                menu <= tts_settings_item

        # Add button + menu to the message container (positioned by CSS).
        try:
            container <= menu_button
            container <= menu
        except Exception:
            pass
