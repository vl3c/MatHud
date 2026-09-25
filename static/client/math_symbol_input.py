"""
MatHud Math Symbol Input

Wires math-symbol entry into the chat input:

    - the Σ palette button and popover (MathSymbolPalette), also toggled with Ctrl+Up
    - Alt shortcuts (Alt+P inserts π, Alt+2 inserts ², ...)
    - backslash completion: typing ``\\alp`` offers α (SymbolCompletionPopup);
      Tab or Enter accepts, and a space after an exact name converts it

Every insertion replaces the selection at the caret, keeps focus in the input
and, where the browser supports it, goes through ``insertText`` so Ctrl+Z
undoes it.

Coexistence with the slash-command autocomplete: this handler is bound after
CommandAutocomplete and ignores keydown events that one already handled
(``defaultPrevented``); backslash completion is off while the input starts
with "/". Enter keeps sending the message unless a symbol popup is using it.

Dependencies:
    - browser: DOM events and localStorage
    - math_symbols: symbol table and pure editing helpers
    - math_symbol_palette, symbol_completion_popup: the two popups
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from browser import document, window

from math_symbol_palette import MathSymbolPalette
from math_symbols import (
    MathSymbol,
    exact_latex_match,
    find_latex_token,
    index_to_utf16_offset,
    insert_text,
    lookup_alt_shortcut,
    sanitize_recent,
    search_symbols,
    update_recent,
    utf16_offset_to_index,
)
from symbol_completion_popup import SymbolCompletionPopup

RECENT_STORAGE_KEY: str = "mathud.symbols.recent"
_ARROW_KEYS: Tuple[str, ...] = ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown")
_CARET_KEYS: Tuple[str, ...] = ("ArrowLeft", "ArrowRight", "Home", "End")


class MathSymbolInput:
    """Symbol palette, Alt shortcuts and backslash completion for the chat input."""

    def __init__(self, input_element: Any, button_element: Any, container: Any) -> None:
        """Attach to the chat input.

        Args:
            input_element: The chat ``<input>``.
            button_element: The Σ palette button.
            container: The positioned chat input container that hosts the popups.
        """
        self._input: Any = input_element
        self._recent: List[str] = self._load_recent()
        self._right_alt_down: bool = False
        self.palette = MathSymbolPalette(input_element, button_element, container, self._on_palette_pick)
        self.palette.set_recent(self._recent)
        self.completion = SymbolCompletionPopup(container, self._accept_completion)
        self._bind_events()

    @classmethod
    def attach(cls) -> Optional["MathSymbolInput"]:
        """Attach to the page's chat input, or return None if the markup is missing."""
        if "chat-input" not in document or "symbol-palette-button" not in document:
            return None
        container = document.querySelector(".chat-input-container")
        if container is None:
            return None
        return cls(document["chat-input"], document["symbol-palette-button"], container)

    # ----- insertion -----

    def insert(self, text: str, start: Optional[int] = None, end: Optional[int] = None) -> None:
        """Insert ``text`` at the caret, replacing the selection or ``start..end``.

        ``start`` and ``end`` are string indices; by default the current
        selection is replaced. Focus returns to the input with the caret after
        the inserted text.
        """
        value = str(self._input.value)
        if start is None or end is None:
            start, end = self._selection_indices(value)
        expected, caret = insert_text(value, start, end, text)
        self._input.focus()
        self._input.setSelectionRange(index_to_utf16_offset(value, start), index_to_utf16_offset(value, end))
        if not self._exec_insert_text(text) or str(self._input.value) != expected:
            self._input.value = expected
            caret_offset = index_to_utf16_offset(expected, caret)
            self._input.setSelectionRange(caret_offset, caret_offset)
            self._dispatch_input_event()

    def _exec_insert_text(self, text: str) -> bool:
        """Insert through the browser's editing command so the edit is undoable."""
        try:
            return bool(document.execCommand("insertText", False, text))
        except Exception:
            return False

    def _dispatch_input_event(self) -> None:
        try:
            self._input.dispatchEvent(window.Event.new("input", {"bubbles": True}))
        except Exception:
            pass

    def _selection_indices(self, value: str) -> Tuple[int, int]:
        start = self._input.selectionStart
        end = self._input.selectionEnd
        if start is None or end is None:
            return len(value), len(value)
        return utf16_offset_to_index(value, int(start)), utf16_offset_to_index(value, int(end))

    def _insert_symbol(self, entry: MathSymbol, start: Optional[int] = None, end: Optional[int] = None) -> None:
        self.insert(entry.insert, start, end)
        self._remember(entry.symbol)

    # ----- recent symbols -----

    def _remember(self, symbol: str) -> None:
        self._recent = update_recent(self._recent, symbol)
        self.palette.set_recent(self._recent)
        try:
            window.localStorage.setItem(RECENT_STORAGE_KEY, " ".join(self._recent))
        except Exception:
            pass

    def _load_recent(self) -> List[str]:
        try:
            stored = window.localStorage.getItem(RECENT_STORAGE_KEY)
        except Exception:
            return []
        if not stored:
            return []
        return sanitize_recent(str(stored).split(" "))

    # ----- events -----

    def _bind_events(self) -> None:
        self._input.bind("keydown", self._on_keydown)
        self._input.bind("keyup", self._on_keyup)
        self._input.bind("input", self._on_input)
        self._input.bind("click", self._on_caret_moved)
        self._input.bind("blur", self._on_blur)

    def _on_keydown(self, event: Any) -> None:
        try:
            if event.defaultPrevented or event.isComposing:
                return
            if event.code == "AltRight":
                self._right_alt_down = True
            if self._handle_completion_key(event):
                return
            if self._handle_space_conversion(event):
                return
            if self._handle_palette_key(event):
                return
            self._handle_alt_shortcut(event)
        except Exception as e:
            print(f"Error handling math symbol keydown: {e}")

    def _on_keyup(self, event: Any) -> None:
        if event.code == "AltRight":
            self._right_alt_down = False
        if event.key in _CARET_KEYS and not self.palette.visible:
            self._refresh_completion()

    def _on_input(self, event: Any) -> None:
        self._refresh_completion()

    def _on_caret_moved(self, event: Any) -> None:
        self._refresh_completion()

    def _on_blur(self, event: Any) -> None:
        self._right_alt_down = False
        window.setTimeout(self._hide_completion_if_unfocused, 150)

    def _hide_completion_if_unfocused(self) -> None:
        if not self._input.isSameNode(document.activeElement):
            self.completion.hide()

    # ----- backslash completion -----

    def _current_token(self) -> Optional[Tuple[int, int, str]]:
        """Return (start, caret, name) of the ``\\name`` token before the caret."""
        value = str(self._input.value)
        if value.startswith("/"):
            return None  # slash commands own the input
        start, end = self._selection_indices(value)
        if start != end:
            return None
        token = find_latex_token(value, end)
        if token is None:
            return None
        return token[0], end, token[1]

    def _refresh_completion(self) -> None:
        token = self._current_token()
        if token is None:
            self.completion.hide()
            return
        self.completion.show(search_symbols(token[2]))
        if self.completion.visible:
            self.palette.close()  # both popups share the space above the input

    def _accept_completion(self, entry: MathSymbol) -> None:
        token = self._current_token()
        self.completion.hide()
        if token is None:
            return
        self._insert_symbol(entry, token[0], token[1])

    def _handle_completion_key(self, event: Any) -> bool:
        if not self.completion.visible or event.ctrlKey or event.altKey or event.metaKey:
            return False
        key = event.key
        if key == "ArrowDown":
            self.completion.select_next()
        elif key == "ArrowUp":
            self.completion.select_previous()
        elif key in ("Enter", "Tab"):
            entry = self.completion.selected_entry()
            if entry is None:
                return False
            self._accept_completion(entry)
        elif key == "Escape":
            self.completion.hide()
        else:
            return False
        event.preventDefault()
        return True

    def _handle_space_conversion(self, event: Any) -> bool:
        if event.key != " " or event.ctrlKey or event.altKey or event.metaKey:
            return False
        token = self._current_token()
        if token is None:
            return False
        entry = exact_latex_match(token[2])
        if entry is None:
            return False
        event.preventDefault()
        self.completion.hide()
        self.insert(entry.insert + " ", token[0], token[1])
        self._remember(entry.symbol)
        return True

    # ----- palette -----

    def _on_palette_pick(self, entry: MathSymbol) -> None:
        self._insert_symbol(entry)

    def _handle_palette_key(self, event: Any) -> bool:
        key = event.key
        if key == "ArrowUp" and event.ctrlKey and not (event.altKey or event.shiftKey or event.metaKey):
            self.completion.hide()
            self.palette.toggle(keyboard=True)
            event.preventDefault()
            return True
        if not self.palette.visible or event.ctrlKey or event.altKey or event.metaKey:
            return False
        if key in _ARROW_KEYS:
            self.palette.move(key)
        elif key == "Tab":
            self.palette.switch_group(-1 if event.shiftKey else 1)
        elif key == "Escape":
            self.palette.close()
        elif key == "Enter":
            entry = self.palette.selected_entry() if self.palette.keyboard_active else None
            if entry is None:
                self.palette.close()
                return False  # let Enter send the message
            self._insert_symbol(entry)
        else:
            return False
        event.preventDefault()
        return True

    # ----- Alt shortcuts -----

    def _handle_alt_shortcut(self, event: Any) -> bool:
        if not event.altKey or event.ctrlKey or event.metaKey or self._right_alt_down:
            return False
        try:
            if event.getModifierState("AltGraph"):
                return False
        except Exception:
            pass
        symbol = lookup_alt_shortcut(str(event.code), bool(event.shiftKey))
        if symbol is None:
            return False
        event.preventDefault()
        self.insert(symbol)
        self._remember(symbol)
        return True
