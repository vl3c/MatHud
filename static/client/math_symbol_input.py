"""
MatHud Math Symbol Input

Wires math-symbol entry into the chat input:

    - the Σ palette button and popover (MathSymbolPalette), also toggled with Ctrl+Up
    - Alt shortcuts (Alt+P inserts π, Alt+2 inserts ², ...)
    - backslash completion: typing ``\\alp`` offers α (SymbolCompletionPopup);
      Tab accepts, Enter accepts once a suggestion was picked with the arrow
      keys, and a space after an exact name converts it

Every insertion replaces the selection at the caret, keeps focus in the input
and, where the browser supports it, goes through ``insertText`` so Ctrl+Z
undoes it.

Positions are DOM selection offsets (UTF-16 code units) throughout. Only the
few characters around the caret are read, through JavaScript ``substring``, so
emoji or pasted astral characters (𝑥) never shift an insertion and the cost
of a keystroke does not grow with the message length.

Coexistence with the slash-command autocomplete: this handler is bound after
CommandAutocomplete and ignores keydown events that one already handled
(``defaultPrevented``); backslash completion is off and the palette closes
when the input starts with "/", and opening the palette fires
SYMBOL_POPUP_OPEN_EVENT on the input, which closes the slash-command list. So
only one popup is open at a time. Enter keeps sending the message unless a
symbol popup is using it.

Dependencies:
    - browser: DOM events and localStorage
    - javascript: String.substring for UTF-16 reads around the caret
    - math_symbols: symbol table and pure editing helpers
    - math_symbol_palette, symbol_completion_popup: the two popups
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import javascript
from browser import document, window

from command_autocomplete import SYMBOL_POPUP_OPEN_EVENT
from math_symbol_palette import GROUP_GRID_ID, MathSymbolPalette
from math_symbols import (
    MathSymbol,
    exact_latex_match,
    find_latex_token,
    latex_word_tail,
    lookup_alt_shortcut,
    sanitize_recent,
    search_symbols,
    update_recent,
)
from symbol_completion_popup import COMPLETION_POPUP_ID, SymbolCompletionPopup

RECENT_STORAGE_KEY: str = "mathud.symbols.recent"
_ARROW_KEYS: Tuple[str, ...] = ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown")
_CARET_KEYS: Tuple[str, ...] = ("ArrowLeft", "ArrowRight", "Home", "End")
# UTF-16 units read on each side of the caret; longer than any backslash name.
_TOKEN_WINDOW: int = 40


def _js_string(value: Any) -> Any:
    """Wrap a DOM string so substring and length count UTF-16 units."""
    return javascript.String.new(value)


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
        self._inserting: bool = False  # True while insert() edits the input
        self._resize_observer: Any = None
        # Kept so destroy() can unbind the same function object.
        self._window_resize_handler: Any = self._on_window_resize
        self.palette = MathSymbolPalette(
            input_element, button_element, container, self._on_palette_pick, self._on_palette_toggled
        )
        self.palette.set_recent(self._recent)
        self.completion = SymbolCompletionPopup(container, input_element, self._accept_completion, self._sync_aria)
        self._sync_aria()
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

    def destroy(self) -> None:
        """Remove the popups and the window-level listeners (used by tests)."""
        try:
            window.unbind("resize", self._window_resize_handler)
        except Exception:
            pass
        if self._resize_observer is not None:
            try:
                self._resize_observer.disconnect()
            except Exception:
                pass
            self._resize_observer = None
        self.completion.destroy()
        self.palette.destroy()

    # ----- insertion -----

    def insert(self, text: str, start: Optional[int] = None, end: Optional[int] = None) -> None:
        """Insert ``text`` at the caret, replacing the selection or ``start..end``.

        ``start`` and ``end`` are DOM selection offsets (UTF-16 units); by
        default the current selection is replaced. Focus returns to the input
        with the caret after the inserted text.
        """
        if start is None or end is None:
            start, end = self._selection()
        self._inserting = True
        try:
            self._input.focus()
            self._input.setSelectionRange(start, end)
            if self._exec_insert_text(text) and self._inserted_at(start, text):
                return
            self._input.setRangeText(text, start, end, "end")
            self._dispatch_input_event()
        finally:
            self._inserting = False

    def _exec_insert_text(self, text: str) -> bool:
        """Insert through the browser's editing command so the edit is undoable.

        The command edits whatever element has focus, so it is used only when
        that is this input (focus() does nothing on a detached element).
        """
        try:
            if not self._input.isSameNode(document.activeElement):
                return False
            return bool(document.execCommand("insertText", False, text))
        except Exception:
            return False

    def _inserted_at(self, start: int, text: str) -> bool:
        """Whether ``text`` now sits at ``start`` with the caret right after it."""
        units = int(_js_string(text).length)
        if self._input.selectionStart != start + units:
            return False
        return str(_js_string(self._input.value).substring(start, start + units)) == text

    def _dispatch_input_event(self) -> None:
        try:
            self._input.dispatchEvent(window.Event.new("input", {"bubbles": True}))
        except Exception:
            pass

    def _selection(self) -> Tuple[int, int]:
        """The selection as DOM offsets, or the end of the value if there is none."""
        start = self._input.selectionStart
        end = self._input.selectionEnd
        if start is None or end is None:
            length = int(_js_string(self._input.value).length)
            return length, length
        return int(start), int(end)

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
        self._input.bind("focus", self._on_focus)
        self._input.bind("blur", self._on_blur)
        window.bind("resize", self._window_resize_handler)
        self._observe_chat_resize()

    def _observe_chat_resize(self) -> None:
        """Refit open popups when the chat pane changes size (separator drags, layout switches)."""
        try:
            chat = document.querySelector(".chat-container")
            if chat is None or not window.ResizeObserver:
                return
            self._resize_observer = window.ResizeObserver.new(self._on_chat_resized)
            self._resize_observer.observe(chat)
        except Exception:
            self._resize_observer = None

    def _on_chat_resized(self, *args: Any) -> None:
        self._refit_popups()

    def _on_window_resize(self, event: Any) -> None:
        self._refit_popups()

    def _refit_popups(self) -> None:
        if self.palette.visible:
            self.palette.fit_height()
        if self.completion.visible:
            self.completion.fit_height()

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
        if self._is_slash_command():
            self.palette.close()  # the slash-command list takes the space above the input
        elif not self._inserting:
            # The user typed: drop the palette's keyboard highlight so Enter sends again.
            self.palette.clear_keyboard_selection()
        self._refresh_completion()

    def _on_caret_moved(self, event: Any) -> None:
        self._refresh_completion()

    def _on_focus(self, event: Any) -> None:
        self._refit_popups()

    def _on_blur(self, event: Any) -> None:
        self._right_alt_down = False
        window.setTimeout(self._hide_completion_if_unfocused, 150)

    def _hide_completion_if_unfocused(self) -> None:
        if not self._input.isSameNode(document.activeElement):
            self.completion.hide()

    def _is_slash_command(self) -> bool:
        return str(self._input.value).startswith("/")

    # ----- popups -----

    def _on_palette_toggled(self, visible: bool) -> None:
        if visible:
            # One popup at a time: close the suggestions and ask the slash-command list to close
            self.completion.hide()
            try:
                self._input.dispatchEvent(window.CustomEvent.new(SYMBOL_POPUP_OPEN_EVENT))
            except Exception:
                pass
        self._sync_aria()

    def _sync_aria(self) -> None:
        """Expose the open popup on the input (a combobox) for assistive technology."""
        controls = GROUP_GRID_ID if self.palette.visible else COMPLETION_POPUP_ID if self.completion.visible else ""
        self._input.attrs["aria-expanded"] = "true" if controls else "false"
        if controls:
            self._input.attrs["aria-controls"] = controls
        else:
            self._input.removeAttribute("aria-controls")

    # ----- backslash completion -----

    def _current_token(self) -> Optional[Tuple[int, int, str]]:
        """Return (start, caret, name) of the ``\\name`` token before the caret, in DOM offsets."""
        if self._is_slash_command():
            return None  # slash commands own the input
        start, caret = self._selection()
        if start != caret:
            return None
        before = str(_js_string(self._input.value).substring(max(0, caret - _TOKEN_WINDOW), caret))
        # A name is ASCII, so its length in characters is its length in UTF-16 units.
        token = find_latex_token(before, len(before))
        if token is None:
            return None
        name = token[1]
        return caret - len(name) - 1, caret, name

    def _word_end(self, caret: int, name: str) -> int:
        """End offset of the ``\\name`` word the caret is in (past letters typed after the caret)."""
        after = str(_js_string(self._input.value).substring(caret, caret + _TOKEN_WINDOW))
        return caret + latex_word_tail(name, after)

    def _refresh_completion(self) -> None:
        token = self._current_token()
        entries = search_symbols(token[2]) if token is not None else []
        if not entries:
            self.completion.hide()
            return
        self.palette.close()  # both popups share the space above the input
        self.completion.show(entries)

    def _accept_completion(self, entry: MathSymbol) -> None:
        token = self._current_token()
        self.completion.hide()
        if token is None:
            return
        start, caret, name = token
        self._insert_symbol(entry, start, self._word_end(caret, name))

    def _handle_completion_key(self, event: Any) -> bool:
        if not self.completion.visible or event.ctrlKey or event.altKey or event.metaKey:
            return False
        key = event.key
        if key == "ArrowDown":
            self.completion.select_next()
        elif key == "ArrowUp":
            self.completion.select_previous()
        elif key == "Tab" or (key == "Enter" and self.completion.navigated):
            entry = self.completion.selected_entry()
            if entry is None:
                return False
            self._accept_completion(entry)
        elif key == "Enter":
            self.completion.hide()
            return False  # no suggestion was picked: Enter sends the message
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
        """Insert the symbol of an Alt shortcut.

        Matched on ``event.key``, the character the key types without Alt on
        Windows and Linux (so AZERTY or Dvorak users press the key labelled P
        for π). On macOS ``key`` is the character Option produces, which never
        matches, so Option keeps its native characters. AltGr (reported as
        Ctrl+Alt or AltGraph), right Alt and Meta combinations are left alone.
        """
        if not event.altKey or event.ctrlKey or event.metaKey or self._right_alt_down:
            return False
        try:
            if event.getModifierState("AltGraph"):
                return False
        except Exception:
            pass
        symbol = lookup_alt_shortcut(str(event.key), bool(event.shiftKey))
        if symbol is None:
            return False
        event.preventDefault()
        self.insert(symbol)
        self._remember(symbol)
        return True
