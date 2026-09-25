"""
MatHud Backslash Symbol Completion Popup

Suggestion list shown above the chat input while the user types a ``\\name``
token (for example ``\\alp`` offers α). It shares the look of the slash
command popup (``.command-autocomplete``) and lists the symbol, its name and
its backslash names.

The popup only renders and tracks the highlighted row; MathSymbolInput decides
when to show it and handles the keys.

Dependencies:
    - browser: DOM construction and events
    - math_symbols: symbol entries
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from browser import html

from math_symbols import MathSymbol, shortcut_label


class SymbolCompletionPopup:
    """List of symbols matching the ``\\name`` token before the caret.

    Attributes:
        entries: Symbols currently listed.
        selected_index: Highlighted row.
        visible: Whether the popup is shown.
    """

    def __init__(self, container: Any, on_accept: Callable[[MathSymbol], None]) -> None:
        """Create the popup inside ``container`` (the chat input container).

        Args:
            container: Positioned element the popup is appended to.
            on_accept: Called with the symbol a user clicks.
        """
        self._on_accept: Callable[[MathSymbol], None] = on_accept
        self.entries: List[MathSymbol] = []
        self.selected_index: int = 0
        self.visible: bool = False
        self._items: List[Any] = []

        self._root = html.DIV(Class="command-autocomplete symbol-completion", id="symbol-completion-popup")
        self._root.attrs["role"] = "listbox"
        self._root.attrs["aria-label"] = "Symbol suggestions"
        self._root.style.display = "none"
        # Keep focus in the chat input when a row is pressed.
        self._root.bind("mousedown", self._prevent_focus_steal)
        container <= self._root

    def show(self, entries: List[MathSymbol]) -> None:
        """Show ``entries`` with the first row highlighted (hides when empty)."""
        if not entries:
            self.hide()
            return
        self.entries = list(entries)
        self.selected_index = 0
        self._render()
        self._root.style.display = "block"
        self.visible = True

    def hide(self) -> None:
        """Hide the popup."""
        self._root.style.display = "none"
        self.visible = False
        self.entries = []
        self._items = []
        self.selected_index = 0

    def select_next(self) -> None:
        """Highlight the next row, wrapping to the first."""
        if self.entries:
            self._highlight((self.selected_index + 1) % len(self.entries))

    def select_previous(self) -> None:
        """Highlight the previous row, wrapping to the last."""
        if self.entries:
            self._highlight((self.selected_index - 1) % len(self.entries))

    def selected_entry(self) -> Optional[MathSymbol]:
        """Return the highlighted symbol, or None."""
        if 0 <= self.selected_index < len(self.entries):
            return self.entries[self.selected_index]
        return None

    def _prevent_focus_steal(self, event: Any) -> None:
        event.preventDefault()

    def _render(self) -> None:
        self._root.clear()
        self._items = []
        for index, entry in enumerate(self.entries):
            item = html.DIV(Class="command-autocomplete-item symbol-completion-item")
            item.attrs["role"] = "option"
            item.attrs["aria-selected"] = "true" if index == self.selected_index else "false"
            if index == self.selected_index:
                item.classList.add("selected")
            item <= html.SPAN(entry.symbol, Class="symbol-completion-symbol")
            item <= html.SPAN(entry.name, Class="symbol-completion-name")
            item <= html.SPAN(" ".join("\\" + name for name in entry.latex), Class="symbol-completion-latex")
            shortcut = shortcut_label(entry.symbol)
            if shortcut:
                item <= html.KBD(shortcut, Class="symbol-completion-key")
            item.bind("mousedown", self._make_click_handler(entry))
            item.bind("mouseenter", self._make_hover_handler(index))
            self._items.append(item)
            self._root <= item

    def _make_click_handler(self, entry: MathSymbol) -> Callable[[Any], None]:
        def handler(event: Any) -> None:
            event.preventDefault()
            event.stopPropagation()
            self._on_accept(entry)

        return handler

    def _make_hover_handler(self, index: int) -> Callable[[Any], None]:
        def handler(event: Any) -> None:
            self._highlight(index)

        return handler

    def _highlight(self, index: int) -> None:
        if 0 <= self.selected_index < len(self._items):
            self._items[self.selected_index].classList.remove("selected")
            self._items[self.selected_index].attrs["aria-selected"] = "false"
        self.selected_index = index
        if 0 <= index < len(self._items):
            item = self._items[index]
            item.classList.add("selected")
            item.attrs["aria-selected"] = "true"
            try:
                item.scrollIntoView({"block": "nearest"})
            except Exception:
                pass
