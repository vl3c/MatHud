"""
MatHud Backslash Symbol Completion Popup

Suggestion list shown above the chat input while the user types a ``\\name``
token (for example ``\\alp`` offers α). It shares the look of the slash
command popup (``.command-autocomplete``) and lists the symbol, its name and
its backslash names.

The popup only renders and tracks the highlighted row; MathSymbolInput decides
when to show it and handles the keys. The highlighted row is exposed through
the input's ``aria-activedescendant``, and the list is capped to the space
above the input because the chat pane clips overflow.

Dependencies:
    - browser: DOM construction and events
    - math_symbols: symbol entries
    - math_symbol_palette: shared popup sizing and scrolling helpers
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from browser import html

from math_symbol_palette import keep_in_view, space_above
from math_symbols import MathSymbol, shortcut_label

COMPLETION_POPUP_ID: str = "symbol-completion-popup"
COMPLETION_ITEM_ID_PREFIX: str = "symbol-suggestion-"
# Same cap as the slash-command popup (.command-autocomplete max-height).
COMPLETION_MAX_HEIGHT: int = 300
COMPLETION_MIN_HEIGHT: int = 48


class SymbolCompletionPopup:
    """List of symbols matching the ``\\name`` token before the caret.

    Attributes:
        entries: Symbols currently listed.
        selected_index: Highlighted row.
        visible: Whether the popup is shown.
        navigated: Whether the highlight was moved with the arrow keys since
            the list was shown (Enter accepts only then).
    """

    def __init__(
        self,
        container: Any,
        input_element: Any,
        on_accept: Callable[[MathSymbol], None],
        on_visibility_change: Optional[Callable[[], None]] = None,
    ) -> None:
        """Create the popup inside ``container`` (the chat input container).

        Args:
            container: Positioned element the popup is appended to.
            input_element: The chat input, which carries ``aria-activedescendant``.
            on_accept: Called with the symbol a user clicks.
            on_visibility_change: Called after the popup is shown or hidden.
        """
        self._input: Any = input_element
        self._on_accept: Callable[[MathSymbol], None] = on_accept
        self._on_visibility_change: Optional[Callable[[], None]] = on_visibility_change
        self.entries: List[MathSymbol] = []
        self.selected_index: int = 0
        self.visible: bool = False
        self.navigated: bool = False
        self._items: List[Any] = []

        self._root = html.DIV(Class="command-autocomplete symbol-completion", id=COMPLETION_POPUP_ID)
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
        was_visible = self.visible
        self.entries = list(entries)
        self.selected_index = 0
        self.navigated = False
        self._render()
        self._root.style.display = "block"
        self.visible = True
        self.fit_height()
        self._root.scrollTop = 0
        self._input.attrs["aria-activedescendant"] = self._items[0].id
        if not was_visible:
            self._notify()

    def hide(self) -> None:
        """Hide the popup."""
        was_visible = self.visible
        self._root.style.display = "none"
        self.visible = False
        self.navigated = False
        self.entries = []
        self._items = []
        self.selected_index = 0
        if was_visible:
            self._input.removeAttribute("aria-activedescendant")
            self._notify()

    def fit_height(self) -> None:
        """Cap the height to the space above the input so no row is clipped by the chat pane."""
        available = space_above(self._root)
        if available is None:
            return
        height = min(COMPLETION_MAX_HEIGHT, max(COMPLETION_MIN_HEIGHT, available))
        self._root.style.maxHeight = f"{height}px"
        self._keep_selected_in_view()

    def destroy(self) -> None:
        """Remove the popup from the page."""
        self.hide()
        self._root.remove()

    def select_next(self) -> None:
        """Highlight the next row, wrapping to the first."""
        if self.entries:
            self.navigated = True
            self._highlight((self.selected_index + 1) % len(self.entries))

    def select_previous(self) -> None:
        """Highlight the previous row, wrapping to the last."""
        if self.entries:
            self.navigated = True
            self._highlight((self.selected_index - 1) % len(self.entries))

    def selected_entry(self) -> Optional[MathSymbol]:
        """Return the highlighted symbol, or None."""
        if 0 <= self.selected_index < len(self.entries):
            return self.entries[self.selected_index]
        return None

    def _notify(self) -> None:
        if self._on_visibility_change is not None:
            self._on_visibility_change()

    def _prevent_focus_steal(self, event: Any) -> None:
        event.preventDefault()

    def _render(self) -> None:
        self._root.clear()
        self._items = []
        for index, entry in enumerate(self.entries):
            item = html.DIV(
                Class="command-autocomplete-item symbol-completion-item",
                id=f"{COMPLETION_ITEM_ID_PREFIX}{index}",
            )
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
            self._input.attrs["aria-activedescendant"] = item.id
            self._keep_selected_in_view()

    def _keep_selected_in_view(self) -> None:
        if 0 <= self.selected_index < len(self._items):
            keep_in_view(self._root, self._items[self.selected_index])
