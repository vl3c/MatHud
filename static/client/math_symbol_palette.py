"""
MatHud Math Symbol Palette

Popover of math symbols opened from the Σ button next to the chat input.

Layout (top to bottom): a "Recent" row of the last symbols used, group tabs,
the symbol grid of the selected group, and a one-line keyboard hint. Every
cell carries a tooltip and aria-label with its name, Alt shortcut and
backslash names.

The palette never takes focus: mousedown on the popover and on the Σ button is
cancelled, so the caret stays in the chat input. Keyboard navigation is
driven by MathSymbolInput, which forwards arrow keys, Tab and Enter while the
palette is open; the highlighted cell is announced through the input's
aria-activedescendant.

The chat pane clips overflow, so the palette's height is capped to the space
above the input: the group grid scrolls, and the keyboard hint and section
label are dropped when space is short.

Dependencies:
    - browser: DOM construction and events
    - math_symbols: symbol table and grid navigation
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from browser import document, html, window

from math_symbols import (
    GROUPS,
    group_index_for,
    MathSymbol,
    describe_symbol,
    get_symbol,
    move_grid_selection,
    shortcut_label,
    symbols_in_group,
)

PALETTE_HINT: str = "Arrows + Enter insert · Tab: next group · type \\name to search"

PALETTE_ID: str = "symbol-palette"
GROUP_GRID_ID: str = "symbol-palette-grid"
RECENT_GRID_ID: str = "symbol-palette-recent-grid"

# localStorage key of the last group tab the user opened (the first tab until one is chosen).
GROUP_STORAGE_KEY: str = "mathud.symbols.group"

# Gap kept between a popup and the top of the chat pane.
POPUP_TOP_MARGIN: int = 12
# Below this height the palette drops its keyboard hint and "Recent" label.
PALETTE_COMPACT_HEIGHT: int = 240
PALETTE_MIN_HEIGHT: int = 80


def space_above(element: Any) -> Optional[int]:
    """Pixels between the top of ``element``'s container and the top of the chat pane.

    Popups sit above the chat input container, and the chat pane clips
    overflow, so this is the most height a popup can use. Returns None when
    the container is not inside the chat pane (for example in tests).
    """
    try:
        chat = document.querySelector(".chat-container")
        container = element.parentElement
        if chat is None or container is None or not chat.contains(container):
            return None
        top = container.getBoundingClientRect().top - chat.getBoundingClientRect().top
        return int(top) - POPUP_TOP_MARGIN
    except Exception:
        return None


def keep_in_view(scroller: Any, item: Any) -> None:
    """Scroll ``scroller`` (vertically and sideways) just enough to show ``item``.

    Unlike ``scrollIntoView`` this never scrolls the chat pane or the page.
    """
    try:
        outer = scroller.getBoundingClientRect()
        inner = item.getBoundingClientRect()
        if inner.top < outer.top:
            scroller.scrollTop -= outer.top - inner.top
        elif inner.bottom > outer.bottom:
            scroller.scrollTop += inner.bottom - outer.bottom
        if inner.left < outer.left:
            scroller.scrollLeft -= outer.left - inner.left
        elif inner.right > outer.right:
            scroller.scrollLeft += inner.right - outer.right
    except Exception:
        pass


class MathSymbolPalette:
    """Symbol palette popover anchored above the chat input.

    Attributes:
        visible: Whether the popover is shown.
        keyboard_active: Whether a cell is highlighted for keyboard selection.
    """

    def __init__(
        self,
        input_element: Any,
        button_element: Any,
        container: Any,
        on_pick: Callable[[MathSymbol], None],
        on_toggle: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Build the popover inside ``container`` (the chat input container).

        Args:
            input_element: The chat input; keeps focus while the palette is open.
            button_element: The Σ toggle button.
            container: Positioned element the popover is appended to.
            on_pick: Called with the symbol a user clicks or confirms.
            on_toggle: Called with the new visibility after the palette opens
                (before a cell is highlighted) or closes.
        """
        self._input: Any = input_element
        self._button: Any = button_element
        self._on_pick: Callable[[MathSymbol], None] = on_pick
        self._on_toggle: Optional[Callable[[bool], None]] = on_toggle
        self._document_mousedown: Callable[[Any], None] = self._on_document_mousedown
        self.visible: bool = False
        self._group_index: int = self._load_group_index()
        self._recent: List[str] = []
        self._sections: List[List[MathSymbol]] = []
        self._cells: List[List[Any]] = []
        self._selection: Optional[Tuple[int, int]] = None

        self._build(container)
        self._bind_events()

    @property
    def keyboard_active(self) -> bool:
        return self.visible and self._selection is not None

    # ----- public API -----

    def clear_keyboard_selection(self) -> None:
        """Remove the keyboard highlight; the palette stays open for mouse picks."""
        if self._selection is not None:
            self._set_selection(None)

    def toggle(self, keyboard: bool = False) -> None:
        """Open the palette if closed, close it if open."""
        if self.visible:
            self.close()
        else:
            self.open(keyboard)

    def open(self, keyboard: bool = False) -> None:
        """Show the palette; with ``keyboard`` the first cell is highlighted."""
        self._render_sections()
        self._root.style.display = ""
        self.visible = True
        self.fit_height()
        self._button.classList.add("active")
        self._button.attrs["aria-expanded"] = "true"
        self._notify_toggle()
        self._set_selection((0, 0) if keyboard else None)

    def close(self) -> None:
        """Hide the palette and clear the keyboard highlight."""
        if not self.visible:
            return
        self._set_selection(None)
        self._root.style.display = "none"
        self.visible = False
        self._button.classList.remove("active")
        self._button.attrs["aria-expanded"] = "false"
        self._notify_toggle()

    def fit_height(self) -> None:
        """Cap the height to the space above the input (the chat pane clips overflow)."""
        available = space_above(self._root)
        if available is None:
            return
        self._root.style.maxHeight = f"{max(PALETTE_MIN_HEIGHT, available)}px"
        if available < PALETTE_COMPACT_HEIGHT:
            self._root.classList.add("compact")
        else:
            self._root.classList.remove("compact")

    def destroy(self) -> None:
        """Remove the popover and its document listener."""
        try:
            document.unbind("mousedown", self._document_mousedown)
        except Exception:
            pass
        self._root.remove()

    def _notify_toggle(self) -> None:
        if self._on_toggle is not None:
            self._on_toggle(self.visible)

    def set_recent(self, recent: List[str]) -> None:
        """Replace the recent list, keeping the highlight on the same symbol."""
        selected = self.selected_entry()
        was_recent = self._selection is not None and self._has_recent() and self._selection[0] == 0
        group_index = self._selection[1] if self._selection is not None and not was_recent else 0
        self._recent = list(recent)
        if not self.visible:
            return
        self._render_sections()
        if self._selection is None or selected is None:
            return
        if was_recent and selected.symbol in self._recent:
            self._set_selection((0, self._recent.index(selected.symbol)))
        else:
            self._set_selection((len(self._sections) - 1, group_index))

    def move(self, key: str) -> None:
        """Move the keyboard highlight with an arrow key (highlights the first cell if none)."""
        if not self._sections:
            return
        if self._selection is None:
            self._set_selection((0, 0))
            return
        sizes = [len(section) for section in self._sections]
        columns = [self._column_count(index) for index in range(len(self._sections))]
        section, index = self._selection
        self._set_selection(move_grid_selection(sizes, columns, section, index, key))

    def switch_group(self, delta: int) -> None:
        """Show the next (delta=1) or previous (delta=-1) group."""
        self._show_group((self._group_index + delta) % len(GROUPS))

    def selected_entry(self) -> Optional[MathSymbol]:
        """Return the highlighted symbol, or None."""
        if self._selection is None:
            return None
        section, index = self._selection
        if section < len(self._sections) and index < len(self._sections[section]):
            return self._sections[section][index]
        return None

    def contains(self, element: Any) -> bool:
        """Whether a DOM node is inside the palette or its toggle button."""
        try:
            return bool(self._root.contains(element) or self._button.contains(element))
        except Exception:
            return False

    # ----- DOM construction -----

    def _build(self, container: Any) -> None:
        self._root = html.DIV(Class="symbol-palette", id=PALETTE_ID)
        self._root.attrs["role"] = "dialog"
        self._root.attrs["aria-label"] = "Math symbols"
        self._root.style.display = "none"

        self._recent_section = html.DIV(Class="symbol-palette-recent")
        self._recent_section <= html.SPAN("Recent", Class="symbol-palette-label")
        self._recent_grid = self._make_grid("Recent symbols", RECENT_GRID_ID)
        self._recent_section <= self._recent_grid

        self._tabs = html.DIV(Class="symbol-palette-tabs")
        self._tabs.attrs["role"] = "tablist"
        self._tabs.attrs["aria-label"] = "Symbol groups"
        self._tab_buttons: List[Any] = []
        for index, (group_id, label) in enumerate(GROUPS):
            tab = html.BUTTON(label, Class="symbol-palette-tab", id=f"symbol-palette-tab-{group_id}")
            tab.attrs["type"] = "button"
            tab.attrs["tabindex"] = "-1"
            tab.attrs["role"] = "tab"
            tab.attrs["aria-controls"] = GROUP_GRID_ID
            tab.bind("click", self._make_tab_handler(index))
            self._tab_buttons.append(tab)
            self._tabs <= tab

        self._group_grid = self._make_grid("Symbols", GROUP_GRID_ID)
        hint = html.DIV(PALETTE_HINT, Class="symbol-palette-hint")

        self._root <= self._recent_section
        self._root <= self._tabs
        self._root <= self._group_grid
        self._root <= hint
        container <= self._root
        self._update_tabs()

    def _make_grid(self, label: str, grid_id: str) -> Any:
        grid = html.DIV(Class="symbol-palette-grid", id=grid_id)
        grid.attrs["role"] = "listbox"
        grid.attrs["aria-label"] = label
        return grid

    def _make_cell(self, entry: MathSymbol, cell_id: str) -> Any:
        cell = html.BUTTON(entry.symbol, Class="symbol-cell", id=cell_id)
        if len(entry.symbol) > 1:
            cell.classList.add("symbol-cell-text")
        cell.attrs["type"] = "button"
        cell.attrs["tabindex"] = "-1"
        cell.attrs["role"] = "option"
        cell.attrs["aria-selected"] = "false"
        cell.attrs["title"] = describe_symbol(entry)
        shortcut = shortcut_label(entry.symbol)
        cell.attrs["aria-label"] = f"{entry.name}, {shortcut}" if shortcut else entry.name
        cell.bind("click", self._make_cell_handler(entry))
        return cell

    def _make_cell_handler(self, entry: MathSymbol) -> Callable[[Any], None]:
        def handler(event: Any) -> None:
            event.preventDefault()
            self._set_selection(None)
            self._on_pick(entry)

        return handler

    def _make_tab_handler(self, index: int) -> Callable[[Any], None]:
        def handler(event: Any) -> None:
            event.preventDefault()
            self._show_group(index)

        return handler

    def _bind_events(self) -> None:
        # Keep focus (and the caret) in the chat input when the popover or button is pressed.
        self._root.bind("mousedown", self._prevent_focus_steal)
        self._button.bind("mousedown", self._prevent_focus_steal)
        self._button.bind("click", self._on_button_click)
        document.bind("mousedown", self._document_mousedown)

    def _prevent_focus_steal(self, event: Any) -> None:
        event.preventDefault()

    def _on_button_click(self, event: Any) -> None:
        event.preventDefault()
        self.toggle(keyboard=False)
        # Focus the input so arrow keys work, except on touch screens where that
        # would pop up the virtual keyboard over the palette.
        if self.visible and not self._is_coarse_pointer():
            self._input.focus()

    def _is_coarse_pointer(self) -> bool:
        try:
            return bool(window.matchMedia("(pointer: coarse)").matches)
        except Exception:
            return False

    def _on_document_mousedown(self, event: Any) -> None:
        if self.visible and not self.contains(event.target):
            self.close()

    # ----- rendering -----

    def _has_recent(self) -> bool:
        return any(get_symbol(symbol) is not None for symbol in self._recent)

    def _render_sections(self) -> None:
        recent_entries = [entry for entry in (get_symbol(s) for s in self._recent) if entry is not None]
        group_entries = symbols_in_group(GROUPS[self._group_index][0])
        self._sections = []
        self._cells = []

        self._recent_grid.clear()
        if recent_entries:
            self._recent_section.style.display = ""
            self._sections.append(recent_entries)
            self._cells.append(self._fill_grid(self._recent_grid, recent_entries, "recent"))
        else:
            self._recent_section.style.display = "none"

        self._group_grid.clear()
        self._sections.append(group_entries)
        self._cells.append(self._fill_grid(self._group_grid, group_entries, "group"))

    def _fill_grid(self, grid: Any, entries: List[MathSymbol], prefix: str) -> List[Any]:
        cells = []
        for index, entry in enumerate(entries):
            cell = self._make_cell(entry, f"symbol-cell-{prefix}-{index}")
            cells.append(cell)
            grid <= cell
        return cells

    def _show_group(self, group_index: int) -> None:
        self._group_index = group_index
        self._save_group_index()
        self._update_tabs()
        keep_highlight = self._selection is not None
        self._render_sections()
        self._set_selection((len(self._sections) - 1, 0) if keep_highlight else None)

    @staticmethod
    def _load_group_index() -> int:
        """Tab to open on: the one used last, or the first (Operators) the first time."""
        try:
            return group_index_for(window.localStorage.getItem(GROUP_STORAGE_KEY))
        except Exception:
            return 0

    def _save_group_index(self) -> None:
        try:
            window.localStorage.setItem(GROUP_STORAGE_KEY, GROUPS[self._group_index][0])
        except Exception:
            pass

    def _update_tabs(self) -> None:
        for index, tab in enumerate(self._tab_buttons):
            selected = index == self._group_index
            if selected:
                tab.classList.add("selected")
            else:
                tab.classList.remove("selected")
            tab.attrs["aria-selected"] = "true" if selected else "false"
        self._group_grid.attrs["aria-label"] = GROUPS[self._group_index][1]

    def _column_count(self, section: int) -> int:
        """Cells in the first row of a section, measured from the layout."""
        cells = self._cells[section] if section < len(self._cells) else []
        if not cells:
            return 1
        top = cells[0].offsetTop
        count = 0
        for cell in cells:
            if cell.offsetTop != top:
                break
            count += 1
        return max(1, count)

    def _set_selection(self, selection: Optional[Tuple[int, int]]) -> None:
        previous = self._cell_at(self._selection)
        if previous is not None:
            previous.classList.remove("selected")
            previous.attrs["aria-selected"] = "false"
        self._selection = selection
        current = self._cell_at(selection)
        if current is not None:
            current.classList.add("selected")
            current.attrs["aria-selected"] = "true"
            self._input.attrs["aria-activedescendant"] = current.id
            keep_in_view(current.parentElement, current)
        else:
            self._input.removeAttribute("aria-activedescendant")

    def _cell_at(self, selection: Optional[Tuple[int, int]]) -> Optional[Any]:
        if selection is None:
            return None
        section, index = selection
        if section < len(self._cells) and index < len(self._cells[section]):
            return self._cells[section][index]
        return None
