"""
Behaviour tests for the chat input's math symbol entry (MathSymbolInput).

Each test builds a detached <input>, Σ button and container, attaches a
MathSymbolInput to them and drives it with synthetic DOM events, checking the
resulting value, caret and ``defaultPrevented`` (a keydown whose default is
not prevented goes on to the chat's Enter-to-send handler).

Values containing emoji or astral math letters are assigned through the DOM
input and compared as UTF-16 code units read back through JavaScript, so the
tests see exactly what the browser holds.

TestEverySymbol loops over the symbol table, so a symbol added to SYMBOLS is
covered automatically; the count and snapshot assertions flag removals.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Tuple

import javascript
from browser import html, window

from command_autocomplete import CommandAutocomplete
from math_symbol_input import RECENT_STORAGE_KEY, MathSymbolInput
from math_symbol_palette import GROUP_GRID_ID, GROUP_STORAGE_KEY
from math_symbols import ALT_SHORTCUTS, GROUP_EXTRAS, GROUPS, SYMBOLS, search_symbols
from symbol_completion_popup import COMPLETION_POPUP_ID

EMOJI = window.String.fromCodePoint(0x1F600)  # 😀, two UTF-16 units
MATH_X = window.String.fromCodePoint(0x1D465)  # 𝑥 (mathematical italic x), two UTF-16 units

EXPECTED_SYMBOL_COUNT = 84
EXPECTED_SYMBOLS = (
    "α β γ δ ε ζ η θ κ λ μ ξ π ρ σ τ φ χ ψ ω Γ Δ Θ Π Σ Φ Ω "
    "× ÷ ± √ ² ³ ⁿ ≠ ≤ ≥ ≈ ∞ ° · ⁰ ¹ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ ⁻ "
    "∠ ⊥ ∥ △ ≅ ∼ ′ ″ ∈ ∉ ⊂ ⊆ ∪ ∩ ∅ ℝ ℕ ℤ ℚ ℂ ∧ ∨ ¬ → ⇒ ⇔ ∀ ∃ "
    "∫ ∑ ∏ ∂ ∇ lim"
)

_PUNCTUATION_CODES: Dict[str, str] = {",": "Comma", ".": "Period", "=": "Equal", "-": "Minus"}


def code_units(text: Any) -> List[int]:
    """UTF-16 code units of a string, read through JavaScript."""
    js = javascript.String.new(text)
    return [int(js.charCodeAt(index)) for index in range(int(js.length))]


def unit_length(text: Any) -> int:
    return int(javascript.String.new(text).length)


def key_code(key: str) -> str:
    """KeyboardEvent.code of the US-layout key that types ``key``."""
    if key.isdigit():
        return f"Digit{key}"
    if key.isalpha():
        return f"Key{key.upper()}"
    return _PUNCTUATION_CODES.get(key, "")


class _SymbolInputFixture(unittest.TestCase):
    """Detached chat input with a MathSymbolInput attached; localStorage is restored afterwards."""

    STORAGE_KEYS = (RECENT_STORAGE_KEY, GROUP_STORAGE_KEY)

    def setUp(self) -> None:
        self._saved_storage = {key: window.localStorage.getItem(key) for key in self.STORAGE_KEYS}
        for key in self.STORAGE_KEYS:
            window.localStorage.removeItem(key)
        self.container = html.DIV()
        self.button = html.BUTTON("Σ")
        self.input = html.INPUT(type="text")
        self.container <= self.button
        self.container <= self.input
        self.symbols = self._attach()

    def _attach(self) -> MathSymbolInput:
        return MathSymbolInput(self.input, self.button, self.container)

    def tearDown(self) -> None:
        self.symbols.destroy()
        for key, value in self._saved_storage.items():
            if value:
                window.localStorage.setItem(key, value)
            else:
                window.localStorage.removeItem(key)

    # ----- helpers -----

    def set_value(self, value: Any, start: Optional[int] = None, end: Optional[int] = None) -> None:
        """Assign the value through the DOM and place the selection (default: caret at the end)."""
        self.input.value = value
        length = unit_length(self.input.value)
        start = length if start is None else start
        end = start if end is None else end
        self.input.setSelectionRange(start, end)

    def type_text(self, value: Any, start: Optional[int] = None) -> None:
        """Set the value and caret, then fire the input event typing would fire."""
        self.set_value(value, start)
        self.input.dispatchEvent(window.Event.new("input", {"bubbles": True}))

    def key(self, key: str, code: str = "", **modifiers: Any) -> Any:
        """Dispatch a keydown on the input and return the event."""
        init: Dict[str, Any] = {"key": key, "code": code, "bubbles": True, "cancelable": True}
        init.update(modifiers)
        event = window.KeyboardEvent.new("keydown", init)
        self.input.dispatchEvent(event)
        return event

    def key_up(self, key: str, code: str = "") -> None:
        init = {"key": key, "code": code, "bubbles": True, "cancelable": True}
        self.input.dispatchEvent(window.KeyboardEvent.new("keyup", init))

    def assert_value(self, expected: Any, caret: int, message: str = "") -> None:
        self.assertEqual(code_units(self.input.value), code_units(expected), message)
        self.assertEqual(int(self.input.selectionStart), caret, message)
        self.assertEqual(int(self.input.selectionEnd), caret, message)

    def palette_cells(self) -> List[Any]:
        return list(self.container.querySelectorAll(f"#{GROUP_GRID_ID} .symbol-cell"))

    def show_group(self, group_index: int) -> None:
        self.container.querySelectorAll(".symbol-palette-tab")[group_index].click()


class TestMathSymbolInsertion(_SymbolInputFixture):
    def test_alt_shortcut_after_emoji_inserts_between_code_points(self) -> None:
        # 😀😀😀b with the caret before b (UTF-16 offset 6)
        self.set_value(EMOJI + EMOJI + EMOJI + "b", 6)
        event = self.key("p", "KeyP", altKey=True)
        self.assertTrue(event.defaultPrevented)
        self.assert_value(EMOJI + EMOJI + EMOJI + "πb", 7)

    def test_insert_before_equals_after_astral_math_letter(self) -> None:
        # 𝑥😀=y with the caret before "=" (offset 4)
        self.set_value(MATH_X + EMOJI + "=y", 4)
        self.key("p", "KeyP", altKey=True)
        self.assert_value(MATH_X + EMOJI + "π=y", 5)

    def test_completion_after_emoji_leaves_no_stray_letters(self) -> None:
        self.type_text(EMOJI + " \\al")
        self.assertTrue(self.symbols.completion.visible)
        self.assertTrue(self.key("Tab").defaultPrevented)
        self.assert_value(EMOJI + " α", 4)

    def test_space_after_exact_name_after_emoji(self) -> None:
        self.type_text(EMOJI + "\\theta")
        self.assertTrue(self.key(" ").defaultPrevented)
        self.assert_value(EMOJI + "θ ", 4)

    def test_insert_replaces_selection_containing_emoji(self) -> None:
        self.set_value("a" + EMOJI + "b", 1, 3)
        self.symbols.insert("≤")
        self.assert_value("a≤b", 2)

    def test_insert_into_long_message(self) -> None:
        text = "x" * 20000
        self.set_value(text + EMOJI + "\\al")
        self.input.dispatchEvent(window.Event.new("input", {"bubbles": True}))
        self.assertTrue(self.symbols.completion.visible)
        self.key("Tab")
        self.assert_value(text + EMOJI + "α", 20003)

    def test_completing_inside_a_word_replaces_the_whole_word(self) -> None:
        self.type_text("\\alpha + 1", 3)  # \al|pha
        self.assertTrue(self.symbols.completion.visible)
        self.key("Tab")
        self.assert_value("α + 1", 1)

    def test_completion_keeps_digits_that_follow_the_word(self) -> None:
        self.type_text("\\pi2", 3)  # \pi|2
        self.key("Tab")
        self.assert_value("π2", 1)


class TestMathSymbolInputKeys(_SymbolInputFixture):
    def test_enter_without_popup_is_left_to_send(self) -> None:
        self.type_text("x^2")
        self.assertFalse(self.key("Enter").defaultPrevented)
        self.assert_value("x^2", 3)

    def test_enter_after_arrow_key_accepts_highlighted_suggestion(self) -> None:
        self.type_text("\\al")
        second = search_symbols("al")[1]
        self.assertTrue(self.key("ArrowDown").defaultPrevented)
        self.assertTrue(self.symbols.completion.navigated)
        self.assertTrue(self.key("Enter").defaultPrevented)
        self.assert_value(second.insert, unit_length(second.insert))
        self.assertFalse(self.symbols.completion.visible)

    def test_enter_without_navigation_sends_the_backslash_text(self) -> None:
        # a message ending in "\n" is sent as typed rather than completed to ≠
        self.type_text("a\\n")
        self.assertTrue(self.symbols.completion.visible)
        self.assertFalse(self.key("Enter").defaultPrevented)
        self.assert_value("a\\n", 3)
        self.assertFalse(self.symbols.completion.visible)

    def test_enter_after_clicking_a_restored_message_sends_it(self) -> None:
        self.set_value("a\\n")
        self.input.dispatchEvent(window.MouseEvent.new("click", {"bubbles": True}))
        self.assertTrue(self.symbols.completion.visible)
        self.assertFalse(self.key("Enter").defaultPrevented)

    def test_tab_accepts_first_suggestion_without_navigation(self) -> None:
        self.type_text("a\\n")
        self.assertTrue(self.key("Tab").defaultPrevented)
        self.assert_value("a≠", 2)

    def test_escape_dismisses_suggestions(self) -> None:
        self.type_text("\\al")
        self.assertTrue(self.key("Escape").defaultPrevented)
        self.assertFalse(self.symbols.completion.visible)

    def test_enter_with_palette_opened_by_mouse_is_left_to_send(self) -> None:
        self.button.click()
        self.assertTrue(self.symbols.palette.visible)
        self.assertFalse(self.key("Enter").defaultPrevented)
        self.assertFalse(self.symbols.palette.visible)

    def test_ctrl_up_opens_palette_and_enter_inserts_highlighted_symbol(self) -> None:
        self.set_value("x")
        self.assertTrue(self.key("ArrowUp", "ArrowUp", ctrlKey=True).defaultPrevented)
        self.assertTrue(self.symbols.palette.visible)
        first = self.palette_cells()[0].text
        self.assertTrue(self.key("Enter").defaultPrevented)
        self.assert_value("x" + first, 1 + unit_length(first))

    def test_symbol_handler_defers_to_an_earlier_handler(self) -> None:
        # The slash-command autocomplete is bound first and cancels the keys it uses
        self.symbols.destroy()
        self.input = html.INPUT(type="text")
        self.container <= self.input
        self.input.bind("keydown", lambda event: event.preventDefault())
        self.symbols = self._attach()
        self.type_text("\\al")
        self.key("Tab")
        self.assert_value("\\al", 3)

    def test_no_completion_in_slash_commands(self) -> None:
        self.type_text("/load \\al")
        self.assertFalse(self.symbols.completion.visible)

    def test_alt_letter_inserts_symbol(self) -> None:
        self.set_value("2")
        self.assertTrue(self.key("p", "KeyP", altKey=True).defaultPrevented)
        self.assert_value("2π", 2)
        self.assertTrue(self.key("P", "KeyP", altKey=True, shiftKey=True).defaultPrevented)
        self.assert_value("2πΠ", 3)

    def test_alt_follows_key_labels_not_positions(self) -> None:
        # AZERTY: the key labelled A sits where QWERTY has Q
        self.set_value("")
        self.assertTrue(self.key("a", "KeyQ", altKey=True).defaultPrevented)
        self.assert_value("α", 1)

    def test_alt_digit_typed_with_shift_inserts_superscript(self) -> None:
        self.set_value("x")
        self.assertTrue(self.key("2", "Digit2", altKey=True, shiftKey=True).defaultPrevented)
        self.assert_value("x²", 2)

    def test_macos_option_characters_are_left_alone(self) -> None:
        self.set_value("x")
        for key, code in (("π", "KeyP"), ("@", "KeyL"), ("[", "Digit5"), ("Dead", "KeyU"), ("ß", "KeyS")):
            self.assertFalse(self.key(key, code, altKey=True).defaultPrevented, key)
        self.assert_value("x", 1)

    def test_altgraph_ctrl_alt_and_meta_are_left_alone(self) -> None:
        self.set_value("x")
        self.assertFalse(self.key("p", "KeyP", altKey=True, modifierAltGraph=True).defaultPrevented)
        self.assertFalse(self.key("p", "KeyP", altKey=True, ctrlKey=True).defaultPrevented)
        self.assertFalse(self.key("p", "KeyP", altKey=True, metaKey=True).defaultPrevented)
        self.assert_value("x", 1)

    def test_right_alt_is_left_alone(self) -> None:
        self.set_value("x")
        self.key("Alt", "AltRight", altKey=True)
        self.assertFalse(self.key("p", "KeyP", altKey=True).defaultPrevented)
        self.key_up("Alt", "AltRight")
        self.assertTrue(self.key("p", "KeyP", altKey=True).defaultPrevented)
        self.assert_value("xπ", 2)


class TestMathSymbolPopups(_SymbolInputFixture):
    def test_typing_a_backslash_name_closes_the_palette(self) -> None:
        self.button.click()
        self.type_text("\\al")
        self.assertFalse(self.symbols.palette.visible)
        self.assertTrue(self.symbols.completion.visible)

    def test_opening_the_palette_hides_suggestions(self) -> None:
        self.type_text("\\al")
        self.button.click()
        self.assertTrue(self.symbols.palette.visible)
        self.assertFalse(self.symbols.completion.visible)

    def test_ctrl_up_hides_suggestions(self) -> None:
        self.type_text("\\al")
        self.key("ArrowUp", "ArrowUp", ctrlKey=True)
        self.assertTrue(self.symbols.palette.visible)
        self.assertFalse(self.symbols.completion.visible)

    def test_slash_command_closes_the_palette(self) -> None:
        self.button.click()
        self.type_text("/")
        self.assertFalse(self.symbols.palette.visible)

    def test_aria_follows_the_open_popup(self) -> None:
        self.assertEqual(self.input.attrs.get("aria-expanded"), "false")
        self.type_text("\\al")
        self.assertEqual(self.input.attrs["aria-expanded"], "true")
        self.assertEqual(self.input.attrs["aria-controls"], COMPLETION_POPUP_ID)
        self.assertEqual(self.input.attrs["aria-activedescendant"], "symbol-suggestion-0")
        self.key("ArrowDown")
        self.assertEqual(self.input.attrs["aria-activedescendant"], "symbol-suggestion-1")
        self.key("Escape")
        self.assertEqual(self.input.attrs["aria-expanded"], "false")
        self.assertFalse(self.input.hasAttribute("aria-controls"))
        self.assertFalse(self.input.hasAttribute("aria-activedescendant"))

        self.key("ArrowUp", "ArrowUp", ctrlKey=True)
        self.assertEqual(self.input.attrs["aria-expanded"], "true")
        self.assertEqual(self.input.attrs["aria-controls"], GROUP_GRID_ID)
        self.assertEqual(self.input.attrs["aria-activedescendant"], self.palette_cells()[0].id)
        self.key("Escape")
        self.assertEqual(self.input.attrs["aria-expanded"], "false")

    def test_palette_tabs_are_a_tablist_controlling_the_grid(self) -> None:
        tabs = list(self.container.querySelectorAll(".symbol-palette-tab"))
        self.assertEqual([tab.text for tab in tabs], [label for _, label in GROUPS])
        self.assertEqual(tabs[0].parentElement.getAttribute("role"), "tablist")
        for tab in tabs:
            self.assertEqual(tab.getAttribute("role"), "tab")
            self.assertEqual(tab.getAttribute("aria-controls"), GROUP_GRID_ID)
        self.assertEqual([tab.getAttribute("aria-selected") for tab in tabs].count("true"), 1)
        self.assertEqual(tabs[0].getAttribute("aria-selected"), "true")  # Operators first
        self.show_group(2)
        self.assertEqual(tabs[2].getAttribute("aria-selected"), "true")
        self.assertEqual(window.localStorage.getItem(GROUP_STORAGE_KEY), GROUPS[2][0])


class TestEverySymbol(_SymbolInputFixture):
    # (case, text before the selection, selected text, text after it)
    CASES: List[Tuple[str, Any, Any, Any]] = [
        ("start", "", "", "x+1"),
        ("middle", "a", "", "b"),
        ("end", "2x", "", ""),
        ("selection", "a", "bc", "d"),
        ("after emoji", EMOJI + EMOJI, "", "b"),
        ("between emoji", EMOJI, "", EMOJI),
        ("after math italic", MATH_X + EMOJI, "", "=y"),
        ("replacing emoji", "a", EMOJI, "b"),
    ]

    def test_symbol_table_snapshot(self) -> None:
        # Update both when symbols are added on purpose; a mismatch otherwise means one went missing
        self.assertEqual(len(SYMBOLS), EXPECTED_SYMBOL_COUNT)
        self.assertEqual(" ".join(entry.symbol for entry in SYMBOLS), EXPECTED_SYMBOLS)

    def test_every_symbol_inserts_at_the_caret(self) -> None:
        for entry in SYMBOLS:
            for case, before, selected, after in self.CASES:
                start = unit_length(before)
                self.set_value(before + selected + after, start, start + unit_length(selected))
                self.symbols.insert(entry.insert)
                self.assert_value(
                    before + entry.insert + after,
                    start + unit_length(entry.insert),
                    f"{entry.name} ({case})",
                )

    def test_every_backslash_name_completes_to_its_symbol(self) -> None:
        for entry in SYMBOLS:
            for name in entry.latex:
                self.type_text(EMOJI + " \\" + name)
                self.assertTrue(self.symbols.completion.visible, name)
                self.assertTrue(self.key("Tab").defaultPrevented, name)
                self.assert_value(EMOJI + " " + entry.insert, 3 + unit_length(entry.insert), f"\\{name}")

    def test_every_backslash_name_converts_on_space(self) -> None:
        for entry in SYMBOLS:
            for name in entry.latex:
                self.type_text("x=\\" + name)
                self.assertTrue(self.key(" ").defaultPrevented, name)
                self.assert_value("x=" + entry.insert + " ", 3 + unit_length(entry.insert), f"\\{name}")

    def test_every_palette_symbol_has_a_labelled_cell_that_inserts_it(self) -> None:
        self.button.click()
        shown = set()
        for group_index, (group_id, _) in enumerate(GROUPS):
            self.show_group(group_index)
            expected = [entry for entry in SYMBOLS if entry.group == group_id]
            expected += [entry for entry in SYMBOLS if entry.symbol in GROUP_EXTRAS.get(group_id, [])]
            for entry in expected:
                cells = {cell.text: cell for cell in self.palette_cells()}
                self.assertIn(entry.symbol, cells, f"{entry.name} in {group_id}")
                cell = cells[entry.symbol]
                self.assertIn(entry.name, cell.getAttribute("title"), entry.name)
                self.assertIn(entry.name, cell.getAttribute("aria-label"), entry.name)
                self.set_value("x")
                cell.click()
                self.assert_value("x" + entry.insert, 1 + unit_length(entry.insert), f"click {entry.name}")
                shown.add(entry.symbol)
        self.assertEqual(shown, {entry.symbol for entry in SYMBOLS if entry.group})

    def test_every_alt_shortcut_inserts_its_symbol(self) -> None:
        covered = set()
        for key, (plain, shifted) in ALT_SHORTCUTS.items():
            self.set_value("x")
            self.assertTrue(self.key(key, key_code(key), altKey=True).defaultPrevented, key)
            self.assert_value("x" + plain, 1 + unit_length(plain), f"Alt+{key}")
            covered.add(plain)
            if shifted is not None:
                self.set_value("x")
                self.assertTrue(self.key(key.upper(), key_code(key), altKey=True, shiftKey=True).defaultPrevented)
                self.assert_value("x" + shifted, 1 + unit_length(shifted), f"Alt+Shift+{key}")
                covered.add(shifted)
        with_shortcut = {
            entry.symbol for entry in SYMBOLS if any(entry.symbol in pair for pair in ALT_SHORTCUTS.values())
        }
        self.assertEqual(covered, with_shortcut)


class _StubCommandHandler:
    def get_commands_list(self) -> List[Tuple[str, str]]:
        return [("/help", "Show help"), ("/history", "Show history")]


class TestCommandAutocompleteModifiers(unittest.TestCase):
    def setUp(self) -> None:
        self.input = html.INPUT(type="text")
        self.autocomplete = CommandAutocomplete(self.input, _StubCommandHandler())

    def tearDown(self) -> None:
        self.autocomplete.destroy()

    def key(self, key: str, **modifiers: Any) -> Any:
        init: Dict[str, Any] = {"key": key, "bubbles": True, "cancelable": True}
        init.update(modifiers)
        event = window.KeyboardEvent.new("keydown", init)
        self.input.dispatchEvent(event)
        return event

    def test_modified_keys_pass_through_the_open_list(self) -> None:
        self.input.value = "/h"
        self.input.dispatchEvent(window.Event.new("input", {"bubbles": True}))
        self.assertTrue(self.autocomplete.visible)
        self.assertFalse(self.key("ArrowUp", ctrlKey=True).defaultPrevented)
        self.assertFalse(self.key("ArrowDown", metaKey=True).defaultPrevented)
        self.assertFalse(self.key("Enter", altKey=True).defaultPrevented)
        self.assertEqual(self.autocomplete.selected_index, 0)
        self.assertTrue(self.autocomplete.visible)
        self.assertTrue(self.key("ArrowDown").defaultPrevented)
        self.assertEqual(self.autocomplete.selected_index, 1)

    def test_ctrl_up_opens_the_palette_in_place_of_the_list(self) -> None:
        saved = {key: window.localStorage.getItem(key) for key in (RECENT_STORAGE_KEY, GROUP_STORAGE_KEY)}
        container = html.DIV()
        container <= self.input
        symbols = MathSymbolInput(self.input, html.BUTTON("Σ"), container)
        try:
            self.input.value = "/h"
            self.input.dispatchEvent(window.Event.new("input", {"bubbles": True}))
            self.assertTrue(self.autocomplete.visible)
            self.assertTrue(self.key("ArrowUp", ctrlKey=True).defaultPrevented)
            self.assertTrue(symbols.palette.visible)
            self.assertFalse(self.autocomplete.visible)
        finally:
            symbols.destroy()
            for key, value in saved.items():
                if value:
                    window.localStorage.setItem(key, value)
                else:
                    window.localStorage.removeItem(key)
