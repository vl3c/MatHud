"""
Tests for the math symbol table and the pure helpers behind the chat input's
symbol palette, Alt shortcuts and backslash completion.
"""

from __future__ import annotations

import unittest

from math_symbols import (
    ALT_SHORTCUTS,
    GROUP_EXTRAS,
    GROUPS,
    RECENT_LIMIT,
    SYMBOLS,
    describe_symbol,
    exact_latex_match,
    find_latex_token,
    get_symbol,
    index_to_utf16_offset,
    insert_text,
    lookup_alt_shortcut,
    move_grid_selection,
    replace_latex_token,
    sanitize_recent,
    search_symbols,
    shortcut_label,
    symbols_in_group,
    update_recent,
    utf16_offset_to_index,
)


class TestMathSymbolTable(unittest.TestCase):
    def test_symbols_are_unique(self) -> None:
        symbols = [entry.symbol for entry in SYMBOLS]
        self.assertEqual(len(symbols), len(set(symbols)))

    def test_names_are_unique_and_non_empty(self) -> None:
        names = [entry.name for entry in SYMBOLS]
        self.assertTrue(all(names))
        self.assertEqual(len(names), len(set(names)))

    def test_latex_names_map_to_one_symbol(self) -> None:
        owners: dict = {}
        for entry in SYMBOLS:
            self.assertTrue(entry.latex, f"{entry.name} has no backslash name")
            for name in entry.latex:
                self.assertNotIn(name, owners, f"\\{name} used by {owners.get(name)} and {entry.symbol}")
                owners[name] = entry.symbol

    def test_latex_names_are_completable_tokens(self) -> None:
        # Every name must be found by find_latex_token (letters, then optional digits)
        for entry in SYMBOLS:
            for name in entry.latex:
                self.assertEqual(find_latex_token("\\" + name, len(name) + 1), (0, name))

    def test_groups_are_known_and_non_empty(self) -> None:
        group_ids = [group_id for group_id, _ in GROUPS]
        for entry in SYMBOLS:
            self.assertIn(entry.group, group_ids + [""], entry.symbol)
        for group_id in group_ids:
            self.assertTrue(symbols_in_group(group_id), group_id)

    def test_group_extras_are_table_entries(self) -> None:
        for group_id, extras in GROUP_EXTRAS.items():
            for symbol in extras:
                self.assertIsNotNone(get_symbol(symbol), symbol)
                self.assertIn(symbol, [entry.symbol for entry in symbols_in_group(group_id)])

    def test_infinity_is_listed_in_operators_and_calculus(self) -> None:
        self.assertIn("∞", [entry.symbol for entry in symbols_in_group("operators")])
        self.assertIn("∞", [entry.symbol for entry in symbols_in_group("calculus")])

    def test_look_alike_greek_letters_are_left_out(self) -> None:
        for symbol in ("ι", "ν", "ο", "υ", "Λ", "Ξ", "Ψ", "Α"):
            self.assertIsNone(get_symbol(symbol), symbol)

    def test_every_alt_shortcut_maps_to_a_table_entry(self) -> None:
        for code, (plain, shifted) in ALT_SHORTCUTS.items():
            self.assertIsNotNone(get_symbol(plain), f"{code} -> {plain}")
            if shifted is not None:
                self.assertIsNotNone(get_symbol(shifted), f"Shift+{code} -> {shifted}")

    def test_alt_shortcut_codes_are_keyboard_event_codes(self) -> None:
        for code in ALT_SHORTCUTS:
            self.assertTrue(
                code.startswith("Key") or code.startswith("Digit") or code in ("Comma", "Period", "Equal", "Minus"),
                code,
            )

    def test_lim_inserts_text(self) -> None:
        entry = get_symbol("lim")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.insert, "lim")

    def test_describe_symbol_lists_name_shortcut_and_latex(self) -> None:
        text = describe_symbol(get_symbol("≤"))
        self.assertIn("less than or equal to", text)
        self.assertIn("Alt+,", text)
        self.assertIn("\\le", text)
        self.assertIn("\\leq", text)

    def test_shortcut_labels(self) -> None:
        self.assertEqual(shortcut_label("π"), "Alt+P")
        self.assertEqual(shortcut_label("Π"), "Alt+Shift+P")
        self.assertEqual(shortcut_label("²"), "Alt+2")
        self.assertIsNone(shortcut_label("∫"))


class TestAltShortcutLookup(unittest.TestCase):
    def test_letters(self) -> None:
        self.assertEqual(lookup_alt_shortcut("KeyP", False), "π")
        self.assertEqual(lookup_alt_shortcut("KeyP", True), "Π")
        self.assertEqual(lookup_alt_shortcut("KeyA", False), "α")
        self.assertEqual(lookup_alt_shortcut("KeyT", True), "Θ")

    def test_shift_without_capital_is_not_handled(self) -> None:
        self.assertIsNone(lookup_alt_shortcut("KeyA", True))
        self.assertIsNone(lookup_alt_shortcut("Digit2", True))

    def test_digits_and_punctuation(self) -> None:
        self.assertEqual(lookup_alt_shortcut("Digit0", False), "⁰")
        self.assertEqual(lookup_alt_shortcut("Digit9", False), "⁹")
        self.assertEqual(lookup_alt_shortcut("Comma", False), "≤")
        self.assertEqual(lookup_alt_shortcut("Period", False), "≥")
        self.assertEqual(lookup_alt_shortcut("Equal", False), "≠")
        self.assertEqual(lookup_alt_shortcut("Minus", False), "⁻")
        self.assertEqual(lookup_alt_shortcut("KeyO", False), "°")

    def test_unmapped_codes(self) -> None:
        self.assertIsNone(lookup_alt_shortcut("KeyZ", False))
        self.assertIsNone(lookup_alt_shortcut("Enter", False))
        self.assertIsNone(lookup_alt_shortcut("", False))


class TestInsertText(unittest.TestCase):
    def test_insert_at_caret(self) -> None:
        self.assertEqual(insert_text("a+b", 1, 1, "π"), ("aπ+b", 2))

    def test_insert_at_end_and_start(self) -> None:
        self.assertEqual(insert_text("x", 1, 1, "²"), ("x²", 2))
        self.assertEqual(insert_text("x", 0, 0, "√"), ("√x", 1))

    def test_replaces_selection(self) -> None:
        self.assertEqual(insert_text("a+b=c", 2, 3, "β"), ("a+β=c", 3))

    def test_multi_character_text(self) -> None:
        self.assertEqual(insert_text("x→0", 0, 0, "lim "), ("lim x→0", 4))

    def test_reversed_and_out_of_range_selection(self) -> None:
        self.assertEqual(insert_text("abc", 3, 1, "θ"), ("aθ", 2))
        self.assertEqual(insert_text("abc", -4, 99, "θ"), ("θ", 1))

    def test_empty_value(self) -> None:
        self.assertEqual(insert_text("", 0, 0, "∞"), ("∞", 1))

    def test_utf16_offsets_without_astral_characters(self) -> None:
        self.assertEqual(utf16_offset_to_index("απβ", 2), 2)
        self.assertEqual(index_to_utf16_offset("απβ", 3), 3)
        self.assertEqual(utf16_offset_to_index("ab", 10), 2)

    def test_utf16_offsets_with_astral_characters(self) -> None:
        # The DOM counts U+1D465 (mathematical italic x) as two UTF-16 units
        value = "\U0001d465=β"
        self.assertEqual(utf16_offset_to_index(value, 2), 1)
        self.assertEqual(utf16_offset_to_index(value, 3), 2)
        self.assertEqual(index_to_utf16_offset(value, 3), 4)


class TestLatexCompletion(unittest.TestCase):
    def test_token_before_caret(self) -> None:
        self.assertEqual(find_latex_token("x = \\alp", 8), (4, "alp"))
        self.assertEqual(find_latex_token("\\pi", 3), (0, "pi"))

    def test_token_in_the_middle(self) -> None:
        value = "a \\thet + b"
        self.assertEqual(find_latex_token(value, 7), (2, "thet"))
        # Caret inside the word: only the letters before the caret count
        self.assertEqual(find_latex_token(value, 5), (2, "th"))

    def test_trailing_digits(self) -> None:
        self.assertEqual(find_latex_token("x\\sup2", 6), (1, "sup2"))
        self.assertEqual(exact_latex_match("sup2").symbol, "²")

    def test_no_token(self) -> None:
        self.assertIsNone(find_latex_token("alpha", 5))
        self.assertIsNone(find_latex_token("x \\", 3))
        self.assertIsNone(find_latex_token("\\alpha ", 7))
        self.assertIsNone(find_latex_token("\\2", 2))
        self.assertIsNone(find_latex_token("\\a2b", 4))
        self.assertIsNone(find_latex_token("", 0))

    def test_caret_is_clamped(self) -> None:
        self.assertEqual(find_latex_token("\\mu", 99), (0, "mu"))

    def test_exact_match_is_case_sensitive(self) -> None:
        self.assertEqual(exact_latex_match("delta").symbol, "δ")
        self.assertEqual(exact_latex_match("Delta").symbol, "Δ")
        self.assertEqual(exact_latex_match("leq").symbol, "≤")
        self.assertIsNone(exact_latex_match("alp"))
        self.assertIsNone(exact_latex_match("DELTA"))

    def test_search_puts_exact_and_prefix_matches_first(self) -> None:
        results = [entry.symbol for entry in search_symbols("in")]
        self.assertEqual(results[0], "∈")
        self.assertLess(results.index("∞"), results.index("⁻"))  # prefix \infty before substring "minus"

    def test_search_prefers_matching_case(self) -> None:
        self.assertEqual([entry.symbol for entry in search_symbols("D")][:2], ["Δ", "δ"])
        self.assertEqual([entry.symbol for entry in search_symbols("d")][0], "δ")

    def test_search_prefix(self) -> None:
        self.assertEqual(search_symbols("alp")[0].symbol, "α")
        self.assertEqual(search_symbols("sq")[0].symbol, "√")

    def test_search_matches_readable_names(self) -> None:
        self.assertEqual(search_symbols("union")[0].symbol, "∪")
        self.assertIn("≠", [entry.symbol for entry in search_symbols("equal")])

    def test_search_empty_and_limit(self) -> None:
        self.assertEqual(search_symbols(""), [])
        self.assertEqual(search_symbols("zzzz"), [])
        self.assertLessEqual(len(search_symbols("s")), 8)
        self.assertEqual(len(search_symbols("s", limit=3)), 3)

    def test_replace_token(self) -> None:
        value = "angle \\alp = 30"
        start, _ = find_latex_token(value, 10)
        self.assertEqual(replace_latex_token(value, start, 10, "α"), ("angle α = 30", 7))


class TestRecentSymbols(unittest.TestCase):
    def test_moves_symbol_to_front(self) -> None:
        self.assertEqual(update_recent(["α", "β", "γ"], "γ"), ["γ", "α", "β"])

    def test_adds_new_symbol(self) -> None:
        self.assertEqual(update_recent([], "π"), ["π"])
        self.assertEqual(update_recent(["α"], "π"), ["π", "α"])

    def test_limit(self) -> None:
        recent = [str(i) for i in range(RECENT_LIMIT)]
        updated = update_recent(recent, "π")
        self.assertEqual(len(updated), RECENT_LIMIT)
        self.assertEqual(updated[0], "π")
        self.assertNotIn(recent[-1], updated)
        self.assertEqual(update_recent(["α", "β"], "γ", limit=2), ["γ", "α"])

    def test_does_not_mutate_input(self) -> None:
        recent = ["α", "β"]
        update_recent(recent, "β")
        self.assertEqual(recent, ["α", "β"])

    def test_sanitize_recent(self) -> None:
        self.assertEqual(sanitize_recent(["π", "nope", "π", "lim", "", 3]), ["π", "lim"])
        self.assertEqual(sanitize_recent(None), [])
        self.assertEqual(sanitize_recent("π"), [])
        self.assertEqual(len(sanitize_recent([entry.symbol for entry in SYMBOLS])), RECENT_LIMIT)


class TestPaletteGridNavigation(unittest.TestCase):
    # Recent row of 3 cells above a 20-cell grid with 9 columns
    SIZES = [3, 20]
    COLUMNS = [3, 9]

    def move(self, section: int, index: int, key: str) -> tuple:
        return move_grid_selection(self.SIZES, self.COLUMNS, section, index, key)

    def test_left_right_within_and_across_sections(self) -> None:
        self.assertEqual(self.move(1, 0, "ArrowRight"), (1, 1))
        self.assertEqual(self.move(0, 2, "ArrowRight"), (1, 0))
        self.assertEqual(self.move(1, 0, "ArrowLeft"), (0, 2))
        self.assertEqual(self.move(1, 19, "ArrowRight"), (0, 0))
        self.assertEqual(self.move(0, 0, "ArrowLeft"), (1, 19))

    def test_down(self) -> None:
        self.assertEqual(self.move(1, 1, "ArrowDown"), (1, 10))
        self.assertEqual(self.move(1, 12, "ArrowDown"), (1, 19))  # shorter last row
        self.assertEqual(self.move(1, 19, "ArrowDown"), (1, 19))  # bottom edge
        self.assertEqual(self.move(0, 2, "ArrowDown"), (1, 2))
        self.assertEqual(move_grid_selection([12, 5], [12, 9], 0, 11, "ArrowDown"), (1, 4))

    def test_up(self) -> None:
        self.assertEqual(self.move(1, 10, "ArrowUp"), (1, 1))
        self.assertEqual(self.move(1, 1, "ArrowUp"), (0, 1))
        self.assertEqual(self.move(1, 7, "ArrowUp"), (0, 2))  # clamped to the recent row
        self.assertEqual(self.move(0, 1, "ArrowUp"), (0, 1))  # top edge
        # Up from the grid lands in the last row of a wrapped section
        self.assertEqual(move_grid_selection([12, 20], [9, 9], 1, 1, "ArrowUp"), (0, 10))

    def test_single_section(self) -> None:
        self.assertEqual(move_grid_selection([20], [9], 0, 19, "ArrowRight"), (0, 0))
        self.assertEqual(move_grid_selection([20], [9], 0, 0, "ArrowUp"), (0, 0))

    def test_invalid_input(self) -> None:
        self.assertEqual(move_grid_selection([], [], 0, 0, "ArrowDown"), (0, 0))
        self.assertEqual(self.move(5, 50, "Enter"), (1, 19))
