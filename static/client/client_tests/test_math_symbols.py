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
    group_index_for,
    RECENT_LIMIT,
    SYMBOLS,
    describe_symbol,
    exact_latex_match,
    find_latex_token,
    get_symbol,
    latex_word_tail,
    lookup_alt_shortcut,
    move_grid_selection,
    sanitize_recent,
    search_symbols,
    shortcut_label,
    symbols_in_group,
    update_recent,
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

    def test_group_tab_order(self) -> None:
        # The first group is the tab the palette opens on.
        self.assertEqual(
            [label for _, label in GROUPS],
            ["Operators", "Calculus", "Sets & logic", "Geometry", "Greek"],
        )

    def test_group_index_for_known_and_unknown_ids(self) -> None:
        self.assertEqual(group_index_for("operators"), 0)
        self.assertEqual(group_index_for("greek"), len(GROUPS) - 1)
        for unknown in (None, "", "nope", 3):
            self.assertEqual(group_index_for(unknown), 0)

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
        for key, (plain, shifted) in ALT_SHORTCUTS.items():
            self.assertIsNotNone(get_symbol(plain), f"{key} -> {plain}")
            if shifted is not None:
                self.assertIsNotNone(get_symbol(shifted), f"Shift+{key} -> {shifted}")

    def test_alt_shortcut_keys_are_unshifted_characters(self) -> None:
        # Keys are what KeyboardEvent.key reports with Alt on Windows/Linux: one lower-case character
        for key in ALT_SHORTCUTS:
            self.assertEqual(len(key), 1, key)
            self.assertEqual(key, key.lower(), key)
            self.assertTrue(key.isalnum() or key in ",.=-", key)

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
        self.assertEqual(lookup_alt_shortcut("p", False), "π")
        self.assertEqual(lookup_alt_shortcut("P", True), "Π")
        self.assertEqual(lookup_alt_shortcut("a", False), "α")
        self.assertEqual(lookup_alt_shortcut("T", True), "Θ")

    def test_caps_lock_letter_without_shift_is_lower_case_symbol(self) -> None:
        self.assertEqual(lookup_alt_shortcut("P", False), "π")

    def test_shift_without_capital_is_not_handled(self) -> None:
        self.assertIsNone(lookup_alt_shortcut("A", True))

    def test_digits_and_punctuation(self) -> None:
        self.assertEqual(lookup_alt_shortcut("0", False), "⁰")
        self.assertEqual(lookup_alt_shortcut("9", False), "⁹")
        self.assertEqual(lookup_alt_shortcut(",", False), "≤")
        self.assertEqual(lookup_alt_shortcut(".", False), "≥")
        self.assertEqual(lookup_alt_shortcut("=", False), "≠")
        self.assertEqual(lookup_alt_shortcut("-", False), "⁻")
        self.assertEqual(lookup_alt_shortcut("o", False), "°")

    def test_digits_typed_with_shift_still_match(self) -> None:
        # AZERTY types digits with Shift, so key is "2" with shiftKey set
        self.assertEqual(lookup_alt_shortcut("2", True), "²")
        # US Shift+2 reports "@", which is not a shortcut
        self.assertIsNone(lookup_alt_shortcut("@", True))

    def test_characters_produced_by_option_on_macos_are_not_handled(self) -> None:
        for key in ("π", "∏", "@", "[", "ß", "¬", "≤", "Dead"):
            self.assertIsNone(lookup_alt_shortcut(key, False), key)
            self.assertIsNone(lookup_alt_shortcut(key, True), key)

    def test_unmapped_keys(self) -> None:
        self.assertIsNone(lookup_alt_shortcut("z", False))
        self.assertIsNone(lookup_alt_shortcut("Enter", False))
        self.assertIsNone(lookup_alt_shortcut("", False))
        self.assertIsNone(lookup_alt_shortcut("KeyP", False))


class TestLatexWordTail(unittest.TestCase):
    def test_letters_after_the_caret_belong_to_the_word(self) -> None:
        self.assertEqual(latex_word_tail("al", "pha + 1"), 3)
        self.assertEqual(latex_word_tail("al", "pha"), 3)

    def test_digits_after_letters(self) -> None:
        self.assertEqual(latex_word_tail("su", "p2 x"), 2)

    def test_nothing_continues_the_word(self) -> None:
        self.assertEqual(latex_word_tail("pi", " x"), 0)
        self.assertEqual(latex_word_tail("pi", ""), 0)
        self.assertEqual(latex_word_tail("pi", "+1"), 0)

    def test_digits_alone_do_not_continue_the_word(self) -> None:
        self.assertEqual(latex_word_tail("pi", "2x"), 0)

    def test_name_ending_in_a_digit_does_not_continue(self) -> None:
        self.assertEqual(latex_word_tail("sup2", "abc"), 0)
        self.assertEqual(latex_word_tail("", "abc"), 0)


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
