"""
MatHud Math Symbol Table

One data table of the math symbols the chat input can insert, plus the pure
(DOM-free) helpers the symbol palette, the Alt shortcuts and the backslash
completion share.

Each symbol has a readable name (tooltips and screen readers), one or more
LaTeX-style names for ``\\name`` completion, a palette group, and optionally
an Alt shortcut. The text inserted is the symbol itself, except for entries
such as ``lim`` whose ``insert`` text differs.

Key Features:
    - SYMBOLS / GROUPS: the table and the palette tabs
    - ALT_SHORTCUTS: Alt / Alt+Shift key codes mapped to symbols
    - insert_text: replace the selection with text, returning value and caret
    - find_latex_token / search_symbols: backslash completion
    - update_recent: most-recently-used list
    - move_grid_selection: arrow-key movement across palette sections

This module has no browser imports, so every helper is unit-testable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


class MathSymbol:
    """One insertable math symbol.

    Attributes:
        symbol: The character(s) shown in the palette.
        name: Readable name used for tooltips and aria-labels.
        latex: Names accepted after a backslash (``\\le``, ``\\leq``).
        group: Palette group id (see GROUPS), or "" for symbols that are
            reachable only by shortcut or completion.
        insert: Text inserted into the input (defaults to ``symbol``).
    """

    __slots__ = ("symbol", "name", "latex", "group", "insert")

    def __init__(
        self,
        symbol: str,
        name: str,
        latex: Sequence[str],
        group: str,
        insert: Optional[str] = None,
    ) -> None:
        self.symbol: str = symbol
        self.name: str = name
        self.latex: Tuple[str, ...] = tuple(latex)
        self.group: str = group
        self.insert: str = insert if insert is not None else symbol

    def __repr__(self) -> str:
        return f"MathSymbol({self.symbol!r}, {self.name!r})"


# Palette groups in display order: (id, tab label).
GROUPS: List[Tuple[str, str]] = [
    ("greek", "Greek"),
    ("operators", "Operators"),
    ("geometry", "Geometry"),
    ("sets", "Sets & logic"),
    ("calculus", "Calculus"),
]

# Symbols shown in a group besides the ones whose primary group it is.
GROUP_EXTRAS: Dict[str, List[str]] = {
    "calculus": ["∞"],
}


def _s(symbol: str, name: str, latex: Sequence[str], group: str, insert: Optional[str] = None) -> MathSymbol:
    return MathSymbol(symbol, name, latex, group, insert)


SYMBOLS: List[MathSymbol] = [
    # Greek (letters that look like Latin letters or logic symbols are left out)
    _s("α", "alpha", ["alpha"], "greek"),
    _s("β", "beta", ["beta"], "greek"),
    _s("γ", "gamma", ["gamma"], "greek"),
    _s("δ", "delta", ["delta"], "greek"),
    _s("ε", "epsilon", ["epsilon", "varepsilon"], "greek"),
    _s("ζ", "zeta", ["zeta"], "greek"),
    _s("η", "eta", ["eta"], "greek"),
    _s("θ", "theta", ["theta"], "greek"),
    _s("κ", "kappa", ["kappa"], "greek"),
    _s("λ", "lambda", ["lambda"], "greek"),
    _s("μ", "mu", ["mu"], "greek"),
    _s("ξ", "xi", ["xi"], "greek"),
    _s("π", "pi", ["pi"], "greek"),
    _s("ρ", "rho", ["rho"], "greek"),
    _s("σ", "sigma", ["sigma"], "greek"),
    _s("τ", "tau", ["tau"], "greek"),
    _s("φ", "phi", ["phi", "varphi"], "greek"),
    _s("χ", "chi", ["chi"], "greek"),
    _s("ψ", "psi", ["psi"], "greek"),
    _s("ω", "omega", ["omega"], "greek"),
    _s("Γ", "capital gamma", ["Gamma"], "greek"),
    _s("Δ", "capital delta", ["Delta"], "greek"),
    _s("Θ", "capital theta", ["Theta"], "greek"),
    _s("Π", "capital pi", ["Pi"], "greek"),
    _s("Σ", "capital sigma", ["Sigma"], "greek"),
    _s("Φ", "capital phi", ["Phi"], "greek"),
    _s("Ω", "capital omega", ["Omega"], "greek"),
    # Operators
    _s("×", "times", ["times"], "operators"),
    _s("÷", "divided by", ["div"], "operators"),
    _s("±", "plus or minus", ["pm"], "operators"),
    _s("√", "square root", ["sqrt"], "operators"),
    _s("²", "squared", ["squared", "sup2"], "operators"),
    _s("³", "cubed", ["cubed", "sup3"], "operators"),
    _s("ⁿ", "to the power n", ["supn"], "operators"),
    _s("≠", "not equal to", ["ne", "neq"], "operators"),
    _s("≤", "less than or equal to", ["le", "leq"], "operators"),
    _s("≥", "greater than or equal to", ["ge", "geq"], "operators"),
    _s("≈", "approximately equal to", ["approx"], "operators"),
    _s("∞", "infinity", ["infty", "infinity"], "operators"),
    _s("°", "degree", ["deg", "degree"], "operators"),
    _s("·", "dot product", ["cdot"], "operators"),
    # Superscripts reached with Alt+digit (and completion), not shown in the palette
    _s("⁰", "superscript 0", ["sup0"], ""),
    _s("¹", "superscript 1", ["sup1"], ""),
    _s("⁴", "superscript 4", ["sup4"], ""),
    _s("⁵", "superscript 5", ["sup5"], ""),
    _s("⁶", "superscript 6", ["sup6"], ""),
    _s("⁷", "superscript 7", ["sup7"], ""),
    _s("⁸", "superscript 8", ["sup8"], ""),
    _s("⁹", "superscript 9", ["sup9"], ""),
    _s("⁻", "superscript minus", ["supminus"], ""),
    # Geometry
    _s("∠", "angle", ["angle"], "geometry"),
    _s("⊥", "perpendicular to", ["perp"], "geometry"),
    _s("∥", "parallel to", ["parallel"], "geometry"),
    _s("△", "triangle", ["triangle"], "geometry"),
    _s("≅", "congruent to", ["cong"], "geometry"),
    _s("∼", "similar to", ["sim"], "geometry"),
    _s("′", "prime", ["prime"], "geometry"),
    _s("″", "double prime", ["dprime"], "geometry"),
    # Sets and logic
    _s("∈", "element of", ["in"], "sets"),
    _s("∉", "not an element of", ["notin"], "sets"),
    _s("⊂", "subset of", ["subset"], "sets"),
    _s("⊆", "subset of or equal to", ["subseteq"], "sets"),
    _s("∪", "union", ["cup", "union"], "sets"),
    _s("∩", "intersection", ["cap", "intersection"], "sets"),
    _s("∅", "empty set", ["emptyset", "varnothing"], "sets"),
    _s("ℝ", "real numbers", ["R", "reals"], "sets"),
    _s("ℕ", "natural numbers", ["N", "naturals"], "sets"),
    _s("ℤ", "integers", ["Z", "integers"], "sets"),
    _s("ℚ", "rational numbers", ["Q", "rationals"], "sets"),
    _s("ℂ", "complex numbers", ["C", "complexes"], "sets"),
    _s("∧", "logical and", ["land", "wedge"], "sets"),
    _s("∨", "logical or", ["lor", "vee"], "sets"),
    _s("¬", "logical not", ["neg", "lnot"], "sets"),
    _s("→", "right arrow", ["to", "rightarrow"], "sets"),
    _s("⇒", "implies", ["implies", "Rightarrow"], "sets"),
    _s("⇔", "if and only if", ["iff", "Leftrightarrow"], "sets"),
    _s("∀", "for all", ["forall"], "sets"),
    _s("∃", "there exists", ["exists"], "sets"),
    # Calculus
    _s("∫", "integral", ["int", "integral"], "calculus"),
    _s("∑", "summation", ["sum"], "calculus"),
    _s("∏", "product", ["prod"], "calculus"),
    _s("∂", "partial derivative", ["partial"], "calculus"),
    _s("∇", "nabla", ["nabla", "grad"], "calculus"),
    _s("lim", "limit", ["lim"], "calculus"),
]

SYMBOLS_BY_CHAR: Dict[str, MathSymbol] = {entry.symbol: entry for entry in SYMBOLS}

# Alt shortcuts keyed by KeyboardEvent.code: (Alt symbol, Alt+Shift symbol or None).
# Capitals are mapped only where they differ from Latin letters.
ALT_SHORTCUTS: Dict[str, Tuple[str, Optional[str]]] = {
    "KeyA": ("α", None),
    "KeyB": ("β", None),
    "KeyD": ("δ", "Δ"),
    "KeyF": ("φ", "Φ"),
    "KeyG": ("γ", "Γ"),
    "KeyL": ("λ", None),
    "KeyM": ("μ", None),
    "KeyO": ("°", None),
    "KeyP": ("π", "Π"),
    "KeyR": ("√", None),
    "KeyS": ("σ", "Σ"),
    "KeyT": ("θ", "Θ"),
    "KeyU": ("∞", None),
    "KeyW": ("ω", "Ω"),
    "Digit0": ("⁰", None),
    "Digit1": ("¹", None),
    "Digit2": ("²", None),
    "Digit3": ("³", None),
    "Digit4": ("⁴", None),
    "Digit5": ("⁵", None),
    "Digit6": ("⁶", None),
    "Digit7": ("⁷", None),
    "Digit8": ("⁸", None),
    "Digit9": ("⁹", None),
    "Comma": ("≤", None),
    "Period": ("≥", None),
    "Equal": ("≠", None),
    "Minus": ("⁻", None),
}

RECENT_LIMIT: int = 10
COMPLETION_LIMIT: int = 8


def _key_label(code: str) -> str:
    """Readable key name for a KeyboardEvent.code value."""
    if code.startswith("Key"):
        return code[3:]
    if code.startswith("Digit"):
        return code[5:]
    return {"Comma": ",", "Period": ".", "Equal": "=", "Minus": "-"}.get(code, code)


def _build_shortcut_labels() -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for code, (plain, shifted) in ALT_SHORTCUTS.items():
        key = _key_label(code)
        labels[plain] = f"Alt+{key}"
        if shifted:
            labels[shifted] = f"Alt+Shift+{key}"
    return labels


SHORTCUT_LABELS: Dict[str, str] = _build_shortcut_labels()


def get_symbol(symbol: str) -> Optional[MathSymbol]:
    """Return the table entry for a symbol, or None."""
    return SYMBOLS_BY_CHAR.get(symbol)


def symbols_in_group(group_id: str) -> List[MathSymbol]:
    """Return the palette symbols of a group in display order."""
    entries = [entry for entry in SYMBOLS if entry.group == group_id]
    for extra in GROUP_EXTRAS.get(group_id, []):
        entry = SYMBOLS_BY_CHAR.get(extra)
        if entry is not None and entry not in entries:
            entries.append(entry)
    return entries


def lookup_alt_shortcut(code: str, shift: bool) -> Optional[str]:
    """Return the symbol for an Alt (or Alt+Shift) key code, or None."""
    mapping = ALT_SHORTCUTS.get(code)
    if mapping is None:
        return None
    return mapping[1] if shift else mapping[0]


def shortcut_label(symbol: str) -> Optional[str]:
    """Return the Alt shortcut label of a symbol (``"Alt+P"``), or None."""
    return SHORTCUT_LABELS.get(symbol)


def describe_symbol(entry: MathSymbol) -> str:
    """Tooltip text: name, Alt shortcut and backslash names."""
    parts = [entry.name]
    label = shortcut_label(entry.symbol)
    if label:
        parts.append(label)
    if entry.latex:
        parts.append(" ".join("\\" + name for name in entry.latex))
    return " · ".join(parts)


def insert_text(value: str, selection_start: int, selection_end: int, text: str) -> Tuple[str, int]:
    """Replace the selected range of ``value`` with ``text``.

    Out-of-range or reversed selections are clamped. Returns the new value and
    the caret position just after the inserted text.
    """
    length = len(value)
    start = max(0, min(selection_start, length))
    end = max(0, min(selection_end, length))
    if end < start:
        start, end = end, start
    new_value = value[:start] + text + value[end:]
    return new_value, start + len(text)


def utf16_offset_to_index(value: str, offset: int) -> int:
    """Convert a DOM (UTF-16) selection offset to a string index."""
    units = 0
    for index, char in enumerate(value):
        if units >= offset:
            return index
        units += 2 if ord(char) > 0xFFFF else 1
    return len(value)


def index_to_utf16_offset(value: str, index: int) -> int:
    """Convert a string index to a DOM (UTF-16) selection offset."""
    return sum(2 if ord(char) > 0xFFFF else 1 for char in value[:index])


def _is_ascii_letter(char: str) -> bool:
    return ("a" <= char <= "z") or ("A" <= char <= "Z")


def find_latex_token(value: str, caret: int) -> Optional[Tuple[int, str]]:
    """Find a ``\\name`` token ending at the caret.

    A name is ASCII letters optionally followed by digits (``\\sup2``).
    Returns ``(start, name)`` where ``start`` is the index of the backslash and
    ``name`` the text typed after it, or None when the text before the caret
    is not such a token.
    """
    caret = max(0, min(caret, len(value)))
    index = caret
    while index > 0 and "0" <= value[index - 1] <= "9":
        index -= 1
    letters_end = index
    while index > 0 and _is_ascii_letter(value[index - 1]):
        index -= 1
    if index == letters_end or index == 0 or value[index - 1] != "\\":
        return None
    return index - 1, value[index:caret]


def exact_latex_match(name: str) -> Optional[MathSymbol]:
    """Return the symbol whose backslash name is exactly ``name`` (case-sensitive)."""
    for entry in SYMBOLS:
        if name in entry.latex:
            return entry
    return None


def _match_rank(entry: MathSymbol, query: str) -> Optional[int]:
    """Rank how well a symbol matches a query; lower is better, None is no match."""
    lowered = query.lower()
    if query in entry.latex:
        return 0
    if any(name.startswith(query) for name in entry.latex):
        return 1
    if any(name.lower().startswith(lowered) for name in entry.latex):
        return 2
    if any(word.startswith(lowered) for word in entry.name.lower().split()):
        return 3
    if any(lowered in name.lower() for name in entry.latex):
        return 4
    if lowered in entry.name.lower():
        return 5
    return None


def search_symbols(query: str, limit: int = COMPLETION_LIMIT) -> List[MathSymbol]:
    """Return symbols matching a backslash query: exact, then prefix, then substring."""
    if not query:
        return []
    ranked: List[Tuple[int, int, MathSymbol]] = []
    for position, entry in enumerate(SYMBOLS):
        rank = _match_rank(entry, query)
        if rank is not None:
            ranked.append((rank, position, entry))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [entry for _, _, entry in ranked[:limit]]


def replace_latex_token(value: str, token_start: int, caret: int, text: str) -> Tuple[str, int]:
    """Replace the ``\\name`` token spanning ``token_start..caret`` with ``text``."""
    return insert_text(value, token_start, caret, text)


def update_recent(recent: Sequence[str], symbol: str, limit: int = RECENT_LIMIT) -> List[str]:
    """Move ``symbol`` to the front of the recent list, dropping duplicates."""
    updated = [symbol] + [item for item in recent if item != symbol]
    return updated[: max(0, limit)]


def sanitize_recent(items: object, limit: int = RECENT_LIMIT) -> List[str]:
    """Keep only known, unique symbols from a stored recent list."""
    if not isinstance(items, (list, tuple)):
        return []
    result: List[str] = []
    for item in items:
        if isinstance(item, str) and item in SYMBOLS_BY_CHAR and item not in result:
            result.append(item)
    return result[: max(0, limit)]


def move_grid_selection(
    sizes: Sequence[int],
    columns: Sequence[int],
    section: int,
    index: int,
    key: str,
) -> Tuple[int, int]:
    """Move a palette selection with an arrow key.

    The palette is a vertical stack of sections (the recent row, then the
    group grid); each section is laid out row by row with ``columns[i]`` cells
    per row. Left/Right step through cells and cross into the neighbouring
    section, wrapping at the ends; Up/Down move by a row and cross sections
    keeping the column where possible, stopping at the top and bottom.

    Args:
        sizes: Number of cells per section; every section must be non-empty.
        columns: Cells per row for each section.
        section: Current section index.
        index: Current cell index within the section.
        key: "ArrowLeft", "ArrowRight", "ArrowUp" or "ArrowDown".

    Returns:
        The new (section, index).
    """
    if not sizes:
        return 0, 0
    count = len(sizes)
    section = max(0, min(section, count - 1))
    index = max(0, min(index, sizes[section] - 1))
    cols = max(1, columns[section])

    if key == "ArrowRight":
        if index + 1 < sizes[section]:
            return section, index + 1
        return (section + 1) % count, 0
    if key == "ArrowLeft":
        if index > 0:
            return section, index - 1
        previous = (section - 1) % count
        return previous, sizes[previous] - 1

    row, col = divmod(index, cols)
    if key == "ArrowDown":
        last_row = (sizes[section] - 1) // cols
        if row < last_row:
            return section, min(index + cols, sizes[section] - 1)
        if section + 1 < count:
            return section + 1, min(col, sizes[section + 1] - 1)
        return section, index
    if key == "ArrowUp":
        if row > 0:
            return section, index - cols
        if section > 0:
            previous = section - 1
            previous_cols = max(1, columns[previous])
            last_row_start = ((sizes[previous] - 1) // previous_cols) * previous_cols
            return previous, min(last_row_start + col, sizes[previous] - 1)
        return section, index
    return section, index
