"""Read equations and inequalities in x and y for implicit curves and shaded regions.

``parse_equation("x^2 + y^2 = 25")`` gives an ``ImplicitEquation``: g(x, y) = LHS - RHS,
whose zero set is the curve, plus the top-level factors of g when one side is 0.

``parse_inequality("y > x^2 - 2 and y < 1")`` gives an ``InequalityTree``: comparisons
(chains such as -2 < x < 3 are the "and" of each neighbouring pair) joined by "and",
"or" and top-level commas (systems: x + y <= 10, x >= 0, y >= 0), grouped by
parentheses. Its ``value`` is negative exactly inside the region: each comparison A < B
gives A - B (B - A for >), "and" takes the largest of its parts and "or" the smallest.
A comparison that is undefined at a point (the square root of a negative) is false there.

Input follows the tools' standard syntax (functions with parentheses, explicit products);
calculator spellings (sin x, sinx, xy) are refused with the form to write, never guessed.

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple, Union

import compiled_expression
from expression_validator import ExpressionValidator

VARIABLES = ("x", "y")
MAX_ATOMS = 8
MAX_TEXT_LENGTH = 500

_NAN = float("nan")
_INF = float("inf")

# Sample points where an equation that holds everywhere (x = x) is caught
_IDENTITY_PROBES = ((0.3711, -1.2093), (2.4417, 0.8813), (-3.1416, 2.7183), (5.5021, -4.0373), (-0.6931, -7.389))

_UNICODE_OPERATORS = (("≤", "<="), ("≦", "<="), ("≥", ">="), ("≧", ">="), ("≠", "!="), ("−", "-"))
# Comparison operators, longest first ("=<" and "=>" have one reading: <= and >=)
_COMPARISON = re.compile(r"<=|>=|=<|=>|==|!=|<|>|=")
_OPERATOR_NAMES = {"<": "<", "<=": "<=", "=<": "<=", ">": ">", ">=": ">=", "=>": ">="}
_BOOLEAN_WORD = re.compile(r"\b(and|or|not)\b", re.IGNORECASE)
_EQUATION_EXAMPLE = "Examples: x^2 + y^2 = 25, x^2/9 + y^2/4 = 1, y^2 = x^3 - x, x = 3."
_INEQUALITY_EXAMPLE = "Examples: y < x^2, x^2 + y^2 <= 4, y > x - 1 and y < 3, x + y <= 10, x >= 0, y >= 0."
_SYNTAX_HINT = (
    "Write every function with parentheses and every product with *: sin(x), 2*x, x*y, x*(x + 1), (x + 1)*(y - 1)."
)


class ImplicitEquation:
    """An equation LHS = RHS in x and y; the curve is where g = LHS - RHS is zero.

    Attributes:
        text: The stored form, both sides as fix_math_expression writes them.
        g: LHS - RHS as a function of (x, y); NaN where undefined.
        factors: Functions whose zero sets together are the curve: the factors of a
            product or power when one side is 0, else (g,).
    """

    def __init__(self, text: str, g: Callable[[float, float], float], factors: Tuple[Callable[..., float], ...]):
        self.text = text
        self.g = g
        self.factors = factors


class InequalityAtom:
    """One comparison A op B; the region is g < 0 (g <= 0 when not strict)."""

    __slots__ = ("left", "op", "right", "g")

    def __init__(self, left: str, op: str, right: str, g: Callable[[float, float], float]) -> None:
        self.left = left
        self.op = op
        self.right = right
        self.g = g

    @property
    def strict(self) -> bool:
        return self.op in ("<", ">")


_Node = Union[int, Tuple[str, List["_Node"]]]


class InequalityTree:
    """Comparisons joined by and/or.

    Attributes:
        text: The stored form ("y > x^2 - 2 and y < 1").
        atoms: The comparisons, in the order written.
    """

    def __init__(self, text: str, atoms: List[InequalityAtom], tree: _Node) -> None:
        self.text = text
        self.atoms = atoms
        self._tree = tree
        self.value: Callable[[float, float], float] = _build_value(tree, atoms)

    def atom_of(self, x: float, y: float) -> int:
        """The index of the comparison whose value is the region's value at (x, y)."""
        index, _ = _active(self._tree, self.atoms, x, y)
        return index

    def contains(self, x: float, y: float) -> bool:
        """Whether (x, y) satisfies the inequalities (a non-strict boundary point does)."""
        return bool(_holds(self._tree, self.atoms, x, y))


# ----------------------------------------------------------------------
# Equations
# ----------------------------------------------------------------------


def parse_equation(text: str) -> ImplicitEquation:
    """Read an equation in x and y.

    Raises:
        ValueError: With what to write instead, when the text is not one equation in
            standard syntax with x or y (no "=", an inequality, a free name, a calculator
            spelling, an equation that holds everywhere).
    """
    source = _prepared(text, "equation")
    if re.search(r"<|>|!=", source):
        raise ValueError(
            f"'{text}' is an inequality, not an equation; shade its region with shade_inequality. {_EQUATION_EXAMPLE}"
        )
    parts = re.split(r"==|=", source)
    if len(parts) == 1:
        raise ValueError(f"'{text}' has no '='; give an equation in x and y. {_EQUATION_EXAMPLE}")
    if len(parts) > 2:
        raise ValueError(f"'{text}' has more than one '='; give one equation, such as x^2 + y^2 = 25.")
    left, right = (_side(part, text) for part in parts)
    combined = compiled_expression.compile_expression(f"({left}) - ({right})", VARIABLES, _FUNCTIONS)
    _check_names(combined, text, "equation")
    if not combined.variables_read:
        raise ValueError(f"'{text}' has no x or y; give an equation in x and y. {_EQUATION_EXAMPLE}")
    g = combined.bind()
    if _holds_everywhere(g):
        raise ValueError(f"'{text}' holds for every x and y (both sides are equal); there is no curve to draw.")
    return ImplicitEquation(f"{left} = {right}", g, _equation_factors(left, right, g))


def _equation_factors(left: str, right: str, g: Callable[..., float]) -> Tuple[Callable[..., float], ...]:
    """The factors (with x or y) of the non-zero side when the other side is 0, else (g,)."""
    if _is_zero(right):
        side = left
    elif _is_zero(left):
        side = right
    else:
        return (g,)
    compiled = compiled_expression.compile_expression(side, VARIABLES, _FUNCTIONS)
    factors = compiled.factors()
    if len(factors) == 1 and factors[0] is compiled:
        return (g,)
    return tuple(factor.bind() for factor in factors if factor.variables_read) or (g,)


def _is_zero(side: str) -> bool:
    return bool(re.fullmatch(r"\s*\(?\s*0+(\.0*)?\s*\)?\s*", side))


def _holds_everywhere(g: Callable[[float, float], float]) -> bool:
    for x, y in _IDENTITY_PROBES:
        try:
            if float(g(x, y)) != 0.0:
                return False
        except Exception:
            return False
    return True


# ----------------------------------------------------------------------
# Inequalities
# ----------------------------------------------------------------------


def parse_inequality(text: str) -> InequalityTree:
    """Read inequalities in x and y joined by and/or; top-level commas separate conditions
    that must all hold (y < -1 or y > 1, x > 0 is (y < -1 or y > 1) and x > 0).

    Raises:
        ValueError: With what to write instead (an equation, !=, not, a comparison
            without x or y, more than MAX_ATOMS comparisons, a free name, a calculator
            spelling, unbalanced parentheses).
    """
    source = _prepared(text, "inequality")
    atoms: List[InequalityAtom] = []
    tree = _parse_system(source, text, atoms)
    if len(atoms) > MAX_ATOMS:
        raise ValueError(f"'{text}' has {len(atoms)} comparisons; at most {MAX_ATOMS} can be shaded together.")
    return InequalityTree(_tree_text(tree, atoms, top=True), atoms, tree)


def _parse_system(source: str, text: str, atoms: List[InequalityAtom]) -> _Node:
    parts = _split_top_level(source, (",", ";"))
    if len(parts) > 1:
        return ("and", [_parse_or(part, text, atoms) for part in parts])
    return _parse_or(source, text, atoms)


def _parse_or(source: str, text: str, atoms: List[InequalityAtom]) -> _Node:
    parts = _split_top_level(source, ("or",))
    if len(parts) > 1:
        return ("or", [_parse_and(part, text, atoms) for part in parts])
    return _parse_and(source, text, atoms)


def _parse_and(source: str, text: str, atoms: List[InequalityAtom]) -> _Node:
    parts = _split_top_level(source, ("and",))
    if len(parts) > 1:
        return ("and", [_parse_operand(part, text, atoms) for part in parts])
    return _parse_operand(source, text, atoms)


def _parse_operand(source: str, text: str, atoms: List[InequalityAtom]) -> _Node:
    operand = source.strip()
    if not operand:
        raise ValueError(f"'{text}' has an empty part between 'and', 'or' or commas. {_INEQUALITY_EXAMPLE}")
    inner = _wrapped_group(operand)
    if inner is not None:
        return _parse_system(inner, text, atoms)
    if re.match(r"not\b", operand, re.IGNORECASE):
        raise ValueError(f"'not' is not supported in '{text}'; write the opposite comparison (not y < 1 is y >= 1).")
    return _parse_chain(operand, text, atoms)


def _parse_chain(operand: str, text: str, atoms: List[InequalityAtom]) -> _Node:
    """A < B or a chain A < B <= C: the "and" of each neighbouring pair."""
    pieces: List[str] = []
    operators: List[str] = []
    depth = 0
    start = 0
    position = 0
    while position < len(operand):
        character = operand[position]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif depth == 0 and character in "<>=!":
            match = _COMPARISON.match(operand, position)
            if match is not None:
                pieces.append(operand[start:position])
                operators.append(match.group(0))
                position = match.end()
                start = position
                continue
        position += 1
    pieces.append(operand[start:])
    if not operators:
        raise ValueError(f"'{operand.strip()}' in '{text}' has no comparison (<, <=, >, >=). {_INEQUALITY_EXAMPLE}")
    for operator in operators:
        if operator in ("=", "=="):
            raise ValueError(
                f"'{operand.strip()}' is an equation, not an inequality; draw it with draw_implicit_curve, "
                f"or write <= or >= for a region with its boundary."
            )
        if operator == "!=":
            raise ValueError(f"'!=' is not supported in '{text}'; write the two sides, such as y < 1 or y > 1.")
    if len(operators) > 2:
        raise ValueError(f"'{operand.strip()}' has more than two signs; a chain has at most two, as in 1 < x + y <= 5.")
    if len({_OPERATOR_NAMES[operator][0] for operator in operators}) > 1:
        raise ValueError(
            f"The two signs of a chain must point the same way, as in 1 < x + y <= 5; write '{operand.strip()}' "
            f"as comparisons joined by and."
        )
    sides = [_side(piece, text) for piece in pieces]
    indices: List[_Node] = []
    for index, operator in enumerate(operators):
        atoms.append(_atom(sides[index], _OPERATOR_NAMES[operator], sides[index + 1], text))
        indices.append(len(atoms) - 1)
    if len(indices) == 1:
        return indices[0]
    return ("chain", indices)


def _atom(left: str, operator: str, right: str, text: str) -> InequalityAtom:
    if operator in ("<", "<="):
        difference = f"({left}) - ({right})"
    else:
        difference = f"({right}) - ({left})"
    compiled = compiled_expression.compile_expression(difference, VARIABLES, _FUNCTIONS)
    _check_names(compiled, text, "inequality")
    if not compiled.variables_read:
        raise ValueError(f"'{left} {operator} {right}' in '{text}' has no x or y; every comparison needs x or y.")
    return InequalityAtom(left, operator, right, compiled.bind())


def _wrapped_group(operand: str) -> Optional[str]:
    """The inside of (...) when the parentheses wrap the whole operand and hold a comparison
    or and/or at their top level, i.e. they group inequalities, not arithmetic."""
    if not (operand.startswith("(") and operand.endswith(")")):
        return None
    depth = 0
    for position, character in enumerate(operand):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0 and position != len(operand) - 1:
                return None
    inner = operand[1:-1]
    if _has_top_level(inner, lambda part: bool(_COMPARISON.search(part) or _BOOLEAN_WORD.search(part) or "," in part)):
        return inner
    return None


def _has_top_level(text: str, test: Callable[[str], bool]) -> bool:
    """Whether test holds for the text outside every parenthesis."""
    outside: List[str] = []
    depth = 0
    for character in text:
        if character == "(":
            depth += 1
            outside.append(" ")
        elif character == ")":
            depth -= 1
            outside.append(" ")
        else:
            outside.append(character if depth == 0 else " ")
    return test("".join(outside))


def _split_top_level(source: str, separators: Sequence[str]) -> List[str]:
    """Split at the separators (whole words or punctuation) outside parentheses."""
    parts: List[str] = []
    depth = 0
    start = 0
    position = 0
    words = [separator for separator in separators if separator.isalpha()]
    marks = [separator for separator in separators if not separator.isalpha()]
    while position < len(source):
        character = source[position]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif depth == 0:
            if character in marks:
                parts.append(source[start:position])
                start = position + 1
            else:
                for word in words:
                    end = position + len(word)
                    if (
                        source[position:end].lower() == word
                        and (position == 0 or not _is_word_character(source[position - 1]))
                        and (end == len(source) or not _is_word_character(source[end]))
                    ):
                        parts.append(source[start:position])
                        start = end
                        position = end - 1
                        break
        position += 1
    parts.append(source[start:])
    return parts


def _is_word_character(character: str) -> bool:
    return character.isalnum() or character == "_"


def _tree_text(node: _Node, atoms: List[InequalityAtom], top: bool = False) -> str:
    if isinstance(node, int):
        atom = atoms[node]
        return f"{atom.left} {atom.op} {atom.right}"
    kind, children = node
    if kind == "chain":
        first = atoms[children[0]]  # type: ignore[index]
        parts = [first.left]
        for child in children:
            atom = atoms[child]  # type: ignore[index]
            parts.extend((atom.op, atom.right))
        return " ".join(parts)
    joined = f" {kind} ".join(_tree_text(child, atoms) for child in children)
    return joined if top or kind == "and" else f"({joined})"


# ----------------------------------------------------------------------
# Values
# ----------------------------------------------------------------------


def _atom_value(atom: InequalityAtom, x: float, y: float) -> float:
    """The comparison's value, +inf where it is undefined (false there)."""
    try:
        value = float(atom.g(x, y))
    except Exception:
        return _INF
    if value != value:
        return _INF
    return value


def _build_value(tree: _Node, atoms: List[InequalityAtom]) -> Callable[[float, float], float]:
    """value(x, y) < 0 exactly inside the region; NaN where nothing defines it."""
    if isinstance(tree, int):
        return atoms[tree].g

    def value(x: float, y: float) -> float:
        result = _node_value(tree, atoms, x, y)
        return _NAN if result == _INF else result

    return value


def _node_value(node: _Node, atoms: List[InequalityAtom], x: float, y: float) -> float:
    if isinstance(node, int):
        return _atom_value(atoms[node], x, y)
    kind, children = node
    values = [_node_value(child, atoms, x, y) for child in children]
    return min(values) if kind == "or" else max(values)


def _active(node: _Node, atoms: List[InequalityAtom], x: float, y: float) -> Tuple[int, float]:
    if isinstance(node, int):
        return node, _atom_value(atoms[node], x, y)
    kind, children = node
    best: Optional[Tuple[int, float]] = None
    for child in children:
        candidate = _active(child, atoms, x, y)
        if best is None or (candidate[1] < best[1] if kind == "or" else candidate[1] > best[1]):
            best = candidate
    assert best is not None
    return best


def _holds(node: _Node, atoms: List[InequalityAtom], x: float, y: float) -> bool:
    if isinstance(node, int):
        atom = atoms[node]
        value = _atom_value(atom, x, y)
        return value < 0 or (value == 0 and not atom.strict)
    kind, children = node
    results = [_holds(child, atoms, x, y) for child in children]
    return any(results) if kind == "or" else all(results)


# ----------------------------------------------------------------------
# Shared
# ----------------------------------------------------------------------

_FUNCTIONS = compiled_expression.FUNCTIONS_REAL


# Symbols for and / or that other notations use (| alone is the absolute value bars |x|)
_LOGIC_SYMBOLS = ("&", "||", "∧", "∨", "∩", "∪")


def _prepared(text: object, what: str) -> str:
    if not isinstance(text, str) or not text.strip():
        example = _EQUATION_EXAMPLE if what == "equation" else _INEQUALITY_EXAMPLE
        raise ValueError(f"Give the {what} as text in x and y. {example}")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"The {what} is longer than {MAX_TEXT_LENGTH} characters.")
    source = text
    for symbol, replacement in _UNICODE_OPERATORS:
        source = source.replace(symbol, replacement)
    depth = 0
    for character in source:
        depth += 1 if character == "(" else -1 if character == ")" else 0
        if depth < 0:
            break
    if depth != 0:
        raise ValueError(f"The parentheses in '{text}' do not balance.")
    for symbol in _LOGIC_SYMBOLS:
        if symbol in source:
            raise ValueError(f"Join comparisons with the words and / or, not '{symbol}': x > 0 and y > 0.")
    return source


def _side(text: str, whole: str) -> str:
    """One side of a comparison in stored form, refusing calculator spellings and syntax errors."""
    side = text.strip()
    if not side:
        raise ValueError(f"'{whole}' has an empty side next to a comparison sign.")
    _refuse_misspellings(side, whole)
    try:
        stored = ExpressionValidator.fix_math_expression(side)
        compiled_expression.compile_expression(stored, VARIABLES, _FUNCTIONS)
    except Exception as exc:
        reason = str(exc)
        if "Disallowed function" in reason or "Call(" in reason:
            reason = (
                "a name or a bracket followed by ( reads as a function call (for a function on the canvas, "
                "write its expression)"
            )
        elif "|" in side:
            reason = "join comparisons with or; | is only the absolute value bars |x|"
        raise ValueError(f"Could not read '{side}' in '{whole}': {reason}. {_SYNTAX_HINT}") from exc
    return stored


def _refuse_misspellings(side: str, whole: str) -> None:
    from expression_functions import CALCULATOR_SPELLINGS, ENGINE_FUNCTION_NAMES, GLUED_CALL_NAMES
    from utils.cas_text import misspellings

    found = misspellings(side, CALCULATOR_SPELLINGS, GLUED_CALL_NAMES, ENGINE_FUNCTION_NAMES, VARIABLES)
    if found:
        raise ValueError(f"In '{whole}', {found[0].reason}.")


def _check_names(compiled: compiled_expression.CompiledExpression, text: str, what: str) -> None:
    free = compiled.free_names
    if not free:
        return
    names = ", ".join(f"'{name}'" for name in free)
    message = f"The {what} '{text}' may use only x and y; {names} is not defined."
    if set(free) & {"X", "Y"}:
        raise ValueError(f"{message} Use lowercase x and y.")
    for name in free:
        if set(name) <= set("xy"):
            raise ValueError(f"{message} Write products explicitly: {'*'.join(name)}.")
    raise ValueError(f"{message} {_SYNTAX_HINT}")
