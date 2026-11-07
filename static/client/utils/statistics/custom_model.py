"""The model of a custom regression: an expression in x with named constants.

``read_model(text)`` checks a model such as ``a*exp(-b*x)*cos(c*x) + d`` and finds its
constants: every name it reads other than x, in order of first appearance. Unclear input
is refused with a hint, never guessed (``ax`` is not read as a*x). ``CustomModel``
evaluates the model at many x values for given constants.

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

import compiled_expression
import expression_functions as ef

MAX_CONSTANTS = 10
EXAMPLE = "a*exp(-b*x)*cos(c*x) + d"

# "y =", "y ~" or "f(x) =" before the model
_MODEL_PREFIX = re.compile(r"^\s*(?:[yY]|[A-Za-z]\w*\s*\(\s*x\s*\))\s*(?:=|~)\s*")
_NAME = re.compile(r"[^\W\d]\w*")
# Two operands side by side with only spaces between them (a sin(b x), b x): a product without '*'
_SPACED_PRODUCT = re.compile(r"([A-Za-z0-9_.]+)\s+([A-Za-z_][A-Za-z0-9_]*)")
_LAMBDA = re.compile(r"\blambda\b")
# Function names never mistaken for constants in the hints below (constants e and pi are values)
_VALUE_NAMES = frozenset(ef.CONSTANTS)

Batch = Callable[[Sequence[float], Sequence[float]], List[float]]


class CustomModel:
    """A model in x with named constants, compiled once.

    Attributes:
        text: The model as given, without a leading "y =".
        constants: The constants, in order of first appearance.
        power_constants: Constants read in the exponent of a power of x itself (x^n).
        growth_constants: Constants read in the base of a power whose exponent reads x (b^x).
    """

    __slots__ = ("text", "constants", "power_constants", "growth_constants", "_batch", "_trig")

    def __init__(
        self,
        text: str,
        constants: Tuple[str, ...],
        power_constants: FrozenSet[str] = frozenset(),
        growth_constants: FrozenSet[str] = frozenset(),
    ) -> None:
        self.text = text
        self.constants = constants
        self.power_constants = power_constants
        self.growth_constants = growth_constants
        compiled = compiled_expression.compile_expression(text, ("x",) + constants)
        self._batch = _batch_evaluator(compiled.bind(), len(constants))
        self._trig: Optional[List[Callable[..., float]]] = None

    def predictions(self, x_data: Sequence[float], params: Sequence[float]) -> Optional[List[float]]:
        """The model's values at x_data, or None if any is not a finite real number."""
        try:
            values: List[float] = self._batch(x_data, params)
        except Exception:
            return None
        total = sum(values)
        # inf - inf and nan - nan are nan; a finite sum of finite values gives 0
        return values if total - total == 0.0 else None

    def trig_arguments(self) -> List[Callable[..., float]]:
        """The arguments of the model's trigonometric calls (b*x + c in sin(b*x + c)), each as f(x, *params).

        Compiled on first use; an argument that does not compile on its own is left out.
        """
        if self._trig is None:
            self._trig = []
            for argument in _trig_argument_texts(self.text):
                try:
                    compiled = compiled_expression.compile_expression(argument, ("x",) + self.constants)
                except Exception:
                    continue
                self._trig.append(compiled.bind())
        return self._trig

    def with_values(self, values: Mapping[str, float], digits: Optional[int] = None) -> str:
        """The model text with each constant replaced by its value.

        digits rounds each value to that many significant digits; the text never uses
        exponent notation (1e-05 is written 0.00001). A negative value, or one written
        right after a number (2b), goes in parentheses.
        """
        text = self.text
        for name in self.constants:
            value = float(values[name])
            if digits is not None and value != 0.0:
                value = float(f"{value:.{digits}g}")
            value += 0.0  # -0.0 is written 0
            number = compiled_expression.plain_number_text(value)
            if number.endswith(".0"):
                number = number[:-2]
            if value < 0.0 or re.search(r"[0-9.]\s*" + re.escape(name) + r"(?!\w)", text):
                number = f"({number})"
            text = compiled_expression.substitute_variable(text, name, number)
        return text


_TRIG_CALL = re.compile(r"\b(?:sin|cos|tan|sec|csc|cot)\s*\(")


def _trig_argument_texts(text: str) -> List[str]:
    """The argument texts of sin(...), cos(...) and the other trigonometric calls in text."""
    arguments = []
    for match in _TRIG_CALL.finditer(text):
        depth, start = 1, match.end()
        for index in range(start, len(text)):
            depth += {"(": 1, ")": -1}.get(text[index], 0)
            if depth == 0:
                arguments.append(text[start:index])
                break
    return arguments


def _batch_evaluator(evaluate: Callable[..., float], count: int) -> Batch:
    """A function giving evaluate(x, *params) for every x, with the constants passed one by one.

    Brython calls a function with *params two to three times more slowly than with the
    arguments written out, and fits spend most of their time here.
    """
    if count == 1:
        return lambda xs, p: [evaluate(x, p[0]) for x in xs]
    if count == 2:
        return lambda xs, p: [evaluate(x, p[0], p[1]) for x in xs]
    if count == 3:
        return lambda xs, p: [evaluate(x, p[0], p[1], p[2]) for x in xs]
    if count == 4:
        return lambda xs, p: [evaluate(x, p[0], p[1], p[2], p[3]) for x in xs]
    if count == 5:
        return lambda xs, p: [evaluate(x, p[0], p[1], p[2], p[3], p[4]) for x in xs]
    if count == 6:
        return lambda xs, p: [evaluate(x, p[0], p[1], p[2], p[3], p[4], p[5]) for x in xs]
    if count == 7:
        return lambda xs, p: [evaluate(x, p[0], p[1], p[2], p[3], p[4], p[5], p[6]) for x in xs]
    if count == 8:
        return lambda xs, p: [evaluate(x, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]) for x in xs]
    if count == 9:
        return lambda xs, p: [evaluate(x, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]) for x in xs]
    return lambda xs, p: [evaluate(x, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]) for x in xs]


def read_model(text: Any) -> CustomModel:
    """The model, checked; its constants are every name it reads other than x.

    Raises:
        ValueError: With a hint, for an empty or unreadable model, one that does not read
            x, reads no constant or more than MAX_CONSTANTS, uses a function name as a
            value (gamma), or reads a name that looks like a glued product or call (ax,
            xc, sinx) or cannot name a constant.
    """
    from utils.parameter_expressions import called_name_hint

    source = _MODEL_PREFIX.sub("", str(text if text is not None else "")).strip()
    if not source:
        raise ValueError(f"Give the model as an expression in x with named constants, such as {EXAMPLE}.")
    if "=" in source or "~" in source:
        raise ValueError(f"Give the model as an expression in x without '=', such as {EXAMPLE}.")
    if _LAMBDA.search(source):
        raise ValueError(f"In '{source}', lambda is a reserved word; name the constant k or λ instead.")
    try:
        compiled = compiled_expression.compile_expression(source, ())
    except Exception as exc:
        hint = called_name_hint(source)
        if hint is not None:
            # The fit creates the constants: drop the hint's advice to create them first
            if " (and create " in hint:
                hint = hint.split(" (and create ")[0] + ", or use a known function such as sin(x)."
            raise ValueError(hint) from exc
        spaced = _SPACED_PRODUCT.search(source)
        if spaced is not None:
            raise ValueError(
                f"In '{source}', '{spaced.group(0)}' has no operator between its parts; write products with '*' "
                f"({spaced.group(1)}*{spaced.group(2)}), such as {EXAMPLE}."
            ) from exc
        raise ValueError(
            f"Could not read the model '{source}': {exc}. Write every function with parentheses and every "
            f"product with '*': {EXAMPLE}."
        ) from exc
    tree = ast.parse(compiled.python_text, mode="eval")
    _refuse_bare_functions(source, tree)
    _refuse_bare_e(source, tree)
    names = compiled.free_names
    constants = _in_order(source, [name for name in names if name != "x"])
    for name in constants:
        _check_constant_name(source, name, constants)
    if "x" not in names:
        raise ValueError(f"The model '{source}' does not read x. Write it in x (rewrite t as x), such as {EXAMPLE}.")
    if not constants:
        raise ValueError(
            f"The model '{source}' has no constants to fit. Name them (a*x^2 + b), or plot it with draw_function."
        )
    if len(constants) > MAX_CONSTANTS:
        raise ValueError(f"The model has {len(constants)} constants; at most {MAX_CONSTANTS} can be fitted.")
    return CustomModel(source, constants, _power_constants(tree, constants), _growth_constants(tree, constants))


def _in_order(source: str, names: List[str]) -> Tuple[str, ...]:
    """The names in order of first appearance in source."""
    first: Dict[str, int] = {}
    for index, token in enumerate(_NAME.findall(source)):
        first.setdefault(token, index)
    return tuple(sorted(names, key=lambda name: first.get(name, len(first))))


def _refuse_bare_functions(source: str, tree: ast.AST) -> None:
    """A function name read as a value (gamma in exp(-gamma*x)) is not a constant: refuse it."""
    nodes = list(ast.walk(tree))
    called = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
    for node in nodes:
        if isinstance(node, ast.Name) and id(node) not in called:
            name = node.id
            if name in ef.RESERVED_NAMES and name not in _VALUE_NAMES and name not in ef.IMAGINARY_UNITS:
                name = _written_name(source, name)
                raise ValueError(
                    f"In '{source}', '{name}' is a function, so it cannot name a constant; "
                    f"use another name, such as {name[0]} or {name[0]}1."
                )


def _refuse_bare_e(source: str, tree: ast.AST) -> None:
    """e outside e^(...) is Euler's number, never fitted: refuse it, as a..e naming five constants is common."""
    nodes = list(ast.walk(tree))
    bases = {id(node.left) for node in nodes if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)}
    if any(isinstance(node, ast.Name) and node.id == "e" and id(node) not in bases for node in nodes):
        raise ValueError(
            f"In '{source}', e is Euler's number (2.718...), so it cannot name a constant to fit; "
            f"use another name, such as e1 or k (e is only read in powers such as e^(k*x))."
        )


def _written_name(source: str, name: str) -> str:
    """The spelling the user wrote for a function name the compiler renamed (ln for log)."""
    tokens = set(_NAME.findall(source))
    if name in tokens:
        return name
    aliases = {**ef.CALL_ALIASES, **ef.WORD_ALIASES}
    return next((alias for alias in sorted(tokens) if aliases.get(alias) == name), name)


def _power_constants(tree: ast.AST, constants: Tuple[str, ...]) -> FrozenSet[str]:
    """Constants read in the exponent of x itself (x^n, x^(n+1))."""
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if isinstance(node.left, ast.Name) and node.left.id == "x":
                found.update(
                    child.id for child in ast.walk(node.right) if isinstance(child, ast.Name) and child.id in constants
                )
    return frozenset(found)


def _growth_constants(tree: ast.AST, constants: Tuple[str, ...]) -> FrozenSet[str]:
    """Constants read in the base of a power whose exponent reads x (b^x, (1 + r)^(x/2))."""
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if any(isinstance(child, ast.Name) and child.id == "x" for child in ast.walk(node.right)):
                found.update(
                    child.id for child in ast.walk(node.left) if isinstance(child, ast.Name) and child.id in constants
                )
    return frozenset(found)


def _check_constant_name(source: str, name: str, constants: Tuple[str, ...]) -> None:
    """Refuse a constant name that hides a product or call with x (ax, xc, sinx).

    A name ending in x is refused when what comes before the x is one letter, a function
    or another constant of the model (ax, sinx, kx with k a constant); Vmax and ymax are
    names. A name starting with x is refused when the rest is one letter or another
    constant (xc, xk); xmax and x0 are names.
    """
    from drawables.parameter import check_parameter_name

    message = f"In '{source}', '{name}' reads as one constant."
    others = set(constants) - {name}
    head = name[:-1]
    if name.endswith("x") and head and (len(head) == 1 or head in ef.ALLOWED_FUNCTIONS or head in others):
        if head in ef.ALLOWED_FUNCTIONS:
            raise ValueError(f"{message} Functions need parentheses: write {head}(x).")
        raise ValueError(f"{message} For {head} times x write {head}*x (products need '*').")
    tail = name[1:]
    if name.startswith("x") and tail[:1].isalpha() and (len(tail) == 1 or tail in others):
        raise ValueError(
            f"{message} For x times {tail} write x*{tail}; to name a constant, do not start it "
            f"with x and one letter (x0, xmax and h are fine)."
        )
    try:
        check_parameter_name(name)
    except ValueError as exc:
        message = str(exc).replace("cannot name a parameter", "cannot name a constant")
        if name == "y":
            message += " The model gives y: write it in x only."
        raise ValueError(message) from exc
