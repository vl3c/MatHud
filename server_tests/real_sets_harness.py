"""The browser's numeric inequality path (InequalitySolver without SymPy) in CPython.

The browser evaluates with math.js; here compile_expression's Python evaluators stand in,
with numpy float64 arithmetic so that an overflow is +-inf, inf - inf is NaN and 1/0 is inf
as in math.js, and log, gamma, floor, ceil and sqrt as math.js has them (log(0) is
-Infinity; floor(1.999999999998) is 2; sqrt(-1) and (-2)^(1/2) are complex, so not real). The relations are InequalitySolver._relations' own; the seeds (the real roots of the
polynomial parts) are found as InequalitySolver._seeds finds them, with SymPy for nerdamer.
Used by test_real_sets_battery.py.
"""

from __future__ import annotations

import contextlib
import math
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from server_tests import client_renderer  # noqa: F401  (browser stub, so the client modules import)
import compiled_expression as ce
from utils.cas_numeric import polynomial_roots
from utils.cas_text import split_relation
from utils.inequality_solver import _TRIG_CALL, _TRIG_WINDOW, InequalitySolver
from utils.real_sets import Piece, Relation, SignChartFailure, Window, format_solution, sign_chart
from utils.symbolic_checks import SymbolicChecks

MAX_SEED_DEGREE = 40  # MathUtils._MAX_NUMERIC_POLYNOMIAL_DEGREE

RealFunction = Callable[[float], Optional[float]]

_GLOBALS: Dict[str, Any] = dict(ce._lambda_globals(ce.FUNCTIONS_ALL))
_GLOBALS.update({"exp": np.exp, "sinh": np.sinh, "cosh": np.cosh})


def _logarithm(function: Callable[..., float]) -> Callable[..., float]:
    """math.js's log at 0: -Infinity, where Python raises."""

    def log(value: float, *base: float) -> float:
        if value == 0:
            return -math.inf
        return function(value, *base)

    return log


def _gamma(value: float) -> float:
    """math.js's gamma at 0 and the negative integers: Infinity, where Python raises."""
    if value <= 0 and value == math.floor(value):
        return math.inf
    return math.gamma(value)


def _nearly_integer(value: float) -> Optional[float]:
    """The integer math.js 14 takes value for in floor and ceil (relTol 1e-12, absTol 1e-15), else None."""
    nearest = float(round(value))
    return nearest if abs(value - nearest) <= max(1e-12 * max(abs(value), abs(nearest)), 1e-15) else None


def _floor(value: float) -> float:
    """math.js floor: 1.999999999998 is 2."""
    nearest = _nearly_integer(value)
    return nearest if nearest is not None else float(math.floor(value))


def _ceil(value: float) -> float:
    nearest = _nearly_integer(value)
    return nearest if nearest is not None else float(math.ceil(value))


def _square_root(function: Callable[[float], float]) -> Callable[[float], float]:
    """math.js's sqrt of a negative number: complex, so not real (MathUtils.sqrt gives NaN)."""

    def sqrt(value: float) -> float:
        if value < 0:
            raise ValueError("not real")
        return function(value)

    return sqrt


_GLOBALS.update({name: _logarithm(_GLOBALS[name]) for name in ("log", "log10", "log2") if name in _GLOBALS})
_GLOBALS.update({"gamma": _gamma, "floor": _floor, "ceil": _ceil, "sqrt": _square_root(_GLOBALS["sqrt"])})


def evaluator(text: str) -> RealFunction:
    """text as f(x), evaluated like math.js: None where it is not real or cannot be evaluated."""
    compiled = ce.compile_expression(text, ("x",), ce.FUNCTIONS_ALL)
    code = compile(ce._lambda_tree(compiled._parsed.tree, ("x",)), "<string>", mode="eval")
    bound = eval(code, _GLOBALS)

    def evaluate(point: float) -> Optional[float]:
        with np.errstate(all="ignore"):
            try:
                value = bound(np.float64(point))
            except OverflowError:
                return math.inf
            except (ArithmeticError, ValueError, TypeError):
                return None
        if isinstance(value, complex):
            return None
        if value != value and not _real_in_python(bound, point):
            return None  # (-2)^(1/2): complex in math.js, NaN in numpy
        return float(value)

    return evaluate


def _real_in_python(bound: Callable[[float], Any], point: float) -> bool:
    """Whether a NaN is real arithmetic's (inf - inf, 0/0), not a complex power: Python's own
    floats give a complex number for (-2)**0.5 and raise on overflow and division by zero."""
    try:
        value = bound(float(point))
    except (OverflowError, ZeroDivisionError):
        return True
    except (ArithmeticError, ValueError, TypeError):
        return False
    return not isinstance(value, complex)


def _numeric_value(text: Any) -> Optional[float]:
    try:
        value = evaluator(str(text))(0.0)
    except Exception:
        return None
    if value is None or not math.isfinite(value):
        return None
    try:
        free = ce.compile_expression(str(text), (), ce.FUNCTIONS_ALL).free_names
    except Exception:
        return None
    return None if free else value


def _real_function(expression: str, variable: str, others: Optional[Sequence[str]] = None) -> Optional[RealFunction]:
    try:
        return evaluator(str(expression))
    except Exception:
        return None


@contextlib.contextmanager
def _evaluated_here() -> Iterator[None]:
    """SymbolicChecks evaluating sides, constants and interval ends with the evaluator, not math.js."""
    replaced: Dict[str, Callable[..., Any]] = {"numeric_value": _numeric_value, "real_function": _real_function}
    saved = {name: SymbolicChecks.__dict__[name] for name in replaced}
    for name, function in replaced.items():
        setattr(SymbolicChecks, name, staticmethod(function))
    try:
        yield
    finally:
        for name, original in saved.items():
            setattr(SymbolicChecks, name, original)


def relations(parts: Sequence[str], signs: Sequence[str], roots: Sequence[float] = ()) -> List[Relation]:
    """InequalitySolver._relations, with the evaluator for math.js."""
    return list(InequalitySolver._relations(parts, signs, "x", roots))


def seeds(parts: Sequence[str], signs: Sequence[str]) -> List[float]:
    """InequalitySolver._seeds: the real roots of lhs - rhs, of it with its denominators cleared
    and of the denominators (MathUtils._denominators), each when it is a polynomial of degree 40
    at most, by cas_numeric.polynomial_roots. SymPy stands in for nerdamer's coefficients."""
    from utils.math_utils import MathUtils

    found: List[float] = []
    for index in range(len(signs)):
        difference = f"({parts[index]})-({parts[index + 1]})"
        denominators = MathUtils._denominators(difference)
        cleared = "*".join([f"({difference})"] + [f"({denominator})" for denominator in denominators])
        for polynomial in [difference, cleared] + list(denominators):
            coefficients = _coefficients(polynomial)
            if coefficients is None or not 1 <= len(coefficients) - 1 <= MAX_SEED_DEGREE:
                continue
            found += [
                root.real
                for root in polynomial_roots(coefficients)
                if abs(root.imag) <= 1e-9 * max(1.0, abs(root.real))
            ]
    return sorted(set(found))


def _coefficients(text: str) -> Optional[List[float]]:
    """[c0, c1, ...] of text as a polynomial in x, or None."""
    import sympy as sp

    from static.sympy_cas import ExpressionError, parse_expression

    x = sp.Symbol("x", real=True)
    try:
        expression = parse_expression(text, real=True)
    except ExpressionError:
        return None
    expression = sp.expand(expression.subs({symbol: x for symbol in expression.free_symbols}))
    if not expression.has(x) or not expression.is_polynomial(x):
        return None
    try:
        return [float(value) for value in reversed(sp.Poly(expression, x).all_coeffs())]
    except (TypeError, ValueError, sp.PolynomialError):
        return None


def prepared(text: str, interval: Optional[str]) -> Tuple[List[Relation], Optional[Window], List[float], str]:
    """(relations, window, seeds, relation text) as InequalitySolver.solve prepares them."""
    with _evaluated_here():
        parts, signs = split_relation(InequalitySolver.prepared(text))
        window = InequalitySolver.window(interval)
        if window is None and any(_TRIG_CALL.search(part) for part in parts):
            window = Window(_TRIG_WINDOW[0], _TRIG_WINDOW[1], True, True, _TRIG_WINDOW[2], _TRIG_WINDOW[3])
        roots = seeds(parts, signs)
        return relations(parts, signs, roots), window, roots, InequalitySolver._relation_text(parts, signs)


def solve(text: str, interval: Optional[str] = None) -> Tuple[Optional[List[Piece]], str]:
    """(the sign chart's pieces, or None on failure; the answer text without its note)."""
    found, window, roots, relation_text = prepared(text, interval)
    try:
        pieces = sign_chart(found, window, roots)
    except SignChartFailure as error:
        return None, f"Error: could not solve {relation_text} reliably ({error})."
    return pieces, str(format_solution("x", pieces, relation_text, window))


# ----- expected sets --------------------------------------------------------------------------

_INFINITY = {"Infinity": math.inf, "-Infinity": -math.inf}


def _end(text: str) -> float:
    text = text.strip()
    if text in _INFINITY:
        return _INFINITY[text]
    value = _numeric_value(text)
    if value is None:
        raise ValueError(f"cannot read the end {text!r}")
    return value


def parse_set(notation: str) -> List[Tuple[float, float, bool, bool]]:
    """ "(-1, 0) ∪ [2, Infinity) ∪ {5}" as (lower, upper, lower closed, upper closed); "{}" is empty."""
    if notation.strip() == "{}":
        return []
    result = []
    for part in notation.split(" ∪ "):
        part = part.strip()
        if part.startswith("{"):
            value = _end(part[1:-1])
            result.append((value, value, True, True))
            continue
        match = re.fullmatch(r"([\[(])(.*), (.*)([\])])", part)
        if match is None:
            raise ValueError(f"cannot read {part!r}")
        result.append((_end(match.group(2)), _end(match.group(3)), match.group(1) == "[", match.group(4) == "]"))
    return result


def same_set(
    pieces: Sequence[Piece], expected: Sequence[Tuple[float, float, bool, bool]], tolerance: float = 1e-9
) -> bool:
    """Whether pieces are the expected set: the same intervals, ends within tolerance (relative)."""
    if len(pieces) != len(expected):
        return False
    for piece, (lower, upper, lower_closed, upper_closed) in zip(pieces, expected):
        for got, want in ((piece.lower, lower), (piece.upper, upper)):
            if math.isinf(want) or math.isinf(got):
                if got != want:
                    return False
            elif abs(got - want) > tolerance * max(1e-300, abs(want)):
                return False
        if math.isfinite(lower) and piece.lower_closed != lower_closed:
            return False
        if math.isfinite(upper) and piece.upper_closed != upper_closed:
            return False
    return True


def pieces_of(expected: Sequence[Tuple[float, float, bool, bool]]) -> List[Piece]:
    return [Piece(lower, upper, lower_closed, upper_closed) for lower, upper, lower_closed, upper_closed in expected]
