"""Sequences u(n) and orbits of a function g, for plot_sequence and plot_cobweb.

A sequence is given by its term u(n), in the index ``n``:

- explicit: an expression in ``n`` only, such as ``1/n``, ``(-1)^n/n`` or ``2^n``;
- recursive: an expression naming the previous terms as ``u(n-1)`` and ``u(n-2)`` (and
  maybe ``n``), such as ``u(n-1) + u(n-2)`` or ``0.5*u(n-1) + 1``, computed from given
  first terms.

A leading ``u(n) =`` is dropped. Other spellings (another index letter, another sequence
letter, ``u_{n-1}``, ``u(n+1)``, named parameters) are refused with a hint naming the
accepted form, never guessed. Terms are floats: n is passed as a float (an exact big
integer for ``2^(2^n)`` would hang the page), and only a ``TypeError`` (``factorial`` of a
float) retries with an int n, for |n| <= 170.

An orbit is x0, g(x0), g(g(x0)), ... of an expression g in ``x``. ``classify_orbit`` says
what the orbit does, judged on up to ``CLASSIFY_STEPS`` iterations (more than are drawn):
it settles on a fixed point (refined by a secant solve, with the slope there and whether it
attracts), settles into a cycle (with its multiplier), escapes (grows past the bound),
diverges (grows steadily), becomes undefined (g fails), or none of these. These are
statements about the computed iterations only, and the results say so.

Pure Python (no browser imports): server-side pytest imports it directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from compiled_expression import FUNCTIONS_ALL, compile_expression

# Terms in one sequence and steps in one drawn orbit; the data plots also stop at 1000 values.
MAX_SEQUENCE_TERMS = 1000
MAX_ITERATION_STEPS = 500
DEFAULT_ITERATION_STEPS = 20
# Iterations classify_orbit may run, beyond the drawn ones, to find what the orbit does.
CLASSIFY_STEPS = 2000
# Largest |n| a sequence may start or end at.
MAX_INDEX = 10**6
# Largest |n| for which factorial(n) is retried with an int n (171! is not a finite float).
MAX_FACTORIAL_INDEX = 170
# An orbit value beyond max(ESCAPE_BOUND, ESCAPE_FACTOR * |x0|) in magnitude escapes.
ESCAPE_BOUND = 1e12
ESCAPE_FACTOR = 1e6
# The web is drawn up to the first value beyond DRAW_FACTOR * max(1, |x0|) in magnitude.
DRAW_FACTOR = 1e3
# Longest cycle classify_orbit looks for, and how many final values decide "diverges".
MAX_PERIOD = 12
DIVERGENCE_WINDOW = 20
# Relative tolerance for "equal" orbit values, and for trend ties.
ORBIT_TOLERANCE = 1e-9
TREND_TOLERANCE = 1e-12
# Undefined indices listed in a result (the count says how many there are).
MAX_LISTED_UNDEFINED = 10

INDEX = "n"
_PREVIOUS = ("u_prev1", "u_prev2")
_IMAGINARY_UNITS = ("i", "j")
_SEQUENCE_EXAMPLE = "for example 1/n, (-1)^n/n, or u(n-1) + u(n-2) with two initial terms"
_FUNCTION_EXAMPLE = "for example cos(x) or 3.2*x*(1-x)"
_SEQUENCE_PRODUCT = "3.2*u(n-1)*(1-u(n-1))"
_FUNCTION_PRODUCT = "3.2*x*(1-x)"


# ---------------------------------------------------------------------------
# Scanning the text (character scans: Brython's re silently fails some patterns)
# ---------------------------------------------------------------------------
def _is_name_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _starts_token(text: str, index: int) -> bool:
    """Whether a name starting at ``index`` is not the tail of a longer name (a digit before it is a factor)."""
    return index == 0 or not (text[index - 1].isalpha() or text[index - 1] == "_")


def _skip_spaces(text: str, index: int) -> int:
    while index < len(text) and text[index] == " ":
        index += 1
    return index


def _read_call_argument(text: str, open_index: int) -> Optional[Tuple[str, int]]:
    """The text inside the parentheses that open at ``open_index``, and the index after them."""
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index], index + 1
    return None


def _strip_prefix(text: str, names: Sequence[str]) -> str:
    """Drop a leading ``<name> =`` (one reading: the right-hand side follows)."""
    stripped = text.strip()
    for name in names:
        if stripped.replace(" ", "").startswith(name + "=") and not stripped.replace(" ", "").startswith(name + "=="):
            return stripped[stripped.index("=") + 1 :].strip()
    return stripped


def _lag_of(argument: str) -> Optional[int]:
    """1 for ``n-1``, 2 for ``n-2`` (spaces allowed), None for anything else."""
    compact = argument.replace(" ", "")
    if compact == "n-1":
        return 1
    if compact == "n-2":
        return 2
    return None


def _replace_previous_terms(text: str) -> Tuple[str, int]:
    """Replace each ``u(n-1)``/``u(n-2)`` by an internal name; return the text and the order.

    Raises:
        ValueError: ``u(...)`` with another argument (``u(n+1)``, ``u(n)``, ``u(n-3)``).
    """
    parts: List[str] = []
    order = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "u" and _starts_token(text, index) and not _is_name_char(text[index + 1 : index + 2] or " "):
            after = _skip_spaces(text, index + 1)
            if after < len(text) and text[after] == "(":
                call = _read_call_argument(text, after)
                if call is None:
                    raise ValueError(f"u( has no closing parenthesis in '{text}'.")
                argument, end = call
                lag = _lag_of(argument)
                if lag is None:
                    raise ValueError(
                        f"u({argument.strip()}) is not a previous term. Give u(n) in terms of u(n-1) and "
                        f"u(n-2) (and n), {_SEQUENCE_EXAMPLE}."
                    )
                parts.append(_PREVIOUS[lag - 1])
                order = max(order, lag)
                index = end
                continue
        parts.append(char)
        index += 1
    return "".join(parts), order


def _whole_tokens(text: str) -> List[str]:
    """The names in ``text``, each read whole (letters, digits and _ after a letter)."""
    tokens: List[str] = []
    index = 0
    while index < len(text):
        if (text[index].isalpha() or text[index] == "_") and _starts_token(text, index):
            end = index
            while end < len(text) and _is_name_char(text[end]):
                end += 1
            tokens.append(text[index:end])
            index = end
        else:
            index += 1
    return tokens


def _other_notation_hint(text: str) -> Optional[str]:
    """A hint when the text writes previous terms in a form other than u(n-1).

    Catches a letter followed by ``_`` or ``[`` and an n (``u_{n-1}``, ``a_n``, ``u[n-1]``)
    and another letter called on ``n - k`` or ``n + k`` (``a(n-1)``, ``x(n+1)``).
    """
    compact = text.replace(" ", "")
    for index, char in enumerate(compact):
        if not char.isalpha() or not _starts_token(compact, index):
            continue
        rest = compact[index + 1 :]
        if rest[:1] in ("_", "[") and "n" in rest[:4]:
            return f"Write the previous terms as u(n-1) and u(n-2), not '{char}{rest[:6]}', {_SEQUENCE_EXAMPLE}."
        if char != "u" and (rest.startswith("(n-") or rest.startswith("(n+")):
            return (
                f"Name the sequence u: write the previous terms as u(n-1) and u(n-2), not "
                f"'{char}{rest[:5]}', {_SEQUENCE_EXAMPLE}."
            )
    return None


def _forward_rule_hint(text: str) -> Optional[str]:
    """A hint for the forward form ``u(n+1) = f(u(n))`` (Casio RECUR, NumWorks)."""
    compact = text.replace(" ", "")
    if compact.startswith("u(n+1)=") or compact.startswith("u(n+2)="):
        return (
            "Shift the index: write u(n) = ... in terms of u(n-1) and u(n-2) (for u(n+1) = 2*u(n), "
            "give u(n-1) in place of u(n): 2*u(n-1))."
        )
    return None


def _free_name_error(names: Sequence[str], text: str, variable: str, example: str) -> str:
    """'write the index as n' for a lone name in a text without n or u(...); else 'write a number'."""
    tokens = _whole_tokens(text)
    if (
        variable == INDEX
        and len(names) == 1
        and INDEX not in tokens
        and not any(_PREVIOUS[i] in tokens for i in range(2))
    ):
        return f"'{names[0]}' is not defined: write the index as n ({example})."
    listed = ", ".join(f"'{name}'" for name in names)
    return (
        f"{listed} {'is' if len(names) == 1 else 'are'} not defined: write a number in place of each "
        f"(named parameters are not supported), {example}."
    )


def _compile_error(exc: ValueError, shown: str, example: str, product: str) -> str:
    """The validator's message, or a plain hint where it would show internals or an AST dump."""
    message = str(exc)
    if "Disallowed function" in message or "u_prev" in message:
        return (
            f"Cannot read '{shown}': a name or a parenthesis is followed by '(' (as in x(1-x)). "
            f"Write every product with *, as in {product}, and call functions as sin(x)."
        )
    return f"Cannot read '{shown}': {message}. Write it {example}."


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SequenceRule:
    """A parsed sequence term.

    Attributes:
        expression: The text as stored (the ``u(n) =`` prefix dropped).
        order: 0 for an explicit term, 1 or 2 for a recursion (the largest lag used).
        evaluate: Takes (n, u(n-1), u(n-2)) and returns u(n); explicit terms ignore the
            previous terms.
    """

    expression: str
    order: int
    evaluate: Callable[[Any, float, float], float]


def parse_sequence(text: Any) -> SequenceRule:
    """Parse u(n) (see the module docstring).

    Raises:
        ValueError: Empty text, another ``=``, a refused spelling or an undefined name,
            each with a hint naming the accepted form.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"The sequence is empty. Give u(n), {_SEQUENCE_EXAMPLE}.")
    forward = _forward_rule_hint(text)
    if forward is not None:
        raise ValueError(forward)
    expression = _strip_prefix(text, ("u(n)",))
    if not expression:
        raise ValueError(f"The sequence is empty. Give the right-hand side of u(n) = ..., {_SEQUENCE_EXAMPLE}.")
    if "=" in expression.replace("==", ""):
        raise ValueError(f"'{text}' has an '='. Give only the right-hand side of u(n) = ..., {_SEQUENCE_EXAMPLE}.")
    hint = _other_notation_hint(expression)
    if hint is not None:
        raise ValueError(hint)
    rewritten, order = _replace_previous_terms(expression)
    units = [token for token in _whole_tokens(rewritten) if token in _IMAGINARY_UNITS]
    if units:
        raise ValueError(f"'{units[0]}' is the imaginary unit: write the index as n ({_SEQUENCE_EXAMPLE}).")
    try:
        compiled = compile_expression(rewritten, (INDEX,) + _PREVIOUS, FUNCTIONS_ALL)
    except ValueError as exc:
        raise ValueError(_compile_error(exc, expression, _SEQUENCE_EXAMPLE, _SEQUENCE_PRODUCT)) from exc
    free = compiled.free_names
    if free:
        raise ValueError(_free_name_error(free, rewritten, INDEX, _SEQUENCE_EXAMPLE))
    return SequenceRule(expression=expression, order=order, evaluate=compiled.bind())


def whole_number(value: Any, name: str) -> int:
    """``value`` as an int: a finite whole number (5.0 accepted, bools refused)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if not math.isfinite(value) or float(value) != math.floor(value):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    return int(value)


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _real_value(evaluate: Callable[..., float], *args: Any) -> Optional[float]:
    """The value, or None when evaluating raises or gives a complex or non-finite number."""
    try:
        value = evaluate(*args)
    except Exception:
        return None
    return _as_real(value)


def _as_real(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _term(rule: SequenceRule, n: int, previous: float, before: float) -> Optional[float]:
    """u(n) with a float n; a TypeError (factorial of a float) retries with an int n, for |n| <= 170."""
    try:
        return _as_real(rule.evaluate(float(n), previous, before))
    except TypeError:
        if abs(n) > MAX_FACTORIAL_INDEX:
            return None
        return _real_value(rule.evaluate, n, previous, before)
    except Exception:
        return None


@dataclass
class SequenceTerms:
    """The computed terms of one sequence.

    Attributes:
        n_values: The indices with a defined term, ascending.
        terms: u(n) for each of them.
        undefined_at: Indices of an explicit sequence whose term is undefined.
        stopped_at: The index where a recursion stopped (its term undefined), or None.
    """

    n_values: List[int]
    terms: List[float]
    undefined_at: List[int] = field(default_factory=list)
    stopped_at: Optional[int] = None


def check_index_range(start: Any, end: Any) -> Tuple[int, int]:
    first = whole_number(start, "start")
    last = whole_number(end, "end")
    for name, value in (("start", first), ("end", last)):
        if abs(value) > MAX_INDEX:
            raise ValueError(f"{name} must be between -{MAX_INDEX} and {MAX_INDEX}, got {value}")
    if last < first:
        raise ValueError(f"end ({last}) must not be less than start ({first})")
    count = last - first + 1
    if count > MAX_SEQUENCE_TERMS:
        raise ValueError(
            f"n from {first} to {last} is {count} terms; a sequence plot takes at most {MAX_SEQUENCE_TERMS}"
        )
    return first, last


def check_initial_terms(rule: SequenceRule, initial_terms: Any) -> List[float]:
    if rule.order == 0:
        if initial_terms:
            raise ValueError("An explicit u(n) takes no initial_terms; give null, or write u(n) using u(n-1).")
        return []
    if not isinstance(initial_terms, (list, tuple)) or len(initial_terms) != rule.order:
        needed = (
            "one initial term, [u(start)]"
            if rule.order == 1
            else "two initial terms, [u(start), u(start+1)] in that order"
        )
        uses = "u(n-1)" if rule.order == 1 else "u(n-2)"
        raise ValueError(f"u(n) = {rule.expression} uses {uses}, so give {needed} in initial_terms")
    return [finite_number(value, "initial_terms") for value in initial_terms]


def sequence_terms(rule: SequenceRule, start: Any, end: Any, initial_terms: Any = None) -> SequenceTerms:
    """The terms u(start) .. u(end) (see the module docstring for undefined terms).

    Raises:
        ValueError: Bad bounds or initial terms, or no term defined in the range.
    """
    first, last = check_index_range(start, end)
    initial = check_initial_terms(rule, initial_terms)
    terms = _explicit_terms(rule, first, last) if rule.order == 0 else _recursive_terms(rule, first, last, initial)
    if not terms.terms:
        raise ValueError(
            f"u(n) = {rule.expression} is not defined (or too large) for any n from {first} to {last}; nothing to plot"
        )
    return terms


def _explicit_terms(rule: SequenceRule, first: int, last: int) -> SequenceTerms:
    result = SequenceTerms(n_values=[], terms=[])
    for n in range(first, last + 1):
        value = _term(rule, n, 0.0, 0.0)
        if value is None:
            result.undefined_at.append(n)
        else:
            result.n_values.append(n)
            result.terms.append(value)
    return result


def _recursive_terms(rule: SequenceRule, first: int, last: int, initial: List[float]) -> SequenceTerms:
    result = SequenceTerms(n_values=[], terms=[])
    previous, before = 0.0, 0.0  # u(n-1), u(n-2)
    for offset, n in enumerate(range(first, last + 1)):
        value = initial[offset] if offset < len(initial) else _term(rule, n, previous, before)
        if value is None:
            result.stopped_at = n
            break
        result.n_values.append(n)
        result.terms.append(value)
        previous, before = value, previous
    return result


def trend(values: Sequence[float]) -> Optional[str]:
    """ "increasing", "decreasing", "constant" or "neither" (None for no values).

    Non-strict: equal neighbours (within TREND_TOLERANCE relative) do not break a trend.
    """
    if not values:
        return None
    rises = falls = False
    for before, after in zip(values, values[1:]):
        tolerance = TREND_TOLERANCE * max(1.0, abs(before), abs(after))
        if after > before + tolerance:
            rises = True
        elif after < before - tolerance:
            falls = True
    if rises and falls:
        return "neither"
    if rises:
        return "increasing"
    if falls:
        return "decreasing"
    return "constant"


def sequence_summary(terms: SequenceTerms) -> Dict[str, Any]:
    """What a sequence plot saves and reports beyond its inputs."""
    values = terms.terms
    return {
        "n_values": list(terms.n_values),
        "terms": list(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "trend": trend(values),
        "undefined_at": list(terms.undefined_at),
        "stopped_at": terms.stopped_at,
    }


# ---------------------------------------------------------------------------
# Orbits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IterationFunction:
    """A parsed g(x): the stored text and its callable."""

    expression: str
    evaluate: Callable[[float], float]


def parse_iteration_function(text: Any) -> IterationFunction:
    """Parse g(x): an expression in x; a leading ``y =``, ``g(x) =`` or ``f(x) =`` is dropped."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"The function is empty. Give g(x), {_FUNCTION_EXAMPLE}.")
    expression = _strip_prefix(text, ("g(x)", "f(x)", "y"))
    if not expression or "=" in expression.replace("==", ""):
        raise ValueError(f"Give only the right-hand side of g(x) = ..., in x, {_FUNCTION_EXAMPLE}.")
    try:
        compiled = compile_expression(expression, ("x",), FUNCTIONS_ALL)
    except ValueError as exc:
        raise ValueError(_compile_error(exc, expression, _FUNCTION_EXAMPLE, _FUNCTION_PRODUCT)) from exc
    if compiled.free_names:
        raise ValueError(_free_name_error(compiled.free_names, expression, "x", _FUNCTION_EXAMPLE))
    return IterationFunction(expression=expression, evaluate=compiled.bind())


def check_steps(steps: Any) -> int:
    if steps is None:
        return DEFAULT_ITERATION_STEPS
    count = whole_number(steps, "steps")
    if not 1 <= count <= MAX_ITERATION_STEPS:
        raise ValueError(f"steps must be from 1 to {MAX_ITERATION_STEPS}, got {count}")
    return count


def escape_bound(x0: float) -> float:
    return max(ESCAPE_BOUND, ESCAPE_FACTOR * abs(x0))


@dataclass
class Orbit:
    """x0, x1 = g(x0), ... as computed.

    Attributes:
        values: The defined values, x0 first.
        stop: None when every step was computed; "escapes" when a value passed the escape
            bound or was infinite; "undefined" when g raised or gave a complex or NaN value.
        stopped_at: The step that stopped the orbit, or None.
    """

    values: List[float]
    stop: Optional[str] = None
    stopped_at: Optional[int] = None


def iterate(g: Callable[[float], float], x0: float, steps: int) -> Orbit:
    """Apply g ``steps`` times from x0, stopping at the first undefined or escaping value."""
    orbit = Orbit(values=[x0])
    bound = escape_bound(x0)
    current = x0
    for step in range(1, steps + 1):
        try:
            raw = g(current)
        except OverflowError:
            raw = math.inf
        except Exception:
            raw = None
        value = None if isinstance(raw, bool) or not isinstance(raw, (int, float)) else float(raw)
        if value is None or math.isnan(value):
            orbit.stop, orbit.stopped_at = "undefined", step
            break
        if math.isinf(value) or abs(value) > bound:
            orbit.stop, orbit.stopped_at = "escapes", step
            break
        orbit.values.append(value)
        current = value
    return orbit


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= ORBIT_TOLERANCE * max(1.0, abs(a), abs(b))


def _slope(g: Callable[[float], float], x: float) -> Optional[float]:
    h = 1e-6 * max(1.0, abs(x))
    right = _real_value(g, x + h)
    left = _real_value(g, x - h)
    if right is None or left is None:
        return None
    return (right - left) / (2.0 * h)


def _stability(multiplier: Optional[float]) -> Optional[bool]:
    """True when |multiplier| < 1, False when > 1, None when unknown or within 1e-6 of 1."""
    if multiplier is None or abs(abs(multiplier) - 1.0) <= 1e-6:
        return None
    return abs(multiplier) < 1.0


def classify_orbit(g: Callable[[float], float], orbit: Orbit) -> Dict[str, Any]:
    """What the orbit does (see the module docstring); keys depend on ``behaviour``.

    Every result has ``iterations``, the number of values of g computed to decide.
    """
    values = orbit.values
    iterations = len(values) - 1
    if orbit.stop == "escapes":
        return {"behaviour": "escapes", "escaped_at": orbit.stopped_at, "iterations": iterations}
    if orbit.stop == "undefined":
        return {
            "behaviour": "undefined",
            "undefined_at": orbit.stopped_at,
            "undefined_input": values[-1],
            "iterations": iterations,
        }
    result = _settled(g, values)
    if result is None:
        result = _unsettled(values)
    result["iterations"] = iterations
    return result


def _settled(g: Callable[[float], float], values: List[float]) -> Optional[Dict[str, Any]]:
    """A fixed point or cycle the last values have settled on, or None."""
    if len(values) >= 2 and _close(values[-1], values[-2]):
        image = _real_value(g, values[-1])
        if image is not None and _close(image, values[-1]):
            return _fixed_point(g, _refined_fixed_point(g, values[-2], values[-1]))
    cycle = _cycle(g, values)
    if cycle is None:
        return None
    multiplier: Optional[float] = 1.0
    for value in cycle:
        slope = _slope(g, value)
        multiplier = None if slope is None or multiplier is None else multiplier * slope
    return {
        "behaviour": "cycle",
        "period": len(cycle),
        "cycle": cycle,
        "multiplier": multiplier,
        "stable": _stability(multiplier),
    }


def _unsettled(values: List[float]) -> Dict[str, Any]:
    """``diverges`` for steady growth, else ``none`` with what the last values are doing.

    ``pattern`` is "monotone" (still moving one way), "shrinking" (the changes are shrinking)
    or "irregular" (neither: the orbit may be chaotic).
    """
    change = abs(values[-1] - values[-2]) if len(values) >= 2 else None
    tail = values[-(DIVERGENCE_WINDOW + 1) :]
    if len(tail) <= DIVERGENCE_WINDOW:
        return {"behaviour": "none", "pattern": "irregular", "last_change": change}
    sizes = [abs(v) for v in tail]
    growth = [b - a for a, b in zip(sizes, sizes[1:])]
    if all(g > 0 for g in growth) and all(b >= a * (1 - 1e-9) for a, b in zip(growth, growth[1:])):
        return {"behaviour": "diverges", "last_change": change}
    diffs = [b - a for a, b in zip(tail, tail[1:])]
    if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
        return {"behaviour": "none", "pattern": "monotone", "last_change": change}
    if abs(diffs[-1]) < 0.5 * abs(diffs[0]):
        return {"behaviour": "none", "pattern": "shrinking", "last_change": change}
    return {"behaviour": "none", "pattern": "irregular", "last_change": change}


def _refined_fixed_point(g: Callable[[float], float], before: float, last: float) -> float:
    """A secant solve of g(x) - x from the last two values, kept when it agrees with ``last``.

    Snapped to 0 when it is within tolerance of 0 and g(0) = 0 (x/2 reports 0, not 8.9e-16).
    """
    point = last
    a, b = before, last
    for _ in range(20):
        fa, fb = _real_value(g, a), _real_value(g, b)
        if fa is None or fb is None:
            break
        ha, hb = fa - a, fb - b
        if hb == 0.0 or ha == hb:
            break
        a, b = b, b - hb * (b - a) / (hb - ha)
        if not math.isfinite(b):
            break
        if abs(b - a) <= 1e-15 * max(1.0, abs(b)):
            break
    if math.isfinite(b) and abs(b - last) <= 1e-6 * max(1.0, abs(last)):
        at_root, at_last = _real_value(g, b), _real_value(g, last)
        if at_root is not None and at_last is not None and abs(at_root - b) <= abs(at_last - last):
            point = b
    if abs(point) <= ORBIT_TOLERANCE:
        zero = _real_value(g, 0.0)
        if zero is not None and abs(zero) <= ORBIT_TOLERANCE:
            point = 0.0
    return point


def _fixed_point(g: Callable[[float], float], point: float) -> Dict[str, Any]:
    slope = _slope(g, point)
    return {"behaviour": "fixed_point", "fixed_point": point, "slope": slope, "stable": _stability(slope)}


def _cycle(g: Callable[[float], float], values: List[float]) -> Optional[List[float]]:
    """The last ``p`` values when they repeat with the smallest period p in 2..MAX_PERIOD."""
    for period in range(2, MAX_PERIOD + 1):
        if len(values) < 2 * period:
            break
        tail = values[-period:]
        earlier = values[-2 * period : -period]
        if not all(_close(a, b) for a, b in zip(tail, earlier)):
            continue
        if any(_close(tail[i], tail[j]) for i in range(period) for j in range(i + 1, period)):
            continue  # a shorter cycle (or a fixed point) repeated
        image = _real_value(g, tail[-1])
        if image is None or not _close(image, tail[0]):
            continue
        start = tail.index(min(tail))
        return tail[start:] + tail[:start]
    return None


def orbit_until_settled(g: Callable[[float], float], x0: float, steps: int) -> Tuple[Orbit, Dict[str, Any]]:
    """The drawn orbit (``steps`` iterations) and the behaviour from up to CLASSIFY_STEPS.

    The longer run is extended in rounds and stops as soon as it settles on a fixed point or
    a cycle, escapes or becomes undefined.
    """
    drawn = iterate(g, x0, steps)
    if drawn.stop is not None:
        return drawn, classify_orbit(g, drawn)
    run = drawn
    while True:
        behaviour = classify_orbit(g, run)
        if behaviour["behaviour"] != "none" and behaviour["behaviour"] != "diverges":
            return drawn, behaviour
        total = len(run.values) - 1
        if total >= CLASSIFY_STEPS:
            return drawn, behaviour
        more = iterate(g, run.values[-1], min(max(total, 50), CLASSIFY_STEPS - total))
        run = Orbit(values=run.values + more.values[1:], stop=more.stop)
        if more.stopped_at is not None:
            run.stopped_at = total + more.stopped_at
        if run.stop is not None:
            return drawn, classify_orbit(g, run)


def drawn_orbit(values: Sequence[float]) -> List[float]:
    """The orbit values the web is drawn through: up to the first beyond DRAW_FACTOR * max(1, |x0|)."""
    bound = DRAW_FACTOR * max(1.0, abs(values[0]))
    shown = [values[0]]
    for value in values[1:]:
        if abs(value) > bound:
            break
        shown.append(value)
    return shown


def orbit_x_range(values: Sequence[float], extra: Sequence[float] = ()) -> Tuple[float, float]:
    """The drawn curve's interval: the values (and any fixed point or cycle), widened by 50% of the span (1 when 0)."""
    points = [float(v) for v in values] + [float(v) for v in extra if v is not None and math.isfinite(v)]
    low, high = min(points), max(points)
    span = high - low
    margin = 0.5 * span if span > 0 else 1.0
    return low - margin, high + margin
