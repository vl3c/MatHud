"""simplify's closed forms of sums: summation(term, k, a, b), from the server's SymPy engine.

1. MathUtils.simplify hands over any input that cas_text.sum_call reads as one sum (or
   refuses as a sum written another way). The term is read as the solve tool reads it
   (InequalitySolver.prepared), with calculator spellings and ambiguous divisions refused;
   an index named like a constant (i, e, pi) is renamed in the term first, so no engine
   reads it as the constant.
2. SymPy finds the closed form (operation summation, static/sympy_cas.py): bounds are
   integers, a symbol n (or c*n + d), or Infinity for the upper one.
3. series_checks compares it with the sum of the terms evaluated by math.js: term by term
   for numeric bounds, at several n for a symbolic bound, against extrapolated partial sums
   for an infinite one, with the other symbols given sample values (inside the condition a
   geometric sum states). A refuted result is an error; one the check cannot judge is given.

There is no nerdamer fallback (its sum is not exact): without SymPy, numeric bounds are
summed numerically and other bounds get an error that says so.

The browser parts (math.js, SymPy requests) are imported when first used, so pytest
covers the module's texts with Python stand-ins.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from utils.cas_text import glued_factorial, glued_index, names_in, replace_name, sum_call
from utils.series_checks import (
    divergence_verdict,
    finite_sum,
    numeric_sum_verdict,
    series_estimate,
    series_verdict,
    symbolic_sum_verdict,
)

MAX_EXPRESSION_LENGTH = 1000
# An index named like a constant (or j, Python's imaginary unit, which the client's
# expression compiler refuses as a name) is renamed in the term to the first of these it lacks.
_CONSTANT_NAMES = ("i", "j", "e", "E", "pi")
_INDEX_NAMES = ("k", "m", "p", "q", "t", "u", "v", "w")
_INFINITY_NAMES = ("Infinity", "+Infinity", "inf", "+inf", "oo", "+oo")
# Values given to the other symbols of a term, one set per check (x^k/k! is checked at two x).
SAMPLE_SETS = ((0.53, 1.27, 0.71), (-0.61, 0.29, 1.9))
# A value inside a condition's interval keeps this share of a finite interval's width from its ends.
INTERIOR_MARGIN = 0.1
# A closed form in N evaluated at the upper bound is shown up to this length.
MAX_SHOWN_FORM_LENGTH = 60
# The largest sum given without SymPy: the term limit of evaluate_expression's summation.
MAX_NUMERIC_TERMS = 100000

_NO_SYMPY = (
    "a closed form of a sum needs the server's SymPy engine, which is not available "
    "(install the packages in requirements.txt and restart the app)"
)
_NUMERIC_HINT = "for its value at a given n, give the bounds as numbers, such as summation(1/k, k, 1, 100)"

Values = Dict[str, float]


class SumInput:
    """A sum ready for the engines: the prepared term, the index, the bounds' texts and the
    names of the bound symbol and the other symbols."""

    __slots__ = ("text", "term", "index", "lower", "upper", "bound", "others")

    def __init__(
        self, text: str, term: str, index: str, lower: str, upper: str, bound: Optional[str], others: List[str]
    ) -> None:
        self.text = text
        self.term = term
        self.index = index
        self.lower = lower
        self.upper = upper
        self.bound = bound
        self.others = others

    @property
    def infinite(self) -> bool:
        return self.upper == "Infinity"


class ClosedFormSums:
    """Every method is static."""

    @staticmethod
    def simplify(expression: Any) -> Optional[str]:
        """The closed form of a sum written as the whole expression, as simplify's text
        (an "Error: ..." on failure); None when expression is not a sum."""
        if not isinstance(expression, str):
            return None
        try:
            call = sum_call(expression)
        except ValueError as error:
            return f"Error: {error}"
        if call is None:
            return None
        try:
            given = ClosedFormSums._input(expression.strip(), *call)
        except ValueError as error:
            return f"Error: {error}"
        response = ClosedFormSums._ask_sympy(given)
        if response is None:
            return ClosedFormSums._without_sympy(given)
        if response.get("timeout"):
            return f"Error: SymPy ran out of time finding a closed form of {given.text}."
        if response.get("error"):
            return f"Error: {response['error']}"
        result = response.get("result")
        if not isinstance(result, dict):
            return f"Error: SymPy found no closed form of {given.text}."
        return ClosedFormSums._answer(given, result)

    # ----- input --------------------------------------------------------------------------

    @staticmethod
    def _input(text: str, term: str, index: str, lower: str, upper: str) -> SumInput:
        """The sum's parts as the engines read them.

        Raises:
            ValueError: for an index that is not a name, a term with a calculator spelling, an
                ambiguous division or a relation, or bounds math.js cannot read.
        """
        from expression_validator import ExpressionValidator
        from utils.inequality_solver import InequalitySolver
        from utils.math_utils import MathUtils

        if len(text) > MAX_EXPRESSION_LENGTH:
            raise ValueError(f"the expression is longer than {MAX_EXPRESSION_LENGTH} characters.")
        name = MathUtils._normalize_variable(index)
        if not isinstance(name, str) or not name.strip().isidentifier():
            raise ValueError(f"the index must be a variable name such as k, not {index!r}.")
        name = name.strip()
        low, high = ClosedFormSums._bound_text(lower), ClosedFormSums._bound_text(upper)
        bound_names = [found for found in names_in(low) + names_in(high) if found not in _INFINITY_NAMES]
        if name in _CONSTANT_NAMES:
            taken = set(names_in(term)) | set(bound_names)
            renamed = next(candidate for candidate in _INDEX_NAMES if candidate not in taken)
            term, name = replace_name(term, name, renamed), renamed
        if any(sign in term for sign in "=<>"):
            raise ValueError(f"give the term of the sum alone, such as k^2, not {term!r}.")
        factorial = glued_factorial(term)
        if factorial is not None:
            raise ValueError(f"{factorial[0]} is unclear; {factorial[1]}.")
        term = ExpressionValidator._replace_factorials(term)  # 1/k! as 1/factorial(k)
        ClosedFormSums._refuse_unclear_term(term, name)
        prepared = str(InequalitySolver.prepared(term))
        bound = bound_names[0] if bound_names else None
        try:
            others = InequalitySolver.other_names([prepared], name, strict=True)
        except ValueError as error:
            raise ValueError(str(error)) from None
        others = [other for other in others if other != bound]
        return SumInput(text, prepared, name, low, high, bound, others)

    @staticmethod
    def _refuse_unclear_term(term: str, index: str) -> None:
        from utils.inequality_solver import InequalitySolver

        try:
            InequalitySolver.refuse_misspellings(term, index, subject="term")
        except ValueError as error:
            # a term without the index is a constant summed (summation(1, k, 1, n) is n)
            if str(error) != f"{index} does not appear in the term.":
                raise
        InequalitySolver.refuse_ambiguous_division(term)
        glued = glued_index(term, index, _CONSTANT_NAMES + _INFINITY_NAMES)
        if glued is not None:
            raise ValueError(f"{glued[0]} is read as one variable; write {glued[1]} for a product, or rename it.")

    @staticmethod
    def _bound_text(bound: str) -> str:
        """A bound as the engines read it, with every spelling of infinity as Infinity."""
        from utils.math_utils import MathUtils

        text = str(MathUtils._cas_input(str(bound).strip())).replace(" ", "")
        return "Infinity" if text in _INFINITY_NAMES else text

    # ----- engines ------------------------------------------------------------------------

    @staticmethod
    def _ask_sympy(given: SumInput) -> Optional[Dict[str, Any]]:
        """The server's answer ({"result"}, {"error"}, {"timeout"}), None when SymPy is unavailable."""
        from utils.sympy_client import SympyClient

        return cast(
            Optional[Dict[str, Any]],
            SympyClient.compute("summation", term=given.term, index=given.index, lower=given.lower, upper=given.upper),
        )

    @staticmethod
    def _without_sympy(given: SumInput) -> str:
        """The sum of the terms for numeric bounds, else an error that SymPy is needed."""
        bounds = _numeric_bounds(given)
        if bounds is None or given.others:
            return f"Error: {_NO_SYMPY}; for a numeric value use evaluate_expression with summation('k^2', 'k', 1, 10)."
        low, high = bounds
        if high - low + 1 > MAX_NUMERIC_TERMS:
            return f"Error: {_NO_SYMPY}, and the sum has more than {MAX_NUMERIC_TERMS} terms to add numerically."
        summed = finite_sum(_term_function(given, {}), low, high)
        if summed is None:
            return f"Error: a term of {given.text} is not a real number for some {given.index} in the range."
        return f"{_number_text(summed[0])} (summed numerically: {_NO_SYMPY})"

    # ----- answers ------------------------------------------------------------------------

    @staticmethod
    def _answer(given: SumInput, result: Dict[str, Any]) -> str:
        if result.get("empty"):
            return "0 (an empty sum: the lower bound is above the upper bound)"
        if result.get("diverges"):
            return ClosedFormSums._divergent(given, result.get("limit"))
        value = result.get("value")
        if isinstance(value, str):
            return ClosedFormSums._checked(given, result, value)
        if (result.get("too_long") or result.get("evaluated")) and isinstance(result.get("decimal"), str):
            return ClosedFormSums._decimal(given, result)
        return ClosedFormSums._no_closed_form(given, result)

    @staticmethod
    def _checked(given: SumInput, result: Dict[str, Any], value: str) -> str:
        """value with its condition and decimal, or an error when the numeric check refutes it."""
        if _verdict(given, result, value) is False:
            return (
                f"Error: SymPy gave {value} as the sum {given.text}, but a numeric check of its terms "
                "shows it is wrong, so no result is given."
            )
        text = value
        if isinstance(result.get("condition"), str):
            text += f" (for {result['condition']})"
        decimal = result.get("decimal")
        if isinstance(decimal, str):
            text += f" (about {_decimal_text(decimal)})"
        return text

    @staticmethod
    def _decimal(given: SumInput, result: Dict[str, Any]) -> str:
        """The decimal of a constant too long to show exactly, or of a closed form in N
        evaluated at the upper bound."""
        decimal = str(result["decimal"])
        if _verdict(given, result, decimal) is False:
            return (
                f"Error: SymPy gave about {_decimal_text(decimal)} as the sum {given.text}, but a numeric check "
                "of its terms shows it is wrong, so no result is given."
            )
        if result.get("evaluated"):
            symbol, general = result.get("general_symbol"), str(result.get("general"))
            form = general if len(general) <= MAX_SHOWN_FORM_LENGTH else f"in {symbol}"
            return f"about {_decimal_text(decimal)} (the closed form {form} at {symbol} = {given.upper})"
        return f"about {_decimal_text(decimal)} (the exact value is too long to show)"

    @staticmethod
    def _divergent(given: SumInput, limit: Any) -> str:
        estimate = series_estimate(_term_function(given, _samples(given, {})[0]), _lower_value(given) or 0)
        if not given.others and estimate is not None and divergence_verdict(limit, estimate) is False:
            return f"Error: SymPy says {given.text} diverges, but its partial sums settle, so no result is given."
        if limit in ("Infinity", "-Infinity"):
            return f"The series diverges: its partial sums tend to {limit}."
        return "The series diverges: its partial sums do not approach a finite value."

    @staticmethod
    def _no_closed_form(given: SumInput, result: Dict[str, Any]) -> str:
        reason = result.get("reason")
        form = result.get("sympy_form")
        named = f" (SymPy writes {form})" if isinstance(form, str) else ""
        if reason == "undefined_term":
            return (
                f"Error: a term of {given.text} is undefined (a division by zero or the like) for some {given.index}."
            )
        if reason == "too_long":
            return f"Error: the closed form of {given.text} is too long to show."
        if reason == "too_many_terms":
            return ClosedFormSums._summed_numerically(given)
        decimal = result.get("decimal")
        if isinstance(decimal, str) and _verdict(given, result, decimal) is not False:
            return f"No closed form in calculator functions{named}; the sum is about {_decimal_text(decimal)}."
        if given.infinite and result.get("converges"):
            return "The series converges, but SymPy found no closed form for it in calculator functions."
        hint = f" {_NUMERIC_HINT[0].upper()}{_NUMERIC_HINT[1:]}." if given.bound is not None else ""
        return f"Error: SymPy found no closed form of {given.text} in calculator functions{named}.{hint}"

    @staticmethod
    def _summed_numerically(given: SumInput) -> str:
        bounds = _numeric_bounds(given)
        if bounds is None or given.others or bounds[1] - bounds[0] + 1 > MAX_NUMERIC_TERMS:
            return f"Error: SymPy found no closed form of {given.text}, and it has too many terms to add exactly."
        summed = finite_sum(_term_function(given, {}), bounds[0], bounds[1])
        if summed is None:
            return f"Error: a term of {given.text} is not a real number for some {given.index} in the range."
        return (
            f"about {summed[0]:.12g} (summed numerically: SymPy found no closed form, and the exact sum has "
            "too many terms to add)"
        )


# ----- checks ---------------------------------------------------------------------------------


def _verdict(given: SumInput, result: Dict[str, Any], value: str) -> Optional[bool]:
    """False when a sample refutes value, True when one confirms it and none refutes it,
    None when none can tell."""
    from utils.symbolic_checks import SymbolicChecks

    closed = SymbolicChecks.scope_function(value)
    if closed is None:
        return None
    verdicts = [_sample_verdict(given, result, closed, values) for values in _samples(given, result)]
    if False in verdicts:
        return False
    return True if True in verdicts else None


def _sample_verdict(
    given: SumInput, result: Dict[str, Any], closed: Callable[[Values], Optional[float]], values: Values
) -> Optional[bool]:
    term = _term_function(given, values)
    if given.infinite:
        return _infinite_verdict(given, closed, values)
    bounds = _numeric_bounds(given)
    if bounds is not None:
        general = result.get("general")
        if isinstance(general, str) and isinstance(result.get("general_symbol"), str):
            return _general_verdict(given, general, str(result["general_symbol"]), bounds[0], values)
        if bounds[1] - bounds[0] + 1 > MAX_NUMERIC_TERMS:
            return None
        value = closed(values)
        return None if value is None else numeric_sum_verdict(value, term, bounds[0], bounds[1])
    bound = given.bound or "n"
    verdict = symbolic_sum_verdict(
        lambda n: closed(dict(values, **{bound: n})),
        lambda k, n: _term_function(given, dict(values, **{bound: n}))(k),
        _bound_function(given.lower, bound),
        _bound_function(given.upper, bound),
    )
    return cast(Optional[bool], verdict)


def _infinite_verdict(given: SumInput, closed: Callable[[Values], Optional[float]], values: Values) -> Optional[bool]:
    """The closed form against the partial sums, at up to three values of a symbolic lower bound."""
    starts: List[Tuple[Values, int]] = []
    if given.bound is None:
        low = _lower_value(given)
        if low is None:
            return None
        starts.append((values, low))
    else:
        lower = _bound_function(given.lower, given.bound)
        for n in range(0, 12):
            start = lower(n)
            if start is not None and float(start).is_integer():
                starts.append((dict(values, **{given.bound: n}), int(start)))
            if len(starts) == 3:
                break
    verdicts = []
    for scope, low in starts:
        value = closed(scope)
        estimate = series_estimate(_term_function(given, scope), low)
        if value is None or estimate is None or not math.isfinite(value):
            continue
        verdicts.append(series_verdict(value, estimate))
    if False in verdicts:
        return False
    return True if True in verdicts else None


def _general_verdict(given: SumInput, general: str, symbol: str, low: int, values: Values) -> Optional[bool]:
    """The closed form in N of a sum with many terms, checked as a sum up to N."""
    from utils.symbolic_checks import SymbolicChecks

    closed = SymbolicChecks.scope_function(general)
    if closed is None:
        return None
    verdict = symbolic_sum_verdict(
        lambda n: closed(dict(values, **{symbol: n})),
        lambda k, n: _term_function(given, values)(k),
        lambda n: float(low),
        lambda n: float(n),
    )
    return cast(Optional[bool], verdict)


def _samples(given: SumInput, result: Dict[str, Any]) -> List[Values]:
    """Values for the other symbols: one set per SAMPLE_SETS row, a symbol with a condition
    taking values inside it."""
    if not given.others:
        return [{}]
    condition_symbol = result.get("condition_symbol")
    inside = _interior_values(result.get("condition_intervals")) if condition_symbol else []
    sets: List[Values] = []
    for row, row_values in enumerate(SAMPLE_SETS):
        values: Values = {}
        for position, name in enumerate(given.others):
            if name == condition_symbol:
                if not inside:
                    return [{}] if not sets else sets
                values[name] = inside[row % len(inside)]
            else:
                values[name] = row_values[position % len(row_values)]
        sets.append(values)
    return sets


def _interior_values(intervals: Any) -> List[float]:
    """Values inside the condition's intervals, away from their ends (and from a removed point)."""
    from utils.symbolic_checks import SymbolicChecks

    values: List[float] = []
    for record in intervals or []:
        low = _end_value(record.get("lower"), SymbolicChecks)
        high = _end_value(record.get("upper"), SymbolicChecks)
        if low is None or high is None:
            continue
        if math.isinf(low) and math.isinf(high):
            values.extend((0.53, -0.61))
        elif math.isinf(low):
            values.extend((high - 0.7, high - 1.9))
        elif math.isinf(high):
            values.extend((low + 0.7, low + 1.9))
        else:
            width = high - low
            values.extend((low + width * 0.35, low + width * (1 - INTERIOR_MARGIN - 0.2)))
    return values


def _end_value(text: Any, checks: Any) -> Optional[float]:
    if text == "Infinity":
        return math.inf
    if text == "-Infinity":
        return -math.inf
    value = checks.numeric_value(str(text))
    return None if value is None else float(value)


# ----- evaluators -------------------------------------------------------------------------------


def _term_function(given: SumInput, values: Values) -> Callable[[int], Optional[float]]:
    """The term as a function of the index, the other symbols fixed at values."""
    from utils.symbolic_checks import SymbolicChecks

    evaluate = SymbolicChecks.scope_function(given.term)
    scope, name = dict(values), given.index

    def term(index: int) -> Optional[float]:
        if evaluate is None:
            return None
        scope[name] = index
        # no typing.cast in this hot loop: Optional[float] costs 30 us per call in Brython
        value: Optional[float] = evaluate(scope)
        return value

    return term


def _bound_function(text: str, symbol: str) -> Callable[[int], Optional[float]]:
    from utils.symbolic_checks import SymbolicChecks

    evaluate = SymbolicChecks.scope_function(text)

    def bound(n: int) -> Optional[float]:
        return None if evaluate is None else evaluate({symbol: n})

    return bound


def _lower_value(given: SumInput) -> Optional[int]:
    from utils.symbolic_checks import SymbolicChecks

    value = SymbolicChecks.numeric_value(given.lower)
    return int(value) if value is not None and float(value).is_integer() else None


def _numeric_bounds(given: SumInput) -> Optional[Tuple[int, int]]:
    """The bounds as integers, or None when either is symbolic, infinite or not whole."""
    from utils.symbolic_checks import SymbolicChecks

    if given.infinite or given.bound is not None:
        return None
    ends = [SymbolicChecks.numeric_value(text) for text in (given.lower, given.upper)]
    if any(end is None or not float(end).is_integer() for end in ends):
        return None
    return int(cast(float, ends[0])), int(cast(float, ends[1]))


# ----- texts -----------------------------------------------------------------------------------


def _decimal_text(decimal: str) -> str:
    """A decimal from the server to 12 significant digits (as text when beyond a float's range)."""
    try:
        value = float(decimal)
    except (TypeError, ValueError, OverflowError):  # Brython raises TypeError for 1.9e+30103
        return decimal
    return f"{value:.12g}" if math.isfinite(value) else decimal


def _number_text(total: float) -> str:
    """A numeric sum: an integer when it is whole and exact in a float, else 12 significant digits."""
    if total.is_integer() and abs(total) < 2.0**53:
        return str(int(total))
    return f"about {total:.12g}"
