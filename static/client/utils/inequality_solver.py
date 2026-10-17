"""Inequalities for the solve tool: SymPy's solution set checked with math.js, else a numeric sign chart.

1. The text goes through the same input pipeline for both engines (Unicode notation,
   sin^2(x), ln, |x| bars, x(x+2) as a product), is split into its expressions and signs
   (a chain such as 1 < 2x+1 <= 5 has two) and must hold one variable only.
2. The server's SymPy engine solves it (operation solve_inequality, static/sympy_cas.py).
3. Its set is checked on a dense grid, at its ends and beside them (utils.real_sets).
   A confirmed (or uncheckable) set is the answer; a refuted one, or none (SymPy
   unavailable, no answer, out of time), gives way to the numeric sign chart, whose answer
   says it was found numerically.

Periodic inequalities without an interval are answered on [-2*pi, 2*pi], as solve lists
periodic roots.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, cast

from utils.cas_text import (
    insert_implicit_products,
    insert_letter_call_products,
    misspellings,
    parse_interval,
    split_relation,
    split_top_level_terms,
)
from utils.real_sets import (
    Magnitude,
    Piece,
    Relation,
    SignChartFailure,
    Window,
    format_solution,
    grid_reach,
    is_plain_number,
    nice_number,
    number_text,
    reach_text,
    chart,
    verify_solution_set,
)

_TRIG_CALL = re.compile(r"(?<![A-Za-z_])(?:sin|cos|tan|sec|csc|cot)\s*\(")
# Without an interval, an inequality with trig functions is solved numerically on this window.
_TRIG_WINDOW = (-2 * math.pi, 2 * math.pi, "-2*pi", "2*pi")
# A number written before a name or "(" (2pi, 3(x+1)), but not an exponent (1e5)
_IMPLICIT_PRODUCT = re.compile(r"(?<![A-Za-z_\d.])(\d+\.?\d*)(?![eE][-+]?\d)\s*(?=[A-Za-z(])")
# A number (2, 1.5, 1e20, kept whole so e20 is no name) or a name (x, cost, θ), as the
# misspelling checks read them
_NUMBER_OR_NAME = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|([^\W\d]\w*)")
# Spellings of an infinite interval end (any case, with an optional sign)
_INFINITE_ENDS = ("inf", "infinity", "oo", "∞")


class InequalitySolver:
    """Every method is static."""

    @staticmethod
    def solve(inequality: str, variable: str, interval: Optional[str] = None) -> str:
        """The real solution set of an inequality in one variable, as format_solution writes it, or "Error: ..."."""
        from utils.math_utils import MathUtils

        try:
            variable = str(MathUtils._normalize_variable(variable))
            InequalitySolver.refuse_misspellings(inequality, variable)
            parts, signs = split_relation(InequalitySolver.prepared(inequality))
            window = InequalitySolver.window(interval)
            InequalitySolver._check_variable(parts, variable)
            seeds = InequalitySolver._seeds(parts, signs, variable)
            relations = InequalitySolver._relations(parts, signs, variable, seeds)
        except ValueError as error:
            return f"Error: {error}"
        relation_text = InequalitySolver._relation_text(parts, signs)
        response = InequalitySolver._ask_sympy(relation_text, variable, window, interval)
        result = response.get("result") if response is not None else None
        sympy_window = window
        period: Optional[str] = None
        if isinstance(result, dict):
            pieces = InequalitySolver._pieces(result.get("intervals"))
            sympy_window = InequalitySolver._server_window(result.get("window")) or window
            # The pattern repeats beyond a window SymPy chose ([-2*pi, 2*pi], or one period)
            period = result.get("period") if not InequalitySolver._same_window(sympy_window, window) else None
            if pieces is not None and verify_solution_set(relations, pieces, sympy_window, seeds) != "refuted":
                return str(format_solution(variable, pieces, relation_text, sympy_window, period))
            reason = "SymPy's answer failed the numeric check"
        elif response is None:
            reason = "SymPy was not available"
        else:
            reason = "SymPy found no exact answer"
        return InequalitySolver._numeric_answer(
            relations, relation_text, variable, sympy_window, period, parts, seeds, reason
        )

    # ----- input --------------------------------------------------------------------------

    @staticmethod
    def prepared(text: str) -> str:
        """text read the same way by SymPy and math.js: bars as abs(), MathUtils._cas_input
        (after the bars, so ln|x| becomes log(abs(x))), x(x+2) as x*(x+2), and (x-1)(x+2), (x+1)2
        and (x-2)^2 (x+1) as products. Calculator spellings (sin x, sinx, xe^x) are not
        rewritten; refuse_misspellings refuses them."""
        from expression_functions import BY_NAME, CALL_ALIASES, replace_absolute_bars
        from utils.math_utils import MathUtils

        if not isinstance(text, str):
            raise ValueError("The expression must be given as text.")
        prepared = replace_absolute_bars(str(MathUtils._normalize_symbols(text)))
        prepared = str(MathUtils._cas_input(prepared))
        prepared = insert_letter_call_products(prepared, list(BY_NAME) + list(CALL_ALIASES))
        return str(insert_implicit_products(prepared))

    @staticmethod
    def misspellings(text: str, keep: Sequence[str] = ()) -> List[Any]:
        """The calculator spellings in text that the tools refuse (cas_text.misspellings):
        sin x, sinx, xlnx, xe^x, xpi, log2x. Names in keep are never misspellings."""
        from expression_functions import CALCULATOR_SPELLINGS, ENGINE_FUNCTION_NAMES, GLUED_CALL_NAMES
        from utils.math_utils import MathUtils

        normalized = str(MathUtils._normalize_symbols(text))
        return list(misspellings(normalized, CALCULATOR_SPELLINGS, GLUED_CALL_NAMES, ENGINE_FUNCTION_NAMES, keep))

    @staticmethod
    def refuse_misspellings(text: str, variable: str) -> None:
        """Raise ValueError with the standard form to write when text has:

        - a function without parentheses (sin x), or a function and digits (sin2, sqrt2);
        - a glued name (sinx, xlnx, xe^x, xpi) that holds variable or another variable of text,
          or e glued to a letter and raised to a power (ke^x);
        - a name made of calculator keys, e and pi, with variable and digits (sinxe, sinxcosx,
          exe, sin2);
        - no variable at all once it is prepared (y = mx + b solved for x).

        Other names are variables: cost in profit = revenue - cost, solved for revenue, x0 and
        xmax, and costs or revenue, though they start with a function name.

        Raises:
            ValueError: naming the misspelling and what to write instead.
        """
        from expression_functions import CALCULATOR_SPELLINGS, ENGINE_FUNCTION_NAMES, GLUED_CALL_NAMES
        from utils.cas_text import NAME_HINT, decomposes, has_inequality

        if not isinstance(text, str):
            return
        tokens = list(dict.fromkeys(_names(text)))
        names = set(tokens) | {variable}
        for misspelling in InequalitySolver.misspellings(text, keep=(variable,)):
            if misspelling.bare or misspelling.powered or (misspelling.call and not misspelling.letters):
                raise ValueError(f"{misspelling.reason}.")
            if set(misspelling.letters) & names:
                raise ValueError(f"{misspelling.reason}, or rename the variable.")
        # calculator keys, e and pi, with variable and digits between them: xmax and x0 are names
        keys = set(CALCULATOR_SPELLINGS) | set(GLUED_CALL_NAMES) | {"e", "pi"}
        functions = set(ENGINE_FUNCTION_NAMES) | keys
        for token in tokens:
            if token != variable and token not in functions and decomposes(token, keys | {variable}, keys):
                raise ValueError(f"{token} is read as one variable; {NAME_HINT}.")
        try:
            prepared = InequalitySolver.prepared(text)
        except Exception:
            return  # the engines say what they cannot read
        if variable not in _names(prepared):
            kind = "inequality" if has_inequality(prepared) else "equation"
            raise ValueError(f"{variable} does not appear in the {kind}.")

    @staticmethod
    def window(interval: Optional[str]) -> Optional[Window]:
        """The interval argument as a Window ([0, 2*pi), ...), None when there is none.

        Raises:
            ValueError: when it is not two real ends with the lower one first.
        """
        from utils.math_utils import MathUtils

        if interval is None or not str(interval).strip():
            return None
        lower_text, upper_text, lower_closed, upper_closed = parse_interval(str(MathUtils._cas_input(str(interval))))
        lower, upper = InequalitySolver._end_value(lower_text), InequalitySolver._end_value(upper_text)
        if lower is None or upper is None:
            raise ValueError(f"the interval's ends must be real numbers, not {interval!r}.")
        if not lower < upper:
            raise ValueError(f"the interval's lower end must be below its upper end, not {interval!r}.")
        return Window(
            lower,
            upper,
            lower_closed,
            upper_closed,
            _end_text(lower_text, lower),
            _end_text(upper_text, upper),
        )

    @staticmethod
    def _end_value(text: str) -> Optional[float]:
        """An interval end's value: +-inf for Infinity, inf, oo or ∞; None unless it is a real number."""
        from utils.symbolic_checks import SymbolicChecks

        compact = text.replace(" ", "")
        body = compact.lstrip("+-")
        if body.lower() in _INFINITE_ENDS and len(compact) - len(body) <= 1:
            return -math.inf if compact.startswith("-") else math.inf
        value = SymbolicChecks.numeric_value(compact)
        return None if value is None else float(value)

    @staticmethod
    def _check_variable(parts: Sequence[str], variable: str) -> None:
        """Refuse an inequality without the variable, or with other free names (a parameter, a second variable)."""
        others = InequalitySolver.other_names(parts, variable, strict=True)
        if others:
            raise ValueError(
                f"solve handles inequalities in one variable; besides {variable} it also contains {', '.join(others)}."
            )
        if not any(re.search(rf"(?<![A-Za-z_]){re.escape(variable)}(?![A-Za-z0-9_])", part) for part in parts):
            raise ValueError(f"the inequality does not contain {variable}.")

    @staticmethod
    def other_names(parts: Sequence[str], variable: str, strict: bool = False) -> List[str]:
        """The free names of the expressions besides variable (a parameter, a second variable).

        Raises:
            ValueError: when strict and an expression cannot be read; otherwise it is skipped.
        """
        from compiled_expression import FUNCTIONS_ALL, compile_expression

        others: List[str] = []
        for part in parts:
            try:
                compiled = compile_expression(part, (variable,), FUNCTIONS_ALL)
            except Exception as error:
                if strict:
                    raise ValueError(f"cannot read {part}: {error}") from None
                continue
            others.extend(name for name in compiled.free_names if name not in others)
        return others

    @staticmethod
    def _relations(
        parts: Sequence[str], signs: Sequence[str], variable: str, seeds: Sequence[float] = ()
    ) -> List[Relation]:
        """One Relation per sign, evaluated by math.js, undefined at the seeds where a
        denominator is 0."""
        from utils.symbolic_checks import SymbolicChecks

        relations: List[Relation] = []
        for index, sign in enumerate(signs):
            left, right = parts[index], parts[index + 1]
            sides = [SymbolicChecks.real_function(side, variable) for side in (left, right)]
            if sides[0] is None or sides[1] is None:
                raise ValueError(f"the expressions of {left} {sign} {right} cannot be evaluated.")
            difference_text = f"({left})-({right})"
            balanced = InequalitySolver._balanced_sides(left, right)
            if balanced is not None:
                moved = [SymbolicChecks.real_function(side, variable) for side in balanced]
                if moved[0] is not None and moved[1] is not None:
                    left, right, sides = balanced[0], balanced[1], moved
            relation = InequalitySolver._relation(left, right, sign, sides[0], sides[1], variable, difference_text)
            zeros = InequalitySolver._denominator_zeros(parts[index], parts[index + 1], variable, seeds)
            if zeros:
                relation.difference = _undefined_at(relation.difference, zeros)
            relations.append(relation)
        return relations

    @staticmethod
    def _denominator_zeros(left: str, right: str, variable: str, seeds: Sequence[float]) -> Set[float]:
        """The seeds where a denominator of lhs - rhs computes to exactly 0: lhs - rhs divides
        by zero there, so it is undefined, though math.js may give a finite value (exp(-1/x) is
        exp(-Infinity) = 0 at 0) or one that looks like an overflow (exp(1/x^2) at 0)."""
        from compiled_expression import substitute_variable
        from utils.math_utils import MathUtils
        from utils.symbolic_checks import SymbolicChecks

        zeros: Set[float] = set()
        if not seeds:
            return zeros
        difference = f"({left})-({right})"
        if variable != "x":
            difference = substitute_variable(difference, variable, "x")
        for denominator in MathUtils._denominators(difference):
            function = SymbolicChecks.real_function(denominator, "x")
            if function is None:
                continue
            zeros.update(seed for seed in seeds if function(seed) == 0)
        return zeros

    @staticmethod
    def _balanced_sides(left: str, right: str) -> Optional[Tuple[str, str]]:
        """The same inequality with every term on the side where it is added, or None when
        each already is: x^3 - 2x^2 + x > 0 as x^3 + x > 2x^2.

        Rounding error in a sum is relative to its largest terms, not to the sum, so the
        sides' magnitude (2 here, near the double root 1) measures the noise of lhs - rhs
        where a side that is 0 cannot. A constant side keeps the difference as written:
        1 - cos(x) < 1e-6 is not evaluated as 1 - (cos(x) + 1e-6), which rounds 1e-6 against 1."""
        left_terms, right_terms = split_top_level_terms(left), split_top_level_terms(right)
        if all(sign > 0 for sign, _ in left_terms + right_terms):
            return None
        added = [term for sign, term in left_terms if sign > 0] + [term for sign, term in right_terms if sign < 0]
        taken = [term for sign, term in left_terms if sign < 0] + [term for sign, term in right_terms if sign > 0]

        def sum_text(terms: List[str]) -> str:
            return " + ".join(f"({term})" for term in terms) if terms else "0"

        return sum_text(added), sum_text(taken)

    @staticmethod
    def _relation(
        left_text: str,
        right_text: str,
        sign: str,
        left: Any,
        right: Any,
        variable: str,
        difference_text: Optional[str] = None,
    ) -> Relation:
        """lhs - rhs with max(|lhs|, |rhs|) as its magnitude.

        A constant side gives the magnitude without evaluating the other side, and the
        difference is difference_text (by default "(lhs)-(rhs)"); otherwise the difference is
        computed from both sides, which the magnitude at the same point reuses (math.js would
        subtract the same two values in "(lhs)-(rhs)").
        """
        from utils.symbolic_checks import SymbolicChecks

        for constant_text, constant_sign in ((right_text, 1.0), (left_text, -1.0)):
            constant_value = SymbolicChecks.numeric_value(constant_text)
            if constant_value is None:
                continue
            difference = SymbolicChecks.real_function(difference_text or f"({left_text})-({right_text})", variable)
            if difference is None:
                raise ValueError(f"the expressions of {left_text} {sign} {right_text} cannot be evaluated.")
            return Relation(difference, sign, _constant_side_magnitude(float(constant_value), constant_sign))
        last: List[Any] = [None, None, None]  # the last point and both sides' values there

        def sides_at(point: float) -> List[Any]:
            if last[0] is None or last[0] != point:
                last[0], last[1], last[2] = point, left(point), right(point)
            return last

        def difference_from_sides(point: float) -> Optional[float]:
            _, lhs, rhs = sides_at(point)
            return None if lhs is None or rhs is None else float(lhs - rhs)

        def magnitude(point: float, value: float) -> Optional[float]:
            _, lhs, rhs = sides_at(point)
            largest: Optional[float] = None
            for side in (lhs, rhs):
                if side is not None and side == side and side != math.inf and side != -math.inf:
                    size = side if side >= 0 else -side
                    if largest is None or size > largest:
                        largest = size
            return largest

        return Relation(difference_from_sides, sign, magnitude)

    @staticmethod
    def _relation_text(parts: Sequence[str], signs: Sequence[str]) -> str:
        text = parts[0]
        for index, sign in enumerate(signs):
            text += f" {sign} {parts[index + 1]}"
        return text

    @staticmethod
    def _seeds(parts: Sequence[str], signs: Sequence[str], variable: str) -> List[float]:
        """Real roots of the polynomial pieces of lhs - rhs (its denominators, and its numerator with them
        cleared): points a grid could step over (a double root, a pole of 1/(x-1)^2)."""
        from compiled_expression import substitute_variable
        from utils.math_utils import MathUtils

        seeds: List[float] = []
        for index in range(len(signs)):
            difference = f"({parts[index]})-({parts[index + 1]})"
            if variable != "x":
                difference = substitute_variable(difference, variable, "x")
            denominators = MathUtils._denominators(difference)
            cleared = "*".join([f"({difference})"] + [f"({denominator})" for denominator in denominators])
            for polynomial in [difference, cleared] + denominators:
                try:
                    roots = MathUtils._polynomial_root_values([polynomial], "x") or []
                except Exception:
                    continue
                seeds.extend(root.real for root in roots if abs(root.imag) <= 1e-9 * max(1.0, abs(root.real)))
        return sorted(set(seeds))

    # ----- SymPy --------------------------------------------------------------------------

    @staticmethod
    def _ask_sympy(
        relation_text: str, variable: str, window: Optional[Window], interval: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """The server's response ({"result": ...}, {"error": ...}, {"timeout": True}), or None when unavailable."""
        from utils.sympy_client import SympyClient

        args: Dict[str, Optional[str]] = {"inequality": relation_text.replace(" ", ""), "variable": variable}
        args.update(InequalitySolver.sympy_interval_args(window))
        return cast(Optional[Dict[str, Any]], SympyClient.compute("solve_inequality", **args))

    @staticmethod
    def sympy_interval_args(window: Optional[Window]) -> Dict[str, Optional[str]]:
        """The window as the SymPy engine's lower, upper and bounds arguments; none without a window."""
        if window is None:
            return {}
        return {
            "lower": window.lower_text if math.isfinite(window.lower) else "-Infinity",
            "upper": window.upper_text if math.isfinite(window.upper) else "Infinity",
            "bounds": ("[" if window.lower_closed else "(") + ("]" if window.upper_closed else ")"),
        }

    @staticmethod
    def _pieces(intervals: Any) -> Optional[List[Piece]]:
        """SymPy's interval records as Pieces with math.js values of their ends; None if one cannot be evaluated."""
        if not isinstance(intervals, list):
            return None
        pieces: List[Piece] = []
        for record in intervals:
            try:
                lower_text, upper_text = str(record["lower"]), str(record["upper"])
                lower, upper = InequalitySolver._end_value(lower_text), InequalitySolver._end_value(upper_text)
                closed = (bool(record["lower_closed"]), bool(record["upper_closed"]))
            except (KeyError, TypeError):
                return None
            if lower is None or upper is None:
                return None
            pieces.append(Piece(lower, upper, closed[0], closed[1], lower_text, upper_text))
        return pieces

    @staticmethod
    def _same_window(first: Optional[Window], second: Optional[Window]) -> bool:
        if first is None or second is None:
            return first is second
        return (first.lower, first.upper, first.lower_closed, first.upper_closed) == (
            second.lower,
            second.upper,
            second.lower_closed,
            second.upper_closed,
        )

    @staticmethod
    def _server_window(text: Any) -> Optional[Window]:
        if not isinstance(text, str):
            return None
        try:
            return InequalitySolver.window(text)
        except ValueError:
            return None

    # ----- the numeric answer -------------------------------------------------------------

    @staticmethod
    def _numeric_answer(
        relations: Sequence[Relation],
        relation_text: str,
        variable: str,
        window: Optional[Window],
        period: Optional[str],
        parts: Sequence[str],
        seeds: Sequence[float],
        reason: str,
    ) -> str:
        if window is None and any(_TRIG_CALL.search(part) for part in parts):
            window = Window(_TRIG_WINDOW[0], _TRIG_WINDOW[1], True, True, _TRIG_WINDOW[2], _TRIG_WINDOW[3])
        try:
            pieces, checked = chart(relations, window, seeds)
        except SignChartFailure as error:
            return f"Error: could not solve {relation_text} reliably ({error})."
        unbounded = window is None or not (math.isfinite(window.lower) and math.isfinite(window.upper))
        reach = f"; checked out to |{variable}| = {reach_text(grid_reach(seeds))}" if unbounded else ""
        if checked is not None:
            # Overflow, underflow or rounding left the values unknown beyond these samples
            reach = f"; checked for {_checked_end(checked[0])} <= {variable} <= {_checked_end(checked[1])}"
        note = f"found numerically because {reason}{reach}"
        return str(format_solution(variable, pieces, relation_text, window, period, note))


def _constant_side_magnitude(constant: float, constant_sign: float) -> Magnitude:
    """max(|lhs|, |rhs|) from lhs - rhs when one side is constant: lhs = value + constant for a
    constant right side (constant_sign 1), rhs = constant - value for a constant left side (-1).
    Written without calls, as the browser runs it at every sample."""
    constant_size = constant if constant >= 0 else -constant

    def magnitude(point: float, value: float) -> float:
        other = constant + constant_sign * value
        other = other if other >= 0 else -other
        return other if other > constant_size else constant_size

    return magnitude


def _checked_end(value: float) -> str:
    """A sample bounding the checked range: -10^6 for a grid end, else 6 significant digits."""
    exponent = round(math.log10(abs(value))) if value != 0 else 0
    if value != 0 and abs(abs(value) - 10.0**exponent) <= 1e-9 * abs(value):
        return ("-" if value < 0 else "") + str(reach_text(abs(value)))
    return str(number_text(value, 6))


def _undefined_at(
    function: Callable[[float], Optional[float]], points: Set[float]
) -> Callable[[float], Optional[float]]:
    """function, undefined (None) at the given points."""

    def restricted(point: float) -> Optional[float]:
        return None if point in points else function(point)

    return restricted


def _end_text(text: str, value: float) -> str:
    """An interval end as the answer writes it: Infinity, a number as given (0.5, 1e-300, 1/3),
    else p/q or p*pi/q (1+1 as 2, 2pi as 2*pi), an exact form as given (sqrt(2)), or decimals."""
    if math.isinf(value):
        return str(number_text(value))
    compact = _IMPLICIT_PRODUCT.sub(r"\1*", text.replace(" ", ""))
    if is_plain_number(compact):
        return compact
    nice = nice_number(value)
    if nice is not None:
        return str(nice[0])
    if re.search(r"[A-Za-z]", compact):
        return compact
    return str(number_text(value))


def _names(text: str) -> List[str]:
    """The names in text, in order (2x has x; 1e20 has none)."""
    return [name for name in _NUMBER_OR_NAME.findall(text) if name]
