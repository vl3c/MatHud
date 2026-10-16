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
from typing import Any, Dict, List, Optional, Sequence, cast

from utils.cas_text import insert_letter_call_products, parse_interval, split_relation
from utils.real_sets import (
    Piece,
    Relation,
    SignChartFailure,
    Window,
    format_solution,
    sign_chart,
    verify_solution_set,
)

_TRIG_CALL = re.compile(r"(?<![A-Za-z_])(?:sin|cos|tan|sec|csc|cot)\s*\(")
# Without an interval, an inequality with trig functions is solved numerically on this window.
_TRIG_WINDOW = (-2 * math.pi, 2 * math.pi, "-2*pi", "2*pi")
# A number written before a name or "(" (2pi, 3(x+1)), but not an exponent (1e5)
_IMPLICIT_PRODUCT = re.compile(r"(?<![A-Za-z_\d.])(\d+\.?\d*)(?![eE][-+]?\d)\s*(?=[A-Za-z(])")
_INFINITE_ENDS = {"Infinity": math.inf, "-Infinity": -math.inf, "inf": math.inf, "-inf": -math.inf}


class InequalitySolver:
    """Every method is static."""

    @staticmethod
    def solve(inequality: str, variable: str, interval: Optional[str] = None) -> str:
        """The real solution set of an inequality in one variable, as format_solution writes it, or "Error: ..."."""
        from utils.math_utils import MathUtils

        try:
            parts, signs = split_relation(InequalitySolver.prepared(inequality))
            variable = str(MathUtils._normalize_variable(variable))
            window = InequalitySolver.window(interval)
            InequalitySolver._check_variable(parts, variable)
            relations = InequalitySolver._relations(parts, signs, variable)
        except ValueError as error:
            return f"Error: {error}"
        relation_text = InequalitySolver._relation_text(parts, signs)
        seeds = InequalitySolver._seeds(parts, signs, variable)
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
        """text read the same way by SymPy and math.js: MathUtils._cas_input, bars as abs(), x(x+2) as x*(x+2)."""
        from expression_functions import BY_NAME, CALL_ALIASES, replace_absolute_bars
        from utils.math_utils import MathUtils

        if not isinstance(text, str):
            raise ValueError("The inequality must be given as text.")
        prepared = replace_absolute_bars(str(MathUtils._cas_input(text)))
        return str(insert_letter_call_products(prepared, list(BY_NAME) + list(CALL_ALIASES)))

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
        return Window(lower, upper, lower_closed, upper_closed, _end_text(lower_text), _end_text(upper_text))

    @staticmethod
    def _end_value(text: str) -> Optional[float]:
        from utils.symbolic_checks import SymbolicChecks

        compact = text.replace(" ", "")
        if compact in _INFINITE_ENDS:
            return _INFINITE_ENDS[compact]
        value = SymbolicChecks.numeric_value(compact)
        return None if value is None else float(value)

    @staticmethod
    def _check_variable(parts: Sequence[str], variable: str) -> None:
        """Refuse an inequality without the variable, or with other free names (a parameter, a second variable)."""
        from compiled_expression import FUNCTIONS_ALL, compile_expression

        others: List[str] = []
        found = False
        for part in parts:
            try:
                compiled = compile_expression(part, (variable,), FUNCTIONS_ALL)
            except Exception as error:
                raise ValueError(f"cannot read {part}: {error}") from None
            others.extend(name for name in compiled.free_names if name not in others)
            found = found or re.search(rf"(?<![A-Za-z_]){re.escape(variable)}(?![A-Za-z0-9_])", part) is not None
        if others:
            raise ValueError(
                f"solve handles inequalities in one variable; besides {variable} it also contains {', '.join(others)}."
            )
        if not found:
            raise ValueError(f"the inequality does not contain {variable}.")

    @staticmethod
    def _relations(parts: Sequence[str], signs: Sequence[str], variable: str) -> List[Relation]:
        """One Relation per sign, evaluated by math.js."""
        from utils.symbolic_checks import SymbolicChecks

        relations: List[Relation] = []
        for index, sign in enumerate(signs):
            left, right = parts[index], parts[index + 1]
            difference = SymbolicChecks.real_function(f"({left})-({right})", variable)
            sides = [SymbolicChecks.real_function(side, variable) for side in (left, right)]
            if difference is None or sides[0] is None or sides[1] is None:
                raise ValueError(f"the expressions of {left} {sign} {right} cannot be evaluated.")
            relations.append(Relation(difference, sign, InequalitySolver._scale(sides[0], sides[1])))
        return relations

    @staticmethod
    def _scale(left: Any, right: Any) -> Any:
        def scale(point: float) -> Optional[float]:
            values = [left(point), right(point)]
            finite = [abs(value) for value in values if value is not None and math.isfinite(value)]
            return float(max([1.0] + finite))

        return scale

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
        if window is not None:
            args["lower"] = window.lower_text if math.isfinite(window.lower) else "-Infinity"
            args["upper"] = window.upper_text if math.isfinite(window.upper) else "Infinity"
            args["bounds"] = ("[" if window.lower_closed else "(") + ("]" if window.upper_closed else ")")
        return cast(Optional[Dict[str, Any]], SympyClient.compute("solve_inequality", **args))

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
            pieces = sign_chart(relations, window, seeds)
        except SignChartFailure as error:
            return f"Error: could not solve {relation_text} reliably ({error})."
        unbounded = window is None or not (math.isfinite(window.lower) and math.isfinite(window.upper))
        reach = "; checked out to |x| = 10^6".replace("x", variable) if unbounded else ""
        note = f"(found numerically because {reason}{reach})"
        return f"{format_solution(variable, pieces, relation_text, window, period)} {note}"


def _end_text(text: str) -> str:
    """An interval end as the answer writes it: no spaces, 2pi as 2*pi."""
    return _IMPLICIT_PRODUCT.sub(r"\1*", text.replace(" ", ""))
