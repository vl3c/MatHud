"""The real roots of an equation in one variable, found numerically: solve's answer when SymPy has none.

lhs = rhs holds where lhs - rhs <= 0 and lhs - rhs >= 0 both hold, so the inequality
sign chart (utils.real_sets.chart) finds its solution set: isolated points are the roots, a
stretch is a range of solutions, the whole window an identity. The chart already leaves out
samples that are rounding noise or underflowed to 0, tells poles from roots, finds touching
roots (x^2 = 0) and snaps ends to nice numbers (1/2, pi/4) when the equation holds there no
worse.

Without an interval, an equation whose variable occurs only inside trig calls (sin(x) = 1/2),
or one with trig calls that keeps changing sign far from 0 (x*sin(x) = 1), is solved on
[-2*pi, 2*pi] (as SymPy's periodic answers list their roots), any other (x + sin(x) = 10) on the
sign chart's grid out to |x| = 10^6.
Roots closer together than the grid without a sign change between samples, and roots
beyond the grid's reach, can be missed; the answer says the roots were found numerically.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

from utils.real_sets import Piece, SignChartFailure, Window, chart, format_solution, grid_reach, reach_text

# A root's text from the chart is kept when it is exact (2, 1/2, 3*pi/4); decimals get 15 digits.
_EXACT_TEXT = re.compile(r"^-?\d+(?:\*pi)?(?:/\d+)?$|^-?pi(?:/\d+)?$")


class NumericEquations:
    """Every method is static."""

    @staticmethod
    def solve(sides: Sequence[str], variable: str, window: Optional[Window], reason: str) -> str:
        """solve()'s answer for lhs = rhs from the sign chart, its note naming how the roots
        were found and why (reason: "SymPy found no closed form", ...); "Error: ..." when the
        equation has other names or the chart fails."""
        from utils.inequality_solver import (
            _TRIG_CALL,
            _TRIG_WINDOW,
            InequalitySolver,
            _checked_end,
            only_in_trig_calls,
        )

        parts = [sides[0], sides[1] if len(sides) == 2 else "0"]
        try:
            others = InequalitySolver.other_names(parts, variable, strict=True)
        except ValueError as error:
            return f"Error: {reason}, and the numeric root finder {error}"
        if others:
            return (
                f"Error: {reason}, and the numeric root finder needs an equation in {variable} alone; "
                f"besides {variable} it contains {', '.join(others)}."
            )
        trig_window = Window(_TRIG_WINDOW[0], _TRIG_WINDOW[1], True, True, _TRIG_WINDOW[2], _TRIG_WINDOW[3])
        has_trig = window is None and bool(_TRIG_CALL.search(" ".join(parts)))
        # x*sin(x) = 1 keeps crossing far from 0: its roots in the window are listed
        more = f" (it has more beyond, as |{variable}| grows)"
        if has_trig and only_in_trig_calls(parts, variable):
            window, more = trig_window, ""
        elif has_trig and NumericEquations._oscillates(parts, variable):
            window = trig_window
        try:
            try:
                pieces, checked, seeds = NumericEquations._chart(parts, variable, window)
            except SignChartFailure:
                if not has_trig or window is not None:
                    raise
                window = trig_window
                pieces, checked, seeds = NumericEquations._chart(parts, variable, window)
        except (ValueError, SignChartFailure) as error:
            return f"Error: {reason}, and the numeric root finder could not solve the equation reliably ({error})."
        if window is None or not has_trig:
            more = ""
        if window is not None and math.isfinite(window.lower) and math.isfinite(window.upper):
            where = f" in {window.text()}"
        else:
            where = f" for |{variable}| up to {reach_text(grid_reach(seeds))}"
            if window is not None:
                where = f" in {window.text()}, checked out to |{variable}| = {reach_text(grid_reach(seeds))}"
        if checked is not None:
            # Overflow, underflow or rounding left the values unknown beyond these samples
            where = f" for {_checked_end(checked[0])} <= {variable} <= {_checked_end(checked[1])}"
        note = f"found numerically{where}{more}, because {reason}"
        if all(piece.is_point for piece in pieces):
            roots = ",".join(NumericEquations._root_text(piece) for piece in pieces)
            kind = "the real roots" if pieces else "no real root was"
            return f"[{roots}] ({kind} {note})"
        relation_text = f"{parts[0]} = {parts[1]}"
        return str(format_solution(variable, pieces, relation_text, window, None, note))

    @staticmethod
    def _root_text(piece: Piece) -> str:
        """A root as solve lists it: the chart's exact text (1/2, 3*pi/4) or 15 significant digits."""
        from utils.cas_numeric import format_number

        text = str(piece.lower_text).replace(" ", "")
        return text if _EXACT_TEXT.match(text) else str(format_number(piece.lower))

    @staticmethod
    def roots(sides: Sequence[str], variable: str, window: Optional[Window] = None) -> Optional[List[float]]:
        """The real roots as numbers when the solution set is isolated points, else None."""
        parts = [sides[0], sides[1] if len(sides) == 2 else "0"]
        try:
            pieces, _, _ = NumericEquations._chart(parts, variable, window)
        except (ValueError, SignChartFailure):
            return None
        if not all(piece.is_point for piece in pieces):
            return None
        return [piece.lower for piece in pieces]

    @staticmethod
    def _oscillates(parts: Sequence[str], variable: str) -> bool:
        """True when lhs - rhs keeps changing sign far from 0 (x*sin(x) = 1, tan(x) = x)."""
        from utils.inequality_solver import InequalitySolver
        from utils.real_sets import keeps_changing

        try:
            seeds = InequalitySolver._seeds(parts, ["="], variable)
            return bool(keeps_changing(InequalitySolver._relations(parts, ["<="], variable, seeds), seeds))
        except (ValueError, SignChartFailure):
            return False

    @staticmethod
    def _chart(
        parts: Sequence[str], variable: str, window: Optional[Window]
    ) -> Tuple[List[Piece], Optional[Tuple[float, float]], List[float]]:
        """(the solution set's pieces, the range the chart read or None, the seeds) of lhs = rhs.

        Raises:
            ValueError: when a side cannot be evaluated.
            SignChartFailure: when the chart finds no reliable answer.
        """
        from utils.inequality_solver import InequalitySolver

        seeds = InequalitySolver._seeds(parts, ["="], variable)
        relations = InequalitySolver._relations(parts, ["<="], variable, seeds)
        relations += InequalitySolver._relations(parts, [">="], variable, seeds)
        pieces, checked = chart(relations, window, seeds)
        return pieces, checked, seeds
