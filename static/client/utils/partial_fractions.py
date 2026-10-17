"""Partial fractions for the expand tool: expand(expression, variable), as calculators' expand(expr, x).

1. The expression is read as the solve tool reads it (InequalitySolver.prepared); a
   calculator spelling (sinx, sin x) is refused with the form to write.
2. The server's SymPy engine splits a ratio of polynomials in the variable alone, with the
   variable in the denominator, into partial fractions over the rationals (operation apart,
   static/sympy_cas.py). Anything else (a polynomial, sin(x)/x, a second variable, a variable
   the expression does not contain) is expanded as usual, so a variable given by mistake
   never turns a valid expansion into an error.
3. The result is checked numerically (utils.equivalence.compare, the step checker's check);
   a refuted result is an error.

There is no nerdamer fallback (its partfrac gives decimals): without SymPy a ratio of
polynomials gets an error that says so, and anything else the usual expansion.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, cast

from utils.equivalence import NOT_EQUIVALENT, compare

# Appended when the result is a single fraction: 1/(x^2 - 2) does not split over the
# rationals, and (x-1)/(x^2-1) cancels to 1/(x+1).
UNCHANGED_NOTE = " (a single fraction: it does not split further over the rationals)"
# Names math.js's rationalize lists as variables although they are constants
_CONSTANTS = frozenset({"e", "pi"})
# A number (kept whole, so 1e20 holds no name e20) or a name (x, cost, θ), as the solve tool reads them
_NUMBER_OR_NAME = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|([^\W\d]\w*)")


class PartialFractions:
    """Every method is static."""

    @staticmethod
    def expand(expression: Any, variable: Any) -> str:
        """The partial fractions of expression in variable, else its ordinary expansion; "Error: ..." on failure."""
        from utils.math_utils import MathUtils

        name = MathUtils._normalize_variable(variable)
        if not isinstance(name, str) or not name.strip().isidentifier():
            return PartialFractions._ordinary(expression)
        name = name.strip()
        try:
            prepared = PartialFractions._prepared(expression)
        except Exception:
            return PartialFractions._ordinary(expression)
        try:
            PartialFractions._refuse_unclear_input(expression, name)
        except ValueError as error:
            # a calculator spelling (1/(sinx+1)) or an ambiguous division (1/(x+1)(x+2)) is
            # refused; a variable the expression lacks is not
            if str(error) != f"{name} does not appear in the expression.":
                return f"Error: {error}"
        if name not in _names(prepared):
            return PartialFractions._ordinary(expression)
        response = PartialFractions._ask_sympy(prepared, name)
        if response is None:
            if PartialFractions._is_rational_in(prepared, name):
                return (
                    "Error: partial fractions need the server's SymPy engine, which is not available "
                    "(install the packages in requirements.txt and restart the app)."
                )
            return PartialFractions._ordinary(expression)
        if response.get("timeout"):
            return f"Error: SymPy ran out of time splitting {expression} into partial fractions."
        if response.get("error"):
            return PartialFractions._ordinary(expression)  # SymPy cannot read it: not a ratio of polynomials
        result = response.get("result")
        if not isinstance(result, dict):
            return PartialFractions._ordinary(expression)  # SymPy could not do it
        fractions = result.get("fractions")
        if not isinstance(fractions, str):
            refusal = result.get("refusal")
            return f"Error: {refusal}" if refusal else PartialFractions._ordinary(expression)
        if PartialFractions._refuted(prepared, fractions, name):
            return (
                f"Error: SymPy gave {fractions} as the partial fractions of {expression}, but a numeric check "
                "shows it is wrong, so no result is given."
            )
        return fractions + (UNCHANGED_NOTE if result.get("unchanged") else "")

    @staticmethod
    def _ordinary(expression: Any) -> str:
        from utils.math_utils import MathUtils

        return str(MathUtils._checked_rewrite("expand", expression))

    @staticmethod
    def _prepared(expression: Any) -> str:
        from utils.inequality_solver import InequalitySolver

        return str(InequalitySolver.prepared(expression))

    @staticmethod
    def _refuse_unclear_input(expression: str, variable: str) -> None:
        from utils.inequality_solver import InequalitySolver

        InequalitySolver.refuse_ambiguous_division(expression)
        InequalitySolver.refuse_misspellings(expression, variable, subject="expression")

    @staticmethod
    def _ask_sympy(expression: str, variable: str) -> Optional[Dict[str, Any]]:
        """The server's answer ({"result"}, {"error"}, {"timeout"}), None when SymPy is unavailable."""
        from utils.sympy_client import SympyClient

        return cast(Optional[Dict[str, Any]], SympyClient.compute("apart", expression=expression, variable=variable))

    @staticmethod
    def _is_rational_in(expression: str, variable: str) -> bool:
        """Whether math.js's rationalize reads expression as a ratio of polynomials in variable
        alone with variable in the denominator."""
        from browser import window

        try:
            detail = window.math.rationalize(expression, {}, True)
            names = set(str(name) for name in detail.variables) - _CONSTANTS
            denominator = detail.denominator
            if names != {variable} or not denominator:
                return False
            text = str(denominator.toString())
        except Exception:
            return False
        return variable in _names(text)

    @staticmethod
    def _refuted(expression: str, fractions: str, variable: str) -> bool:
        from utils.symbolic_checks import SymbolicChecks

        first = SymbolicChecks.scope_function(expression)
        second = SymbolicChecks.scope_function(fractions)
        if first is None or second is None:
            return False
        return bool(compare(first, second, [variable]).verdict == NOT_EQUIVALENT)


def _names(text: str) -> List[str]:
    """The names in text (with a character scan, not a lookbehind, which Brython's re mishandles)."""
    return [name for name in _NUMBER_OR_NAME.findall(text) if name]
