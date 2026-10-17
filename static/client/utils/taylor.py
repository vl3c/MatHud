"""The taylor_polynomial tool: a function's Taylor polynomial at a point, from the server's SymPy engine.

1. The function is read as the solve tool reads it (InequalitySolver.prepared: Unicode
   notation, sin^2(x), ln, |x| bars, x(x+2) as a product); calculator spellings (sinx,
   sin x, xe^x) are refused with the form to write, and it must have one variable.
2. The point is a real number in standard syntax (0, -1, 1/2, pi/4), the order a whole
   number from 0 to MAX_ORDER.
3. SymPy computes the polynomial (operation taylor, static/sympy_cas.py), refusing points
   where the function is not smooth (abs(x), sqrt(x), 1/x or ln(x) at 0).
4. series_checks.taylor_matches checks it against the function's math.js values beside the
   point; a refuted polynomial is an error, one the check cannot judge is given.

There is no nerdamer fallback: without SymPy the tool says so.

The browser parts (math.js, SymPy requests) are imported when first used, so pytest
covers the module's texts with Python stand-ins.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, cast

from utils.series_checks import taylor_matches

MAX_ORDER = 10
MAX_EXPRESSION_LENGTH = 1000

_NO_SYMPY = (
    "taylor_polynomial needs the server's SymPy engine, which is not available "
    "(install the packages in requirements.txt and restart the app)."
)


class Taylor:
    """Every method is static."""

    @staticmethod
    def taylor_polynomial(expression: Any, variable: Any, point: Any, order: Any) -> Any:
        """The Taylor polynomial of expression at variable = point up to order, as a dict, or "Error: ...".

        Returns {"polynomial", "center", "order", "next_term", "remainder_note"}: the
        polynomial in powers of (x - a), ready for draw_function, the first omitted term
        (None when there is none, or none within a few orders) and a sentence on the error.
        """
        try:
            name = Taylor._variable(variable)
            degree = Taylor._order(order)
            prepared = Taylor._prepared(expression, name)
            center_text, center = Taylor._center(point)
        except ValueError as error:
            return f"Error: {error}"
        response = Taylor._ask_sympy(prepared, name, center_text, degree)
        if response is None:
            return f"Error: {_NO_SYMPY}"
        if response.get("timeout"):
            return (
                f"Error: SymPy ran out of time for the Taylor polynomial of {expression} of order {degree}; "
                "try a lower order."
            )
        if response.get("error"):
            return f"Error: {response['error']}"
        result = response.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("polynomial"), str):
            return f"Error: SymPy found no Taylor polynomial of {expression} at {name} = {center_text}."
        polynomial = str(result["polynomial"])
        if Taylor._matches(prepared, polynomial, name, center, degree) is False:
            return (
                f"Error: SymPy gave {polynomial} as the Taylor polynomial of {expression}, but a numeric check "
                "shows it is wrong, so no polynomial is given."
            )
        return {
            "polynomial": polynomial,
            "center": str(result.get("center", center_text)),
            "order": degree,
            "next_term": result.get("next_term"),
            "remainder_note": remainder_note(result, name),
        }

    # ----- input --------------------------------------------------------------------------

    @staticmethod
    def _variable(variable: Any) -> str:
        from utils.math_utils import MathUtils

        name = MathUtils._normalize_variable(variable)
        if not isinstance(name, str) or not name.strip().isidentifier():
            raise ValueError("variable must be the function's variable, such as x.")
        return name.strip()

    @staticmethod
    def _order(order: Any) -> int:
        if isinstance(order, float) and order.is_integer():
            order = int(order)
        if isinstance(order, bool) or not isinstance(order, int) or not 0 <= order <= MAX_ORDER:
            raise ValueError(f"order must be a whole number from 0 to {MAX_ORDER}, not {order!r}.")
        return order

    @staticmethod
    def _prepared(expression: Any, variable: str) -> str:
        """The function as both engines read it.

        Raises:
            ValueError: for a calculator spelling, an equation, or other variables.
        """
        from utils.inequality_solver import InequalitySolver

        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression must be the function as an expression, such as sin(x).")
        if len(expression) > MAX_EXPRESSION_LENGTH:
            raise ValueError(f"expression is longer than {MAX_EXPRESSION_LENGTH} characters.")
        if "=" in expression or "<" in expression or ">" in expression:
            raise ValueError(f"give the function's expression alone, such as sin(x), not {expression.strip()!r}.")
        InequalitySolver.refuse_misspellings(expression, variable, subject="function")
        prepared = str(InequalitySolver.prepared(expression))
        try:
            others = InequalitySolver.other_names([prepared], variable, strict=True)
        except ValueError as error:
            raise ValueError(str(error)) from None
        if others:
            raise ValueError(
                f"taylor_polynomial takes a function of {variable} alone; it also contains {', '.join(others)}."
            )
        return prepared

    @staticmethod
    def _center(point: Any) -> Tuple[str, float]:
        """(the point's text for SymPy, its value).

        Raises:
            ValueError: unless it is a finite real number.
        """
        from utils.math_utils import MathUtils
        from utils.symbolic_checks import SymbolicChecks

        if isinstance(point, (int, float)) and not isinstance(point, bool):
            text = repr(point)
        elif isinstance(point, str) and point.strip():
            text = str(MathUtils._cas_input(point.strip()))
        else:
            raise ValueError("point must be a real number, such as 0, 1 or pi/4.")
        value = SymbolicChecks.numeric_value(text)
        if value is None:
            raise ValueError(f"point must be a real number, such as 0, 1 or pi/4, not {point!r}.")
        return text, float(value)

    # ----- engines ------------------------------------------------------------------------

    @staticmethod
    def _ask_sympy(expression: str, variable: str, point: str, order: int) -> Optional[Dict[str, Any]]:
        """The server's answer ({"result"}, {"error"}, {"timeout"}), None when SymPy is unavailable."""
        from utils.sympy_client import SympyClient

        return cast(
            Optional[Dict[str, Any]],
            SympyClient.compute("taylor", expression=expression, variable=variable, point=point, order=str(order)),
        )

    @staticmethod
    def _matches(expression: str, polynomial: str, variable: str, center: float, order: int) -> Optional[bool]:
        """taylor_matches on math.js values; None when math.js cannot evaluate one of them."""
        from utils.symbolic_checks import SymbolicChecks

        function = SymbolicChecks.real_function(expression, variable)
        approximation = SymbolicChecks.real_function(polynomial, variable)
        if function is None or approximation is None:
            return None
        return cast(Optional[bool], taylor_matches(function, approximation, center, order))


def remainder_note(result: Dict[str, Any], variable: str) -> str:
    """A sentence on the error of the polynomial in a taylor result from the server."""
    center = str(result.get("center", "0"))
    base = str(result.get("base", variable))
    where = f"{variable} = {center}"
    if result.get("exact"):
        return "Exact: the function is its own Taylor polynomial of this order."
    if result.get("one_sided"):
        return (
            f"Beyond this order the function is not smooth at {where}: its series differ on the two sides, "
            "so the error has no single leading term."
        )
    next_term = result.get("next_term")
    if next_term:
        return f"Near {where} the error is about {next_term} (it grows away from {center})."
    power = f"{base}^{result.get('lookahead')}"
    if result.get("flat"):
        return (
            f"Every Taylor coefficient up to {power} is 0, though the function is not 0: near {where} it is "
            "smaller than any of those powers, and the polynomial says little about it away from that point."
        )
    return f"There are no further nonzero terms up to {power}, so near {where} the error is smaller than that power."
