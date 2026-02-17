"""
Expression utilities for the numeric solver.

Provides variable detection, residual conversion, and evaluation functions.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from browser import window


# Math function names to exclude from variable detection
MATH_FUNCTIONS = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "log",
        "ln",
        "log10",
        "log2",
        "exp",
        "sqrt",
        "cbrt",
        "abs",
        "sign",
        "floor",
        "ceil",
        "round",
        "min",
        "max",
        "mod",
        "pow",
        "pi",
        "e",
    }
)


def detect_variables(equations: Sequence[str]) -> List[str]:
    """Extract single-letter variable names from equations.

    Excludes math function names like sin, cos, log, etc.

    Args:
        equations: List of equation strings.

    Returns:
        Sorted list of unique variable names.
    """
    variables: set[str] = set()

    # Pattern matches single letters that are not part of longer words
    # Uses negative lookbehind and lookahead to ensure it's a standalone letter
    pattern = r"(?<![a-zA-Z])([a-zA-Z])(?![a-zA-Z])"

    for eq in equations:
        # Find all single letters
        matches = re.findall(pattern, eq)
        for match in matches:
            if match.lower() not in MATH_FUNCTIONS:
                variables.add(match)

    return sorted(variables)


def equation_to_residual(equation: str) -> str:
    """Convert an equation to residual form.

    If the equation contains '=', converts 'LHS = RHS' to '(LHS) - (RHS)'.
    Otherwise returns the equation as-is (assumed equal to 0).

    Args:
        equation: Equation string, possibly containing '='.

    Returns:
        Residual expression string.
    """
    if "=" in equation:
        parts = equation.split("=", 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        return f"({lhs}) - ({rhs})"
    return equation


def evaluate_residuals(
    residual_exprs: Sequence[str],
    variables: Sequence[str],
    values: Sequence[float],
) -> Optional[List[float]]:
    """Evaluate all residual expressions at given variable values.

    Args:
        residual_exprs: List of residual expression strings.
        variables: List of variable names.
        values: List of values corresponding to variables.

    Returns:
        List of residual values, or None if any evaluation fails or returns non-finite.
    """
    # Build scope dictionary for math.js
    scope = {}
    for var, val in zip(variables, values):
        scope[var] = val

    residuals: List[float] = []
    for expr in residual_exprs:
        try:
            result = window.math.evaluate(expr, scope)
            # Convert to Python float
            val = float(result)
            # Check for non-finite values
            if not _is_finite(val):
                return None
            residuals.append(val)
        except Exception:
            return None

    return residuals


def _is_finite(value: float) -> bool:
    """Check if a value is finite (not NaN or infinity)."""
    import math

    return math.isfinite(value)
