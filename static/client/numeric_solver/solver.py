"""
Main entry point for the numeric solver.

Orchestrates multi-start Newton-Raphson solving.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence

from .expression_utils import detect_variables, equation_to_residual
from .newton_raphson import evaluate_scaled_residual, newton_raphson
from .utils import deduplicate_solutions, generate_initial_guesses


def solve_numeric(
    equations: Sequence[str],
    variables: Optional[Sequence[str]] = None,
    initial_guesses: Optional[Sequence[Sequence[float]]] = None,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
) -> str:
    """Numerically solve a system of equations using multi-start Newton-Raphson.

    Args:
        equations: List of equation strings. Use '=' for equations (e.g., 'sin(x) + y = 1').
            If no '=' is present, the expression is assumed equal to 0.
        variables: Optional list of variable names. If not provided, auto-detected.
        initial_guesses: Optional list of starting point vectors.
        tolerance: Convergence tolerance for residuals, relative to the scale of
            each equation and the size of the variables.
        max_iterations: Maximum Newton-Raphson iterations per starting point.

    Returns:
        JSON string with solutions, variables, and method information.
    """
    # Validate input
    if not equations:
        return _error_result([], "No equations provided.")

    # Detect variables if not provided
    if variables is None or len(variables) == 0:
        detected_vars = detect_variables(equations)
        if not detected_vars:
            return _error_result([], "No variables detected in equations.")
        var_list = detected_vars
    else:
        var_list = list(variables)

    n_vars = len(var_list)
    n_eqs = len(equations)

    # Warn if system is over/under-determined (non-square systems are solved with
    # Gauss-Newton least-squares or minimum-norm steps)
    warning = None
    if n_eqs > n_vars:
        warning = (
            f"System has {n_eqs} equations and {n_vars} variables; only points satisfying every equation are reported."
        )
    elif n_eqs < n_vars:
        warning = (
            f"System has {n_eqs} equations and {n_vars} variables, so solutions are not unique; "
            "the reported solutions are sample points of the solution set."
        )

    # Convert equations to residual form
    residual_exprs = [equation_to_residual(eq) for eq in equations]

    # Generate initial guesses
    guesses = generate_initial_guesses(n_vars, initial_guesses)

    # Run Newton-Raphson from each starting point
    found_solutions: List[List[float]] = []
    found_residuals: List[float] = []

    for guess in guesses:
        solution = newton_raphson(
            residual_exprs,
            var_list,
            guess,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

        if solution is not None:
            # Verify the solution with the same scale-aware residual as the iteration
            residual = evaluate_scaled_residual(residual_exprs, var_list, solution)
            if residual is not None and residual <= math.sqrt(tolerance):
                found_solutions.append(solution)
                found_residuals.append(residual)

    # Deduplicate solutions (keeping the most accurate of each cluster)
    unique_solutions = deduplicate_solutions(found_solutions, var_list, residuals=found_residuals)

    # Build result
    result: Dict[str, Any] = {
        "solutions": unique_solutions,
        "variables": var_list,
        "method": "newton_raphson",
    }

    if warning:
        result["warning"] = warning

    if not unique_solutions and n_eqs > n_vars:
        result["message"] = (
            f"No point satisfies all {n_eqs} equations simultaneously: the least-squares residual stays "
            "non-zero, so the system appears inconsistent. Try providing initial_guesses if a solution is expected."
        )
    elif not unique_solutions:
        result["message"] = (
            "No solutions found in search range [-10, 10]. Try providing initial_guesses closer to expected solutions."
        )

    return json.dumps(result)


def _error_result(variables: List[str], message: str) -> str:
    """Create an error result JSON string."""
    return json.dumps(
        {
            "solutions": [],
            "variables": variables,
            "method": "newton_raphson",
            "error": message,
        }
    )
