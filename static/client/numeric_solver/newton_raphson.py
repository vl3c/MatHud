"""
Newton-Raphson iteration with Armijo backtracking line search.

Core iteration logic for the numeric solver.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .expression_utils import evaluate_residuals
from .jacobian import compute_jacobian
from .linear_algebra import solve_linear_system_gaussian


def newton_raphson(
    residual_exprs: Sequence[str],
    variables: Sequence[str],
    x0: Sequence[float],
    tolerance: float = 1e-10,
    max_iterations: int = 50,
) -> Optional[List[float]]:
    """Run Newton-Raphson iteration with Armijo backtracking line search.

    Args:
        residual_exprs: List of residual expression strings.
        variables: List of variable names.
        x0: Initial guess for variable values.
        tolerance: Convergence tolerance (max absolute residual).
        max_iterations: Maximum number of iterations.

    Returns:
        Converged solution, or None if iteration fails to converge.
    """
    # Armijo line search parameters
    c = 1e-4  # Sufficient decrease parameter
    rho = 0.5  # Backtracking factor
    max_backtracks = 10

    # Divergence threshold
    divergence_threshold = 1e15

    x = list(x0)
    n = len(x)

    for iteration in range(max_iterations):
        # Evaluate residuals
        F = evaluate_residuals(residual_exprs, variables, x)
        if F is None:
            return None

        # Check convergence
        max_residual = max(abs(f) for f in F)
        if max_residual < tolerance:
            return x

        # Check for divergence
        if any(abs(xi) > divergence_threshold for xi in x):
            return None

        # Compute Jacobian
        J = compute_jacobian(residual_exprs, variables, x)
        if J is None:
            return None

        # Solve J * delta = -F
        neg_F = [-f for f in F]
        delta = solve_linear_system_gaussian(J, neg_F)
        if delta is None:
            # Singular Jacobian
            return None

        # Armijo backtracking line search
        alpha = 1.0
        F_norm_sq = sum(f * f for f in F)

        for _ in range(max_backtracks):
            # Trial point
            x_new = [x[i] + alpha * delta[i] for i in range(n)]

            # Evaluate residuals at trial point
            F_new = evaluate_residuals(residual_exprs, variables, x_new)
            if F_new is None:
                alpha *= rho
                continue

            F_new_norm_sq = sum(f * f for f in F_new)

            # Armijo condition: ||F(x + alpha*delta)||^2 <= (1 - 2*c*alpha) * ||F(x)||^2
            if F_new_norm_sq <= (1 - 2 * c * alpha) * F_norm_sq:
                x = x_new
                break

            alpha *= rho
        else:
            # All backtracks failed, take the full step anyway
            x = [x[i] + delta[i] for i in range(n)]

    # Did not converge within max_iterations
    # Check if we're close enough
    F = evaluate_residuals(residual_exprs, variables, x)
    if F is not None and max(abs(f) for f in F) < tolerance * 100:
        return x

    return None
