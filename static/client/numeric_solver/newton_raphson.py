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

# Armijo line search constants
ARMIJO_C = 1e-4  # Sufficient decrease parameter
ARMIJO_RHO = 0.5  # Backtracking factor
MAX_BACKTRACKS = 10

# Divergence detection
DIVERGENCE_THRESHOLD = 1e15


def scaled_residual(F: Sequence[float], J: Sequence[Sequence[float]], x: Sequence[float]) -> float:
    """Return the largest residual measured relative to the scale of its equation.

    Each residual F_i is divided by sum_j |J_ij| * (1 + |x_j|), the change in F_i
    produced by a relative change of the variables. The result is roughly the
    relative distance to the root, so one tolerance works for equations of any
    magnitude (e.g. x^3 = 2e9 as well as 1e-12*x = 1e-3).
    """
    worst = 0.0
    for i, f in enumerate(F):
        if f == 0:
            continue
        scale = sum(abs(J[i][j]) * (1.0 + abs(x[j])) for j in range(len(x)))
        if scale == 0:
            return math.inf
        worst = max(worst, abs(f) / scale)
    return worst


def evaluate_scaled_residual(
    residual_exprs: Sequence[str],
    variables: Sequence[str],
    x: Sequence[float],
) -> Optional[float]:
    """Evaluate the scaled residual (see scaled_residual) at x, or None on failure."""
    F = evaluate_residuals(residual_exprs, variables, x)
    if F is None:
        return None
    if all(f == 0 for f in F):
        return 0.0
    J = compute_jacobian(residual_exprs, variables, x)
    if J is None:
        return None
    return scaled_residual(F, J, x)


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
        tolerance: Convergence tolerance, relative to the scale of each equation
            (see scaled_residual) and to the size of the variables.
        max_iterations: Maximum number of iterations.

    Returns:
        Converged solution, or None if iteration fails to converge.
    """
    x = list(x0)
    n = len(x)

    for iteration in range(max_iterations):
        # Evaluate residuals
        F = evaluate_residuals(residual_exprs, variables, x)
        if F is None:
            return None
        if all(f == 0 for f in F):
            return x

        # Check for divergence
        if any(abs(xi) > DIVERGENCE_THRESHOLD for xi in x):
            return None

        # Compute Jacobian
        J = compute_jacobian(residual_exprs, variables, x)
        if J is None:
            return None

        # Check convergence (scale-aware residual)
        residual = scaled_residual(F, J, x)
        if residual <= tolerance:
            return x

        # Solve J * delta = -F
        neg_F = [-f for f in F]
        delta = solve_linear_system_gaussian(J, neg_F)
        if delta is None:
            # Singular Jacobian
            return None

        # Step-size convergence: the remaining correction is negligible
        if residual <= math.sqrt(tolerance) and all(abs(delta[i]) <= tolerance * (1.0 + abs(x[i])) for i in range(n)):
            return [x[i] + delta[i] for i in range(n)]

        # Armijo backtracking line search
        alpha = 1.0
        F_norm_sq = sum(f * f for f in F)

        for _ in range(MAX_BACKTRACKS):
            # Trial point
            x_new = [x[i] + alpha * delta[i] for i in range(n)]

            # Evaluate residuals at trial point
            F_new = evaluate_residuals(residual_exprs, variables, x_new)
            if F_new is None:
                alpha *= ARMIJO_RHO
                continue

            F_new_norm_sq = sum(f * f for f in F_new)

            # Armijo condition: ||F(x + alpha*delta)||^2 <= (1 - 2*c*alpha) * ||F(x)||^2
            if F_new_norm_sq <= (1 - 2 * ARMIJO_C * alpha) * F_norm_sq:
                x = x_new
                break

            alpha *= ARMIJO_RHO
        else:
            # All backtracks failed, take the full step anyway
            x = [x[i] + delta[i] for i in range(n)]

    # Did not converge within max_iterations (e.g. slow linear convergence to a
    # multiple root). Accept if the remaining relative correction is small.
    residual_at_end = evaluate_scaled_residual(residual_exprs, variables, x)
    if residual_at_end is not None and residual_at_end <= math.sqrt(tolerance):
        return x

    return None
