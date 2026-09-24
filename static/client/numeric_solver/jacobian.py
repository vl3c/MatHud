"""
Jacobian computation for the numeric solver.

Provides numerical Jacobian via central differences.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .expression_utils import evaluate_residuals


def compute_jacobian(
    residual_exprs: Sequence[str],
    variables: Sequence[str],
    values: Sequence[float],
    h: float = 1e-7,
) -> Optional[List[List[float]]]:
    """Compute the numerical Jacobian matrix via central differences.

    J[i][j] = (F_i(x + h_j*e_j) - F_i(x - h_j*e_j)) / (2*h_j)

    Args:
        residual_exprs: List of residual expression strings.
        variables: List of variable names.
        values: Current values of variables.
        h: Relative step size for finite differences; the step for variable j
            is h * max(1, |x_j|) so it stays meaningful for large values.

    Returns:
        Jacobian matrix (list of rows), or None if evaluation fails.
    """
    n_eqs = len(residual_exprs)
    n_vars = len(variables)
    values_list = list(values)

    jacobian: List[List[float]] = [[0.0] * n_vars for _ in range(n_eqs)]

    # Each column needs only two evaluations of all residuals
    for j in range(n_vars):
        step = h * max(1.0, abs(values_list[j]))

        # Forward point: x + h*e_j
        values_plus = values_list.copy()
        values_plus[j] += step

        # Backward point: x - h*e_j
        values_minus = values_list.copy()
        values_minus[j] -= step

        # Evaluate residuals at both points
        f_plus = evaluate_residuals(residual_exprs, variables, values_plus)
        f_minus = evaluate_residuals(residual_exprs, variables, values_minus)

        if f_plus is None or f_minus is None:
            return None

        # Central difference over the step actually representable in floating point
        width = values_plus[j] - values_minus[j]
        for i in range(n_eqs):
            jacobian[i][j] = (f_plus[i] - f_minus[i]) / width

    return jacobian
