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

    J[i][j] = (F_i(x + h*e_j) - F_i(x - h*e_j)) / (2h)

    Args:
        residual_exprs: List of residual expression strings.
        variables: List of variable names.
        values: Current values of variables.
        h: Step size for finite differences.

    Returns:
        Jacobian matrix (list of rows), or None if evaluation fails.
    """
    n_eqs = len(residual_exprs)
    n_vars = len(variables)
    values_list = list(values)

    jacobian: List[List[float]] = []

    for i in range(n_eqs):
        row: List[float] = []
        for j in range(n_vars):
            # Compute partial derivative of F_i with respect to x_j

            # Forward point: x + h*e_j
            values_plus = values_list.copy()
            values_plus[j] += h

            # Backward point: x - h*e_j
            values_minus = values_list.copy()
            values_minus[j] -= h

            # Evaluate residuals at both points
            f_plus = evaluate_residuals(residual_exprs, variables, values_plus)
            f_minus = evaluate_residuals(residual_exprs, variables, values_minus)

            if f_plus is None or f_minus is None:
                return None

            # Central difference
            derivative = (f_plus[i] - f_minus[i]) / (2 * h)
            row.append(derivative)

        jacobian.append(row)

    return jacobian
