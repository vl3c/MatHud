"""
Linear algebra utilities for the numeric solver.

Provides Gaussian elimination for solving linear systems.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def solve_linear_system_gaussian(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
) -> Optional[List[float]]:
    """Solve a linear system Ax = b using Gaussian elimination with partial pivoting.

    Args:
        A: n*n coefficient matrix (list of rows).
        b: n-element right-hand side vector.

    Returns:
        Solution vector x, or None if the matrix is singular (pivot below
        1e-12 times the largest entry) or non-square.
    """
    n = len(b)

    # Check for square matrix
    if len(A) != n:
        return None
    if any(len(row) != n for row in A):
        return None

    # Create augmented matrix [A|b]
    aug: List[List[float]] = []
    for i in range(n):
        row = [float(A[i][j]) for j in range(n)]
        row.append(float(b[i]))
        aug.append(row)

    # Pivots are judged relative to the matrix magnitude, not an absolute value
    singular_tol = 1e-12 * max((abs(value) for row in aug for value in row[:n]), default=0.0)

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot (maximum absolute value in column)
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row

        # Swap rows if needed
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        # Check for singular matrix
        pivot = aug[col][col]
        if abs(pivot) <= singular_tol:
            return None

        # Eliminate below pivot
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back substitution
    x: List[float] = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(aug[i][i]) <= singular_tol:
            return None
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    return x
