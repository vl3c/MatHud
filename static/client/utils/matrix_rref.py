"""Reduced row echelon form and rank for plain Python matrices.

Pure Python (no browser imports) so it can be tested with regular pytest. Used by
``LinearAlgebraUtils`` to offer ``rref(A)`` and ``rank(A)`` inside linear algebra
expressions, which math.js does not provide.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Matrix = List[List[float]]

# A pivot counts as zero when its magnitude is at most this fraction of the largest
# entry of the matrix
DEFAULT_RELATIVE_TOLERANCE: float = 1e-10
# A result entry counts as round-off (and becomes 0) when its magnitude is at most this
# fraction of the largest result entry. Kept separate from the pivot tolerance: results are
# pivot-normalised, so an input-scaled threshold would erase real small entries.
_RESULT_ZERO_TOLERANCE: float = 1e-12
# Results keep this many significant digits (cleans 0.9999999999999999 and 0.20000000000000012)
_SIGNIFICANT_DIGITS: int = 12


def rref(matrix: Sequence[Sequence[float]], tol: Optional[float] = None) -> Tuple[Matrix, int]:
    """Return the reduced row echelon form of ``matrix`` and its rank.

    Gauss-Jordan elimination with partial pivoting: each column uses the row with the
    largest remaining entry as pivot. Pivots with magnitude <= ``tol`` are treated as zero
    (so the column has no pivot); ``tol`` defaults to ``1e-10`` times the largest absolute
    entry. Result values that are round-off relative to the largest result entry become 0;
    the rest are rounded to 12 significant digits.
    The input is not modified.
    """
    rows = _copy_rows(matrix)
    n_rows, n_cols = len(rows), len(rows[0])
    threshold = _resolve_tolerance(rows, tol)

    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        best = _find_pivot_row(rows, pivot_row, col)
        if abs(rows[best][col]) <= threshold:
            # A negligible column: its remaining entries are zero within the tolerance
            for row in rows[pivot_row:]:
                row[col] = 0.0
            continue
        rows[pivot_row], rows[best] = rows[best], rows[pivot_row]
        _normalize_row(rows[pivot_row], col)
        _eliminate_column(rows, pivot_row, col)
        pivot_row += 1

    largest_result = max(abs(value) for row in rows for value in row)
    zero_threshold = _RESULT_ZERO_TOLERANCE * largest_result
    return [[_clean(value, zero_threshold) for value in row] for row in rows], pivot_row


def matrix_rank(matrix: Sequence[Sequence[float]], tol: Optional[float] = None) -> int:
    """Return the rank of ``matrix`` (number of pivots found by :func:`rref`)."""
    return rref(matrix, tol)[1]


def _copy_rows(matrix: Sequence[Sequence[float]]) -> Matrix:
    if not isinstance(matrix, (list, tuple)) or not matrix:
        raise ValueError("rref requires a non-empty matrix")
    rows: Matrix = []
    for row in matrix:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("rref requires a 2-D matrix (a list of non-empty rows)")
        rows.append([float(value) for value in row])
        if not all(math.isfinite(value) for value in rows[-1]):
            raise ValueError("rref requires finite matrix entries")
    if len({len(row) for row in rows}) != 1:
        raise ValueError("All matrix rows must have the same length")
    return rows


def _resolve_tolerance(rows: Matrix, tol: Optional[float]) -> float:
    if tol is not None:
        if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol < 0:
            raise ValueError("rref tolerance must be a non-negative number")
        return float(tol)
    largest = max(abs(value) for row in rows for value in row)
    return DEFAULT_RELATIVE_TOLERANCE * largest


def _find_pivot_row(rows: Matrix, start: int, col: int) -> int:
    best = start
    for index in range(start + 1, len(rows)):
        if abs(rows[index][col]) > abs(rows[best][col]):
            best = index
    return best


def _normalize_row(row: List[float], col: int) -> None:
    pivot = row[col]
    for index in range(len(row)):
        row[index] /= pivot
    row[col] = 1.0


def _eliminate_column(rows: Matrix, pivot_row: int, col: int) -> None:
    pivot = rows[pivot_row]
    for index, row in enumerate(rows):
        if index == pivot_row:
            continue
        factor = row[col]
        if factor == 0.0:
            continue
        for position in range(len(row)):
            row[position] -= factor * pivot[position]
        row[col] = 0.0


def _clean(value: float, threshold: float) -> float:
    if abs(value) <= threshold:
        return 0.0
    return float(f"{value:.{_SIGNIFICANT_DIGITS}g}")
