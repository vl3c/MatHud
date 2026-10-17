"""A linear system solved numerically: solve_system_of_equations' answer when SymPy is unavailable.

The equations sum_j a_ij * x_j = b_i are reduced by Gauss-Jordan elimination with partial
pivoting. The result has the shape of the server's solve_system answer for a linear system:
{"none": True} when the equations contradict each other, {"solutions": [{name: value}]} for
one solution, or {"infinite": True, "general": {...}, "free": [...], "rank": r} when they are
dependent (each pivot unknown written in the free ones: x = 1 - y).

Pure Python (no browser imports), so server-side pytest covers it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from utils.cas_numeric import format_number as _format_number

# A reduced coefficient this small (relative to the largest one) counts as 0.
PIVOT_TOLERANCE = 1e-10
# Values are given to this many significant digits, so rounding in the elimination (0.700000000000001) is dropped.
SIGNIFICANT_DIGITS = 12


def solve_linear(
    coefficients: Sequence[Sequence[float]], constants: Sequence[float], names: Sequence[str]
) -> Dict[str, Any]:
    """The solutions of coefficients · names = constants (one row per equation)."""
    rows = [list(map(float, row)) + [float(constant)] for row, constant in zip(coefficients, constants)]
    count = len(names)
    scale = max([1.0] + [abs(value) for row in rows for value in row])
    tolerance = PIVOT_TOLERANCE * scale
    pivots: List[int] = []
    for column in range(count):
        rank = len(pivots)
        best = max(range(rank, len(rows)), key=lambda index: abs(rows[index][column]), default=None)
        if best is None or abs(rows[best][column]) <= tolerance:
            continue
        rows[rank], rows[best] = rows[best], rows[rank]
        pivot = rows[rank][column]
        rows[rank] = [value / pivot for value in rows[rank]]
        for index, row in enumerate(rows):
            if index != rank and row[column] != 0:
                factor = row[column]
                rows[index] = [value - factor * lead for value, lead in zip(row, rows[rank])]
        pivots.append(column)
    if any(abs(row[count]) > tolerance for row in rows[len(pivots) :]):
        return {"none": True}
    if len(pivots) == count:
        return {"solutions": [{names[column]: format_number(rows[rank][count]) for rank, column in enumerate(pivots)}]}
    free = [column for column in range(count) if column not in pivots]
    general = {
        names[column]: _affine_text(rows[rank][count], [(-rows[rank][other], names[other]) for other in free])
        for rank, column in enumerate(pivots)
    }
    return {"infinite": True, "general": general, "free": [names[column] for column in free], "rank": len(pivots)}


def format_number(value: float) -> str:
    return str(_format_number(float(f"{value:.{SIGNIFICANT_DIGITS}g}")))


def _affine_text(constant: float, terms: Sequence[Any]) -> str:
    """constant + sum of coefficient*name as text: 1-y, 2*y+0.5, -y."""
    text = format_number(constant) if abs(constant) > PIVOT_TOLERANCE or not terms else ""
    for coefficient, name in terms:
        if abs(coefficient) <= PIVOT_TOLERANCE:
            continue
        size = format_number(abs(coefficient))
        term = name if size == "1" else f"{size}*{name}"
        sign = "-" if coefficient < 0 else ("+" if text else "")
        text += f"{sign}{term}"
    return text or "0"
