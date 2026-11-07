"""Levenberg-Marquardt least squares for a handful of constants, in pure Python.

``levenberg_marquardt(residuals, start, budget)`` minimises the sum of squared residuals
of a model with a few constants (custom-model regression). ``residuals(params)`` returns
the residual list, or None where the model cannot be evaluated (a domain error, an
overflow, a complex value); such trial points are rejected like steps that do not
reduce the sum.

The method follows MINPACK's ``lmder`` in outline: each step solves the damped problem
``[J; sqrt(lambda) D] delta = [-r; 0]`` by Householder QR (never the normal equations,
whose condition number is squared), with D the running maximum of the Jacobian's
column norms (Marquardt's scaling, so the damping does not depend on the constants'
units) and Nielsen's update of lambda from the gain ratio. The Jacobian is taken by
forward differences.

Work is counted in model evaluations at one data point (``EvaluationBudget``) rather than
time, so a fit gives the same answer on every machine. The residual function spends from
the budget; the iteration stops when it is exhausted.

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

Residuals = Callable[[List[float]], Optional[List[float]]]

MAX_ITERATIONS = 200
# Relative reduction of the sum of squares, and relative step, below which a fit has converged
FUNCTION_TOLERANCE = 1e-12
STEP_TOLERANCE = 1e-12
# Forward-difference step: sqrt(machine epsilon), relative to |p| + DIFFERENCE_FLOOR
DIFFERENCE_STEP = 1.4901161193847656e-08
DIFFERENCE_FLOOR = 1e-3
INITIAL_DAMPING = 1e-3
MAX_DAMPING = 1e16
# A column whose QR pivot is below this fraction of the largest (columns scaled to unit
# norm) is not determined by the data
RANK_TOLERANCE = 1e-9


class EvaluationBudget:
    """Model evaluations a fit may still spend (one per data point per residual vector)."""

    __slots__ = ("limit", "used")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def spend(self, count: int) -> None:
        self.used += count

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit


class FitRun:
    """The outcome of one Levenberg-Marquardt run from one start."""

    __slots__ = ("params", "residuals", "sse", "iterations", "converged")

    def __init__(
        self, params: List[float], residuals: List[float], sse: float, iterations: int, converged: bool
    ) -> None:
        self.params = params
        self.residuals = residuals
        self.sse = sse
        self.iterations = iterations
        self.converged = converged


def sum_of_squares(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        total += value * value
    return total


def _norm(values: Sequence[float]) -> float:
    """The Euclidean norm, scaled by the largest entry when squares would underflow or overflow
    (exp(-0.2*x) at x near 2000 is about 1e-174)."""
    largest = 0.0
    for value in values:
        size = abs(value)
        if size > largest:
            largest = size
    if largest == 0.0 or not math.isfinite(largest):
        return largest
    if 1e-150 < largest < 1e150:
        return math.sqrt(sum_of_squares(values))
    return largest * math.sqrt(sum_of_squares([value / largest for value in values]))


def _dot(u: Sequence[float], v: Sequence[float], offset: int) -> float:
    """sum(u[i] * v[offset + i]) over u."""
    total = 0.0
    for i in range(len(u)):
        total += u[i] * v[offset + i]
    return total


def solve_least_squares(columns: List[List[float]], rhs: List[float]) -> Tuple[List[float], List[int]]:
    """Solve A c ~= rhs by Householder QR, A given by its columns (all of rhs's length).

    Columns are scaled to unit norm first, so the result does not depend on their units
    (x^2 at x = 10^5 next to a constant). A column whose part outside the span of the
    earlier columns is below RANK_TOLERANCE of its norm is left out: its coefficient is
    0 and its index is returned among the undetermined ones, so a rank-deficient A still
    gives a least-squares solution.

    Sums are plain loops: Brython runs them several times faster than fsum of a generator.
    """
    width = len(columns)
    norms = [_norm(column) for column in columns]
    a = []
    for j in range(width):
        scale = norms[j]
        a.append([value / scale for value in columns[j]] if scale > 0.0 else list(columns[j]))
    b = list(rhs)
    kept: List[int] = []
    undetermined: List[int] = []
    for j in range(width):
        row = len(kept)
        column = a[j]
        v = column[row:]
        remaining = _norm(v)
        if norms[j] == 0.0 or not remaining > RANK_TOLERANCE:
            undetermined.append(j)
            continue
        v[0] -= -remaining if column[row] >= 0.0 else remaining
        v_norm_sq = sum_of_squares(v)
        length = len(v)
        for k in range(j, width):
            other = a[k]
            factor = 2.0 * _dot(v, other, row) / v_norm_sq
            if factor != 0.0:
                other[row:] = [other[row + i] - factor * v[i] for i in range(length)]
        factor = 2.0 * _dot(v, b, row) / v_norm_sq
        b[row:] = [b[row + i] - factor * v[i] for i in range(length)]
        kept.append(j)
    solution = [0.0] * width
    for index in range(len(kept) - 1, -1, -1):
        j = kept[index]
        total = b[index]
        for k in kept[index + 1 :]:
            total -= a[k][index] * solution[k]
        solution[j] = total / a[j][index]
    for j in kept:
        solution[j] /= norms[j]
    return solution, undetermined


def undetermined_columns(
    columns: List[List[float]], tolerance: float = RANK_TOLERANCE, scales: Optional[Sequence[float]] = None
) -> List[int]:
    """Indices of columns that are (numerically) combinations of the earlier ones.

    A column is undetermined when its part outside the span of the earlier kept columns
    is at most tolerance times its scale (by default its own norm, so units do not matter).
    """
    own = [_norm(column) for column in columns]
    norms = list(scales) if scales is not None else own
    basis: List[List[float]] = []
    undetermined: List[int] = []
    for j in range(len(columns)):
        if norms[j] <= 0.0 or own[j] <= tolerance * norms[j]:
            undetermined.append(j)
            continue
        vector = [value / norms[j] for value in columns[j]]
        for _ in range(2):  # Gram-Schmidt, twice for orthogonality
            for unit in basis:
                dot = _dot(unit, vector, 0)
                vector = [vector[i] - dot * unit[i] for i in range(len(vector))]
        length = _norm(vector)
        if length <= tolerance or length == 0.0:
            undetermined.append(j)
            continue
        basis.append([value / length for value in vector])
    return undetermined


def jacobian(
    residuals: Residuals, params: List[float], base: List[float], budget: EvaluationBudget
) -> List[List[float]]:
    """Forward-difference Jacobian columns of the residuals at params (backward where forward fails).

    A column whose constant cannot be moved either way is all zeros (the fit leaves it).
    """
    columns: List[List[float]] = []
    count = len(base)
    for j in range(len(params)):
        original = params[j]
        step = DIFFERENCE_STEP * (abs(original) + DIFFERENCE_FLOOR)
        column: Optional[List[float]] = None
        for signed in (step, -step):
            moved = list(params)
            moved[j] = original + signed
            actual = moved[j] - original
            values = residuals(moved)
            if values is not None and actual != 0.0:
                column = [(values[i] - base[i]) / actual for i in range(count)]
                break
        columns.append(column if column is not None else [0.0] * count)
    return columns


def _damped_step(columns: List[List[float]], base: List[float], scale: List[float], damping: float) -> List[float]:
    """The step delta minimising |J delta + r|^2 + damping * |D delta|^2."""
    width = len(columns)
    root = math.sqrt(damping)
    augmented = []
    for j, column in enumerate(columns):
        extra = [0.0] * width
        extra[j] = root * scale[j]
        augmented.append(column + extra)
    rhs = [-value for value in base] + [0.0] * width
    delta, _ = solve_least_squares(augmented, rhs)
    return delta


def _linear_prediction(columns: List[List[float]], base: List[float], delta: List[float]) -> float:
    """|r + J delta|^2: the sum of squares the linear model predicts after the step."""
    predicted = list(base)
    for j, column in enumerate(columns):
        step = delta[j]
        if step != 0.0:
            for i in range(len(predicted)):
                predicted[i] += column[i] * step
    return sum_of_squares(predicted)


def levenberg_marquardt(
    residuals: Residuals,
    start: Sequence[float],
    budget: EvaluationBudget,
    max_iterations: int = MAX_ITERATIONS,
) -> Optional[FitRun]:
    """Minimise the sum of squared residuals from start; None when start cannot be evaluated.

    Stops when a step reduces the sum by less than FUNCTION_TOLERANCE relatively, the
    step is below STEP_TOLERANCE relatively, the sum is zero, no damping finds a lower
    sum (a minimum to rounding), or at max_iterations or the end of the budget (then
    ``converged`` is False).
    """
    params = [float(value) for value in start]
    first = residuals(params)
    if first is None:
        return None
    base: List[float] = first
    sse = sum_of_squares(base)
    scale = [0.0] * len(params)
    damping = INITIAL_DAMPING
    growth = 2.0
    converged = False
    iterations = 0
    while iterations < max_iterations and not budget.exhausted and sse > 0.0:
        iterations += 1
        columns = jacobian(residuals, params, base, budget)
        for j, column in enumerate(columns):
            scale[j] = max(scale[j], _norm(column))
        damped_scale = [value if value > 0.0 else 1.0 for value in scale]
        stepped = False
        while not budget.exhausted and damping <= MAX_DAMPING:
            delta = _damped_step(columns, base, damped_scale, damping)
            trial = [params[j] + delta[j] for j in range(len(params))]
            values = residuals(trial)
            if values is not None:
                trial_sse = sum_of_squares(values)
                predicted = sse - _linear_prediction(columns, base, delta)
                if trial_sse < sse:
                    gain = (sse - trial_sse) / predicted if predicted > 0.0 else 1.0
                    reduction = sse - trial_sse
                    step_size = _norm([delta[j] * damped_scale[j] for j in range(len(delta))])
                    size = _norm([params[j] * damped_scale[j] for j in range(len(params))])
                    params, base, sse = trial, values, trial_sse
                    damping *= max(1.0 / 3.0, 1.0 - (2.0 * gain - 1.0) ** 3)
                    growth = 2.0
                    stepped = True
                    if reduction <= FUNCTION_TOLERANCE * (sse + reduction) or step_size <= STEP_TOLERANCE * (
                        size + STEP_TOLERANCE
                    ):
                        converged = True
                    break
                if predicted <= FUNCTION_TOLERANCE * sse:
                    # Even the linearised model promises no real reduction: a minimum to rounding
                    converged = True
                    break
            damping *= growth
            growth *= 2.0
        if not stepped:
            # No damping lowers the sum: a minimum to rounding (or the budget ran out)
            converged = converged or not budget.exhausted
            break
        if converged:
            break
    if sse == 0.0:
        converged = True
    return FitRun(params, list(base), sse, iterations, converged)


def polish(
    residuals: Residuals, params: Sequence[float], budget: EvaluationBudget, tolerance: float, steps: int = 4
) -> Optional[FitRun]:
    """Gauss-Newton steps from params (each halved until it lowers the sum of squares), until a
    full step promises to lower the sum by at most tolerance relatively (that step is still
    taken when it lowers the sum); None when params cannot be evaluated. ``converged`` says
    whether that point was reached.

    Levenberg-Marquardt crawls along a long narrow valley (nearly collinear constants, such
    as a sine's frequency and phase at x near 2000) while its damping shrinks; undamped
    steps cross it at once.
    """
    point = [float(value) for value in params]
    first = residuals(point)
    if first is None:
        return None
    base: List[float] = first
    sse = sum_of_squares(base)
    iterations = 0
    while sse > 0.0 and iterations < steps and not budget.exhausted:
        iterations += 1
        columns = jacobian(residuals, point, base, budget)
        delta, _ = solve_least_squares(columns, [-value for value in base])
        if not all(math.isfinite(value) for value in delta):
            break
        stationary = sse - _linear_prediction(columns, base, delta) <= tolerance * sse
        for _ in range(1 if stationary else 4):
            trial = [point[j] + delta[j] for j in range(len(point))]
            values = residuals(trial)
            if values is not None and sum_of_squares(values) < sse:
                point, base, sse = trial, values, sum_of_squares(values)
                break
            delta = [value / 2.0 for value in delta]
        else:
            if not stationary:
                break
        if stationary:
            return FitRun(point, list(base), sse, iterations, True)
    return FitRun(point, list(base), sse, iterations, sse == 0.0)
