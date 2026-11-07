"""Linear constants of a custom model, and solving them exactly (variable projection).

``linear_constants(model, xs)`` finds the constants the model is jointly affine in, by
evaluating it (a and d in ``a*exp(-b*x)*cos(c*x) + d``). For any values of the other
(nonlinear) constants, ``Projector.project`` solves the linear ones by linear least
squares, so only the nonlinear constants need starting values, and Levenberg-Marquardt
over the nonlinear constants never fights their correlation with the linear ones.

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from utils.statistics.custom_model import CustomModel
from utils.statistics.levenberg_marquardt import (
    EvaluationBudget,
    solve_least_squares,
    sum_of_squares,
    undetermined_columns,
)

# Reference values of the constants for the linearity test (several, as some fail to
# evaluate or hide a constant: exp(b*x) overflows or underflows at x = 2000 unless b is
# small, exp(-x/tau) unless tau is large, b^x unless b is near 1), each as (first, spread)
# for constant k = first + spread*k. Settings scaled by the largest |x| (both ways) and
# settings near 1 are added per data set.
_REFERENCE_STARTS = ((0.53, 0.21), (1.31, -0.17), (0.047, 0.031), (-0.61, 0.29), (2.2, 0.37))
_LINEAR_TOLERANCE = 1e-8
# Offsets from the reference value at which a linear constant's effect must stay on its line:
# wide, so a constant that only looks linear for small changes (atan(b/(x - c)) at large x)
# shows its curvature, or fails to evaluate (exp(b*x) overflows)
_LINEAR_OFFSETS = (1.5, 37.0, -450.0, 6000.0)
# Rounding allowed against the values themselves
_ROUNDING = 1e-12
_SAMPLE_POINTS = 12


def linear_constants(model: CustomModel, xs: List[float]) -> List[int]:
    """Indices of constants the model is jointly affine in.

    A constant is linear when it moves the model's values on a straight line at every
    reference setting where it moves them at all, and it does so at two settings or more
    (one when only one setting can be evaluated).
    """
    count = len(model.constants)
    sample = _sample(xs, _SAMPLE_POINTS)
    settings = [ref for ref in _reference_settings(count, xs) if _alive(model, sample, ref)]
    if not settings:
        return []
    candidates = []
    for k in range(count):
        verdicts = [_affine_in(model, sample, ref, k) for ref in settings]
        if False not in verdicts and verdicts.count(True) >= min(2, len(settings)):
            candidates.append(k)
    joint = [ref for ref in settings if all(_moves(model, sample, ref, k) for k in candidates)][:2] or settings[:2]
    while candidates and not all(_jointly_affine(model, sample, ref, candidates) for ref in joint):
        candidates.pop()
    return candidates


def _reference_settings(count: int, xs: List[float]) -> List[List[float]]:
    scale = max(1.0, max(abs(x) for x in xs))
    starts = list(_REFERENCE_STARTS)
    if scale > 1.0:
        starts += [(first / scale, spread / scale) for first, spread in _REFERENCE_STARTS[:3]]
        starts += [(first * scale, spread * scale) for first, spread in _REFERENCE_STARTS[:2]]
    starts += [(1.0 + first / scale, spread / scale) for first, spread in _REFERENCE_STARTS[:2]]
    return [_reference(count, setting) for setting in starts]


def _moves(model: CustomModel, sample: List[float], reference: List[float], k: int) -> bool:
    return _affine_in(model, sample, reference, k) is not None


def _alive(model: CustomModel, sample: List[float], params: List[float]) -> bool:
    """Whether the model gives finite values there, not all zero (an underflowed exp(-k*x) at x = 2000)."""
    values = model.predictions(sample, params)
    return values is not None and any(value != 0.0 for value in values)


def _reference(count: int, setting: Tuple[float, float]) -> List[float]:
    first, spread = setting
    return [first + spread * k for k in range(count)]


def _sample(xs: List[float], size: int) -> List[float]:
    if len(xs) <= size:
        return list(xs)
    stride = (len(xs) - 1) / (size - 1)
    return [xs[round(i * stride)] for i in range(size)]


def _on_line(value: float, base: float, change: float) -> bool:
    """Whether value equals base + change, to _LINEAR_TOLERANCE of the change and rounding."""
    error = abs(value - base - change)
    return error <= _LINEAR_TOLERANCE * abs(change) + _ROUNDING * (abs(value) + abs(base)) + 1e-300


def _affine_in(model: CustomModel, sample: List[float], reference: List[float], k: int) -> Optional[bool]:
    """Whether the model's values move on a straight line as constant k moves, the others held.

    None when the constant does not move them at the reference (an underflowed exp), which
    is no evidence either way. Treating a linear constant as nonlinear only costs time, the
    other way round gives wrong fits.
    """
    base = model.predictions(sample, reference)
    below = list(reference)
    below[k] -= 1.0
    lower = model.predictions(sample, below)
    if base is None:
        return None
    if lower is None:
        return False  # a linear constant never stops the model evaluating
    slopes = [base[i] - lower[i] for i in range(len(sample))]
    if all(abs(slopes[i]) <= _ROUNDING * abs(base[i]) for i in range(len(sample))):
        return None
    for offset in _LINEAR_OFFSETS:
        params = list(reference)
        params[k] += offset
        values = model.predictions(sample, params)
        if values is None:
            return False
        for i in range(len(sample)):
            if not _on_line(values[i], base[i], slopes[i] * offset):
                return False
    return True


def _jointly_affine(model: CustomModel, sample: List[float], reference: List[float], linear: List[int]) -> bool:
    """Whether the model is affine in the constants linear together (no products a*d)."""
    origin = list(reference)
    for k in linear:
        origin[k] = 0.0
    base = model.predictions(sample, origin)
    if base is None:
        return False
    columns = []
    for k in linear:
        params = list(origin)
        params[k] = 1.0
        values = model.predictions(sample, params)
        if values is None:
            return False
        columns.append([values[i] - base[i] for i in range(len(sample))])
    for spread in (0.41, -37.0):
        weights = [0.37 + spread * (index + 1) for index in range(len(linear))]
        params = list(origin)
        for index, k in enumerate(linear):
            params[k] = weights[index]
        combined = model.predictions(sample, params)
        if combined is None:
            return False
        for i in range(len(sample)):
            change = 0.0
            for index in range(len(linear)):
                change += weights[index] * columns[index][i]
            if not _on_line(combined[i], base[i], change):
                return False
    return True


class Projector:
    """The best linear constants, and the sum of squares, for given nonlinear constants.

    Attributes:
        linear: Indices of the linear constants (solved).
        nonlinear: Indices of the nonlinear constants a theta gives, in order.
        fixed: Constants held at a value (neither solved nor varied).
    """

    def __init__(
        self,
        model: CustomModel,
        xs: List[float],
        ys: List[float],
        linear: List[int],
        nonlinear: List[int],
        budget: EvaluationBudget,
        fixed: Optional[Dict[int, float]] = None,
    ) -> None:
        self.model = model
        self.xs = xs
        self.ys = ys
        self.linear = linear
        self.nonlinear = nonlinear
        self.budget = budget
        self.fixed = dict(fixed or {})
        self.last_remaining: List[float] = []

    def on(self, xs: List[float], ys: List[float], budget: EvaluationBudget) -> "Projector":
        """The same split of constants over other data."""
        return Projector(self.model, xs, ys, self.linear, self.nonlinear, budget, self.fixed)

    def holding(self, held: Dict[int, float]) -> "Projector":
        """A projector with the constants in held fixed at their values (taken out of nonlinear)."""
        fixed = dict(self.fixed)
        fixed.update(held)
        nonlinear = [k for k in self.nonlinear if k not in fixed]
        return Projector(self.model, self.xs, self.ys, self.linear, nonlinear, self.budget, fixed)

    def theta(self, params: Sequence[float]) -> List[float]:
        """The nonlinear constants of a full constant list."""
        return [float(params[k]) for k in self.nonlinear]

    def project(self, theta: Sequence[float]) -> Optional[Tuple[List[float], float]]:
        """(all constants, sum of squares) with the linear ones solved; None if not evaluable."""
        params = [0.0] * len(self.model.constants)
        for k, value in self.fixed.items():
            params[k] = value
        for index, k in enumerate(self.nonlinear):
            params[k] = float(theta[index])
        base = self.model.predictions(self.xs, params)
        self.budget.spend(len(self.xs))
        if base is None:
            return None
        rhs = [self.ys[i] - base[i] for i in range(len(self.ys))]
        if not self.linear:
            self.last_remaining = rhs
            return params, sum_of_squares(rhs)
        columns = self._columns(params, base)
        if columns is None:
            return None
        solution, _ = solve_least_squares(columns, rhs)
        for index, k in enumerate(self.linear):
            params[k] = solution[index]
        remaining = rhs
        for index, column in enumerate(columns):
            weight = solution[index]
            remaining = [remaining[i] - weight * column[i] for i in range(len(remaining))]
        sse = sum_of_squares(remaining)
        if not math.isfinite(sse):
            return None
        self.last_remaining = remaining
        return params, sse

    def _columns(self, params: List[float], base: List[float]) -> Optional[List[List[float]]]:
        """The change of the model per unit of each linear constant (the others at params)."""
        columns = []
        for k in self.linear:
            unit = list(params)
            unit[k] = params[k] + 1.0
            values = self.model.predictions(self.xs, unit)
            self.budget.spend(len(self.xs))
            if values is None:
                return None
            columns.append([values[i] - base[i] for i in range(len(base))])
        return columns

    def residuals(self, theta: List[float]) -> Optional[List[float]]:
        """The residuals (model - y) with the linear constants solved, for Levenberg-Marquardt."""
        if self.project(theta) is None:
            return None
        return [-value for value in self.last_remaining]

    def undetermined_linear(self, params: List[float]) -> List[int]:
        """The linear constants whose columns, at params, are combinations of the others."""
        if not self.linear:
            return []
        base_params = list(params)
        for k in self.linear:
            base_params[k] = 0.0
        base = self.model.predictions(self.xs, base_params)
        if base is None:
            return []
        columns = self._columns(base_params, base)
        if columns is None:
            return []
        return [self.linear[index] for index in undetermined_columns(columns)]
