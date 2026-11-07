"""Starting values for a custom fit: screening candidate values of the nonlinear constants.

Candidate values for each nonlinear constant are 0; magnitudes from 0.01 to 30 in the
units of 1, the x range and 1/(x range); the x quartiles (shifts); the frequencies of 1
to MIN_HALF_CYCLES half cycles over the x range (more for data that crosses its mean
often); and, for a constant raised to a power of x (b^x), growth factors near 1. A coordinate search scans one constant at a time,
twice over, from two seeds, and the best values per constant are combined. Candidates
whose curve swings between the data points (an aliased sine) are ranked after the others
(``Shape.swings``).

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import math
from typing import List, Mapping, Optional, Tuple

from utils.statistics.custom_model import CustomModel
from utils.statistics.levenberg_marquardt import sum_of_squares
from utils.statistics.variable_projection import Projector

# Frequencies tried: at least MIN_HALF_CYCLES half cycles over the x range, more (up to
# MAX_HALF_CYCLES) when the data crosses its mean more often; never beyond the data's own
# sampling limit
MIN_HALF_CYCLES = 40
MAX_HALF_CYCLES = 80
# Candidates per constant kept for the combination stage, and the most combinations tried
COMBINED_PER_CONSTANT = 3
MAX_COMBINATIONS = 243
# Best screened candidates checked for swinging between the data points, and how many more
# are checked for the best one that does not swing when none of those qualifies
RANKED_CANDIDATES = 12
STEADY_CHECKS = 60
# Share of the data's variation left unexplained below which screening stops after one seed
GOOD_SCREEN = 0.01
# Points of the dense grid curves are compared on (at least this many, and 4 per data point)
GRID_POINTS = 400
# A curve whose total variation over the grid exceeds this multiple of its variation through
# the data points swings between them
SWING_RATIO = 3.0

Candidate = Tuple[float, List[float]]


class Shape:
    """Compares candidate curves over the data's x range."""

    def __init__(self, model: CustomModel, xs: List[float]) -> None:
        self.model = model
        self.points = sorted(set(xs))
        low, high = self.points[0], self.points[-1]
        size = max(GRID_POINTS, 4 * len(self.points))
        self.grid = [low + (high - low) * i / (size - 1) for i in range(size)] if high > low else [low]

    def curve(self, params: List[float]) -> Optional[List[float]]:
        values: Optional[List[float]] = self.model.predictions(self.grid, params)
        return values

    def swings(self, params: List[float]) -> bool:
        """Whether the curve swings between the data points much more than through them."""
        dense = self.curve(params)
        through = self.model.predictions(self.points, params)
        if dense is None or through is None:
            return True
        return variation(dense) > SWING_RATIO * variation(through) + 1e-300


def variation(values: List[float]) -> float:
    total = 0.0
    for i in range(len(values) - 1):
        total += abs(values[i + 1] - values[i])
    return total


def subsample(xs: List[float], ys: List[float], size: int) -> Tuple[List[float], List[float]]:
    """At most size data points, evenly spread over the data sorted by x."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    if len(order) > size:
        stride = (len(order) - 1) / (size - 1)
        order = [order[round(i * stride)] for i in range(size)]
    return [xs[i] for i in order], [ys[i] for i in order]


def half_cycles(xs: List[float], ys: List[float]) -> int:
    """How many half cycles of frequency candidates to try (see MIN_HALF_CYCLES).

    Oscillating data crosses its mean about once per half cycle; a quarter more covers the
    estimate's error. The data's sampling limit (n - 1 half cycles) caps it.
    """
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    mean = sum(ys) / len(ys)
    crossings = 0
    for index in range(len(order) - 1):
        if (ys[order[index]] - mean) * (ys[order[index + 1]] - mean) < 0.0:
            crossings += 1
    wanted = max(MIN_HALF_CYCLES, int(crossings * 1.25) + 1)
    return max(1, min(wanted, MAX_HALF_CYCLES, len(xs) - 1))


def candidate_values(xs: List[float], cycles: int) -> List[float]:
    """Values tried for each nonlinear constant: magnitudes, x quartiles and frequencies."""
    low, high = min(xs), max(xs)
    span = high - low
    if span <= 0.0:
        span = abs(high) or 1.0
    magnitudes = [m * 10.0**j for j in range(-2, 2) for m in (1.0, 3.0)]
    values = [0.0]
    for unit in (1.0, 1.0 / span, span):
        for magnitude in magnitudes:
            values.extend((magnitude * unit, -magnitude * unit))
    ordered = sorted(xs)
    values.extend(ordered[round(q * (len(ordered) - 1))] for q in (0.0, 0.25, 0.5, 0.75, 1.0))
    values.extend(math.pi * k / span for k in range(1, cycles + 1))
    return _distinct(values)


def growth_values(xs: List[float]) -> List[float]:
    """Extra values tried for a constant raised to a power of x (b^x): factors near 1 per unit of x,
    as b^x must neither overflow nor flatten over the x range (x in years)."""
    span = (max(xs) - min(xs)) or 1.0
    return [value for m in (0.1, 0.3, 1.0, 3.0) for value in (1.0 + m / span, 1.0 - m / span) if value > 0.0]


def _distinct(values: List[float]) -> List[float]:
    """The values sorted, without near-duplicates."""
    result: List[float] = []
    for value in sorted(values):
        if not result or value - result[-1] > 1e-9 * max(abs(value), abs(result[-1])):
            result.append(value)
    return result


class Screening:
    """The ranked screened candidates: starts to refine, and the best that does not swing."""

    __slots__ = ("starts", "steady")

    def __init__(self, starts: List[List[float]], steady: Optional[List[float]]) -> None:
        self.starts = starts
        self.steady = steady


def screen(
    projector: Projector,
    xs: List[float],
    seed_values: Mapping[int, float],
    cycles: int,
    count: int,
) -> Screening:
    """The count best distinct candidates (full constant lists), curves that swing last.

    A coordinate search runs from two seeds: every nonlinear constant at 1 (or its value in
    seed_values), then every one at the median x, so a peak's centre and width are found
    together when the data lies far from 0. The second seed is skipped when the first
    already explains all but GOOD_SCREEN of the variation.
    """
    nonlinear = projector.nonlinear
    if not nonlinear:
        projected = projector.project([])
        return Screening([projected[0]] if projected is not None else [], None)
    values = candidate_values(xs, cycles)
    growth = growth_values(xs)
    model = projector.model
    per_constant = [
        _distinct(values + growth) if model.constants[k] in model.growth_constants else values for k in nonlinear
    ]
    pool: List[Candidate] = []
    rankings: List[List[Tuple[float, float]]] = [[] for _ in nonlinear]
    middle = sorted(xs)[len(xs) // 2]
    seeds = [[float(seed_values.get(k, 1.0)) for k in nonlinear]]
    if abs(middle - 1.0) > 0.5:
        seeds.append([middle] * len(nonlinear))
    mean = sum(projector.ys) / len(projector.ys)
    total = sum_of_squares([y - mean for y in projector.ys])
    for seed in seeds:
        if pool and min(sse for sse, _ in pool) <= GOOD_SCREEN * total:
            break
        _coordinate_search(projector, per_constant, seed, pool, rankings)
    if len(nonlinear) > 1:
        _combine(projector, rankings, pool)
    distinct = _best_distinct(pool, nonlinear, RANKED_CANDIDATES + STEADY_CHECKS)
    shape = Shape(projector.model, projector.xs)
    cost = len(shape.grid) + len(shape.points)
    flags: List[bool] = []
    for index, params in enumerate(distinct):
        if index >= RANKED_CANDIDATES and (False in flags):
            break  # the ranked candidates hold one that does not swing; the rest are not needed
        flags.append(shape.swings(params))
        projector.budget.spend(cost)
    distinct = distinct[: len(flags)]
    ranked = sorted(range(len(distinct)), key=lambda index: (flags[index], index))
    steady = next((distinct[index] for index in ranked if not flags[index]), None)
    return Screening([distinct[index] for index in ranked[:count]], steady)


def _coordinate_search(
    projector: Projector,
    values: List[List[float]],
    seed: List[float],
    pool: List[Candidate],
    rankings: List[List[Tuple[float, float]]],
) -> None:
    """Scan each nonlinear constant over its values, keeping the best so far (twice when there are several).

    Every evaluated candidate joins pool; the first round's results per constant join rankings.
    """
    current = list(seed)
    best_sse = math.inf
    projected = projector.project(current)
    if projected is not None:
        pool.append((projected[1], projected[0]))
        best_sse = projected[1]
    count = len(current)
    for round_index in range(2 if count > 1 else 1):
        for index in range(count):
            if projector.budget.exhausted:
                return
            best_value = current[index]
            for value in values[index]:
                theta = list(current)
                theta[index] = value
                result = projector.project(theta)
                if result is None:
                    continue
                pool.append((result[1], result[0]))
                if round_index == 0:
                    rankings[index].append((result[1], value))
                if result[1] < best_sse:
                    best_sse, best_value = result[1], value
            current[index] = best_value


def _combine(projector: Projector, rankings: List[List[Tuple[float, float]]], pool: List[Candidate]) -> None:
    """Try the combinations of each constant's best COMBINED_PER_CONSTANT values."""
    tops = [
        [value for _, value in sorted(ranking, key=lambda item: item[0])[:COMBINED_PER_CONSTANT]] or [1.0]
        for ranking in rankings
    ]
    combos: List[List[float]] = [[]]
    for options in tops:
        combos = [combo + [value] for combo in combos for value in options][:MAX_COMBINATIONS]
    for theta in combos:
        if projector.budget.exhausted:
            return
        result = projector.project(theta)
        if result is not None:
            pool.append((result[1], result[0]))


def _best_distinct(pool: List[Candidate], nonlinear: List[int], limit: int) -> List[List[float]]:
    pool.sort(key=lambda item: item[0])
    distinct: List[List[float]] = []
    for sse, params in pool:
        if math.isfinite(sse) and all(_differs(params, other, nonlinear) for other in distinct):
            distinct.append(params)
        if len(distinct) >= limit:
            break
    return distinct


def _differs(a: List[float], b: List[float], indices: List[int]) -> bool:
    """Whether two constant lists differ in any of the constants at indices (beyond 1e-6 relatively)."""
    return any(abs(a[k] - b[k]) > 1e-6 * max(abs(a[k]), abs(b[k]), 1e-12) for k in indices)
