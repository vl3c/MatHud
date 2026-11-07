"""Starting values for a custom fit: screening candidate values of the nonlinear constants.

Candidate values for each nonlinear constant are 0; magnitudes from 0.01 to 30 in the
units of 1, the x range and 1/(x range); and the x deciles (shifts). A constant setting
the frequency of a trigonometric argument (b, f or T in sin(b*x), sin(2*pi*f*x),
sin(2*pi*x/T)) also gets the values giving 1 to MIN_HALF_CYCLES half cycles over the x
range, and more when the data crosses its mean more often (a window around that count
beyond MAX_HALF_CYCLES); a phase or shift gets values across one period (_Trig); a
constant raised to a power of x (b^x) gets growth factors near 1. A coordinate search
scans one constant at a time (frequencies first), twice over, from up to three seeds,
each with its share of the work, and the best values per constant are combined.
Candidates whose curve swings between the data points (an aliased sine) are ranked after
the others (``Shape.swings``).

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from utils.statistics.custom_model import CustomModel
from utils.statistics.levenberg_marquardt import sum_of_squares
from utils.statistics.variable_projection import Projector

# Frequencies tried: at least MIN_HALF_CYCLES half cycles over the x range, more when the
# data crosses its mean more often (every count up to MAX_HALF_CYCLES, beyond it at most
# WINDOW_VALUES counts around the estimate); never beyond the data's own sampling limit
MIN_HALF_CYCLES = 40
MAX_HALF_CYCLES = 80
WINDOW_VALUES = 40
# Candidates per constant kept for the combination stage, and the most combinations tried
COMBINED_PER_CONSTANT = 3
MAX_COMBINATIONS = 243
# Best screened candidates checked for swinging between the data points, and how many more
# are checked for the best one that does not swing when none of those qualifies
RANKED_CANDIDATES = 12
STEADY_CHECKS = 60
# A setting where moving each other constant changes the fit by less than this fraction is a dead end
DEAD_END_CHANGE = 1e-9
# Share of the data's variation left unexplained below which screening stops after one seed
GOOD_SCREEN = 0.01
# Points of the dense grid curves are compared on (at least this many, and 4 per data point)
GRID_POINTS = 400
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
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
        # Unevenly spaced (each point moved by a golden-ratio fraction of a step), so a sine
        # aliased on evenly spaced data is not aliased on the grid too
        step = (high - low) / size
        inner = [low + step * (i + (i * _GOLDEN) % 1.0) for i in range(1, size - 1)]
        self.grid = [low] + inner + [high] if high > low else [low]

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
    """Up to how many half cycles of frequency candidates to try (see MIN_HALF_CYCLES).

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
    return max(1, min(wanted, len(xs) - 1))


def candidate_values(xs: List[float]) -> List[float]:
    """Values tried for each nonlinear constant: magnitudes and x deciles."""
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
    values.extend(ordered[round(q * (len(ordered) - 1))] for q in [i / 10.0 for i in range(11)])
    return _distinct(values)


def half_cycle_counts(cycles: int) -> List[int]:
    """Half-cycle counts whose frequencies are tried: every one up to MAX_HALF_CYCLES; beyond,
    WINDOW_VALUES counts spread over the top 40% of cycles (where the estimate from mean
    crossings lies)."""
    counts = list(range(1, min(cycles, MAX_HALF_CYCLES) + 1))
    if cycles > MAX_HALF_CYCLES:
        low = max(MAX_HALF_CYCLES + 1, int(0.6 * cycles))
        stride = max(1, (cycles - low + 1) // WINDOW_VALUES)
        counts.extend(range(low, cycles + 1, stride))
    return counts


class _Trig:
    """How each nonlinear constant enters the model's trigonometric arguments.

    A constant the argument's slope in x is proportional to (b in sin(b*x + c), B in
    B*(x - C), f in 2*pi*f*x) or inversely proportional to (T in 2*pi*x/T) gets the values
    giving k half cycles over the x range (frequencies). A constant that moves the argument
    without changing its slope (a phase c, a shift C) gets, at each scan, the values one
    eighth of a period apart over the period around its current value, as one period is
    what matters.
    """

    def __init__(self, projector: Projector, xs: List[float], cycles: int) -> None:
        model = projector.model
        self.nonlinear = projector.nonlinear
        self.count = len(model.constants)
        self.arguments = model.trig_arguments()
        self.low, self.high = min(xs), max(xs)
        span = (self.high - self.low) or 1.0
        targets = [math.pi * k / span for k in half_cycle_counts(cycles)]
        self.frequencies: Dict[int, List[float]] = {}
        self.phases: Dict[int, Callable[..., float]] = {}
        base = [1.0] * self.count
        for index, k in enumerate(self.nonlinear):
            for argument in self.arguments:
                first, second = self._slope(argument, base, k, 1.0), self._slope(argument, base, k, 2.0)
                if first is None or second is None:
                    continue
                if abs(first) > 1e-12 and abs(second / first - 2.0) < 1e-6:
                    self.frequencies.setdefault(index, [target / first for target in targets])
                elif abs(first) > 1e-12 and abs(second / first - 0.5) < 1e-6:
                    self.frequencies.setdefault(index, [first / target for target in targets])
                elif abs(second - first) <= 1e-9 * max(1.0, abs(first)):
                    self.phases.setdefault(index, argument)

    def turns(self, index: int, theta: List[float]) -> List[float]:
        """Values one eighth of a period apart around a phase constant's current value."""
        argument = self.phases.get(index)
        if argument is None:
            return []
        params = self._params(theta)
        step = 1e-3 * (abs(params[self.nonlinear[index]]) + 1.0)
        moved = list(params)
        moved[self.nonlinear[index]] += step
        try:
            effect = (argument(self.low, *moved) - argument(self.low, *params)) / step
        except Exception:
            return []
        if not math.isfinite(effect) or effect == 0.0:
            return []
        eighth = (math.pi / 4.0) / abs(effect)
        return [theta[index] + j * eighth for j in range(-4, 4) if j != 0]

    def _params(self, theta: List[float]) -> List[float]:
        params = [1.0] * self.count
        for index, k in enumerate(self.nonlinear):
            params[k] = theta[index]
        return params

    def _slope(self, argument: Callable[..., float], base: List[float], k: int, value: float) -> Optional[float]:
        params = list(base)
        params[k] = value
        try:
            slope = (argument(self.high, *params) - argument(self.low, *params)) / ((self.high - self.low) or 1.0)
        except Exception:
            return None
        return slope if math.isfinite(slope) else None


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

    A coordinate search runs from up to three seeds: every nonlinear constant at 1 (or its
    value in seed_values), then every one at the median x, so a peak's centre and width are
    found together when the data lies far from 0, then every one at 0. A seed is skipped
    when an earlier one already explains all but GOOD_SCREEN of the variation.
    """
    nonlinear = projector.nonlinear
    if not nonlinear:
        projected = projector.project([])
        return Screening([projected[0]] if projected is not None else [], None)
    values = candidate_values(xs)
    growth = growth_values(xs)
    trig = _Trig(projector, xs, cycles)
    model = projector.model
    per_constant = []
    for index, k in enumerate(nonlinear):
        extra = (growth if model.constants[k] in model.growth_constants else []) + trig.frequencies.get(index, [])
        per_constant.append(_distinct(values + extra) if extra else values)
    pool: List[Candidate] = []
    rankings: List[List[Tuple[float, float]]] = [[] for _ in nonlinear]
    middle = sorted(xs)[len(xs) // 2]
    seeds = [[float(seed_values.get(k, 1.0)) for k in nonlinear]]
    if abs(middle - 1.0) > 0.5:
        seeds.append([middle] * len(nonlinear))
    # From 0 a frequency is scanned before a fast decay has hidden it (exp(-b*x)*cos(c*x))
    seeds.append([0.0] * len(nonlinear))
    mean = sum(projector.ys) / len(projector.ys)
    total = sum_of_squares([y - mean for y in projector.ys])
    budget = projector.budget
    for number, seed in enumerate(seeds):
        if pool and min(sse for sse, _ in pool) <= GOOD_SCREEN * total:
            break
        # Each seed scans every constant once, then a second round within its share of what
        # is left, so the last seed is still scanned
        stop = budget.used + max(0, budget.limit - budget.used) // (len(seeds) - number)
        _coordinate_search(projector, per_constant, seed, pool, rankings, trig, stop)
    if len(nonlinear) > 1:
        _combine(projector, rankings, pool)
    distinct = _best_distinct(projector, pool, RANKED_CANDIDATES + STEADY_CHECKS)
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
    trig: "_Trig",
    stop: int,
) -> None:
    """Scan each nonlinear constant over its values, keeping the best so far (twice when there are several),
    constants setting a frequency first.

    A phase constant also scans a period around its current value (_Trig.turns). The second
    round stops once the budget's used count reaches stop. Every evaluated candidate joins
    pool; the first round's results per constant join rankings.
    """
    current = list(seed)
    best_sse = math.inf
    projected = projector.project(current)
    if projected is not None:
        pool.append((projected[1], projected[0]))
        best_sse = projected[1]
    count = len(current)
    # Frequencies first: before a decay or a growth has been fitted to the first points alone
    order = sorted(range(count), key=lambda index: index not in trig.frequencies)
    for round_index in range(2 if count > 1 else 1):
        for index in order:
            if projector.budget.exhausted or (round_index > 0 and projector.budget.used >= stop):
                return
            scanned: List[Tuple[float, float]] = []
            for value in values[index] + trig.turns(index, current):
                theta = list(current)
                theta[index] = value
                result = projector.project(theta)
                if result is None:
                    continue
                pool.append((result[1], result[0]))
                scanned.append((result[1], value))
                if round_index == 0:
                    rankings[index].append((result[1], value))
            for sse, value in sorted(scanned):
                if sse >= best_sse:
                    break
                theta = list(current)
                theta[index] = value
                if count == 1 or not _hides_the_others(projector, theta, index, sse):
                    best_sse, current[index] = sse, value
                    break


def _hides_the_others(projector: Projector, theta: List[float], index: int, sse: float) -> bool:
    """Whether at theta no other nonlinear constant changes the fit (a decay so fast that only
    the first point is left, exp(-200*x)*cos(c*x)): a dead end for the search, as Levenberg-
    Marquardt cannot leave it either."""
    for other in range(len(theta)):
        if other == index:
            continue
        moved = list(theta)
        moved[other] = theta[other] * 1.5 + 0.5
        result = projector.project(moved)
        if result is None or abs(result[1] - sse) > DEAD_END_CHANGE * sse + 1e-300:
            return False
    return True


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


def _best_distinct(projector: Projector, pool: List[Candidate], limit: int) -> List[List[float]]:
    """The best candidates, at most limit, each drawing its own curve over the screened points.

    Candidates whose curves match (a pulse that has died out after the first point, whatever
    its rate: k = 90, 300 or 900) would fill the starts with one dead end.
    """
    pool.sort(key=lambda item: item[0])
    distinct: List[List[float]] = []
    curves: List[List[float]] = []
    nonlinear = projector.nonlinear
    # The screened points and the midpoints between them, where sines aliased on the points differ
    points = sorted(set(projector.xs))
    probe = points + [(points[i] + points[i + 1]) / 2.0 for i in range(len(points) - 1)]
    for sse, params in pool:
        if len(distinct) >= limit:
            break
        if not math.isfinite(sse) or not all(_differs(params, other, nonlinear) for other in distinct):
            continue
        curve = projector.model.predictions(probe, params)
        projector.budget.spend(len(probe))
        if curve is None or all(_curves_differ(curve, other) for other in curves):
            distinct.append(params)
            if curve is not None:
                curves.append(curve)
    return distinct


def _curves_differ(first: List[float], second: List[float]) -> bool:
    extent = (max(second) - min(second)) or max(abs(value) for value in second) or 1.0
    return max(abs(first[i] - second[i]) for i in range(len(first))) > 0.05 * extent


def _differs(a: List[float], b: List[float], indices: List[int]) -> bool:
    """Whether two constant lists differ in any of the constants at indices (beyond 1e-6 relatively)."""
    return any(abs(a[k] - b[k]) > 1e-6 * max(abs(a[k]), abs(b[k]), 1e-12) for k in indices)
