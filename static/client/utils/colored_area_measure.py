"""Measuring the area of a coloured area (``calculate_area`` on a coloured area's name).

Areas between two bounds (functions, segments, constants or the x-axis) are the
integral of |f1(x) - f2(x)| over the area's x-interval. The method is conservative:
all of its work shares one budget of function evaluations (``EVALUATION_BUDGET``,
30000, about 2.5 s in the browser at worst), and whenever it cannot vouch for the
result it says so (``accuracy_limited`` with a warning, and an error estimate that is
a deliberately loose bound, or null) rather than reporting a small, confident number.
Its estimates are a guide only for bounds that are smooth between the cuts and well
resolved by the grid; nothing here proves an error bound.

1. A vertical asymptote that a bounding function lists inside the interval is an
   error ("the area diverges near x ≈ ..."), as is a value that cannot be computed.
2. f1 - f2 is sampled on grids of 512, 1024, 2048, ... cells (up to 8192). The grid
   is trusted only when three successive grids, and a coarser grid of unrelated
   spacing, agree on the number of sign changes and there are 8 cells per sign
   change: nested grids can alias a fast oscillation alike. Exact zeros are not sign
   changes; a run of them is cut at both ends. A grid that never settles gives a
   rough trapezoid value, with accuracy limited and no error estimate.
3. Each sign change is located by the Illinois method. A sharp peak of |f1 - f2|
   (one or two grid points standing well above their neighbours, end points included)
   is refined by golden-section search and cut there. Where |f1 - f2| grows without
   bound, its growth exponent decides: like 1/|x - s| or faster the area diverges
   (an error); slower, the singularity is integrable, the panels next to it get the
   integral of the power law, and the value has accuracy limited and no estimate.
4. The interval is cut at the crossings, zero runs and peaks, every panel is split
   once, and globally adaptive Simpson's rule halves the panel with the largest
   error estimate until the total is below 1e-6 of the area or the budget runs out.
   A panel whose five nodes change sign has a crossing the grid missed: it is located
   and the panel split there.
5. The error estimate is the sum of the panels' Richardson estimates (|S2 - S1|/15).
   A split that shrinks a panel's raw difference less than eightfold (a kink or a
   jump) marks its halves rough: their estimate is twice the raw difference. When
   the budget runs out, or a panel cannot be split further, the estimate given is the
   sum of the raw differences (15 times looser), with accuracy limited; when the
   budget runs out before the panels are set up, the grid is integrated as a
   piecewise linear function with Richardson's step, bounded by the difference from
   every other grid point.

Bounds that are straight lines make the integrand piecewise linear, so those areas
come out exact up to rounding. Closed shapes use exact formulas (the shoelace formula
for straight-edged polygons, pi*r^2 and pi*a*b); a region area is measured from its
region expression, with the drawn outline as a check.

This module has no browser dependencies, so server-side pytest suites can test it
with stand-in drawables.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# All evaluations of f1 - f2 for one measurement (grids, crossings, peaks, panels).
EVALUATION_BUDGET = 30000
# Grids: the first, the coarsest one that may be trusted, the finest, and the cells
# wanted per sign change. Sign-change counts within this share are "the same".
GRID_CELLS = 512
MIN_TRUSTED_GRID_CELLS = 2048
MAX_GRID_CELLS = 8192
CELLS_PER_CROSSING = 8
GRID_COUNT_SLACK = 0.005
# A settled dyadic grid is checked against a coarser one of this many times its cells, whose
# spacing is unrelated to the dyadic grids (it still has about 5 cells per sign change).
INDEPENDENT_GRID_RATIO = 0.6180339887498949
# Illinois iterations locating a crossing; and how much |f1 - f2| at the narrowed bracket
# may exceed its value at the cell ends before the "crossing" counts as a singularity.
CROSSING_ITERATIONS = 80
POLE_GROWTH = 1e3
# A grid peak sharper than this ratio to its lower neighbour (and above the higher one by
# the shoulder ratio; see _sharp_block) is refined; a refined
# maximum this many times the grid value is a singularity. At most this many peaks
# are refined (beyond that accuracy is limited).
PEAK_SHARPNESS = 2.0
PEAK_SHOULDER = 1.2
PEAK_REFINE_STEPS = 64
PEAK_SINGULAR_RATIO = 1e6
MAX_PEAK_PROBES = 50
# Growth exponent p in |f1 - f2| ~ |x - s|^-p at or above which the area diverges.
DIVERGENT_ORDER = 0.98
# Adaptive Simpson: target relative error, and initial panels over the interval. A split
# that shrinks the raw Simpson difference by less than ROUGH_RATIO (16 for a smooth
# integrand, 4 at a kink, 2 at a jump) marks the halves rough: their error is twice the
# raw difference, not a fifteenth of it, and they get no Richardson correction.
ROUGH_RATIO = 8.0
# A rough panel's error estimate is this many times its raw difference (a jump's Simpson
# error is about the raw difference itself; the factor is the safety margin).
ROUGH_ERROR_FACTOR = 2.0
RELATIVE_TOLERANCE = 1e-6
INITIAL_PANELS = 1024

ColoredAreaMeasure = Dict[str, Any]
YFunction = Callable[[float], Optional[float]]
ExpressionArea = Callable[[str], float]
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


class _Undefined(ValueError):
    """f1 - f2 cannot be computed at a point."""


class _OutOfBudget(Exception):
    """The evaluation budget ran out."""


def measure_colored_area(area: Any, expression_area: Optional[ExpressionArea] = None) -> ColoredAreaMeasure:
    """The area of ``area`` with how it was measured.

    Returns ``{"value", "method", "error_estimate", "accuracy_limited", ...}``:
    ``bounds`` for the integrated kinds, ``crossings`` (x values where the bounds
    cross, when at most 50) or ``crossing_count``, ``evaluations``, and ``warning``
    when accuracy is limited (then ``error_estimate`` is a loose bound or null).
    ``expression_area`` measures a region expression (a region area's own).

    Raises:
        ValueError: the area diverges, has no finite x-interval, a bound is undefined
            inside it, or its kind cannot be measured.
    """
    kind = str(area.get_class_name())
    if kind == "FunctionsBoundedColoredArea":
        left, right = area._get_bounds()
        return _between(
            lambda x: area._get_function_y_at_x(area.func1, x),
            lambda x: area._get_function_y_at_x(area.func2, x),
            left,
            right,
            _bound_name(area.func1),
            _bound_name(area.func2),
            [area.func1, area.func2],
        )
    if kind == "FunctionSegmentBoundedColoredArea":
        left, right = area._get_bounds()
        segment = area.segment
        return _between(
            area._get_function_y_at_x,
            _line_through(segment),
            left,
            right,
            _bound_name(area.func),
            str(getattr(segment, "name", "segment")),
            [area.func],
        )
    if kind == "SegmentsBoundedColoredArea":
        return _between_segments(area.segment1, area.segment2)
    if kind == "ClosedShapeColoredArea":
        return _closed_shape(area, expression_area)
    raise ValueError(f"Cannot measure a coloured area of type '{kind}'")


# ----------------------------------------------------------------------
# Areas between two bounds
# ----------------------------------------------------------------------


def _bound_name(bound: Any) -> str:
    if bound is None:
        return "the x-axis"
    if isinstance(bound, (int, float)):
        return f"y = {bound:g}"
    return str(getattr(bound, "name", "f"))


def _line_through(segment: Any) -> YFunction:
    """y(x) on the line through a segment's endpoints (a vertical segment has none)."""
    x1, y1 = float(segment.point1.x), float(segment.point1.y)
    x2, y2 = float(segment.point2.x), float(segment.point2.y)
    if x1 == x2:
        return lambda _x: None
    slope = (y2 - y1) / (x2 - x1)
    return lambda x: y1 + slope * (x - x1)


def _segment_range(segment: Any) -> Tuple[float, float]:
    x1, x2 = float(segment.point1.x), float(segment.point2.x)
    return min(x1, x2), max(x1, x2)


def _between_segments(segment1: Any, segment2: Any) -> ColoredAreaMeasure:
    """Between two segments over the overlap of their x-ranges (between a segment and the x-axis over its range)."""
    left, right = _segment_range(segment1)
    if segment2 is not None:
        other_left, other_right = _segment_range(segment2)
        left, right = max(left, other_left), min(right, other_right)
    lower: YFunction = _line_through(segment2) if segment2 is not None else (lambda _x: 0.0)
    second = str(getattr(segment2, "name", "segment")) if segment2 is not None else "the x-axis"
    return _between(_line_through(segment1), lower, left, right, str(getattr(segment1, "name", "segment")), second, [])


def _listed_asymptote(bounds: Sequence[Any], a: float, b: float) -> Optional[Tuple[str, float]]:
    """A vertical asymptote a bounding function lists in [a, b], with that function's name."""
    for bound in bounds:
        for value in getattr(bound, "vertical_asymptotes", None) or []:
            try:
                x = float(value)
            except (TypeError, ValueError):
                continue
            if a <= x <= b:
                return _bound_name(bound), x
    return None


def _diverges(x: float, why: str) -> ValueError:
    return ValueError(f"The area diverges near x ≈ {x:.6g}: {why}")


class _Counted:
    """f1 - f2 with the shared evaluation budget: raises _OutOfBudget once it is spent."""

    def __init__(self, function: Callable[[float], float], budget: int) -> None:
        self.function = function
        self.left = budget
        self.used = 0

    def __call__(self, x: float) -> float:
        if self.left <= 0:
            raise _OutOfBudget()
        self.left -= 1
        self.used += 1
        return self.function(x)


def _between(
    upper: YFunction,
    lower: YFunction,
    left: Optional[float],
    right: Optional[float],
    upper_name: str,
    lower_name: str,
    bounds: Sequence[Any],
) -> ColoredAreaMeasure:
    """Integral of |upper - lower| over [left, right] (see the module docstring)."""
    if left is None or right is None or not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError("The coloured area has no finite x-interval to measure over")
    a, b = float(left), float(right)
    if not a < b:
        raise ValueError(f"The coloured area's x-interval [{a:g}, {b:g}] is empty")
    listed = _listed_asymptote(bounds, a, b)
    if listed is not None:
        raise _diverges(listed[1], f"{listed[0]} has a vertical asymptote there, inside [{a:g}, {b:g}]")

    def undefined(x: float, name: str) -> _Undefined:
        return _Undefined(
            f"The area diverges or is undefined near x ≈ {x:.6g}: {name} is undefined or not "
            f"finite at x = {x:.12g}, inside [{a:g}, {b:g}]"
        )

    def raw(x: float) -> float:
        # Kept lean: it runs tens of thousands of times in the browser.
        y1 = upper(x)
        if y1 is None or y1 != y1 or y1 in (math.inf, -math.inf):
            raise undefined(x, upper_name)
        y2 = lower(x)
        if y2 is None or y2 != y2 or y2 in (math.inf, -math.inf):
            raise undefined(x, lower_name)
        return float(y1) - float(y2)

    run = _Run(_Counted(raw, EVALUATION_BUDGET), a, b, f"|{upper_name} - {lower_name}|")
    return run.measure()


class _Run:
    """One measurement of the area between two bounds over [a, b]."""

    def __init__(self, difference: _Counted, a: float, b: float, names: str) -> None:
        self.d = difference
        self.a, self.b = a, b
        self.names = names
        self.warnings: List[str] = []
        self.limited = False
        self.unbounded = False  # no error estimate can be given
        self.crossings: List[float] = []
        self.singular: Dict[float, float] = {}  # point: growth exponent p
        self.roots: set = set()  # located crossings (cut points)
        self.xs: List[float] = []
        self.values: List[float] = []

    # -- driver ---------------------------------------------------------

    def measure(self) -> ColoredAreaMeasure:
        try:
            settled = self._sample()
        except _OutOfBudget:  # pragma: no cover - the grids fit in the budget
            settled = False
        if not settled:
            return self._unresolved_grid()
        try:
            cuts = self._cuts()
        except _OutOfBudget:
            self._limit("The evaluation budget ran out while locating crossings and peaks.")
            return self._grid_result()
        try:
            value, estimate, panels = self._adaptive(cuts)
        except _OutOfBudget:
            self._limit("The evaluation budget ran out before every panel was set up.")
            return self._grid_result()
        return self._result(value, estimate, f"adaptive Simpson's rule ({panels} panels)")

    def _grid_result(self) -> ColoredAreaMeasure:
        """The settled grid as a piecewise linear function, improved by Richardson's step.

        The bound given is the difference between the grid and every other point, three
        times the Richardson estimate of a second-order rule: loose on purpose.
        """
        fine = _linear_abs_integral(self.xs, self.values)
        coarse = _linear_abs_integral(self.xs[::2], self.values[::2])
        # Richardson's step for a second-order rule; the bound stays the raw difference.
        value = fine + (fine - coarse) / 3.0
        return self._result(value, abs(fine - coarse), "piecewise linear on the grid with Richardson's step")

    def _limit(self, warning: str) -> None:
        self.limited = True
        if warning not in self.warnings:
            self.warnings.append(warning)

    def _result(self, value: float, estimate: Optional[float], rule: str) -> ColoredAreaMeasure:
        split = f", cut at {len(self.crossings)} crossing(s) of the bounds" if self.crossings else ""
        guide = (
            "; error_estimate is the rule's own estimate, a guide only for smooth bounds resolved between the cuts"
            if not self.limited
            else "; accuracy is limited (see warning)"
        )
        measure: ColoredAreaMeasure = {
            "value": value,
            "method": (
                f"numeric integration of {self.names} over [{self.a:.12g}, {self.b:.12g}] by {rule}{split}, "
                f"{self.d.used} evaluations{guide}"
            ),
            "error_estimate": None if self.unbounded else estimate,
            "accuracy_limited": self.limited,
            "evaluations": self.d.used,
            "bounds": [self.a, self.b],
        }
        crossings = sorted(self.crossings)
        if len(crossings) <= 50:
            if crossings:
                measure["crossings"] = crossings
        else:
            measure["crossing_count"] = len(crossings)
        if self.warnings:
            measure["warning"] = " ".join(self.warnings)
        return measure

    # -- grids ----------------------------------------------------------

    def _sample(self) -> bool:
        """Refine the grid until its sign-change count settles; False when it never does."""
        a, b = self.a, self.b
        cells = GRID_CELLS
        self.xs = [a + (b - a) * i / cells for i in range(cells)] + [b]
        self.values = [self.d(x) for x in self.xs]
        counts: List[int] = []
        while True:
            count = _count_sign_changes(self.values)
            counts.append(count)
            # Three successive grids must agree: two coarse grids can alias a fast oscillation alike.
            stable = len(counts) >= 3 and all(abs(c - count) <= GRID_COUNT_SLACK * count for c in counts[-3:])
            if cells >= MIN_TRUSTED_GRID_CELLS and stable and cells >= CELLS_PER_CROSSING * count:
                # Nested grids can alias alike; a grid of unrelated spacing must agree too.
                if abs(self._independent_count(cells) - count) <= max(1.0, GRID_COUNT_SLACK * count):
                    return True
            if cells >= MAX_GRID_CELLS:
                return False
            cells *= 2
            xs: List[float] = []
            values: List[float] = []
            for i in range(len(self.xs) - 1):
                mid = a + (b - a) * (2 * i + 1) / cells
                xs += [self.xs[i], mid]
                values += [self.values[i], self.d(mid)]
            self.xs, self.values = xs + [b], values + [self.values[-1]]

    def _independent_count(self, cells: int) -> int:
        """Sign changes on a grid whose spacing is unrelated to the dyadic grids'."""
        a, b = self.a, self.b
        other = int(cells * INDEPENDENT_GRID_RATIO)
        return _count_sign_changes([self.d(a + (b - a) * i / other) for i in range(other + 1)])

    def _unresolved_grid(self) -> ColoredAreaMeasure:
        count = _count_sign_changes(self.values)
        self._limit(
            f"The bounds oscillate faster than the finest grid ({len(self.xs) - 1} cells) resolves: the grids "
            "disagree on how often they cross, so the value is a rough approximation with no error estimate."
        )
        self.unbounded = True
        # The trapezoid rule on |f1 - f2|: aliased samples still spread over the phases.
        trapezoid = sum(
            0.5 * (x1 - x0) * (abs(v0) + abs(v1))
            for x0, x1, v0, v1 in zip(self.xs, self.xs[1:], self.values, self.values[1:])
        )
        measure = self._result(trapezoid, None, "the trapezoid rule on |f1 - f2| over the finest grid")
        measure["crossing_count"] = count
        measure.pop("crossings", None)
        return measure

    # -- cuts: crossings, zero runs, peaks --------------------------------

    def _cuts(self) -> List[float]:
        xs, values = self.xs, self.values
        h = xs[1] - xs[0]
        cuts: List[float] = []
        for i in range(len(xs) - 1):
            if values[i] * values[i + 1] < 0:
                x = self._locate_crossing(xs[i], xs[i + 1], values[i], values[i + 1], h)
                cuts.append(x)
        # A run of exact zeros (bounds that meet, a zero branch) is cut at both ends; it is a
        # crossing when the values on its two sides have opposite signs.
        i = 0
        while i < len(xs):
            if values[i] != 0.0:
                i += 1
                continue
            j = i
            while j + 1 < len(xs) and values[j + 1] == 0.0:
                j += 1
            cuts += [xs[i], xs[j]]
            if 0 < i and j < len(xs) - 1 and values[i - 1] * values[j + 1] < 0:
                self.crossings.append(xs[i] if i == j else 0.5 * (xs[i] + xs[j]))
                self.roots.update({xs[i], xs[j]})
            i = j + 1
        cuts += self._peaks(h)
        return sorted({x for x in cuts if self.a < x < self.b})

    def _locate_crossing(self, lo: float, hi: float, f_lo: float, f_hi: float, h: float) -> float:
        """The root of f1 - f2 in a sign-change bracket (Illinois method); a singularity there is classified."""
        scale = max(abs(f_lo), abs(f_hi))
        side = 0
        for _ in range(CROSSING_ITERATIONS):
            if hi - lo <= 1e-15 * max(1.0, abs(lo), abs(hi)):
                break
            x = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
            if not lo < x < hi:
                x = 0.5 * (lo + hi)
            try:
                f_x = self.d(x)
            except _Undefined:
                self._singularity(x, h, "the bounds' difference changes sign through it")
                return x
            if f_x == 0.0:
                self.crossings.append(x)
                self.roots.add(x)
                return x
            if (f_x < 0) == (f_lo < 0):
                lo, f_lo = x, f_x
                if side == -1:
                    f_hi *= 0.5
                side = -1
            else:
                hi, f_hi = x, f_x
                if side == 1:
                    f_lo *= 0.5
                side = 1
        x = 0.5 * (lo + hi)
        try:
            at_x = abs(self.d(x))
        except _Undefined:
            at_x = math.inf
        if at_x > POLE_GROWTH * scale:
            self._singularity(x, h, "the bounds' difference changes sign through it")
            return x
        self.crossings.append(x)
        self.roots.add(x)
        return x

    def _peaks(self, h: float) -> List[float]:
        """Refine the sharp peaks of |f1 - f2| (end points included); returns where they are.

        A peak is one grid point, or two neighbouring ones (a peak between them), that
        stand more than PEAK_SHARPNESS times above every value just outside them. A jump
        next to a crossing does not qualify: one side stays as high.
        """
        sizes = [abs(v) for v in self.values]
        last = len(sizes) - 1
        found: List[float] = []
        probes = 0
        i = 0
        while i <= last:
            block = _sharp_block(sizes, i)
            if block is None:
                i += 1
                continue
            if probes >= MAX_PEAK_PROBES:
                self._limit(f"Only the first {MAX_PEAK_PROBES} sharp peaks were checked for singularities.")
                break
            probes += 1
            lo, hi = self.xs[max(i - 1, 0)], self.xs[min(block + 1, last)]
            peak_x, peak = self._refine_peak(lo, hi)
            if peak is None or peak > PEAK_SINGULAR_RATIO * max(sizes[i : block + 1]):
                self._singularity(peak_x, h, "the bounds' difference grows without bound there")
            found.append(peak_x)
            i = block + 1
        return found

    def _refine_peak(self, lo: float, hi: float) -> Tuple[float, Optional[float]]:
        """Golden-section search for the maximum of |f1 - f2| in [lo, hi]: (x, value), None if not computable."""
        m1, m2 = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
        try:
            v1, v2 = abs(self.d(m1)), abs(self.d(m2))
            for _ in range(PEAK_REFINE_STEPS):
                if v1 < v2:
                    lo, m1, v1 = m1, m2, v2
                    m2 = lo + _GOLDEN * (hi - lo)
                    v2 = abs(self.d(m2))
                else:
                    hi, m2, v2 = m2, m1, v1
                    m1 = hi - _GOLDEN * (hi - lo)
                    v1 = abs(self.d(m1))
        except _Undefined:
            return 0.5 * (m1 + m2), None
        return (m1, v1) if v1 >= v2 else (m2, v2)

    def _singularity(self, s: float, h: float, why: str) -> None:
        """Classify an unbounded point: divergent (an error) or integrable (accuracy limited)."""
        order = self._singular_order(s, h)
        if order >= DIVERGENT_ORDER:
            raise _diverges(s, why)
        self.singular[s] = order
        self.unbounded = True
        self._limit(
            f"|f1 - f2| has an integrable singularity near x ≈ {s:.6g} (it grows like |x - s|^-{order:.2g}): "
            "the value is approximate and has no error estimate."
        )

    def _singular_order(self, s: float, h: float) -> float:
        """p in |f1 - f2| ~ |x - s|^-p, from two distances; infinity when it cannot be computed there."""
        near = max(1e-9 * h, 1e-13 * max(1.0, abs(s)))
        far = max(1e-6 * h, 1e3 * near)

        def size(delta: float) -> float:
            sizes = []
            for x in (s - delta, s + delta):
                try:
                    sizes.append(abs(self.d(x)))
                except _Undefined:
                    continue
            return max(sizes) if sizes else math.inf

        v_far, v_near = size(far), size(near)
        if not math.isfinite(v_near) or not math.isfinite(v_far):
            return math.inf
        if v_far <= 0.0 or v_near <= v_far:
            return 0.0
        return math.log(v_near / v_far) / math.log(far / near)

    # -- adaptive Simpson -------------------------------------------------

    def _edge(self, x: float, toward: float) -> float:
        """f1 - f2 at a panel end; at an integrable singularity, just inside the panel."""
        if x in self.singular:
            span = max(abs(toward - x), 1e-300)
            x = x + (toward - x) * min(1e-9, 1e-12 * max(1.0, abs(x)) / span)
        return self.d(x)

    def _adaptive(self, cuts: List[float]) -> Tuple[float, Optional[float], int]:
        """Globally adaptive Simpson over the panels between the grid points and the cuts."""
        xs, values = self.xs, self.values
        stride = max(1, (len(xs) - 1) // INITIAL_PANELS)
        known = {xs[i]: values[i] for i in range(0, len(xs), stride)}
        known[xs[-1]] = values[-1]
        nodes = sorted(set(known) | set(cuts))
        heap: List[Tuple[float, int, _Panel]] = []
        frozen: List[_Panel] = []
        counter = 0
        # Raises _OutOfBudget when the budget does not cover setting up the panels.
        node_values = {x: known[x] for x in nodes if x in known and x not in self.singular}
        for lo, hi in zip(nodes, nodes[1:]):
            f_lo = node_values[lo] if lo in node_values else self._edge(lo, hi)
            f_hi = node_values[hi] if hi in node_values else self._edge(hi, lo)
            # Every panel is split once, so a kink or jump inside it shows (see ROUGH_RATIO).
            for panel in self._panels(lo, hi, f_lo, f_hi):
                for child in self._split(panel):
                    heapq.heappush(heap, (-child.error, counter, child))
                    counter += 1
        total = sum(p.value for _, _, p in heap)
        error = sum(p.error for _, _, p in heap)
        min_width = 1e-13 * max(1.0, abs(self.a), abs(self.b))
        while heap and error > RELATIVE_TOLERANCE * abs(total):
            _, _, panel = heapq.heappop(heap)
            if panel.b - panel.a <= min_width:
                frozen.append(panel)
                error -= panel.error
                continue
            try:
                children = self._split(panel)
            except _OutOfBudget:
                heapq.heappush(heap, (-panel.error, counter, panel))
                self._limit("The evaluation budget ran out before the integral converged.")
                break
            total += sum(c.value for c in children) - panel.value
            error += sum(c.error for c in children) - panel.error
            for child in children:
                heapq.heappush(heap, (-child.error, counter, child))
                counter += 1
        panels = [p for _, _, p in heap] + frozen
        for panel in panels:
            tail = self._singular_tail(panel)
            if tail is not None:
                panel.value = tail
        total = sum(p.value for p in panels)
        error = sum(p.error for p in panels)
        if frozen and sum(p.error for p in frozen) > RELATIVE_TOLERANCE * abs(total):
            self._limit("Some panels could not be split further (a very narrow or singular feature).")
        if self.limited:
            # A loose bound: the raw Simpson differences, 15 times the usual estimate.
            error = sum(max(p.raw_difference, p.error) for p in panels)
        return total, error, len(panels)

    def _panels(self, a: float, b: float, f_a: float, f_b: float, f_m: Optional[float] = None) -> List["_Panel"]:
        """Panels over [a, b]: one, or two split at a crossing its nodes reveal."""
        m = 0.5 * (a + b)
        if f_m is None:
            f_m = self.d(m)
        f_l, f_r = self.d(0.5 * (a + m)), self.d(0.5 * (m + b))
        xs = [a, 0.5 * (a + m), m, 0.5 * (m + b), b]
        fs = [f_a, f_l, f_m, f_r, f_b]
        if b - a > 1e-12 * max(1.0, abs(a), abs(b)):
            # An end at a located crossing is a root: its value's sign is noise, not a flip.
            checked = [(x, f) for x, f in zip(xs, fs) if x not in self.roots]
            flip = _first_flip([f for _, f in checked])
            if flip is not None:
                # A crossing the grid missed: locate it and split there.
                (x0, f0), (x1, f1) = checked[flip], checked[flip + 1]
                root = self._locate_crossing(x0, x1, f0, f1, b - a)
                if a < root < b:
                    f_root = self._edge(root, a)
                    left = self._panels(a, root, f_a, f_root)
                    return left + self._panels(root, b, self._edge(root, b), f_b)
        return [_Panel(a, b, fs)]

    def _split(self, panel: "_Panel") -> List["_Panel"]:
        f_a, f_l, f_m, f_r, f_b = panel.values
        children = self._panels(panel.a, panel.m, f_a, f_m, f_l) + self._panels(panel.m, panel.b, f_m, f_b, f_r)
        if panel.rough or sum(c.raw_difference for c in children) * ROUGH_RATIO > panel.raw_difference:
            for child in children:
                child.make_rough()
        return children

    def _singular_tail(self, panel: "_Panel") -> Optional[float]:
        """The integral of the power law |f1 - f2| ~ C |x - s|^-p over a panel that ends at a singularity."""
        for s, order in self.singular.items():
            for end, other_value in ((panel.a, panel.values[-1]), (panel.b, panel.values[0])):
                if end == s and order < 1.0:
                    return abs(other_value) * (panel.b - panel.a) / (1.0 - order)
        return None


class _Panel:
    """A Simpson panel: five nodes, the refined value and its error estimate."""

    __slots__ = ("a", "b", "m", "values", "value", "error", "raw_difference", "fine", "rough")

    def __init__(self, a: float, b: float, values: List[float]) -> None:
        self.a, self.b, self.m = a, b, 0.5 * (a + b)
        self.values = values
        f_a, f_l, f_m, f_r, f_b = (abs(v) for v in values)
        coarse = (b - a) / 6.0 * (f_a + 4.0 * f_m + f_b)
        fine = (b - a) / 12.0 * (f_a + 4.0 * f_l + 2.0 * f_m + 4.0 * f_r + f_b)
        self.raw_difference = abs(fine - coarse)
        self.error = self.raw_difference / 15.0
        self.value = fine + (fine - coarse) / 15.0
        self.fine = fine
        self.rough = False

    def make_rough(self) -> None:
        """Not smooth here (a kink or a jump): the raw difference is the error, no Richardson step."""
        self.rough = True
        self.error = ROUGH_ERROR_FACTOR * self.raw_difference
        self.value = self.fine

    def __lt__(self, other: "_Panel") -> bool:
        return self.a < other.a


def _sharp_block(sizes: Sequence[float], i: int) -> Optional[int]:
    """The last index of a sharp peak starting at grid point i (one or two points), or None.

    One point is sharp when it is PEAK_SHARPNESS times its lower neighbour and clearly
    above the higher one (PEAK_SHOULDER); two points are sharp together when both are
    PEAK_SHARPNESS times every value just outside them (a peak between them).
    """
    for j in (i, i + 1):
        if j >= len(sizes):
            return None
        block = min(sizes[i : j + 1])
        outside = [sizes[k] for k in (i - 1, j + 1) if 0 <= k < len(sizes)]
        if block <= 0.0 or not outside:
            continue
        if j == i and block > PEAK_SHARPNESS * min(outside) and block > PEAK_SHOULDER * max(outside):
            return j
        if j == i + 1 and block > PEAK_SHARPNESS * max(outside):
            return j
    return None


def _first_flip(values: Sequence[float]) -> Optional[int]:
    """Index i of the first node pair (i, i + 1) with opposite non-zero signs, or None."""
    for i in range(len(values) - 1):
        if values[i] * values[i + 1] < 0:
            return i
    return None


def _count_sign_changes(values: Sequence[float]) -> int:
    """Sign changes between consecutive non-zero values (exact zeros are skipped)."""
    count = 0
    last = 0.0
    for value in values:
        if value == 0.0:
            continue
        if last * value < 0:
            count += 1
        last = value
    return count


def _linear_abs_integral(xs: Sequence[float], values: Sequence[float]) -> float:
    """Integral of |piecewise linear interpolant| through the points."""
    total = 0.0
    for i in range(len(xs) - 1):
        h, v0, v1 = xs[i + 1] - xs[i], values[i], values[i + 1]
        if v0 * v1 < 0:
            total += 0.5 * h * (v0 * v0 + v1 * v1) / (abs(v0) + abs(v1))
        else:
            total += 0.5 * h * (abs(v0) + abs(v1))
    return total


# ----------------------------------------------------------------------
# Closed shapes
# ----------------------------------------------------------------------


def _closed_shape(area: Any, expression_area: Optional[ExpressionArea]) -> ColoredAreaMeasure:
    shape = str(getattr(area, "shape_type", ""))
    if shape == "circle" and getattr(area, "circle", None) is not None:
        radius = float(area.circle.radius)
        return _exact(math.pi * radius * radius, f"pi * r^2 with r = {radius:.12g}")
    if shape == "ellipse" and getattr(area, "ellipse", None) is not None:
        rx, ry = float(area.ellipse.radius_x), float(area.ellipse.radius_y)
        return _exact(math.pi * rx * ry, f"pi * a * b with a = {rx:.12g} and b = {ry:.12g}")
    if shape == "polygon":
        coords = _loop_vertices(list(getattr(area, "segments", None) or []))
        if len(coords) < 3:
            raise ValueError("The coloured polygon's segments do not form a closed loop")
        return _exact(_shoelace(coords), f"the shoelace formula over its {len(coords)} straight-edged vertices")
    if shape == "region":
        return _region(area, expression_area)
    if shape in ("circle_segment", "ellipse_segment"):
        raise ValueError(
            "Cannot measure a coloured circle or ellipse segment by name; use calculate_area with the "
            "shape and its chord segment, e.g. 'C(5) & AB'"
        )
    raise ValueError(f"Cannot measure a coloured closed shape of type '{shape}'")


def _region(area: Any, expression_area: Optional[ExpressionArea]) -> ColoredAreaMeasure:
    """A region area from its own region expression; the drawn outline gives the error estimate."""
    points = [(float(x), float(y)) for x, y in (getattr(area, "points", None) or [])]
    outline = _shoelace(points) if len(points) >= 3 else None
    expression = getattr(area, "expression", None)
    if isinstance(expression, str) and expression.strip() and expression_area is not None:
        value = float(expression_area(expression))
        measure: ColoredAreaMeasure = {
            "value": value,
            "method": f"the region expression '{expression}' measured by the region engine",
            "error_estimate": abs(value - outline) if outline is not None else None,
        }
        if outline is not None:
            measure["note"] = "error_estimate is the difference from the drawn outline (sampled curved edges)."
        return measure
    if outline is None:
        raise ValueError("The coloured region has no expression and no outline to measure")
    return {
        "value": outline,
        "method": f"the shoelace formula over the {len(points)} points of the drawn outline",
        "error_estimate": None,
        "warning": "The outline samples any curved edges, so this is an approximation of unknown accuracy.",
    }


def _loop_vertices(segments: List[Any]) -> List[Tuple[float, float]]:
    """The vertices of segments that close one loop, in order; empty when they do not."""
    if len(segments) < 3:
        return []
    ends = [((float(s.point1.x), float(s.point1.y)), (float(s.point2.x), float(s.point2.y))) for s in segments]
    start, current = ends[0]
    vertices = [start]
    unused = ends[1:]
    while unused:
        following = next((pair for pair in unused if current in pair), None)
        if following is None:
            return []
        unused.remove(following)
        vertices.append(current)
        current = following[1] if following[0] == current else following[0]
    return vertices if current == start else []


def _exact(value: float, formula: str) -> ColoredAreaMeasure:
    return {"value": value, "method": f"exact: {formula}", "error_estimate": 0.0}


def _shoelace(points: Sequence[Tuple[float, float]]) -> float:
    twice = 0.0
    for (x1, y1), (x2, y2) in zip(points, list(points[1:]) + [points[0]]):
        twice += x1 * y2 - x2 * y1
    return abs(twice) / 2.0
