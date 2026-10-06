"""Numeric roots, local extrema and intersections of plotted functions y = f(x).

This module has no browser/Brython dependencies so it can be validated by the
server-side pytest suites as well as the Brython test runner.

Method:
    The interval is split at the known breakpoints (vertical asymptotes and point
    discontinuities) and each piece is sampled on a uniform grid. Where the grid does not
    resolve the curve (the fourth difference of the samples is comparable to their
    second differences: close or steep features), finer samples are added, at most
    three levels of eight times finer steps, within a budget of one extra evaluation
    per sample.

    - Roots: sign changes between consecutive finite samples are refined with
      Brent's method. A bracket whose refined point is not a true zero (|f| does not
      shrink there, or f blows up right beside it) is a pole or a jump and is dropped,
      so 1/x and tan(x) report no false roots. Samples that are exactly zero are roots
      as they stand; a run of them means f vanishes on a whole interval. An extremum
      that dips across the axis between two samples on the same side splits their step
      into two sign-change brackets ((x - 0.5)(x - 0.5001), x^2 - 1e-10).
    - Extrema: sign changes of the slope between samples are refined with Brent's
      bounded minimizer. A refined value that runs far past the samples, or one beside a
      pole (f is clearly better nearby, or changes as much over one float as over a
      wider offset), is dropped.
    - A local extremum whose value is zero (x^2 at 0) is also reported as a touching
      root, which no sign change can find.
    - Inflection points (only when asked for): sign changes of the second difference
      between samples, above rounding noise, are refined with Brent's method on a
      numeric second derivative, then confirmed: the second derivative must have
      opposite signs on both sides and f must be continuous there, so poles and jumps
      are not inflections. The numeric derivatives use the steps actually taken, so
      rounding x -/+ step costs no accuracy at a large |x| (an inflection at 1e8 is found).
    - Intersections of f and g are the roots of f - g on the shared interval.
    - ``scan_roots`` returns the raw roots for callers that map them to points
      (utils/object_intersections.py), splitting an extremum that dips across zero
      between two samples into its two roots.

Results closer than the sampling can resolve are merged, values are rounded to the
accuracy the methods reach, and at most ``max_results`` features are returned (the
report's ``truncated`` flag says when more were found).
"""

from __future__ import annotations

import bisect
import math
from typing import Callable, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TypedDict

DEFAULT_MAX_RESULTS = 50

# Sample counts: at least MIN_SAMPLES, SAMPLES_PER_UNIT per unit of x, SAMPLES_PER_PIXEL
# per screen pixel and SAMPLES_PER_PERIOD per period, capped at MAX_SAMPLES (Brython
# evaluates every sample through the Python expression evaluator).
MIN_SAMPLES = 500
MAX_SAMPLES = 10000
SAMPLES_PER_UNIT = 25
SAMPLES_PER_PIXEL = 2
SAMPLES_PER_PERIOD = 40
MIN_SEGMENT_SAMPLES = 16

# Local refinement: where the samples do not resolve the curve, each sample step is split
# into _REFINE_FACTOR steps, at most _REFINE_DEPTH times over, spending at most
# _REFINE_BUDGET extra evaluations per sample taken
_REFINE_FACTOR = 8
_REFINE_DEPTH = 3
_REFINE_BUDGET = 1.0
# The curve is unresolved where the fourth difference of the samples is above this fraction
# of their second differences: the curvature changes within about two sample steps
_UNRESOLVED_RATIO = 0.5
# Steps refined either side of an unresolved stretch, and steps left alone next to a
# breakpoint (the curve is steep next to a vertical asymptote, not unresolved)
_REFINE_MARGIN = 1
_BREAKPOINT_SKIP = 4
# Steps are not refined below this many units in the last place of x
_REFINE_MIN_ULPS = 64.0
# Neighbouring steps that differ by less than this fraction (rounding of x) are equal
_UNIFORM_STEP_TOLERANCE = 1e-6

FEATURE_ROOTS = "roots"
FEATURE_EXTREMA = "extrema"
FEATURE_INFLECTIONS = "inflections"
SUPPORTED_FEATURES = (FEATURE_ROOTS, FEATURE_EXTREMA, FEATURE_INFLECTIONS)
# Found when no features are named; inflection points are only found on request
DEFAULT_FEATURES = (FEATURE_ROOTS, FEATURE_EXTREMA)

KIND_ROOT = "root"
KIND_LOCAL_MIN = "local_min"
KIND_LOCAL_MAX = "local_max"
KIND_INTERSECTION = "intersection"
KIND_INFLECTION = "inflection"

_EPS = 2.220446049250313e-16
# Rounding noise of values that underflow: subnormal numbers are spaced 5e-324 apart, so
# their relative precision is far worse than _EPS (exp(-x^2) near x = 27 is a staircase)
_UNDERFLOW_NOISE = 4.0 * 5e-324
_SQRT_EPS = math.sqrt(_EPS)
_GOLDEN = 0.5 * (3.0 - math.sqrt(5.0))

# Breakpoints are approached no closer than this fraction of the interval width
_BREAKPOINT_GAP = 1e-9
# A refined root must have |f| below this fraction of the bracket's end values (loose
# enough for a vertical tangent: cbrt(x) is 8e-6 at Brent's x resolution of 6e-16)...
_ROOT_RESIDUAL_RATIO = 1e-3
# ...plus the change of f to the next float, when that is below this fraction of those values
_FLOAT_STEP_SHARE = 0.1
# ...and |f| must shrink towards it: at _ROOT_PROBE_OFFSET of the interval width (at least
# _ROOT_PROBE_ULPS units in the last place of x) from the point it must be at most
# _ROOT_SHRINK_RATIO of |f| ten times further out. Near a pole |f| grows instead, and
# across a jump it stays level; a steep root (cbrt(x), tanh(1e5 x)) still shrinks.
_ROOT_PROBE_OFFSET = 1e-7
_ROOT_PROBE_ULPS = 64.0
_ROOT_PROBE_FACTOR = 10.0
_ROOT_SHRINK_RATIO = 0.95
# An interval endpoint whose |f| is below this fraction of the largest |f| sampled is a root
# that rounding moved off zero (sin(2*pi) is -2.4e-16)...
_ENDPOINT_ZERO_RATIO = 1e-12
# ...provided f rises away from it like a root: the next sample inwards is at least this many
# times larger (exp(x) at x = -40 is tiny but not a root: its neighbour is about as small)
_ENDPOINT_RISE_FACTOR = 1e6
# An extremum value within this fraction of its bump depth counts as zero (touching root)
_ZERO_EXTREMUM_RATIO = 1e-9
# A refined extremum may move past the best sample by at most this many bump depths
_EXTREMUM_RUNAWAY_FACTOR = 10.0
# An extremum must have no clearly better value at these fractions of its flatness radius
# (better by more than noise and this fraction of its bump depth)
_NEIGHBOUR_DEPTH_RATIO = 1e-3
_NEIGHBOUR_FRACTIONS = (0.25, 0.0625)
# Slope sign changes separated by more flat steps than this are plateaus, not extrema
_MAX_FLAT_STEPS = 3
# Differences below this multiple of the rounding error of the values are flat
_FLAT_NOISE_FACTOR = 16.0
# Results closer than this fraction of the interval width (or this many units in the last
# place of x) are one result
_MERGE_DISTANCE = 1e-7
_MERGE_ULPS = 64.0
# Significant digits reported: Brent's root finder reaches ~1e-15 relative accuracy,
# an extremum position only ~sqrt(eps) (less where f is flatter), so it gets fewer
_ROOT_DIGITS = 10
_EXTREMUM_X_DIGITS = 8
# More than 10^_FAR_X_MAGNITUDES interval widths from 0 an x keeps more significant digits
# (see _round_x): a root up to 14, which leaves over 40 units of margin over the rounding
# of x itself; an extremum or inflection up to 15, its uncertainty radius setting the decimals
_FAR_X_MAGNITUDES = 3
_ROOT_MAX_DIGITS = 14
_MAX_X_DIGITS = 15
_VALUE_DIGITS = 10
# Function values below this are reported as 0 (sin(pi) is 1.2e-16, not a value to show)
_VALUE_ZERO_FLOOR = 1e-12

_KIND_ORDER = {KIND_ROOT: 0, KIND_INTERSECTION: 0, KIND_LOCAL_MIN: 1, KIND_LOCAL_MAX: 1, KIND_INFLECTION: 2}
# A second difference within this many rounding units of its terms counts as zero curvature
_CURVATURE_NOISE_FACTOR = 64.0
# Share of that factor charged for rounding x inside f (eps * |x| times the slope); the
# rounding of the stencil's own arguments is avoided by using the steps really taken
_ARGUMENT_NOISE_SHARE = 0.125
# Rounding-noise estimate (in ulps) used only to decide how many digits of x to report:
# deciding whether an inflection exists stays conservative, reporting its digits does not
_ROUNDING_NOISE_FACTOR = 4.0
# Step of the numeric second derivative as a fraction of the sample spacing (first refinement),
# and the further fraction of that step used to polish the answer
_CURVATURE_STEP_FRACTION = 0.125
_CURVATURE_POLISH_FRACTION = 0.125
# The second derivative must change sign this many sample spacings either side of an
# inflection, and this many polishing steps either side of it
_INFLECTION_CHECK_STEPS = 2.0
_LOCAL_CHECK_STEPS = 4.0
# Where f is flat to rounding at that distance, the check is repeated at up to this many
# halvings of the spacing
_CONFIRM_HALVINGS = 6
# Measured noise: second differences of 2 * _NOISE_PROBE_POINTS + 1 points this fraction of
# the step apart; a curvature (or an unresolved stretch) must exceed it this many times
_NOISE_PROBE_FRACTION = 1.0 / 1024.0
_NOISE_PROBE_POINTS = 3
_NOISE_MARGIN = 4.0
# Noise beside an inflection: probe points this fraction of the spacing apart (not a power
# of two), centred this many spacings either side
_SIDE_PROBE_FRACTION = 0.3713
_SIDE_PROBE_DISTANCE = 3.0
# A bracket end whose curvature is lost in noise moves out by this fraction of the bracket,
# at most this many times
_BRACKET_WIDENING = 0.125
_BRACKET_WIDENINGS = 4
# Halving the polishing step may move an inflection by at most this fraction of the step
_STEP_STABILITY = 0.25
# Continuity check across an inflection: the change of f over a tiny offset (relative to
# max(1, |x|)) must be at most this fraction of its change over a nearby offset
_CONTINUITY_OFFSET = 1e-9
_CONTINUITY_RATIO = 0.1
# ...and at most this fraction of the nearby offset, which a large |x| would otherwise exceed
_CONTINUITY_NEAR_FRACTION = 1e-3
# The same check beside an extremum, with a smaller tiny offset so that cusps pass
_EXTREMUM_CONTINUITY_FRACTION = 1e-4
_INFLECTION_X_DIGITS = 8


class _FunctionFeatureBase(TypedDict):
    x: float
    y: float
    kind: str


class FunctionFeature(_FunctionFeatureBase, total=False):
    """One feature of a function.

    ``touching`` marks a root where f touches zero without crossing it (an extremum on
    the x-axis). ``zero_interval`` is set when f (or f - g) is zero on a whole interval
    [a, b] of samples; ``x`` is then its left end. ``point_name`` is added by callers
    that place a point at the feature.
    """

    touching: bool
    zero_interval: List[float]
    point_name: str


class FeatureReport(TypedDict):
    """Features sorted by x, whether the list was cut at max_results, and the work done."""

    features: List[FunctionFeature]
    total_found: int
    truncated: bool
    samples: int


class RawRoot(NamedTuple):
    """An unrounded root from ``scan_roots``.

    ``end`` is set when f is zero on the whole interval [x, end] of samples; ``touching``
    when f touches zero there without crossing it.
    """

    x: float
    end: Optional[float]
    touching: bool


class _Root:
    """A raw root found in one segment; ``end`` is set for a run of zero samples."""

    def __init__(self, x: float, end: Optional[float] = None, touching: bool = False) -> None:
        self.x = x
        self.end = end
        self.touching = touching


class _Extremum:
    """A raw local extremum: its sample bracket, bump depth, the radius within which x is
    uncertain and the largest |y| that still counts as on the x-axis."""

    def __init__(
        self,
        x: float,
        y: float,
        kind: str,
        bracket: Tuple[float, float],
        depth: float,
        radius: float,
        zero_tolerance: float,
    ) -> None:
        self.x = x
        self.y = y
        self.kind = kind
        self.bracket = bracket
        self.depth = depth
        self.radius = radius
        self.zero_tolerance = zero_tolerance

    def is_on_axis(self) -> bool:
        return abs(self.y) <= self.zero_tolerance


class _Piece:
    """A continuous piece [a, b] of the interval: its samples and the step of its uniform grid.

    The samples are the uniform grid plus finer ones where it did not resolve the curve.
    """

    def __init__(self, a: float, b: float, xs: List[float], ys: List[float], step: float) -> None:
        self.a = a
        self.b = b
        self.xs = xs
        self.ys = ys
        self.step = step

    def flat_limit(self) -> float:
        """Sign changes further apart than this are separated by a plateau (_MAX_FLAT_STEPS grid steps)."""
        return (_MAX_FLAT_STEPS + 1) * self.step * (1.0 + _UNIFORM_STEP_TOLERANCE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_count(span: float, pixel_span: Optional[float] = None, period: Optional[float] = None) -> int:
    """Return how many samples to take over an interval of width ``span``.

    ``pixel_span`` is the interval's width on screen (from the canvas) and ``period`` the
    function's period when it is periodic; both raise the count so no visible wiggle or
    period is skipped.
    """
    count = max(MIN_SAMPLES, math.ceil(span * SAMPLES_PER_UNIT))
    if pixel_span is not None and math.isfinite(pixel_span) and pixel_span > 0:
        count = max(count, math.ceil(pixel_span * SAMPLES_PER_PIXEL))
    if period is not None and math.isfinite(period) and period > 0:
        count = max(count, math.ceil(span / period * SAMPLES_PER_PERIOD))
    return int(min(count, MAX_SAMPLES))


def find_function_features(
    f: Callable[[float], float],
    left: float,
    right: float,
    features: Sequence[str] = DEFAULT_FEATURES,
    breakpoints: Iterable[float] = (),
    samples: Optional[int] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> FeatureReport:
    """Find the roots, local extrema and/or inflection points of f on [left, right].

    Args:
        f: The function; exceptions, NaN, infinities and complex values count as undefined.
        left, right: The interval (left < right, both finite).
        features: Any of "roots", "extrema" and "inflections".
        breakpoints: x values where f is not continuous (vertical asymptotes, holes, jumps).
        samples: Total sample count; defaults to ``sample_count`` of the interval width.
        max_results: Maximum number of features returned.
    """
    left, right = _checked_interval(left, right)
    wanted = _checked_features(features)
    evaluate = _safe_evaluator(f)
    total_samples = samples if samples is not None else sample_count(right - left)
    pieces = _sampled_segments(evaluate, left, right, breakpoints, total_samples)
    span = right - left
    found: List[FunctionFeature] = []
    if FEATURE_ROOTS in wanted or FEATURE_EXTREMA in wanted:
        # Extrema are always scanned: a touching root is found as an extremum on the x-axis
        roots, extrema = _scan(evaluate, left, right, pieces)
        if FEATURE_ROOTS in wanted:
            roots = _with_touching_roots(roots, extrema, span)
            found.extend(_root_feature(root, KIND_ROOT, 0.0, span) for root in roots)
        if FEATURE_EXTREMA in wanted:
            found.extend(_extremum_feature(extremum, span) for extremum in extrema)
    if FEATURE_INFLECTIONS in wanted:
        inflections = _scan_inflections(evaluate, span, pieces)
        found.extend(_inflection_feature(evaluate, x, radius, span) for x, radius in inflections)
    return _report(found, max_results, total_samples)


def find_intersections(
    f: Callable[[float], float],
    g: Callable[[float], float],
    left: float,
    right: float,
    breakpoints: Iterable[float] = (),
    samples: Optional[int] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> FeatureReport:
    """Find where f and g meet on [left, right]: the roots of f - g, each with y = f(x).

    A tangency (f - g touches zero, e.g. x^2 and 0) is found like a touching root.
    ``breakpoints`` should hold the breakpoints of both functions.
    """
    left, right = _checked_interval(left, right)
    evaluate_f = _safe_evaluator(f)
    evaluate_g = _safe_evaluator(g)

    def difference(x: float) -> float:
        return evaluate_f(x) - evaluate_g(x)

    total_samples = samples if samples is not None else sample_count(right - left)
    span = right - left
    pieces = _sampled_segments(difference, left, right, breakpoints, total_samples)
    roots, extrema = _scan(difference, left, right, pieces)
    roots = _with_touching_roots(roots, extrema, span)
    found = [_root_feature(root, KIND_INTERSECTION, evaluate_f(root.x), span) for root in roots]
    return _report(found, max_results, total_samples)


def scan_roots(
    f: Callable[[float], float],
    left: float,
    right: float,
    breakpoints: Iterable[float] = (),
    samples: Optional[int] = None,
) -> List[RawRoot]:
    """Unrounded, uncapped roots of f on [left, right], for callers that map them to points.

    Finds what ``find_function_features`` finds as roots (sign changes, exact zeros and runs
    of them, touching roots) and, in addition, two crossings closer together than the sample
    spacing: a local extremum whose refined value has the other sign than the samples beside
    it is split into its two roots.
    """
    left, right = _checked_interval(left, right)
    evaluate = _safe_evaluator(f)
    total_samples = samples if samples is not None else sample_count(right - left)
    span = right - left
    roots, extrema = _scan(evaluate, left, right, breakpoints, total_samples)
    roots.extend(_close_crossing_pairs(evaluate, roots, extrema, span))
    return [RawRoot(root.x, root.end, root.touching) for root in _with_touching_roots(roots, extrema, span)]


def round_report_value(value: float) -> float:
    """Round a number for a report the way feature values are rounded (10 significant digits)."""
    return _round_value(value, _VALUE_DIGITS, _VALUE_ZERO_FLOOR)


# ---------------------------------------------------------------------------
# Validation and evaluation
# ---------------------------------------------------------------------------


def _checked_interval(left: float, right: float) -> Tuple[float, float]:
    left, right = float(left), float(right)
    if not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError("The interval bounds must be finite numbers.")
    if left >= right:
        raise ValueError(f"The left bound ({left}) must be less than the right bound ({right}).")
    return left, right


def _checked_features(features: Sequence[str]) -> Tuple[str, ...]:
    wanted = tuple(dict.fromkeys(str(feature).strip().lower() for feature in features))
    if not wanted:
        raise ValueError("Ask for at least one feature: 'roots', 'extrema' or 'inflections'.")
    unknown = [feature for feature in wanted if feature not in SUPPORTED_FEATURES]
    if unknown:
        raise ValueError(f"Unknown feature(s) {unknown}; use 'roots', 'extrema' and/or 'inflections'.")
    return wanted


def _safe_evaluator(f: Callable[[float], float]) -> Callable[[float], float]:
    """Wrap f so that every failure or non-real value comes back as NaN."""

    def evaluate(x: float) -> float:
        try:
            value = f(x)
        except Exception:
            return math.nan
        if isinstance(value, complex):
            if value.imag != 0:
                return math.nan
            value = value.real
        try:
            return float(value)
        except Exception:
            return math.nan

    return evaluate


# ---------------------------------------------------------------------------
# Sampling and scanning
# ---------------------------------------------------------------------------


def _segments(left: float, right: float, breakpoints: Iterable[float]) -> List[Tuple[float, float]]:
    """Split [left, right] at the breakpoints inside it, keeping a small gap around each."""
    gap = _BREAKPOINT_GAP * (right - left)
    cuts = sorted({float(b) for b in breakpoints if math.isfinite(float(b)) and left < float(b) < right})
    pieces: List[Tuple[float, float]] = []
    start = left
    for cut in cuts:
        if cut - gap > start:
            pieces.append((start, cut - gap))
        start = cut + gap
    if right > start:
        pieces.append((start, right))
    return pieces


def _sampled_segments(
    evaluate: Callable[[float], float],
    left: float,
    right: float,
    breakpoints: Iterable[float],
    total_samples: int,
) -> List[_Piece]:
    """Each continuous piece of [left, right] with its sample grid and values.

    The grid is uniform, plus finer samples where it does not resolve the curve.
    """
    span = right - left
    cuts = _discontinuities(evaluate, breakpoints, span / max(1, total_samples))
    pieces: List[_Piece] = []
    for a, b in _segments(left, right, cuts):
        count = max(MIN_SEGMENT_SAMPLES, int(round(total_samples * (b - a) / span)))
        xs = [a + (b - a) * i / count for i in range(count)] + [b]
        ys = [evaluate(x) for x in xs]
        budget = int(_REFINE_BUDGET * count)
        xs, ys = _refined_samples(evaluate, xs, ys, budget, a != left, b != right)
        pieces.append(_Piece(a, b, xs, ys, (b - a) / count))
    return pieces


def _discontinuities(evaluate: Callable[[float], float], breakpoints: Iterable[float], step: float) -> List[float]:
    """The breakpoints where f really is discontinuous: undefined, a pole, a jump or a hole.

    A breakpoint where f is defined and continuous (the corner where two pieces of a
    piecewise function meet) is sampled through like any other point, so a root or an
    extremum right there (|x| with a breakpoint at 0) is not lost in the gap around a cut.
    """
    near = _CURVATURE_STEP_FRACTION * step
    cuts: List[float] = []
    for breakpoint in breakpoints:
        x = float(breakpoint)
        value = evaluate(x)
        if not (math.isfinite(value) and _is_continuous_beside(evaluate, x, value, near)):
            cuts.append(x)
    return cuts


def _refined_samples(
    evaluate: Callable[[float], float],
    xs: List[float],
    ys: List[float],
    budget: int,
    cut_before: bool,
    cut_after: bool,
) -> Tuple[List[float], List[float]]:
    """Add finer samples where the uniform grid does not resolve the curve.

    Stretches whose curvature changes within about two steps (close or steep features)
    are resampled with steps _REFINE_FACTOR times finer, and so on _REFINE_DEPTH times,
    worst stretches first, until the evaluation budget is spent. Features are then found
    on the merged grid by the same tests as everywhere else.
    """
    added: List[Tuple[float, float]] = []
    grids = [(xs, ys, cut_before, cut_after)]
    for _ in range(_REFINE_DEPTH):
        # (severity, grid xs, grid ys, starts at a breakpoint, ends at a breakpoint)
        stretches: List[Tuple[float, List[float], List[float], bool, bool]] = []
        for grid_xs, grid_ys, before, after in grids:
            for first, last, severity in _unresolved_stretches(grid_ys, before, after):
                stretches.append(
                    (
                        severity,
                        grid_xs[first : last + 1],
                        grid_ys[first : last + 1],
                        before and first == 0,
                        after and last == len(grid_xs) - 1,
                    )
                )
        stretches.sort(key=lambda stretch: (-stretch[0], stretch[1][0]))
        grids = []
        for _, coarse_xs, coarse_ys, before, after in stretches:
            cost = (len(coarse_xs) - 1) * (_REFINE_FACTOR - 1)
            fine_step = (coarse_xs[1] - coarse_xs[0]) / _REFINE_FACTOR
            if cost > budget or fine_step <= _REFINE_MIN_ULPS * _EPS * max(abs(coarse_xs[0]), abs(coarse_xs[-1])):
                continue
            budget -= 2 * _NOISE_PROBE_POINTS + 1
            if _is_noise(evaluate, coarse_xs, coarse_ys):
                continue
            budget -= cost
            fine_xs, fine_ys = _subdivided(evaluate, coarse_xs, coarse_ys)
            added.extend((x, y) for k, (x, y) in enumerate(zip(fine_xs, fine_ys)) if k % _REFINE_FACTOR)
            grids.append((fine_xs, fine_ys, before, after))
        if not grids:
            break
    if not added:
        return xs, ys
    merged = dict(zip(xs, ys))
    for x, y in added:
        merged.setdefault(x, y)
    ordered = sorted(merged)
    return ordered, [merged[x] for x in ordered]


def _is_noise(evaluate: Callable[[float], float], xs: List[float], ys: List[float]) -> bool:
    """The second differences of an unresolved stretch are no larger than f's measured noise.

    Noise looks unresolved at every step (its fourth differences are as large as its second
    ones); refining it would only multiply the sign changes it makes.
    """
    scale = 0.0
    for i in range(1, len(ys) - 1):
        second = ys[i - 1] - 2.0 * ys[i] + ys[i + 1]
        if math.isfinite(second):
            scale = max(scale, abs(second))
    middle = len(xs) // 2
    return scale <= _NOISE_MARGIN * _measured_noise(evaluate, xs[middle], xs[middle + 1] - xs[middle])


def _subdivided(
    evaluate: Callable[[float], float], xs: List[float], ys: List[float]
) -> Tuple[List[float], List[float]]:
    """Split every step of the grid into _REFINE_FACTOR equal steps, keeping the known samples."""
    fine_xs: List[float] = [xs[0]]
    fine_ys: List[float] = [ys[0]]
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        for k in range(1, _REFINE_FACTOR):
            x = a + (b - a) * k / _REFINE_FACTOR
            fine_xs.append(x)
            fine_ys.append(evaluate(x))
        fine_xs.append(b)
        fine_ys.append(ys[i + 1])
    return fine_xs, fine_ys


def _unresolved_stretches(ys: List[float], cut_before: bool, cut_after: bool) -> List[Tuple[int, int, float]]:
    """Sample index ranges (first, last, severity) of a uniform grid that the samples do not resolve.

    There the fourth difference is comparable to the second differences around it (the
    curvature changes within about two steps), the second differences being above rounding
    noise. Steps next to a breakpoint are left alone, as are undefined samples.
    """
    count = len(ys)
    low = _BREAKPOINT_SKIP if cut_before else 2
    high = count - 1 - (_BREAKPOINT_SKIP if cut_after else 2)
    # seconds[i] and noise[i] belong to sample i + 1; an undefined sample makes its seconds non-finite
    seconds = [ys[i] - 2.0 * ys[i + 1] + ys[i + 2] for i in range(count - 2)]
    sizes = [abs(y) for y in ys]
    noise = [
        _CURVATURE_NOISE_FACTOR * (_EPS * (sizes[i] + 2.0 * sizes[i + 1] + sizes[i + 2]) + _UNDERFLOW_NOISE)
        for i in range(count - 2)
    ]
    flagged: List[Tuple[int, float]] = []
    for i in range(low, high):
        s0, s1, s2 = seconds[i - 2], seconds[i - 1], seconds[i]
        fourth = s0 - 2.0 * s1 + s2
        scale = max(abs(s0), abs(s1), abs(s2))
        # Cheap test first: almost every sample of a resolved curve fails it
        if not abs(fourth) > _UNRESOLVED_RATIO * scale or not math.isfinite(fourth):
            continue
        if scale <= max(noise[i - 2], noise[i - 1], noise[i]):
            continue
        flagged.append((i, abs(fourth) / scale))
    stretches: List[Tuple[int, int, float]] = []
    for i, severity in flagged:
        first = max(0, i - 1 - _REFINE_MARGIN)
        last = min(count - 1, i + 1 + _REFINE_MARGIN)
        if stretches and first <= stretches[-1][1]:
            previous = stretches[-1]
            stretches[-1] = (previous[0], last, max(previous[2], severity))
        else:
            stretches.append((first, last, severity))
    return stretches


def _scan(
    evaluate: Callable[[float], float],
    left: float,
    right: float,
    pieces: List[_Piece],
) -> Tuple[List[_Root], List[_Extremum]]:
    """Collect the roots and extrema of every sampled continuous piece of [left, right]."""
    span = right - left
    roots: List[_Root] = []
    extrema: List[_Extremum] = []
    for piece in pieces:
        xs, ys = piece.xs, piece.ys
        segment_extrema = _segment_extrema(evaluate, xs, ys, span, piece.flat_limit())
        roots.extend(_segment_roots(evaluate, xs, ys, span))
        roots.extend(_roots_beside_extrema(evaluate, xs, ys, segment_extrema, span))
        roots.extend(_endpoint_roots(xs, ys, piece.a == left, piece.b == right))
        extrema.extend(segment_extrema)
    return roots, extrema


def _segment_roots(evaluate: Callable[[float], float], xs: List[float], ys: List[float], span: float) -> List[_Root]:
    roots = _zero_sample_roots(xs, ys)
    for i in range(len(xs) - 1):
        ya, yb = ys[i], ys[i + 1]
        if not (math.isfinite(ya) and math.isfinite(yb)) or ya == 0.0 or yb == 0.0:
            continue
        if (ya < 0.0) == (yb < 0.0):
            continue
        root = _refine_root(evaluate, xs[i], xs[i + 1], ya, yb, span)
        if root is not None:
            roots.append(_Root(root))
    roots.sort(key=lambda root: root.x)
    return roots


def _roots_beside_extrema(
    evaluate: Callable[[float], float],
    xs: List[float],
    ys: List[float],
    extrema: List[_Extremum],
    span: float,
) -> List[_Root]:
    """The two roots either side of an extremum that dips across the axis between two samples.

    Samples on the same side of the axis hide two close roots ((x - 0.5)(x - 0.5001), x^2 - 1e-10)
    when the extremum between them lies on the other side; it splits their step into two
    sign-change brackets. An extremum on the axis within rounding is a touching root instead.
    """
    roots: List[_Root] = []
    for extremum in extrema:
        x_m, y_m = extremum.x, extremum.y
        if not math.isfinite(y_m) or extremum.is_on_axis():
            continue
        k = bisect.bisect_right(xs, x_m) - 1
        if k < 0 or k >= len(xs) - 1 or xs[k] == x_m:
            continue
        for a, b, fa, fb in ((xs[k], x_m, ys[k], y_m), (x_m, xs[k + 1], y_m, ys[k + 1])):
            if not _changes_sign(fa, fb):
                continue
            root = _refine_root(evaluate, a, b, fa, fb, span)
            if root is not None:
                roots.append(_Root(root))
    return roots


def _endpoint_roots(xs: List[float], ys: List[float], at_left: bool, at_right: bool) -> List[_Root]:
    """Roots at the ends of the search interval that rounding moved slightly off zero.

    No sign change brackets them (cos(x) on [0, pi/2] ends at 6e-17), so an end sample
    counts as a root when |f| there is at rounding level next to the largest |f| sampled
    and f rises away from it. Exact zeros are already found by ``_zero_sample_roots``;
    near-duplicates are merged later.
    """
    finite = [abs(y) for y in ys if math.isfinite(y)]
    scale = max(finite) if finite else 0.0
    if scale == 0.0 or len(ys) < 2:
        return []
    roots: List[_Root] = []
    for wanted, index, inner in ((at_left, 0, 1), (at_right, len(ys) - 1, len(ys) - 2)):
        y, neighbour = ys[index], ys[inner]
        if not wanted or not (math.isfinite(y) and math.isfinite(neighbour)) or y == 0.0:
            continue
        if abs(y) <= _ENDPOINT_ZERO_RATIO * scale and abs(y) * _ENDPOINT_RISE_FACTOR <= abs(neighbour):
            roots.append(_Root(xs[index]))
    return roots


def _zero_sample_roots(xs: List[float], ys: List[float]) -> List[_Root]:
    """Samples where f is exactly zero; a run of them is one root spanning an interval."""
    roots: List[_Root] = []
    i = 0
    while i < len(ys):
        if ys[i] != 0.0:
            i += 1
            continue
        j = i
        while j + 1 < len(ys) and ys[j + 1] == 0.0:
            j += 1
        roots.append(_Root(xs[i], end=xs[j] if j > i else None))
        i = j + 1
    return roots


def _refine_root(
    evaluate: Callable[[float], float], a: float, b: float, fa: float, fb: float, span: float
) -> Optional[float]:
    """Refine a sign-change bracket with Brent's method; None for a pole or a jump."""
    x, fx = _brent_root(evaluate, a, b, fa, fb, xtol=_EPS * span)
    if not math.isfinite(fx):
        return None
    scale = max(abs(fa), abs(fb))
    if abs(fx) > _ROOT_RESIDUAL_RATIO * scale + _float_step_change(evaluate, x, fx, scale):
        return None
    if not _shrinks_towards(evaluate, x, max(_ROOT_PROBE_OFFSET * span, _ROOT_PROBE_ULPS * _EPS * abs(x))):
        return None
    return x


def _float_step_change(evaluate: Callable[[float], float], x: float, fx: float, scale: float) -> float:
    """How much f changes from x to the next float either side, if small next to ``scale``.

    Far from 0 the floats are coarse (1.2e-4 apart at 1e12), so no x makes f smaller than
    that change; beside a pole the change is huge, and counts as nothing.
    """
    offset = 0.75 * _EPS * abs(x)
    change = 0.0
    for value in (evaluate(x - offset), evaluate(x + offset)):
        if math.isfinite(value):
            change = max(change, abs(value - fx))
    return change if change <= _FLOAT_STEP_SHARE * scale else 0.0


def _shrinks_towards(evaluate: Callable[[float], float], x: float, offset: float) -> bool:
    """True if |f| gets smaller approaching x from both sides (a root), not larger (a pole) or level (a jump)."""
    for side in (-1.0, 1.0):
        near = evaluate(x + side * offset)
        far = evaluate(x + side * offset * _ROOT_PROBE_FACTOR)
        if not (math.isfinite(near) and math.isfinite(far)):
            return False
        if abs(near) > _ROOT_SHRINK_RATIO * abs(far):
            return False
    return True


def _segment_extrema(
    evaluate: Callable[[float], float], xs: List[float], ys: List[float], span: float, flat_limit: float
) -> List[_Extremum]:
    """Bracket slope sign changes between samples and refine each with Brent's minimizer.

    Sign changes more than ``flat_limit`` apart are the ends of a plateau, not an extremum.
    """
    extrema: List[_Extremum] = []
    last_sign = 0
    last_index = -1
    for i in range(len(xs) - 1):
        sign = _slope_sign(ys[i], ys[i + 1])
        if sign is None:
            last_sign, last_index = 0, -1
            continue
        if sign == 0:
            continue
        if last_sign == -sign and xs[i] - xs[last_index] <= flat_limit:
            kind = KIND_LOCAL_MAX if last_sign > 0 else KIND_LOCAL_MIN
            extremum = _refine_extremum(evaluate, xs, ys, last_index, i + 1, kind, span)
            if extremum is not None:
                extrema.append(extremum)
        last_sign, last_index = sign, i
    return extrema


def _slope_sign(ya: float, yb: float) -> Optional[int]:
    """+1 rising, -1 falling, 0 flat within rounding noise, None where f is undefined."""
    if not (math.isfinite(ya) and math.isfinite(yb)):
        return None
    difference = yb - ya
    if abs(difference) <= _FLAT_NOISE_FACTOR * (_EPS * max(abs(ya), abs(yb)) + _UNDERFLOW_NOISE):
        return 0
    return 1 if difference > 0 else -1


def _refine_extremum(
    evaluate: Callable[[float], float],
    xs: List[float],
    ys: List[float],
    first: int,
    last: int,
    kind: str,
    span: float,
) -> Optional[_Extremum]:
    """Refine the extremum between samples ``first`` and ``last``; None if it runs off to a pole."""
    direction = 1.0 if kind == KIND_LOCAL_MIN else -1.0

    def objective(x: float) -> float:
        value = evaluate(x)
        return direction * value if math.isfinite(value) else math.inf

    inner = ys[first + 1 : last]
    best_sample = min(inner) if kind == KIND_LOCAL_MIN else max(inner)
    depth = max(abs(ys[first] - best_sample), abs(ys[last] - best_sample))
    low, high = xs[first], xs[last]
    x, scaled = _brent_minimize(objective, low, high, xatol=_EPS * span)
    x, scaled = _polish_minimum(objective, x, scaled, low, high)
    if not math.isfinite(scaled) or scaled > direction * best_sample:
        # Never report a point worse than the best sample
        best_index = first + 1 + inner.index(best_sample)
        x, scaled = xs[best_index], direction * best_sample
    y = direction * scaled
    if abs(y - best_sample) > _EXTREMUM_RUNAWAY_FACTOR * depth:
        return None
    radius = _flatness_radius(objective, x, scaled, high - low)
    # How much f changes within the minimizer's resolution of x: large at a cusp
    # (|x|^(2/3) is 4e-12 where x misses 0 by 7e-18), negligible at a smooth extremum
    resolution = 2.0 * (4.0 * _EPS * abs(x) + _EPS * span)
    blur = 0.0
    for value in (evaluate(x - resolution), evaluate(x + resolution)):
        if math.isfinite(value):
            blur = max(blur, abs(value - y))
    if (
        radius >= high - low
        or _has_better_neighbour(objective, x, scaled, radius, blur, depth)
        or not _is_continuous_beside(evaluate, x, y, _CURVATURE_STEP_FRACTION * (high - low) / (last - first))
    ):
        # x is beside a pole (a sample close to tan's asymptote), not an extremum
        return None
    return _Extremum(x, y, kind, (low, high), depth, radius, max(_ZERO_EXTREMUM_RATIO * depth, blur))


def _is_continuous_beside(evaluate: Callable[[float], float], x: float, fx: float, near: float) -> bool:
    """On each side, the change of f from x shrinks with the offset (rules out x beside a pole).

    At a large |x| the float next to a pole is the largest value on its side, so no
    neighbour beats it; but its change over one float is as large as over a wider offset.
    Smooth extrema, kinks and cusps such as |x|^(1/3) change far less over the tiny offset.
    """
    tiny = max(_EXTREMUM_CONTINUITY_FRACTION * near, 0.75 * _EPS * abs(x))
    for side in (-1.0, 1.0):
        tiny_value, near_value = evaluate(x + side * tiny), evaluate(x + side * near)
        if not (math.isfinite(tiny_value) and math.isfinite(near_value)):
            return False
        noise = _CURVATURE_NOISE_FACTOR * (_EPS * max(abs(fx), abs(tiny_value), abs(near_value)) + _UNDERFLOW_NOISE)
        if abs(tiny_value - fx) > _CONTINUITY_RATIO * abs(near_value - fx) + noise:
            return False
    return True


def _has_better_neighbour(
    f: Callable[[float], float], x: float, fx: float, radius: float, blur: float, depth: float
) -> bool:
    """f is clearly lower than fx somewhere inside the flatness radius.

    A minimum has no such point, beyond rounding noise and beyond ``blur``, what f changes
    within the minimizer's resolution of x (a cusp such as |x|^(1/3) is 4e-6 where x misses
    0 by 5e-17). Next to a pole the radius ends just past the pole (where f changes sign),
    so a quarter of it is still on the pole's side, where f runs away by a sizeable part of
    the bump depth; rounding noise of f, which can be far above a few eps (exp(-x/2) *
    sin(5*x) near x = 140 rounds sin's argument 700), stays far below that.
    """
    margin = _FLAT_NOISE_FACTOR * (_EPS * abs(fx) + _UNDERFLOW_NOISE) + blur + _NEIGHBOUR_DEPTH_RATIO * depth
    for fraction in _NEIGHBOUR_FRACTIONS:
        if min(f(x - fraction * radius), f(x + fraction * radius)) < fx - margin:
            return True
    return False


def _polish_minimum(
    f: Callable[[float], float], x: float, fx: float, low: float, high: float, max_iter: int = 80
) -> Tuple[float, float]:
    """Golden-section search close around Brent's answer, keeping the best point seen.

    Brent's minimizer stops at sqrt(eps) relative accuracy, which suits smooth extrema
    but leaves a kink such as |x - 0.3| at about 1e-9 above its minimum.
    """
    width = 4.0 * (_SQRT_EPS * abs(x) + _EPS * (high - low))
    a, b = max(low, x - width), min(high, x + width)
    best_x, best_f = x, fx
    c, d = b - (1.0 - _GOLDEN) * (b - a), a + (1.0 - _GOLDEN) * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        for point, value in ((c, fc), (d, fd)):
            if value < best_f:
                best_x, best_f = point, value
        if b - a <= 4.0 * _EPS * max(abs(a), abs(b), 1e-300):
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (1.0 - _GOLDEN) * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (1.0 - _GOLDEN) * (b - a)
            fd = f(d)
    return best_x, best_f


def _flatness_radius(f: Callable[[float], float], x: float, fx: float, limit: float) -> float:
    """How far from x the objective stays within rounding noise of its minimum fx.

    Any x in that radius is as good a minimizer as any other (x^4 - 1 is flat to
    double precision for |x| < 1e-4), so it bounds the digits worth reporting.
    """
    noise = _FLAT_NOISE_FACTOR * _EPS * abs(fx)
    radius = max(4.0 * _EPS * abs(x), _EPS * limit, 1e-300)
    while radius < limit:
        if min(f(x - radius), f(x + radius)) - fx > noise:
            return radius
        radius *= 2.0
    return limit


# ---------------------------------------------------------------------------
# Brent's methods
# ---------------------------------------------------------------------------


def _scan_inflections(
    evaluate: Callable[[float], float],
    span: float,
    pieces: List[_Piece],
) -> List[Tuple[float, float]]:
    """Inflection points (x, uncertainty radius) of every sampled continuous piece, merged and sorted."""
    found: List[Tuple[float, float]] = []
    for piece in pieces:
        found.extend(_segment_inflections(evaluate, piece.xs, piece.ys, span, piece.flat_limit()))
    merged: List[Tuple[float, float]] = []
    for inflection in sorted(found):
        if merged and _is_near(merged[-1][0], inflection[0], span, merged[-1][1] + inflection[1]):
            continue
        merged.append(inflection)
    return merged


def _segment_inflections(
    evaluate: Callable[[float], float], xs: List[float], ys: List[float], span: float, flat_limit: float
) -> List[Tuple[float, float]]:
    """Bracket sign changes of the second difference between samples and refine each.

    Where the grid was refined the steps differ, so the second difference is the divided
    one and the refinement works on the finest step around the bracket. Sign changes more
    than ``flat_limit`` apart are separated by a straight stretch, not an inflection.
    """
    inflections: List[Tuple[float, float]] = []
    last_sign = 0
    last_index = -1
    for i in range(1, len(xs) - 1):
        sign = _sample_curvature_sign(xs, ys, i)
        if sign is None:
            last_sign, last_index = 0, -1
            continue
        if sign == 0:
            continue
        if last_sign == -sign and xs[i] - xs[last_index] <= flat_limit:
            # The finest step there, unless the bracket spans many steps of flat curvature
            # (x^5 - x near 0 on a refined grid), where a step that small drowns in noise
            spacing = max(
                min(xs[k + 1] - xs[k] for k in range(last_index - 1, i + 1)),
                (xs[i] - xs[last_index]) / (_MAX_FLAT_STEPS + 1),
            )
            inflection = _refine_inflection(evaluate, xs[last_index], xs[i], spacing, span)
            if inflection is not None:
                inflections.append(inflection)
        last_sign, last_index = sign, i
    return inflections


def _sample_curvature_sign(xs: List[float], ys: List[float], i: int) -> Optional[int]:
    """Sign of the second difference of the samples around sample i (weighted where the steps differ)."""
    h1, h2 = xs[i] - xs[i - 1], xs[i + 1] - xs[i]
    if abs(h1 - h2) <= _UNIFORM_STEP_TOLERANCE * max(h1, h2):
        # The uniform grid (its steps differ only by the rounding of x)
        return _curvature_sign(ys[i - 1], ys[i], ys[i + 1])
    return _curvature_sign(ys[i - 1], ys[i], ys[i + 1], h1=h1, h2=h2)


def _curvature_sign(
    y0: float,
    y1: float,
    y2: float,
    extra_noise: float = 0.0,
    factor: float = _CURVATURE_NOISE_FACTOR,
    h1: float = 1.0,
    h2: float = 1.0,
) -> Optional[int]:
    """Sign of the second difference y0 - 2*y1 + y2: 0 within rounding noise, None if undefined.

    With unequal steps h1 (before y1) and h2 (after) it is the weighted difference
    (2*h2*y0 - 2*(h1 + h2)*y1 + 2*h1*y2) / (h1 + h2), which is the same for equal steps.
    """
    if not (math.isfinite(y0) and math.isfinite(y1) and math.isfinite(y2)):
        return None
    if h1 == h2:
        second = y0 - 2.0 * y1 + y2
        size = abs(y0) + 2.0 * abs(y1) + abs(y2)
    else:
        w0, w2 = 2.0 * h2 / (h1 + h2), 2.0 * h1 / (h1 + h2)
        second = w0 * y0 - 2.0 * y1 + w2 * y2
        size = w0 * abs(y0) + 2.0 * abs(y1) + w2 * abs(y2)
    noise = factor * (_EPS * size + _UNDERFLOW_NOISE) + extra_noise
    if abs(second) <= noise:
        return 0
    return 1 if second > 0 else -1


def _second_derivative(
    evaluate: Callable[[float], float], step: float, factor: float = _CURVATURE_NOISE_FACTOR
) -> Callable[[float], float]:
    """Central-difference second derivative: 0 within rounding noise, NaN where f is undefined.

    The steps are the ones actually taken (x -/+ step rounds to a float), so rounding the
    arguments costs no accuracy. The noise still includes the rounding of x inside f
    (sin(2*x + 1) rounds 2*x + 1: about eps * |x| times the slope), which dominates at large |x|.
    """
    argument_factor = factor * _ARGUMENT_NOISE_SHARE

    def second(x: float) -> float:
        stencil = _stencil(evaluate, x, step)
        if stencil is None:
            return 0.0
        h1, h2, y0, y1, y2 = stencil
        argument_noise = 0.0
        if math.isfinite(y0) and math.isfinite(y2):
            argument_noise = argument_factor * _EPS * abs(x) * abs(y2 - y0) / (0.5 * (h1 + h2))
        sign = _curvature_sign(y0, y1, y2, argument_noise, factor, h1, h2)
        if sign is None:
            return math.nan
        if sign == 0:
            return 0.0
        return _divided_second(h1, h2, y0, y1, y2)

    return second


def _stencil(
    evaluate: Callable[[float], float], x: float, step: float
) -> Optional[Tuple[float, float, float, float, float]]:
    """(h1, h2, f(x - h1), f(x), f(x + h2)) with the steps x -/+ step really takes; None below x's precision."""
    before, after = x - step, x + step
    h1, h2 = x - before, after - x
    if h1 <= 0.0 or h2 <= 0.0:
        return None
    return h1, h2, evaluate(before), evaluate(x), evaluate(after)


def _divided_second(h1: float, h2: float, y0: float, y1: float, y2: float) -> float:
    """Second divided difference of samples h1 before and h2 after the middle one."""
    return 2.0 * (h2 * y0 - (h1 + h2) * y1 + h1 * y2) / (h1 * h2 * (h1 + h2))


def _refine_inflection(
    evaluate: Callable[[float], float], low: float, high: float, spacing: float, span: float
) -> Optional[Tuple[float, float]]:
    """Locate the sign change of the second derivative in [low, high].

    Returns (x, radius), radius being how far x is uncertain because of rounding noise,
    or None unless it is a true inflection. A step finer than the sample spacing is tried
    first; the spacing itself always reproduces the sign change seen in the samples. The
    answer is then polished with a smaller step, which must find the sign change again
    right there: across a pole the second difference flips only while its stencil
    straddles the pole.
    """
    x: Optional[float] = None
    for fraction in (_CURVATURE_STEP_FRACTION, 1.0):
        x = _curvature_root(evaluate, spacing * fraction, low, high, span, widen=True)
        if x is not None:
            break
    if x is None:
        return None
    reach = spacing * _CURVATURE_STEP_FRACTION
    polish_step = reach * _CURVATURE_POLISH_FRACTION
    polish = _second_derivative(evaluate, polish_step)
    # Grow the step while the curvature at the bracket ends is lost in rounding noise (a
    # large constant offset in f, or a large |x|), up to the spacing that found the change
    while polish(x - reach) == 0.0 or polish(x + reach) == 0.0:
        if polish_step >= spacing:
            # Below noise this close to x at any usable step: keep the sampled sign change if
            # f is not exactly straight beside x (the flat stretch between two steps of floor
            # sampled right at its jumps) and the change holds still at twice the step (at a
            # large |x| a pole's curvature drowns too)
            if _is_straight_beside(evaluate, x, reach, spacing):
                return None
            if _step_shift(evaluate, x, spacing, 2.0 * spacing, 2.0 * spacing, span) is None:
                return None
            return (x, reach) if _is_confirmed_inflection(evaluate, x, spacing) else None
        polish_step = min(2.0 * polish_step, spacing)
        polish = _second_derivative(evaluate, polish_step)
    x = _curvature_root(evaluate, polish_step, x - reach, x + reach, span)
    if x is None or not _changes_sign_around(polish, x, _LOCAL_CHECK_STEPS * polish_step):
        return None
    # A true inflection moves by O(step^2) when the step halves; a stencil reaching across
    # a pole puts the sign change one step from the pole, so it moves with the step.
    radius = _noise_radius(evaluate, polish_step, x, reach, span, _ROUNDING_NOISE_FACTOR)
    half = _second_derivative(evaluate, 0.5 * polish_step)
    if half(x - reach) != 0.0 and half(x + reach) != 0.0:
        x_half = _step_shift(evaluate, x, polish_step, 0.5 * polish_step, reach, span)
        if x_half is None:
            return None
        # The bias is O(step^2): the half step's is a third of the shift, which Richardson
        # extrapolation removes; a third of the shift still bounds what is left
        shift = abs(x_half - x)
        x, radius = x_half + (x_half - x) / 3.0, max(radius, shift / 3.0)
    else:
        # The half step drowns in rounding noise (a large |x|): compare with twice the step
        x_double = _step_shift(evaluate, x, polish_step, 2.0 * polish_step, reach, span)
        if x_double is None:
            return None
        radius = max(radius, abs(x_double - x))
    if not _is_confirmed_inflection(evaluate, x, spacing):
        return None
    return x, radius


def _is_straight_beside(evaluate: Callable[[float], float], x: float, offset: float, step: float) -> bool:
    """The second difference with this step is exactly 0 at x - offset or at x + offset.

    Not merely below rounding noise: f is exactly linear there, so its concavity cannot
    change sign at x.
    """
    for side in (-1.0, 1.0):
        stencil = _stencil(evaluate, x + side * offset, step)
        if stencil is not None and _divided_second(*stencil) == 0.0:
            return True
    return False


def _step_shift(
    evaluate: Callable[[float], float], x: float, step: float, other_step: float, reach: float, span: float
) -> Optional[float]:
    """The sign change found with ``other_step`` near the one at x found with ``step``, if it holds still.

    None if it is gone, or moved by more than _STEP_STABILITY of the larger step (beyond
    rounding noise): next to a pole it moves with the step.
    """
    x_other = _curvature_root(evaluate, other_step, x - reach, x + reach, span)
    if x_other is None:
        return None
    tolerance = 2.0 * max(
        _noise_radius(evaluate, step, x, reach, span),
        _noise_radius(evaluate, other_step, x_other, reach, span),
    )
    if abs(x_other - x) > _STEP_STABILITY * max(step, other_step) + tolerance:
        return None
    return x_other


def _changes_sign_around(second: Callable[[float], float], x: float, offset: float) -> bool:
    """``second`` is non-zero with opposite signs at x - offset and x + offset."""
    return _changes_sign(second(x - offset), second(x + offset))


def _noise_radius(
    evaluate: Callable[[float], float],
    step: float,
    x: float,
    limit: float,
    span: float,
    factor: float = _CURVATURE_NOISE_FACTOR,
) -> float:
    """Half-width around x where the second derivative is lost in rounding noise (at least a few ulps)."""
    second = _second_derivative(evaluate, step, factor)
    radius = max(_EPS * span, 4.0 * _EPS * abs(x))
    while radius < limit and (second(x - radius) == 0.0 or second(x + radius) == 0.0):
        radius *= 2.0
    return radius


def _curvature_root(
    evaluate: Callable[[float], float], step: float, low: float, high: float, span: float, widen: bool = False
) -> Optional[float]:
    """Where the second derivative (with this step) changes sign in [low, high], else None.

    The ends must differ in sign above rounding noise; the root is then located on the raw
    second difference, which, unlike the noise-zeroed one, has no flat band to stop in.
    With ``widen``, an end whose curvature is lost in noise (the inflection is right next
    to that sample) moves outwards, by at most half the bracket.
    """
    second = _second_derivative(evaluate, step)
    before, after = second(low), second(high)
    widening = (high - low) * _BRACKET_WIDENING
    for _ in range(_BRACKET_WIDENINGS if widen else 0):
        if before != 0.0 and after != 0.0:
            break
        if before == 0.0:
            low -= widening
            before = second(low)
        if after == 0.0:
            high += widening
            after = second(high)
    if not _changes_sign(before, after):
        return None

    def raw(x: float) -> float:
        stencil = _stencil(evaluate, x, step)
        return math.nan if stencil is None else _divided_second(*stencil)

    x, value = _brent_root(raw, low, high, raw(low), raw(high), xtol=_EPS * span)
    return x if math.isfinite(value) else None


def _changes_sign(before: float, after: float) -> bool:
    """Both values are finite and non-zero, with opposite signs."""
    if not (math.isfinite(before) and math.isfinite(after)) or before == 0.0 or after == 0.0:
        return False
    return (before < 0.0) != (after < 0.0)


def _is_confirmed_inflection(evaluate: Callable[[float], float], x: float, spacing: float) -> bool:
    """f is defined and continuous at x, and the second derivative has opposite signs either side."""
    if not math.isfinite(evaluate(x)):
        return False
    if not _curvature_flips_at(evaluate, x, spacing):
        return False
    return _is_continuous_at(evaluate, x, spacing)


def _curvature_flips_at(evaluate: Callable[[float], float], x: float, spacing: float) -> bool:
    """The second derivative has opposite signs either side of x.

    Judged with the sample spacing as step (it resolved the sign change in the samples) at
    two spacings either side. Where f is flat to rounding there (tanh(1e4 x) is exactly
    +/-1 two spacings from 0), the step and offset are halved until both sides are
    resolved; never across a jump, whose sides stay flat at every step.

    Both second differences must also stand clear of the noise measured beside x: f's own
    rounding can be far above the modelled few eps ((x + 1e3) - 1e3 rounds at 1e3, a
    noisy x + 1e-15 * random()), and noise alone changes sign anywhere.
    """
    noise = _side_noise(evaluate, x, spacing)
    step = spacing
    for _ in range(_CONFIRM_HALVINGS + 1):
        second = _second_derivative(evaluate, step)
        before, after = second(x - _INFLECTION_CHECK_STEPS * step), second(x + _INFLECTION_CHECK_STEPS * step)
        if not (math.isfinite(before) and math.isfinite(after)):
            return False
        floor = _NOISE_MARGIN * noise / (step * step)
        if abs(before) > floor and abs(after) > floor:
            return _changes_sign(before, after)
        step *= 0.5
    return False


def _side_noise(evaluate: Callable[[float], float], x: float, spacing: float) -> float:
    """Rounding noise of f beside x, as the size of a second difference.

    Seven points a non-dyadic fraction of the spacing apart, centred three spacings either
    side of x: their fourth differences remove the smooth curve (up to its fourth
    derivative) but not noise, which rounds to a staircase that looks straight at finer
    dyadic offsets ((x + 1e3) - 1e3 near 0.3125). A noise of amplitude a gives fourth
    differences up to 16 a and second differences up to 4 a, hence the quarter.
    """
    offset = spacing * _SIDE_PROBE_FRACTION
    noise = 0.0
    for side in (-1.0, 1.0):
        centre = x + side * _SIDE_PROBE_DISTANCE * spacing
        values = [evaluate(centre + k * offset) for k in range(-3, 4)]
        for k in range(len(values) - 4):
            y0, y1, y2, y3, y4 = values[k : k + 5]
            fourth = y0 - 4.0 * y1 + 6.0 * y2 - 4.0 * y3 + y4
            if math.isfinite(fourth):
                noise = max(noise, 0.25 * abs(fourth))
    return noise


def _measured_noise(evaluate: Callable[[float], float], x: float, spacing: float) -> float:
    """The largest second difference of f among points spaced 1/1024 of the spacing around x.

    So close together a resolved curve contributes about a millionth of its second
    differences at the spacing, so this measures the rounding noise of f there (random
    noise, at least; a staircase of rounded values can look straight at these offsets).
    """
    offset = spacing * _NOISE_PROBE_FRACTION
    values = [evaluate(x + k * offset) for k in range(-_NOISE_PROBE_POINTS, _NOISE_PROBE_POINTS + 1)]
    noise = 0.0
    for k in range(1, len(values) - 1):
        difference = values[k - 1] - 2.0 * values[k] + values[k + 1]
        if math.isfinite(difference):
            noise = max(noise, abs(difference))
    return noise


def _is_continuous_at(evaluate: Callable[[float], float], x: float, spacing: float) -> bool:
    """The change of f across x shrinks as the offset shrinks (rules out jumps and poles).

    Scale-free, so steep but continuous points (cbrt at 0) pass: a jump keeps its size
    and a pole grows, while a continuous function's change tends to zero.
    """
    near = _CURVATURE_STEP_FRACTION * spacing
    # Far below the near offset (at |x| = 1e6 a relative 1e-9 is already half of it), yet a
    # few units of x's last place, so that x -/+ tiny are distinct numbers
    tiny = max(min(_CONTINUITY_OFFSET * max(1.0, abs(x)), _CONTINUITY_NEAR_FRACTION * near), 4.0 * _EPS * abs(x))
    values = (evaluate(x - tiny), evaluate(x + tiny), evaluate(x - near), evaluate(x + near))
    if not all(math.isfinite(value) for value in values):
        return False
    left_value, right_value, near_left, near_right = values
    noise = _CURVATURE_NOISE_FACTOR * (_EPS * max(abs(value) for value in values) + _UNDERFLOW_NOISE)
    return abs(right_value - left_value) <= _CONTINUITY_RATIO * abs(near_right - near_left) + noise


def _brent_root(
    f: Callable[[float], float], a: float, b: float, fa: float, fb: float, xtol: float, max_iter: int = 200
) -> Tuple[float, float]:
    """Brent's root finder on a sign-change bracket [a, b] (as in SciPy's brentq).

    Returns the last iterate and its value; a non-finite value means f is undefined
    somewhere inside the bracket.
    """
    x_pre, x_cur = a, b
    f_pre, f_cur = fa, fb
    x_blk, f_blk = 0.0, 0.0
    s_pre, s_cur = 0.0, 0.0
    for _ in range(max_iter):
        if f_pre != 0.0 and f_cur != 0.0 and (f_pre < 0.0) != (f_cur < 0.0):
            x_blk, f_blk = x_pre, f_pre
            s_pre = s_cur = x_cur - x_pre
        if abs(f_blk) < abs(f_cur):
            x_pre, x_cur, x_blk = x_cur, x_blk, x_cur
            f_pre, f_cur, f_blk = f_cur, f_blk, f_cur
        delta = (xtol + 4.0 * _EPS * abs(x_cur)) / 2.0
        s_bis = (x_blk - x_cur) / 2.0
        if f_cur == 0.0 or abs(s_bis) < delta:
            return x_cur, f_cur
        if abs(s_pre) > delta and abs(f_cur) < abs(f_pre):
            try:
                if x_pre == x_blk:
                    s_try = -f_cur * (x_cur - x_pre) / (f_cur - f_pre)
                else:
                    d_pre = (f_pre - f_cur) / (x_pre - x_cur)
                    d_blk = (f_blk - f_cur) / (x_blk - x_cur)
                    s_try = -f_cur * (f_blk * d_blk - f_pre * d_pre) / (d_blk * d_pre * (f_blk - f_pre))
            except ZeroDivisionError:
                # Values so small their differences underflow: bisect instead
                s_try = math.inf
            if 2.0 * abs(s_try) < min(abs(s_pre), 3.0 * abs(s_bis) - delta):
                s_pre, s_cur = s_cur, s_try
            else:
                s_pre = s_cur = s_bis
        else:
            s_pre = s_cur = s_bis
        x_pre, f_pre = x_cur, f_cur
        x_cur += s_cur if abs(s_cur) > delta else (delta if s_bis > 0 else -delta)
        f_cur = f(x_cur)
        if not math.isfinite(f_cur):
            return x_cur, f_cur
    return x_cur, f_cur


def _brent_minimize(
    f: Callable[[float], float], a: float, b: float, xatol: float, max_iter: int = 500
) -> Tuple[float, float]:
    """Brent's bounded minimizer on [a, b] (golden section plus parabolic steps, as in fminbound)."""
    fulc = nfc = xf = a + _GOLDEN * (b - a)
    fx = f(xf)
    ffulc = fnfc = fx
    rat = e = 0.0
    xm = 0.5 * (a + b)
    tol1 = _SQRT_EPS * abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1
    for _ in range(max_iter):
        if abs(xf - xm) <= tol2 - 0.5 * (b - a):
            break
        golden_step = True
        if abs(e) > tol1:
            golden_step = False
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r, e = e, rat
            if abs(p) < abs(0.5 * q * r) and q * (a - xf) < p < q * (b - xf):
                rat = p / q
                x = xf + rat
                if (x - a) < tol2 or (b - x) < tol2:
                    rat = tol1 if xm >= xf else -tol1
            else:
                golden_step = True
        if golden_step:
            e = (a - xf) if xf >= xm else (b - xf)
            rat = _GOLDEN * e
        step = max(abs(rat), tol1)
        x = xf + (step if rat >= 0 else -step)
        fu = f(x)
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if fu <= fnfc or nfc == xf:
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif fu <= ffulc or fulc == xf or fulc == nfc:
                fulc, ffulc = x, fu
        xm = 0.5 * (a + b)
        tol1 = _SQRT_EPS * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
    return xf, fx


# ---------------------------------------------------------------------------
# Touching roots, merging, rounding and the report
# ---------------------------------------------------------------------------


def _close_crossing_pairs(
    evaluate: Callable[[float], float], roots: List[_Root], extrema: List[_Extremum], span: float
) -> List[_Root]:
    """The two roots of each extremum that dips across zero between two samples of one sign.

    No sign change between samples brackets such a pair; the refined extremum does, with
    each end of its sample bracket. An extremum within rounding of zero is a touching
    root instead (``_with_touching_roots``).
    """
    pairs: List[_Root] = []
    for extremum in extrema:
        if abs(extremum.y) <= _ZERO_EXTREMUM_RATIO * extremum.depth:
            continue
        low, high = extremum.bracket
        f_low, f_high = evaluate(low), evaluate(high)
        if not (_changes_sign(f_low, extremum.y) and _changes_sign(extremum.y, f_high)):
            continue
        if any(low <= root.x <= high for root in roots):
            continue
        for a, b, fa, fb in ((low, extremum.x, f_low, extremum.y), (extremum.x, high, extremum.y, f_high)):
            root = _refine_root(evaluate, a, b, fa, fb, span)
            if root is not None:
                pairs.append(_Root(root))
    return pairs


def _with_touching_roots(roots: List[_Root], extrema: List[_Extremum], span: float) -> List[_Root]:
    """Add (or mark) the roots where an extremum sits on the x-axis, then merge near-equal roots."""
    result = list(roots)
    for extremum in extrema:
        if not extremum.is_on_axis():
            continue
        nearby = [root for root in result if _is_near(root.x, extremum.x, span, extremum.radius)]
        if nearby:
            for root in nearby:
                root.touching = True
            continue
        low, high = extremum.bracket
        if any(low <= root.x <= high for root in result):
            # f crosses zero inside the bracket: two close simple roots, not a touching one
            continue
        result.append(_Root(extremum.x, touching=True))
    return _merged_roots(result, span)


def _merged_roots(roots: List[_Root], span: float) -> List[_Root]:
    merged: List[_Root] = []
    for root in sorted(roots, key=lambda item: item.x):
        if merged and merged[-1].end is None and root.end is None and _is_near(merged[-1].x, root.x, span):
            merged[-1].touching = merged[-1].touching or root.touching
            continue
        merged.append(root)
    return merged


def _is_near(x1: float, x2: float, span: float, slack: float = 0.0) -> bool:
    """Closer than the sampling can tell apart, a few units in the last place of x, or ``slack``.

    Not a fraction of |x|: at x = 1e8 that would be wider than the features to tell apart.
    """
    ulps = _MERGE_ULPS * _EPS * max(abs(x1), abs(x2))
    return abs(x1 - x2) <= max(_MERGE_DISTANCE * span, ulps, slack)


def _root_feature(root: _Root, kind: str, y: float, span: float) -> FunctionFeature:
    feature: FunctionFeature = {
        "x": _round_x(root.x, _ROOT_DIGITS, span, _ROOT_MAX_DIGITS),
        "y": _round_value(y, _VALUE_DIGITS, _VALUE_ZERO_FLOOR),
        "kind": kind,
    }
    if root.touching:
        feature["touching"] = True
    if root.end is not None:
        feature["zero_interval"] = [
            _round_x(root.x, _ROOT_DIGITS, span, _ROOT_MAX_DIGITS),
            _round_x(root.end, _ROOT_DIGITS, span, _ROOT_MAX_DIGITS),
        ]
    return feature


def _extremum_feature(extremum: _Extremum, span: float) -> FunctionFeature:
    y = extremum.y
    if extremum.is_on_axis():
        y = 0.0
    # Keep the digits of x that the function values can tell apart
    decimals = -(math.floor(math.log10(extremum.radius)) + 1)
    return {
        "x": _round_x(round(extremum.x, decimals), _EXTREMUM_X_DIGITS, span),
        "y": _round_value(y, _VALUE_DIGITS, _VALUE_ZERO_FLOOR),
        "kind": extremum.kind,
    }


def _inflection_feature(evaluate: Callable[[float], float], x: float, radius: float, span: float) -> FunctionFeature:
    # Keep the digits of x that rounding noise and the step's bias leave certain
    decimals = -(math.floor(math.log10(radius)) + 1)
    # y at the precise x, to the precision the uncertainty of x allows (0 at sin's inflections)
    y = evaluate(x)
    slope = abs(evaluate(x + radius) - evaluate(x - radius)) / (2.0 * radius)
    y_floor = max(_VALUE_ZERO_FLOOR, slope * radius) if math.isfinite(slope) else _VALUE_ZERO_FLOOR
    return {
        "x": _round_x(round(x, decimals), _INFLECTION_X_DIGITS, span),
        "y": _round_value(y, _VALUE_DIGITS, y_floor),
        "kind": KIND_INFLECTION,
    }


def _round_x(x: float, digits: int, span: float, max_digits: int = _MAX_X_DIGITS) -> float:
    """Round an x position; positions within 10^-(digits+2) of the interval width of 0 are 0.

    Far from 0 (|x| more than 1000 interval widths) x keeps a digit more for every power
    of ten beyond that, up to ``max_digits`` significant digits: at x = 1e6 + 0.3576 in an
    interval of width 10, eight significant digits would keep only two decimals.
    """
    zero_floor = 10.0 ** -(digits + 2) * span
    if math.isfinite(x) and abs(x) > span > 0:
        extra = math.floor(math.log10(abs(x))) - math.floor(math.log10(span)) - _FAR_X_MAGNITUDES
        digits = max(digits, min(max_digits, digits + extra))
    return _round_value(x, digits, zero_floor)


def _round_value(value: float, digits: int, zero_floor: float) -> float:
    """Round to ``digits`` significant digits; |value| <= zero_floor (and -0.0) become 0.0."""
    if not math.isfinite(value):
        return value
    if abs(value) <= zero_floor:
        return 0.0
    return float(f"{value:.{digits}g}") + 0.0


def _report(found: List[FunctionFeature], max_results: int, samples: int) -> FeatureReport:
    limit = max(0, int(max_results))
    ordered = sorted(found, key=lambda feature: (feature["x"], _KIND_ORDER.get(feature["kind"], 2)))
    return {
        "features": ordered[:limit],
        "total_found": len(ordered),
        "truncated": len(ordered) > limit,
        "samples": int(samples),
    }
