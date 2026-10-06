"""Numeric roots, local extrema and intersections of plotted functions y = f(x).

This module has no browser/Brython dependencies so it can be validated by the
server-side pytest suites as well as the Brython test runner.

Method:
    The interval is split at the known breakpoints (vertical asymptotes and point
    discontinuities) and each piece is sampled on a uniform grid.

    - Roots: sign changes between consecutive finite samples are refined with
      Brent's method. A bracket whose refined point is not a true zero (|f| does not
      shrink there, or f blows up right beside it) is a pole or a jump and is dropped,
      so 1/x and tan(x) report no false roots. Samples that are exactly zero are roots
      as they stand; a run of them means f vanishes on a whole interval.
    - Extrema: sign changes of the slope between samples are refined with Brent's
      bounded minimizer. A refined value that runs far past the samples is a pole and
      is dropped.
    - A local extremum whose value is zero (x^2 at 0) is also reported as a touching
      root, which no sign change can find.
    - Inflection points (only when asked for): sign changes of the second difference
      between samples, above rounding noise, are refined with Brent's method on a
      numeric second derivative, then confirmed: the second derivative must have
      opposite signs on both sides and f must be continuous there, so poles and jumps
      are not inflections.
    - Intersections of f and g are the roots of f - g on the shared interval.
    - ``scan_roots`` returns the raw roots for callers that map them to points
      (utils/object_intersections.py), splitting an extremum that dips across zero
      between two samples into its two roots.

Results closer than the sampling can resolve are merged, values are rounded to the
accuracy the methods reach, and at most ``max_results`` features are returned (the
report's ``truncated`` flag says when more were found).
"""

from __future__ import annotations

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
_SQRT_EPS = math.sqrt(_EPS)
_GOLDEN = 0.5 * (3.0 - math.sqrt(5.0))

# Breakpoints are approached no closer than this fraction of the interval width
_BREAKPOINT_GAP = 1e-9
# A refined root must have |f| below this fraction of the bracket's end values (loose
# enough for a vertical tangent: cbrt(x) is 8e-6 at Brent's x resolution of 6e-16)...
_ROOT_RESIDUAL_RATIO = 1e-3
# ...and |f| must shrink towards it: at _ROOT_PROBE_OFFSET (relative) from the point it must be
# at most _ROOT_SHRINK_RATIO of |f| ten times further out. Near a pole |f| grows instead, and
# across a jump it stays level; a steep root (cbrt(x), tanh(1e5 x)) still shrinks.
_ROOT_PROBE_OFFSET = 1e-7
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
# Slope sign changes separated by more flat steps than this are plateaus, not extrema
_MAX_FLAT_STEPS = 3
# Differences below this multiple of the rounding error of the values are flat
_FLAT_NOISE_FACTOR = 16.0
# Results closer than this fraction of the interval width (or of |x|) are one result
_MERGE_DISTANCE = 1e-7
# Significant digits reported: Brent's root finder reaches ~1e-15 relative accuracy,
# an extremum position only ~sqrt(eps) (less where f is flatter), so it gets fewer
_ROOT_DIGITS = 10
_EXTREMUM_X_DIGITS = 8
_VALUE_DIGITS = 10
# Function values below this are reported as 0 (sin(pi) is 1.2e-16, not a value to show)
_VALUE_ZERO_FLOOR = 1e-12

_KIND_ORDER = {KIND_ROOT: 0, KIND_INTERSECTION: 0, KIND_LOCAL_MIN: 1, KIND_LOCAL_MAX: 1, KIND_INFLECTION: 2}
# A second difference within this many rounding units of its terms counts as zero curvature
_CURVATURE_NOISE_FACTOR = 64.0
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
# Halving the polishing step may move an inflection by at most this fraction of the step
_STEP_STABILITY = 0.25
# Continuity check across an inflection: the change of f over a tiny offset (relative to
# max(1, |x|)) must be at most this fraction of its change over a nearby offset
_CONTINUITY_OFFSET = 1e-9
_CONTINUITY_RATIO = 0.1
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
    """A raw local extremum: its sample bracket, bump depth and the radius within which x is uncertain."""

    def __init__(
        self, x: float, y: float, kind: str, bracket: Tuple[float, float], depth: float, radius: float
    ) -> None:
        self.x = x
        self.y = y
        self.kind = kind
        self.bracket = bracket
        self.depth = depth
        self.radius = radius


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
    # Extrema are always scanned: a touching root is found as an extremum on the x-axis
    roots, extrema = _scan(evaluate, left, right, breakpoints, total_samples)
    span = right - left
    found: List[FunctionFeature] = []
    if FEATURE_ROOTS in wanted:
        roots = _with_touching_roots(roots, extrema, span)
        found.extend(_root_feature(root, KIND_ROOT, 0.0, span) for root in roots)
    if FEATURE_EXTREMA in wanted:
        found.extend(_extremum_feature(extremum, span) for extremum in extrema)
    if FEATURE_INFLECTIONS in wanted:
        inflections = _scan_inflections(evaluate, left, right, breakpoints, total_samples)
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
    roots, extrema = _scan(difference, left, right, breakpoints, total_samples)
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
) -> List[Tuple[float, float, List[float], List[float]]]:
    """Each continuous piece (a, b) of [left, right] with its uniform sample grid and values."""
    span = right - left
    pieces: List[Tuple[float, float, List[float], List[float]]] = []
    for a, b in _segments(left, right, breakpoints):
        count = max(MIN_SEGMENT_SAMPLES, int(round(total_samples * (b - a) / span)))
        xs = [a + (b - a) * i / count for i in range(count)] + [b]
        pieces.append((a, b, xs, [evaluate(x) for x in xs]))
    return pieces


def _scan(
    evaluate: Callable[[float], float],
    left: float,
    right: float,
    breakpoints: Iterable[float],
    total_samples: int,
) -> Tuple[List[_Root], List[_Extremum]]:
    """Sample every continuous piece of [left, right] and collect its roots and extrema."""
    span = right - left
    roots: List[_Root] = []
    extrema: List[_Extremum] = []
    for a, b, xs, ys in _sampled_segments(evaluate, left, right, breakpoints, total_samples):
        roots.extend(_segment_roots(evaluate, xs, ys, span))
        roots.extend(_endpoint_roots(xs, ys, a == left, b == right))
        extrema.extend(_segment_extrema(evaluate, xs, ys, span))
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
    if abs(fx) > _ROOT_RESIDUAL_RATIO * scale:
        return None
    if not _shrinks_towards(evaluate, x, _ROOT_PROBE_OFFSET * max(span, abs(x))):
        return None
    return x


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
    evaluate: Callable[[float], float], xs: List[float], ys: List[float], span: float
) -> List[_Extremum]:
    """Bracket slope sign changes between samples and refine each with Brent's minimizer."""
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
        if last_sign == -sign and i - last_index <= _MAX_FLAT_STEPS + 1:
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
    if abs(difference) <= _FLAT_NOISE_FACTOR * _EPS * max(abs(ya), abs(yb)):
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
    return _Extremum(x, y, kind, (low, high), depth, radius)


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
    left: float,
    right: float,
    breakpoints: Iterable[float],
    total_samples: int,
) -> List[Tuple[float, float]]:
    """Inflection points (x, uncertainty radius) of every continuous piece, merged and sorted."""
    span = right - left
    found: List[Tuple[float, float]] = []
    for _, _, xs, ys in _sampled_segments(evaluate, left, right, breakpoints, total_samples):
        found.extend(_segment_inflections(evaluate, xs, ys, span))
    merged: List[Tuple[float, float]] = []
    for inflection in sorted(found):
        if not (merged and _is_near(merged[-1][0], inflection[0], span)):
            merged.append(inflection)
    return merged


def _segment_inflections(
    evaluate: Callable[[float], float], xs: List[float], ys: List[float], span: float
) -> List[Tuple[float, float]]:
    """Bracket sign changes of the second difference between samples and refine each."""
    spacing = xs[1] - xs[0]
    inflections: List[Tuple[float, float]] = []
    last_sign = 0
    last_index = -1
    for i in range(1, len(xs) - 1):
        sign = _curvature_sign(ys[i - 1], ys[i], ys[i + 1])
        if sign is None:
            last_sign, last_index = 0, -1
            continue
        if sign == 0:
            continue
        if last_sign == -sign and i - last_index <= _MAX_FLAT_STEPS + 1:
            inflection = _refine_inflection(evaluate, xs[last_index], xs[i], spacing, span)
            if inflection is not None:
                inflections.append(inflection)
        last_sign, last_index = sign, i
    return inflections


def _curvature_sign(
    y0: float, y1: float, y2: float, extra_noise: float = 0.0, factor: float = _CURVATURE_NOISE_FACTOR
) -> Optional[int]:
    """Sign of the second difference y0 - 2*y1 + y2: 0 within rounding noise, None if undefined."""
    if not (math.isfinite(y0) and math.isfinite(y1) and math.isfinite(y2)):
        return None
    second = y0 - 2.0 * y1 + y2
    noise = factor * _EPS * (abs(y0) + 2.0 * abs(y1) + abs(y2)) + extra_noise
    if abs(second) <= noise:
        return 0
    return 1 if second > 0 else -1


def _second_derivative(
    evaluate: Callable[[float], float], step: float, factor: float = _CURVATURE_NOISE_FACTOR
) -> Callable[[float], float]:
    """Central-difference second derivative: 0 within rounding noise, NaN where f is undefined.

    The noise includes the rounding of the arguments x -/+ step (about eps * |x| times the
    slope), which dominates at large |x|.
    """

    def second(x: float) -> float:
        y0, y1, y2 = evaluate(x - step), evaluate(x), evaluate(x + step)
        argument_noise = 0.0
        if math.isfinite(y0) and math.isfinite(y2):
            argument_noise = factor * _EPS * abs(x) * abs(y2 - y0) / step
        sign = _curvature_sign(y0, y1, y2, argument_noise, factor)
        if sign is None:
            return math.nan
        if sign == 0:
            return 0.0
        return (y0 - 2.0 * y1 + y2) / (step * step)

    return second


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
        x = _curvature_root(evaluate, spacing * fraction, low, high, span)
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
            # Below noise this close to x at any usable step: keep the sampled sign change
            # (a pole's curvature is never lost in noise)
            return (x, reach) if _is_confirmed_inflection(evaluate, x, spacing) else None
        polish_step = min(2.0 * polish_step, spacing)
        polish = _second_derivative(evaluate, polish_step)
    x = _curvature_root(evaluate, polish_step, x - reach, x + reach, span)
    if x is None or not _changes_sign_around(polish, x, _LOCAL_CHECK_STEPS * polish_step):
        return None
    # A true inflection moves by O(step^2) when the step halves; a stencil reaching across
    # a pole puts the sign change one step from the pole, so it moves with the step.
    # (Skipped when the half step drowns in rounding noise, which a pole never does.)
    radius = _noise_radius(evaluate, polish_step, x, reach, span, _ROUNDING_NOISE_FACTOR)
    half = _second_derivative(evaluate, 0.5 * polish_step)
    if half(x - reach) != 0.0 and half(x + reach) != 0.0:
        x_half = _curvature_root(evaluate, 0.5 * polish_step, x - reach, x + reach, span)
        if x_half is None:
            return None
        shift = abs(x_half - x)
        tolerance = 2.0 * max(
            _noise_radius(evaluate, polish_step, x, reach, span),
            _noise_radius(evaluate, 0.5 * polish_step, x_half, reach, span),
        )
        if shift > _STEP_STABILITY * polish_step + tolerance:
            return None
        # The bias is O(step^2): the half step's is a third of the shift, which Richardson
        # extrapolation removes; a third of the shift still bounds what is left
        x, radius = x_half + (x_half - x) / 3.0, max(radius, shift / 3.0)
    if not _is_confirmed_inflection(evaluate, x, spacing):
        return None
    return x, radius


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
    evaluate: Callable[[float], float], step: float, low: float, high: float, span: float
) -> Optional[float]:
    """Where the second derivative (with this step) changes sign in [low, high], else None.

    The ends must differ in sign above rounding noise; the root is then located on the raw
    second difference, which, unlike the noise-zeroed one, has no flat band to stop in.
    """
    second = _second_derivative(evaluate, step)
    if not _changes_sign(second(low), second(high)):
        return None

    def raw(x: float) -> float:
        return (evaluate(x - step) - 2.0 * evaluate(x) + evaluate(x + step)) / (step * step)

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
    # The sample spacing as step: it resolved the sign change in the samples
    second = _second_derivative(evaluate, spacing)
    if not _changes_sign_around(second, x, _INFLECTION_CHECK_STEPS * spacing):
        return False
    return _is_continuous_at(evaluate, x, spacing)


def _is_continuous_at(evaluate: Callable[[float], float], x: float, spacing: float) -> bool:
    """The change of f across x shrinks as the offset shrinks (rules out jumps and poles).

    Scale-free, so steep but continuous points (cbrt at 0) pass: a jump keeps its size
    and a pole grows, while a continuous function's change tends to zero.
    """
    tiny = _CONTINUITY_OFFSET * max(1.0, abs(x))
    near = _CURVATURE_STEP_FRACTION * spacing
    values = (evaluate(x - tiny), evaluate(x + tiny), evaluate(x - near), evaluate(x + near))
    if not all(math.isfinite(value) for value in values):
        return False
    left_value, right_value, near_left, near_right = values
    noise = _CURVATURE_NOISE_FACTOR * _EPS * max(abs(value) for value in values)
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
        if abs(extremum.y) > _ZERO_EXTREMUM_RATIO * extremum.depth:
            continue
        nearby = [root for root in result if _is_near(root.x, extremum.x, span)]
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


def _is_near(x1: float, x2: float, span: float) -> bool:
    return abs(x1 - x2) <= _MERGE_DISTANCE * max(span, abs(x1), abs(x2))


def _root_feature(root: _Root, kind: str, y: float, span: float) -> FunctionFeature:
    feature: FunctionFeature = {
        "x": _round_x(root.x, _ROOT_DIGITS, span),
        "y": _round_value(y, _VALUE_DIGITS, _VALUE_ZERO_FLOOR),
        "kind": kind,
    }
    if root.touching:
        feature["touching"] = True
    if root.end is not None:
        feature["zero_interval"] = [_round_x(root.x, _ROOT_DIGITS, span), _round_x(root.end, _ROOT_DIGITS, span)]
    return feature


def _extremum_feature(extremum: _Extremum, span: float) -> FunctionFeature:
    y = extremum.y
    if abs(y) <= _ZERO_EXTREMUM_RATIO * extremum.depth:
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


def _round_x(x: float, digits: int, span: float) -> float:
    """Round an x position; positions within 10^-(digits+2) of the interval width of 0 are 0."""
    return _round_value(x, digits, 10.0 ** -(digits + 2) * span)


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
