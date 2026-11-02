"""Numeric arc length of a plane curve given as a point function of one parameter.

Pure Python (no browser imports), so server-side pytest covers it.

Method:
    The length of a curve is the limit of the lengths of inscribed polylines. For a smooth
    curve the polyline over n equal parameter steps h falls short by A*h^2 + B*h^4 + ...,
    so each interval keeps five points (its ends and quarters) and combines one, two and
    four chords by Richardson extrapolation (Romberg's first two columns); the difference
    of the last two columns is the interval's error estimate. Intervals with the largest
    error are halved until the total error is below TOLERANCE times the length. Needing no
    derivative, the method handles corners (|x|), cusps (an astroid) and vertical tangents
    (sqrt(x) at 0).

    Extrapolation is trusted only where the interval looks resolved:
    - the chords turn by less than _TURN_LIMIT at every point, including the points just
      outside the interval (a corner or cusp makes the chords cut across it);
    - the gains of halving shrink about fourfold (two then four chords);
    - the quartic through the five points, traced through the quarters' midpoints and any
      extremum of a coordinate inside a quarter, gains little over the four chords (a steep
      coordinate's extremum hides between two points that look collinear in the plane).
    Elsewhere the interval's error is the gain it may still make, so it is halved until that
    is small. A split also checks the parent's value against its halves'.

    The whole measurement runs twice, on start grids of START_INTERVALS and
    CHECK_INTERVALS intervals (coprime), and the error reported is at least their
    difference: a failure one grid's points cannot see (a sampling coincidence) shows as a
    disagreement.

    Breaks make the length infinite or undefined and are refused:
    - a vertical asymptote (|P| grows without bound): found by numeric_integration's
      singularity scan of |P(s)|, which also catches poles that do not change sign (1/x^2);
    - a jump: a chord across one is straight, so the error estimate cannot see it. Pieces
      much faster (chord over width) than their neighbours are zoomed into, halving toward
      the longer half: a continuous curve's chord shrinks to nothing, a jump's stays.
    Where the curve has no real point, a removable point (sin(x)/x at 0) or a bound with a
    finite one-sided limit (x*ln(x) at 0) takes the limit; anywhere else is an error.

Errors are ValueError with a message for the user; ``label`` names the parameter in them.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

from utils.numeric_integration import find_interior_singularity

Point = Tuple[float, float]
PointFn = Callable[[float], Optional[Point]]

# Stop when the error estimate is below TOLERANCE times the length. If the budget (each
# grid's evaluations or the time limit) runs out first, the length is returned with a
# warning when its error is below ACCEPTABLE_TOLERANCE times it, and refused otherwise
TOLERANCE = 1e-10
ACCEPTABLE_TOLERANCE = 1e-6
MAX_EVALUATIONS = 20000
TIME_LIMIT_SECONDS = 4.0
# Each interval has SEGMENTS chords (SEGMENTS + 1 points); the start grids have
# START_INTERVALS and CHECK_INTERVALS intervals (coprime)
SEGMENTS = 8
START_INTERVALS = 32
CHECK_INTERVALS = 31

# The singularity scan of |P| samples this many intervals
SINGULARITY_SCAN_INTERVALS = 1000

# Chords turning by more than this (radians) at a point do not resolve the curve there
_TURN_LIMIT = 1.0
_COS_TURN_LIMIT = math.cos(_TURN_LIMIT)
# A smooth arc's polyline gains a quarter as much at each halving (ratio 4)
_RATIO_LOW = 3.0
_QUARTER_LOW, _QUARTER_HIGH = 3.75, 4.25
_RATIO_HIGH = 5.5
# Extrapolation is trusted only where the gain the quartic predicts for the next halving is at
# most this share of the last gain (a quarter on a resolved arc)
_PREDICTED_SHARE = 0.5
# A split gives each half at least this fraction of the disagreement between the parent's value
# and the halves' sum as its error: a smooth arc's halves are about 2^7 times more accurate
# than the parent, a fooled estimate's are not
_SPLIT_SHARE = 32.0
# Bisection steps locating a coordinate's extremum inside a quarter (to 1/1024 of it)
_EXTREMUM_BISECTIONS = 10
# Rounding: a chord is known to about this many units in the last place of the coordinates
_NOISE_ULPS = 8.0

# Jump test: a piece faster than SPIKE_FACTOR times both neighbours is zoomed into (at most
# MAX_ZOOMS of them, longest chord first). A zoom halves ZOOM_HALVINGS times, needs at least
# MIN_ZOOM_HALVINGS for a verdict, and finds a jump when the chord keeps BREAK_FRACTION of
# its starting length
SPIKE_FACTOR = 4.0
MAX_ZOOMS = 16
ZOOM_HALVINGS = 48
MIN_ZOOM_HALVINGS = 20
BREAK_FRACTION = 0.25

# Retracing: a range that is k whole periods traces the curve k times. k is found factor by
# factor (each factor up to MAX_TRACE_FACTOR, k up to MAX_TRACES); the curve is compared with
# itself one period on at RETRACE_SAMPLES parameters
MAX_TRACES = 10000
MAX_TRACE_FACTOR = 60
RETRACE_SAMPLES = 12
_RETRACE_AGREEMENT = 1e-9
# ... and only where it moves, within the period, by more than this (relative)
_RETRACE_MOVE = 1e-6
# Pieces between breakpoints must meet: their ends may differ by at most this (relative)
_JOIN_AGREEMENT = 1e-6

# The limit at an undefined point: probes PROBE_RATIO times closer each step, from
# _PROBE_START_STEP to _PROBE_STEPS steps in (a fraction 4**-k of the range away); the pair of
# successive probes that agree best gives the limit and its error, which must stay within
# _AGREEMENT of the point's scale
_PROBE_RATIO = 4.0
_PROBE_START_STEP = 4
_PROBE_STEPS = 26
_AGREEMENT = 1e-3
# A range narrower than this share of its bounds is refused (its points round together)
_NARROWEST = 1e-12

# Quartic through points at 0..4: its slope at each point (rows), and its value at each
# quarter's midpoint 0.5, 1.5, 2.5 and 3.5
_NODE_SLOPES = (
    (-25 / 12, 4.0, -3.0, 4 / 3, -1 / 4),
    (-1 / 4, -5 / 6, 3 / 2, -1 / 2, 1 / 12),
    (1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12),
    (-1 / 12, 1 / 2, -3 / 2, 5 / 6, 1 / 4),
    (1 / 4, -4 / 3, 3.0, -4.0, 25 / 12),
)
_MIDPOINTS = (
    (35 / 128, 35 / 32, -35 / 64, 7 / 32, -5 / 128),
    (-5 / 128, 15 / 32, 45 / 64, -5 / 32, 3 / 128),
    (3 / 128, -5 / 32, 45 / 64, 15 / 32, -5 / 128),
    (-5 / 128, 7 / 32, -35 / 64, 35 / 32, 35 / 128),
)


class ArcLengthResult(TypedDict, total=False):
    """length and error_estimate (absolute); warning when TOLERANCE was not reached;
    traced_times when the curve repeats itself over the range (see traced_times)."""

    length: float
    error_estimate: float
    evaluations: int
    warning: str
    traced_times: int


class _Curve:
    """The point function with an evaluation count, the range, the parameter's label and the noise floor."""

    def __init__(self, point: PointFn, lower: float, upper: float, label: str, deadline: float) -> None:
        self._point = point
        self.lower = lower
        self.upper = upper
        self.width = upper - lower
        self.label = label
        self.evaluations = 0
        self.deadline = deadline
        self.noise = 0.0
        self.limit_errors: Dict[float, float] = {}

    def at(self, s: float) -> Optional[Point]:
        """The curve's point at s, or None where it has no real finite point."""
        self.evaluations += 1
        try:
            p = self._point(s)
            if p is None:
                return None
            x, y = float(p[0]), float(p[1])
        except Exception:
            return None
        if math.isfinite(x) and math.isfinite(y):
            return (x, y)
        return None

    def defined_at(self, s: float) -> Point:
        """The point at s, or its limit where s is removable; raises where it is neither.

        A limit's error is kept in limit_errors and counted in the length's error estimate.
        """
        p = self.at(s)
        if p is not None:
            return p
        found = self._limit_with_error(s)
        if found is None:
            raise ValueError(self.undefined_message(s))
        self.limit_errors[s] = found[1]
        return found[0]

    def limit(self, s: float) -> Optional[Point]:
        """The limit of the curve at s from inside the range, or None where it has none."""
        found = self._limit_with_error(s)
        return None if found is None else found[0]

    def _limit_with_error(self, s: float) -> Optional[Tuple[Point, float]]:
        if s <= self.lower:
            return self._one_sided_limit(s, 1.0)
        if s >= self.upper:
            return self._one_sided_limit(s, -1.0)
        left, right = self._one_sided_limit(s, -1.0), self._one_sided_limit(s, 1.0)
        if left is None or right is None:
            return None
        gap = _distance(left[0], right[0])
        if gap > _AGREEMENT * _scale(left[0]):
            return None
        middle = ((left[0][0] + right[0][0]) / 2, (left[0][1] + right[0][1]) / 2)
        return middle, max(left[1], right[1]) + gap / 2

    def _one_sided_limit(self, s: float, direction: float) -> Optional[Tuple[Point, float]]:
        """The limit at s from one side, with its error: the closest pair of successive probes."""
        best: Optional[Tuple[Point, float]] = None
        previous: Optional[Point] = None
        h = self.width * _PROBE_RATIO ** (-_PROBE_START_STEP)
        for _ in range(_PROBE_START_STEP, _PROBE_STEPS + 1):
            if s + direction * h == s:
                break
            p = self.at(s + direction * h)
            if p is not None and previous is not None:
                gap = _distance(p, previous)
                if best is None or gap < best[1]:
                    best = (p, gap)
            previous = p
            h /= _PROBE_RATIO
        if best is None or best[1] > _AGREEMENT * _scale(best[0]):
            return None
        return best

    def undefined_message(self, s: float) -> str:
        where = "at the bound of" if s in (self.lower, self.upper) else "inside"
        return (
            f"The curve has no real point at {self.label} = {_fmt(s)} ({where} [{_fmt(self.lower)}, "
            f"{_fmt(self.upper)}]) and no finite limit there, so it has no length over this range. "
            "Measure only where the curve is defined."
        )


def _distance(p: Point, q: Point) -> float:
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def _scale(p: Point) -> float:
    return max(1.0, abs(p[0]), abs(p[1]))


def _fmt(value: float) -> str:
    return f"{value:.6g}"


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

# The curve points just outside an interval, before and after it (None at the range's ends)
_Neighbours = Tuple[Optional[Point], Optional[Point]]
# An interval: (-error, value, parameters, points, neighbours): five parameters and points
# (ends and quarters)
_Interval = Tuple[float, float, List[float], List[Point], _Neighbours]


def _interval(ts: List[float], ps: List[Point], neighbours: _Neighbours, noise: float) -> _Interval:
    """Value and error estimate of one interval of SEGMENTS chords (see the module docstring)."""
    hypot = math.hypot
    chords = [hypot(ps[k + 1][0] - ps[k][0], ps[k + 1][1] - ps[k][1]) for k in range(SEGMENTS)]
    eight = math.fsum(chords)
    four = (
        hypot(ps[2][0] - ps[0][0], ps[2][1] - ps[0][1])
        + hypot(ps[4][0] - ps[2][0], ps[4][1] - ps[2][1])
        + hypot(ps[6][0] - ps[4][0], ps[6][1] - ps[4][1])
        + hypot(ps[8][0] - ps[6][0], ps[8][1] - ps[6][1])
    )
    two = hypot(ps[4][0] - ps[0][0], ps[4][1] - ps[0][1]) + hypot(ps[8][0] - ps[4][0], ps[8][1] - ps[4][1])
    one = hypot(ps[8][0] - ps[0][0], ps[8][1] - ps[0][1])
    gains = (two - one, four - two, eight - four)
    floor = SEGMENTS * noise  # gains below the rounding of the chords are no gains
    turn = _max_turn(ps, chords, neighbours)
    predicted = _predicted_gain(ps[0:5], chords[0:4]) + _predicted_gain(ps[4:9], chords[4:8])
    last = gains[2]
    if turn > _TURN_LIMIT:
        # A corner, a cusp or a stretch the points do not resolve: the chords may cut across
        # it (a cusp's two branches), so the error is a share of the interval's length
        error = max(last, predicted, eight * turn / math.pi)
        return (-max(0.0, error - floor), eight + last / 3.0, ts, ps, neighbours)
    if predicted <= _PREDICTED_SHARE * last + floor and _asymptotic(gains, floor):
        # Romberg: one, two, four and eight chords, extrapolated in h^2 and then h^4 and h^6
        a2, a4, a8 = two + gains[0] / 3.0, four + gains[1] / 3.0, eight + gains[2] / 3.0
        b4, b8 = a4 + (a4 - a2) / 15.0, a8 + (a8 - a4) / 15.0
        error = abs(b8 - b4) / 63.0
        if neighbours[0] is None or neighbours[1] is None or not _quarters(gains):
            # Where the curvature is infinite (x^(3/2) at 0, |x|^(5/3) at 0) the chords converge
            # at a lower order than Romberg assumes, at an end of the range or inside it
            error = max(error, _order_error(gains[1], last, 3.0), _order_error(a4 - a2, a8 - a4, 15.0))
        return (-max(0.0, error - floor), b8 + (b8 - b4) / 63.0, ts, ps, neighbours)
    return (-max(0.0, max(last, predicted) - floor), eight + last / 3.0, ts, ps, neighbours)


def _quarters(gains: Tuple[float, float, float]) -> bool:
    """True when each halving gained close to a quarter of the one before, as a smooth arc's
    chords do once the interval is short (between _QUARTER_LOW and _QUARTER_HIGH)."""
    for coarse, fine in ((gains[0], gains[1]), (gains[1], gains[2])):
        if fine != 0.0 and not _QUARTER_LOW <= coarse / fine <= _QUARTER_HIGH:
            return False
    return True


def _order_error(coarse: float, fine: float, assumed: float) -> float:
    """What extrapolating fine by fine/assumed misses when the gains shrink by coarse/fine rather
    than by assumed + 1: a curve whose curvature is infinite at a point (x^(3/2) at 0, a
    lemniscate at the pole) converges at a lower order there, which Romberg would overshoot."""
    if fine == 0.0:
        return 0.0
    ratio = coarse / fine
    if ratio <= 1.0:
        return abs(fine)
    return abs(fine) * abs(1.0 / (ratio - 1.0) - 1.0 / assumed)


def _asymptotic(gains: Tuple[float, float, float], floor: float) -> bool:
    """True when each halving of the chords gained about a quarter of the one before, as on a smooth arc."""
    for coarse, fine in ((gains[0], gains[1]), (gains[1], gains[2])):
        if coarse <= floor and fine <= floor:
            continue
        if not (fine > 0.0 and _RATIO_LOW <= coarse / fine <= _RATIO_HIGH):
            return False
    return True


def _max_turn(ps: List[Point], chords: List[float], neighbours: _Neighbours) -> float:
    """The largest angle between consecutive chords through the points and their neighbours (0 to pi).

    The turns at the interval's ends count too, so a corner just inside either end is seen
    from both sides.
    """
    before, after = neighbours
    points = list(ps)
    lengths = list(chords)
    if before is not None:
        points.insert(0, before)
        lengths.insert(0, math.hypot(ps[0][0] - before[0], ps[0][1] - before[1]))
    if after is not None:
        points.append(after)
        lengths.append(math.hypot(after[0] - ps[-1][0], after[1] - ps[-1][1]))
    lowest = 1.0
    for k in range(1, len(points) - 1):
        product = lengths[k - 1] * lengths[k]
        if product == 0.0:
            continue
        a, b, c = points[k - 1], points[k], points[k + 1]
        cosine = ((b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])) / product
        if cosine < lowest:
            lowest = cosine
    if lowest >= 1.0:
        return 0.0
    return math.acos(max(-1.0, lowest))


def _predicted_gain(ps: List[Point], chords: List[float]) -> float:
    """The length the quartic through five points gains over their four chords.

    The quartic of each coordinate is traced through each quarter's midpoint and through any
    extremum a coordinate has inside a quarter, which a chord between two points on the same
    side of it cuts off.
    """
    xs = [p[0] for p in ps]
    ys = [p[1] for p in ps]
    extrema = _quarter_extrema(xs) + _quarter_extrema(ys)
    path = 0.0
    hypot = math.hypot
    for k in range(4):
        w = _MIDPOINTS[k]
        mx = w[0] * xs[0] + w[1] * xs[1] + w[2] * xs[2] + w[3] * xs[3] + w[4] * xs[4]
        my = w[0] * ys[0] + w[1] * ys[1] + w[2] * ys[2] + w[3] * ys[3] + w[4] * ys[4]
        inside = [s for s in extrema if k < s < k + 1]
        if not inside:
            path += hypot(mx - xs[k], my - ys[k]) + hypot(xs[k + 1] - mx, ys[k + 1] - my)
            continue
        cx, cy = _quartic(xs), _quartic(ys)
        last = (xs[k], ys[k])
        for stop in sorted(inside + [k + 0.5]):
            here = (_horner(cx, stop), _horner(cy, stop))
            path += hypot(here[0] - last[0], here[1] - last[1])
            last = here
        path += hypot(xs[k + 1] - last[0], ys[k + 1] - last[1])
    return max(0.0, path - (chords[0] + chords[1] + chords[2] + chords[3]))


def _quarter_extrema(values: List[float]) -> List[float]:
    """Where the quartic through the values has an extremum strictly inside a quarter (by its slope's sign)."""
    v0, v1, v2, v3, v4 = values
    slopes = [w[0] * v0 + w[1] * v1 + w[2] * v2 + w[3] * v3 + w[4] * v4 for w in _NODE_SLOPES]
    found: List[float] = []
    for k in range(4):
        if slopes[k] * slopes[k + 1] < 0.0:
            found.append(_extremum_between(_quartic(values), float(k), float(k + 1), slopes[k]))
    return found


def _quartic(values: List[float]) -> List[float]:
    """Coefficients c0..c4 of the quartic through (0, v0) .. (4, v4)."""
    v0, v1, v2, v3, v4 = values
    return [
        v0,
        -25 / 12 * v0 + 4 * v1 - 3 * v2 + 4 / 3 * v3 - 0.25 * v4,
        35 / 24 * v0 - 13 / 3 * v1 + 19 / 4 * v2 - 7 / 3 * v3 + 11 / 24 * v4,
        -5 / 12 * v0 + 1.5 * v1 - 2 * v2 + 7 / 6 * v3 - 0.25 * v4,
        (v0 - 4 * v1 + 6 * v2 - 4 * v3 + v4) / 24,
    ]


def _horner(c: List[float], s: float) -> float:
    return c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * c[4])))


def _extremum_between(c: List[float], lo: float, hi: float, slope_lo: float) -> float:
    """Where the quartic's slope changes sign between lo and hi, by bisection."""
    rising = slope_lo > 0
    for _ in range(_EXTREMUM_BISECTIONS):
        mid = (lo + hi) / 2
        slope_mid = c[1] + mid * (2 * c[2] + mid * (3 * c[3] + mid * 4 * c[4]))
        if (slope_mid > 0) == rising:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _split(curve: _Curve, item: _Interval) -> Tuple[_Interval, _Interval]:
    """The two halves of an interval (SEGMENTS new points).

    Each half's error is at least a share of the disagreement between the parent's value and
    the halves' sum, so an interval whose own estimate is fooled is still refined.
    """
    ts, ps, (before, after) = item[2], item[3], item[4]
    fine_ts: List[float] = []
    fine_ps: List[Point] = []
    for k in range(SEGMENTS):
        middle = (ts[k] + ts[k + 1]) / 2
        fine_ts.extend((ts[k], middle))
        fine_ps.extend((ps[k], curve.defined_at(middle)))
    fine_ts.append(ts[SEGMENTS])
    fine_ps.append(ps[SEGMENTS])
    half = SEGMENTS
    left = _interval(fine_ts[: half + 1], fine_ps[: half + 1], (before, fine_ps[half + 1]), curve.noise)
    right = _interval(fine_ts[half:], fine_ps[half:], (fine_ps[half - 1], after), curve.noise)
    floor = abs(left[1] + right[1] - item[1]) / _SPLIT_SHARE - 2 * curve.noise
    if -left[0] < floor:
        left = (-floor, left[1], left[2], left[3], left[4])
    if -right[0] < floor:
        right = (-floor, right[1], right[2], right[3], right[4])
    return left, right


def _can_split(item: _Interval) -> bool:
    ts = item[2]
    return all(ts[k] < (ts[k] + ts[k + 1]) / 2 < ts[k + 1] for k in range(SEGMENTS))


# ---------------------------------------------------------------------------
# Breaks: vertical asymptotes and jumps
# ---------------------------------------------------------------------------

# A piece: (t0, t1, p0, p1), one chord of an interval
_Piece = Tuple[float, float, Point, Point]


def _pieces_of(intervals: List[_Interval]) -> List[_Piece]:
    """Every chord of the intervals, in parameter order."""
    pieces = []
    for item in sorted(intervals, key=lambda it: it[2][0]):
        ts, ps = item[2], item[3]
        for k in range(SEGMENTS):
            pieces.append((ts[k], ts[k + 1], ps[k], ps[k + 1]))
    return pieces


def _spikes(pieces: List[_Piece], noise: float) -> List[int]:
    """Indices of pieces whose speed (chord over width) exceeds SPIKE_FACTOR times both neighbours'."""
    speeds = []
    for t0, t1, p0, p1 in pieces:
        chord = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        speeds.append(chord / (t1 - t0) if t1 > t0 and chord > 64 * noise else 0.0)
    found = []
    for i, speed in enumerate(speeds):
        if speed == 0.0:
            continue
        neighbours = [speeds[j] for j in (i - 1, i + 1) if 0 <= j < len(speeds)]
        if neighbours and speed > SPIKE_FACTOR * max(neighbours):
            found.append(i)
    found.sort(key=lambda i: speeds[i] * (pieces[i][1] - pieces[i][0]), reverse=True)
    return found[:MAX_ZOOMS]


def _zoom_for_jump(curve: _Curve, piece: _Piece) -> Optional[float]:
    """Halve a piece toward its longer half; the location of a jump, or None (continuous, or no verdict)."""
    t0, t1, p0, p1 = piece
    start = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if start <= 64 * curve.noise:
        return None
    halvings = 0
    for _ in range(ZOOM_HALVINGS):
        mid = (t0 + t1) / 2
        if mid <= t0 or mid >= t1:
            break
        pm = curve.at(mid)
        if pm is None:
            curve.defined_at(mid)  # raises unless removable
            return None
        halvings += 1
        if math.hypot(pm[0] - p0[0], pm[1] - p0[1]) >= math.hypot(p1[0] - pm[0], p1[1] - pm[1]):
            t1, p1 = mid, pm
        else:
            t0, p0 = mid, pm
    end = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if halvings >= MIN_ZOOM_HALVINGS and end >= BREAK_FRACTION * start and end > 64 * curve.noise:
        return (t0 + t1) / 2
    return None


def _check_jumps(curve: _Curve, pieces: List[_Piece]) -> None:
    for i in _spikes(pieces, curve.noise):
        location = _zoom_for_jump(curve, pieces[i])
        if location is not None:
            raise ValueError(_break_message(curve, location))


def _check_asymptotes(curve: _Curve) -> None:
    """Refuse a range across which |P| grows without bound (1/x, 1/x^2, ln|x|, tan(x))."""

    def magnitude(s: float) -> float:
        p = curve.at(s)
        if p is None:
            raise ValueError("undefined")
        return math.hypot(p[0], p[1])

    location = find_interior_singularity(magnitude, curve.lower, curve.upper, SINGULARITY_SCAN_INTERVALS)
    if location is None:
        return
    if curve.at(location) is None and curve.limit(location) is None:
        raise ValueError(curve.undefined_message(location))
    raise ValueError(_break_message(curve, location))


def _break_message(curve: _Curve, location: float) -> str:
    if abs(location) <= 1e-9 * curve.width:
        location = 0.0
    return (
        f"The curve breaks near {curve.label} = {_fmt(location)} (a jump or a vertical asymptote), so it is "
        f"not one connected curve of finite length on [{_fmt(curve.lower)}, {_fmt(curve.upper)}]. "
        "Measure each side of the break separately."
    )


def _check_finite(curve: _Curve, intervals: List[_Interval]) -> None:
    """A chord too long for a float (values near 1e308) can only come from a vertical asymptote."""
    for item in intervals:
        if not (math.isfinite(item[0]) and math.isfinite(item[1])):
            raise ValueError(_break_message(curve, (item[2][0] + item[2][SEGMENTS]) / 2))


def _check_worst_for_jumps(curve: _Curve, intervals: List[_Interval]) -> None:
    """Zoom into the places that kept the length from converging: a jump there is a break.

    Refinement may have narrowed those intervals to float resolution, too narrow to zoom,
    so each zoom starts from a window one start step wide around the interval.
    """
    window = curve.width / (SEGMENTS * START_INTERVALS)
    centres: List[float] = []
    for item in sorted(intervals, key=lambda it: it[0]):
        centre = (item[2][0] + item[2][SEGMENTS]) / 2
        if any(abs(centre - seen) < window for seen in centres):
            continue
        centres.append(centre)
        if len(centres) > MAX_ZOOMS:
            break
        lo, hi = max(curve.lower, centre - window / 2), min(curve.upper, centre + window / 2)
        p_lo, p_hi = curve.at(lo), curve.at(hi)
        if p_lo is None or p_hi is None:
            continue
        location = _zoom_for_jump(curve, (lo, hi, p_lo, p_hi))
        if location is not None:
            raise ValueError(_break_message(curve, location))


# ---------------------------------------------------------------------------
# One measurement on one start grid
# ---------------------------------------------------------------------------


class _Measurement:
    __slots__ = ("length", "error", "worst")

    def __init__(self, length: float, error: float, worst: float) -> None:
        self.length = length
        self.error = error
        self.worst = worst


def _start_intervals(curve: _Curve, count: int) -> List[_Interval]:
    n = SEGMENTS * count
    step = curve.width / n
    ts = [curve.lower + i * step for i in range(n)] + [curve.upper]
    ps = [curve.defined_at(t) for t in ts]
    if curve.noise == 0.0:
        scale = max(max(abs(p[0]), abs(p[1])) for p in ps)
        curve.noise = _NOISE_ULPS * 2.220446049250313e-16 * scale
    intervals = []
    for i in range(count):
        first = SEGMENTS * i
        neighbours = (ps[first - 1] if i else None, ps[first + SEGMENTS + 1] if i < count - 1 else None)
        intervals.append(
            _interval(ts[first : first + SEGMENTS + 1], ps[first : first + SEGMENTS + 1], neighbours, curve.noise)
        )
    return intervals


def _measure(curve: _Curve, count: int, tolerance: float, max_evaluations: int) -> _Measurement:
    intervals = _start_intervals(curve, count)
    _check_jumps(curve, _pieces_of(intervals))
    _check_finite(curve, intervals)

    limit = curve.evaluations + max_evaluations
    heapq.heapify(intervals)
    total = math.fsum(item[1] for item in intervals)
    error = math.fsum(-item[0] for item in intervals)
    while error > tolerance * total:
        if curve.evaluations >= limit or time.time() > curve.deadline:
            break
        item = heapq.heappop(intervals)
        if not _can_split(item):
            heapq.heappush(intervals, item)
            break  # the worst interval cannot be split further in floating point
        left, right = _split(curve, item)
        total += left[1] + right[1] - item[1]
        error += item[0] - left[0] - right[0]
        heapq.heappush(intervals, left)
        heapq.heappush(intervals, right)
    _check_finite(curve, intervals)
    _check_jumps(curve, _pieces_of(intervals))

    total = math.fsum(item[1] for item in intervals)
    error = math.fsum(-item[0] for item in intervals)
    if error > tolerance * total:
        _check_worst_for_jumps(curve, intervals)
    worst = min(intervals, key=lambda it: it[0])[2]
    return _Measurement(total, error, (worst[0] + worst[SEGMENTS]) / 2)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def arc_length(
    point: PointFn,
    lower: float,
    upper: float,
    label: str = "t",
    tolerance: float = TOLERANCE,
    max_evaluations: int = MAX_EVALUATIONS,
    time_limit: float = TIME_LIMIT_SECONDS,
) -> ArcLengthResult:
    """The length of the curve point(s) for s between lower and upper.

    point returns (x, y), or None (or raises, or gives a non-finite value) where the curve
    has no real point. Reversed bounds give the same length; equal bounds give 0.
    max_evaluations caps each of the two measurements; time_limit (seconds) caps the call.

    Raises:
        ValueError: If the curve is undefined somewhere on the range, breaks (a jump or a
            vertical asymptote), or the length does not converge (it oscillates too fast or
            is infinite).
    """
    lower, upper = float(lower), float(upper)
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError("The bounds must be finite numbers.")
    if lower > upper:
        lower, upper = upper, lower
    if lower == upper:
        return ArcLengthResult(length=0.0, error_estimate=0.0, evaluations=0)
    if upper - lower <= _NARROWEST * max(abs(lower), abs(upper)):
        raise ValueError(
            f"The range [{_fmt(lower)}, {_fmt(upper)}] is narrower than the numbers can resolve, so its "
            "intervals cannot be told apart. Widen the range."
        )

    curve = _Curve(point, lower, upper, label, time.time() + time_limit)
    curve.defined_at(lower)
    curve.defined_at(upper)
    _check_asymptotes(curve)
    first = _measure(curve, START_INTERVALS, tolerance, max_evaluations)
    if first.length == 0.0:
        return ArcLengthResult(length=0.0, error_estimate=0.0, evaluations=curve.evaluations)
    second = _measure(curve, CHECK_INTERVALS, tolerance, max_evaluations)

    best = first if first.error <= second.error else second
    # Twice the larger of the estimates and the grids' disagreement: a safety margin, since an
    # estimate measures the error of the less extrapolated value
    error = 2.0 * max(first.error, second.error, abs(first.length - second.length))
    # A limit standing in for an undefined point moves the length by up to twice its error
    error += 2.0 * sum(curve.limit_errors.values())
    result = ArcLengthResult(length=best.length, error_estimate=error, evaluations=curve.evaluations)
    if error <= tolerance * best.length * 20:
        return result
    if error <= ACCEPTABLE_TOLERANCE * best.length:
        result["warning"] = (
            f"The length is accurate only to about {error:.1g}: the curve has a sharp point or oscillates "
            "quickly, and more digits would need more steps than the tool takes."
        )
        return result
    raise ValueError(
        f"The length did not converge: the curve oscillates too fast or grows too long near "
        f"{label} = {_fmt(best.worst)} (it may be infinitely long there). Narrow the range."
    )


def traced_times(point: PointFn, lower: float, upper: float) -> int:
    """How many times the curve is traced over [lower, upper]: the largest k for which the curve
    one k-th of the range on is the same point (r = cos(θ) on [0, 2π] is traced twice, a circle
    on [0, 20π] ten times); 1 when it does not repeat.

    k grows factor by factor: once the range is k periods, a factor m is taken when the range is
    also k*m periods, so 12 is found as 2, then 2, then 3. A period counts only where the curve
    moves within it, so a short stretch of a curve does not pass for a tiny closed one.
    """
    lower, upper = min(lower, upper), max(lower, upper)
    width = upper - lower
    times = 1
    while True:
        factor = _next_factor(point, lower, width, times)
        if factor is None:
            return times
        times *= factor


def _next_factor(point: PointFn, lower: float, width: float, times: int) -> Optional[int]:
    for factor in range(2, MAX_TRACE_FACTOR + 1):
        if times * factor > MAX_TRACES:
            return None
        period = width / (times * factor)
        if _repeats(point, lower, period) and _moves(point, lower, period):
            return factor
    return None


def _repeats(point: PointFn, lower: float, period: float) -> bool:
    for i in range(RETRACE_SAMPLES):
        s = _retrace_sample(lower, period, i)
        try:
            here, there = point(s), point(s + period)
            if here is None or there is None:
                return False
            if _apart(here, there) > _RETRACE_AGREEMENT * _size(here):
                return False
        except Exception:
            return False
    return True


def _moves(point: PointFn, lower: float, period: float) -> bool:
    """Whether the curve gets well away from where it was within one period: true of a closed
    curve's period, false when the period is too short for the curve to have moved."""
    for i in range(RETRACE_SAMPLES):
        s = _retrace_sample(lower, period, i)
        try:
            here, there = point(s), point(s + _GOLDEN_FRACTION * period)
        except Exception:
            continue
        if here is not None and there is not None and _apart(here, there) > _RETRACE_MOVE * _size(here):
            return True
    return False


def _retrace_sample(lower: float, period: float, i: int) -> float:
    return lower + period * (i + 0.5 * _GOLDEN_FRACTION * (i % 3 + 1)) / RETRACE_SAMPLES


def _apart(here: Point, there: Point) -> float:
    return max(abs(here[0] - there[0]), abs(here[1] - there[1]))


def _size(here: Point) -> float:
    return max(1.0, abs(here[0]), abs(here[1]))


_GOLDEN_FRACTION = (math.sqrt(5.0) - 1.0) / 2.0


def arc_length_in_pieces(
    point: PointFn,
    lower: float,
    upper: float,
    breakpoints: Sequence[float] = (),
    label: str = "t",
    time_limit: float = TIME_LIMIT_SECONDS,
) -> ArcLengthResult:
    """arc_length over [lower, upper] split at the breakpoints inside it (piecewise boundaries,
    holes), checking that consecutive pieces meet: a gap between them is a jump.
    """
    lower, upper = min(lower, upper), max(lower, upper)
    cuts = sorted({float(b) for b in breakpoints if lower < float(b) < upper})
    if not cuts:
        return arc_length(point, lower, upper, label, time_limit=time_limit)
    deadline = time.time() + time_limit
    edges = [lower] + cuts + [upper]
    whole = _Curve(point, lower, upper, label, deadline)
    previous_end: Optional[Point] = None
    for a, b in zip(edges, edges[1:]):
        # Each piece's ends as limits from inside it: the value at a cut may belong to either side
        piece = _Curve(point, a, b, label, deadline)
        start, end = piece.limit(a), piece.limit(b)
        if start is None or end is None:
            raise ValueError(whole.undefined_message(a if start is None else b))
        if previous_end is not None and not _meet(previous_end, start):
            raise ValueError(_break_message(whole, a))
        previous_end = end
    total = ArcLengthResult(length=0.0, error_estimate=0.0, evaluations=0)
    for a, b in zip(edges, edges[1:]):
        part = arc_length(point, a, b, label, time_limit=max(0.0, deadline - time.time()))
        total["length"] += part["length"]
        total["error_estimate"] += part["error_estimate"]
        total["evaluations"] += part["evaluations"]
        if "warning" in part:
            total["warning"] = part["warning"]
    return total


def _meet(p: Point, q: Point) -> bool:
    scale = max(1.0, abs(p[0]), abs(p[1]))
    return abs(p[0] - q[0]) <= _JOIN_AGREEMENT * scale and abs(p[1] - q[1]) <= _JOIN_AGREEMENT * scale
