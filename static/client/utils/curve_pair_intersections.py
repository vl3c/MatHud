"""Where two parametric curves (x1(s), y1(s)) and (x2(t), y2(t)) meet.

This module has no browser/Brython dependencies so it can be validated by the
server-side pytest suites as well as the Brython test runner. ``utils.object_intersections``
turns its result into intersection points and overlaps.

Method:
    1. Each curve is sampled by its complexity (a first grid, then intervals whose midpoint
       strays from the chord halved, up to 256 times finer) into a polyline, and each
       polyline gets a tree of bounding boxes.
    2. Walking both trees together finds the pairs of polyline segments that cross, and the
       pairs that come within 0.12 of a segment length of each other without crossing.
    3. Overlaps: the samples of the first curve that lie along the second are projected onto
       it (Gauss-Newton on the distance); runs of samples on it are shared stretches, their
       ends bisected. The second curve is walked the same way for stretches the first walk
       cannot see (a short piece of a long curve). A short stretch where the curves stay
       within tolerance without coinciding to rounding is a high-order contact or a cusp on
       the other curve: one point, touching or crossing by which side the curve lies on just
       past it.
    4. Crossings: each crossing pair of segments seeds a 2D Newton iteration on
       (x1(s) - x2(t), y1(s) - y2(t)) with a step limit and backtracking. When Newton fails
       (the curves run parallel there) the pair of parameter intervals is halved repeatedly,
       keeping the halves whose chords still meet, then minimised.
    5. Near touches: the closest segment pairs that do not cross (local minima of the gap
       along the first curve) seed a minimisation of the distance; a minimum within
       tolerance is a point.
    6. A point where the curves run (nearly) parallel is polished to where their directions
       are exactly parallel (bisection along the first curve, the second curve's point being
       the foot of the perpendicular) and flagged as a tangency. At a cusp (a curve that all
       but stops) the tangency is whether the curves stay on one side of each other there.
    A curve whose samples all repeat after a period 2*pi*p/q is searched over its first period
    only (``periods`` in the result). The whole search, sampling first, has a work budget
    (about 2.5 seconds in the browser); wherever it cuts work short (sampling, candidate pairs,
    shared stretches, crossings, near touches) the result is marked incomplete.

A point counts when the curves are within 1e-10 of the pair's size there, plus the rounding
of the coordinates and of the parameters.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

Point2 = Tuple[float, float]

_EPS = 2.220446049250313e-16
# Tolerance of a meeting point: this fraction of the pair's size, plus this many rounding
# units of the coordinates and parameters
_ON_OBJECT_TOLERANCE = 1e-10
_ROUNDING_UNITS = 16.0
# An interval is halved while its midpoint is farther than this fraction of the chord from
# the chord's midpoint (about 0.4 radians of turning)
_DEVIATION_RATIO = 0.05
_REFINE_PASSES = 8
# Segments this many of their lengths apart are near (the polylines' error is far smaller)
_NEAR_FACTOR = 0.12
# A segment with both ends this many of the other segment's lengths from it lies along it (the
# polylines stray from their curves by about 0.05 of a chord)
_ALONG_FACTOR = 0.1
# ...and runs within this sine of parallel to it (chords of a curve turn about 0.4 radians)
_ALONG_SINE = 0.3
# Slack of a segment parameter beyond [0, 1] when two polyline segments cross
_SEGMENT_SLACK = 1e-9
_LEAF_SIZE = 2
_NEWTON_STEPS = 40
_BACKTRACK_STEPS = 16
_MINIMISE_STEPS = 100
_PROJECT_STEPS = 40
_SHRINK_LEVELS = 64
_SHRINK_BOXES = 4
# Newton steps are limited to this many sample spacings
_STEP_SPACINGS = 4.0
# A near pair may hide a meeting point when its gap is within this many times the two
# segments' strays from their curves (such stretches are scanned sample by sample)
_MEET_FACTOR = 2.0
# Sine of the angle between the curves below which a meeting point is polished as a
# tangency, and below which it is one without polishing
_POLISH_ANGLE = 1e-3
_TANGENT_ANGLE = 1e-7
_BISECTION_STEPS = 120
# Distinct foot guesses tried per sample when walking one curve along the other
_WALK_GUESSES = 3
# Shared stretches at most this many sample steps apart, with the gap on both curves, are one
_JOIN_SPACINGS = 4.0
# A projection for an on-curve test stops once its distance is this many tolerances away and
# has stalled (a full step, not one cut to the step limit, shrank it by less than half)
_GIVE_UP_FACTOR = 100.0
_GIVE_UP_RATIO = 0.5
# Periods 2*pi*p/q tried for a curve that retraces itself (q up to this), checked at this many
# points within this fraction of its extent (times 1 + |t| + period, for rounding)
_PERIOD_DENOMINATOR = 12
_PERIOD_CHECKS = 12
_PERIOD_TOLERANCE = 1e-9
# Periods that pass the spot checks but are then checked at every sample, at most
_PERIOD_FULL_CHECKS = 2
# A meeting point where a curve moves slower than this fraction of its typical speed is a cusp
_CUSP_RATIO = 1e-3
# Rounding units of the coordinates within which two curves coincide (not just meet)
_NOISE_UNITS = 64.0
# A shared stretch at most this many sample steps long that does not coincide to rounding is
# a high-order contact (one point)
_SHORT_PIECE_SPACINGS = 8.0
_GOLDEN_STEPS = 80
# As in object_intersections: points this close (relative to the size, plus rounding) are one
_MERGE_TOLERANCE = 1e-9
_SAME_POINT_ULPS = 4.0
# First grid of a curve (its own sample count, at most this many) and most samples after
# refinement by turning
_PROBE_SAMPLES = 1500
_MAX_SAMPLES = 32000
# Work allowed for the whole search, sampling included (about 0.1 s in CPython, 2.5 s in the
# browser), in curve evaluations: a derivative counts two, a pair of tree nodes or polyline
# segments compared one, a crossing pair kept _CROSSING_PAIR_WORK (a near one
# _NEAR_PAIR_WORK) and a crossing pair checked against the points found _CHECK_WORK. Long
# ranges of curves that do not retrace themselves (incommensurate frequencies) stop there, the
# result marked incomplete; the first curve's segments are visited in order.
_WORK_BUDGET = 150000
_CROSSING_PAIR_WORK = 5
_NEAR_PAIR_WORK = 2
_CHECK_WORK = 3
# Shares of the budget by which refining the samples, the search for close segment pairs,
# and then the search for shared stretches must be done (the rest refines the crossings).
# Sampling comes first: a well-sampled curve keeps the later stages cheap
_SAMPLE_SHARE = 0.25
_PAIR_SHARE = 0.8
_WALK_SHARE = 0.95
# Found points are filed in grid cells this many typical segment lengths wide (to merge one
# found twice)
_GRID_SEGMENTS = 4.0


class Curve(Protocol):
    """What the search needs of a parametric curve (``ParametricShape`` provides it)."""

    t_min: float
    t_max: float

    def point(self, t: float) -> Point2: ...

    def derivative(self, t: float) -> Point2: ...

    def samples(self, span: float) -> int: ...


class CurveHit:
    """A meeting point: parameters s (first curve) and t (second), location, tangency."""

    def __init__(self, s: float, t: float, x: float, y: float, tangent: bool) -> None:
        self.s = s
        self.t = t
        self.x = x
        self.y = y
        self.tangent = tangent


class CurvePairResult:
    """Meeting points, shared stretches (as parameter intervals of either curve) and sampling.

    ``first_pieces`` are (s_first, s_last) intervals of the first curve lying on the second;
    ``second_pieces`` are (t_first, t_last) intervals of the second curve on the first that
    the first walk did not see. ``spacing`` is each curve's sample step in its parameter;
    ``complete`` is False when the search stopped at its work budget.
    """

    def __init__(self) -> None:
        self.hits: List[CurveHit] = []
        self.first_pieces: List[Tuple[float, float]] = []
        self.second_pieces: List[Tuple[float, float]] = []
        self.spacing: Tuple[float, float] = (0.0, 0.0)
        self.complete = True
        # Period (in its parameter) of a curve searched over its first period only
        self.periods: Tuple[Optional[float], Optional[float]] = (None, None)


def _repeat_period(line: _Polyline, extent: float) -> Optional[float]:
    """The smallest period 2*pi*p/q (p, q up to 12) after which the curve retraces itself all
    along its range, when the range covers at least two of them; None otherwise.

    A period that holds at 12 points spread over the range is then checked at every sample the
    search took of the whole range (refined by its turning): each must match its counterpart
    shifted back into the first period, within 1e-9 of the curve's ``extent`` (more at large
    parameters). A change anywhere the samples see (a radius that grows late in the range, a
    bump) rules it out, and so does sampling cut short (the search is then incomplete anyway).
    """
    curve, low, high = line.curve, line.low, line.high
    span = high - low
    if line.cut_short or not span > 0.0 or not extent > 0.0:
        return None
    candidates = sorted({2.0 * math.pi * p / q for q in range(1, _PERIOD_DENOMINATOR + 1) for p in range(1, 13)})
    full_checks = 0
    for period in candidates:
        if 2.0 * period > span:
            break
        spot_checks = (low + (span - period) * (k + 0.5) / _PERIOD_CHECKS for k in range(_PERIOD_CHECKS))
        if not all(_repeats(curve, t, period, extent) for t in spot_checks):
            continue
        if _repeats_at_every_sample(line, period, extent):
            return period
        full_checks += 1
        if full_checks >= _PERIOD_FULL_CHECKS:
            break
    return None


def _repeats(curve: Curve, t: float, period: float, extent: float) -> bool:
    here, later = curve.point(t), curve.point(t + period)
    if not (_finite(here) and _finite(later)):
        return False
    return _distance(here, later) <= _PERIOD_TOLERANCE * extent * (1.0 + abs(t) + period)


def _repeats_at_every_sample(line: _Polyline, period: float, extent: float) -> bool:
    curve, low = line.curve, line.low
    for t, point in zip(line.ts, line.points):
        turns = math.floor((t - low) / period)
        if turns <= 0:
            continue
        base = curve.point(t - turns * period)
        if _finite(point) != _finite(base):
            return False
        if _finite(point) and _distance(point, base) > _PERIOD_TOLERANCE * extent * (1.0 + abs(t) + period):
            return False
    return True


def _first_period(line: _Polyline, period: float, work: List[int]) -> _Polyline:
    """The polyline cut down to the curve's first period (its range narrowed to match)."""
    curve = line.curve
    end = line.low + period
    keep = 0
    while keep < len(line.ts) and line.ts[keep] < end:
        keep += 1
    ts, points, measured = line.ts[:keep], line.points[:keep], line.measured[: max(keep - 1, 0)]
    if keep:
        ts.append(end)
        points.append(curve.point(end))
        measured.append(-1.0)
    curve.t_max = end
    return _Polyline(curve, work, (ts, points, measured, False))


def intersect_curves(
    first: Curve, second: Curve, size: float, extents: Tuple[float, float] = (0.0, 0.0)
) -> CurvePairResult:
    """Find where two parametric curves meet; ``size`` (the smaller curve's extent) scales
    tolerances. With the curves' ``extents``, a curve that retraces itself is searched over
    one period (``periods`` in the result)."""
    return _PairSearch(first, second, size, extents).run()


# ---------------------------------------------------------------------------
# Polylines and their bounding-box trees
# ---------------------------------------------------------------------------


def _finite(point: Point2) -> bool:
    return math.isfinite(point[0]) and math.isfinite(point[1])


class _Polyline:
    """Samples of a curve, the segments between consecutive finite samples, and a box tree."""

    def __init__(
        self,
        curve: Curve,
        work: List[int],
        samples: Optional[Tuple[List[float], List[Point2], List[float], bool]] = None,
    ) -> None:
        self.curve = curve
        self.low, self.high = float(curve.t_min), float(curve.t_max)
        if samples is None:
            # A first grid of the curve's own sample count (at most _PROBE_SAMPLES), then refined
            # by its turning: the samples follow the curve's complexity, not its length on screen
            count = max(2, min(int(curve.samples(self.high - self.low)), _PROBE_SAMPLES))
            samples = _adaptive_samples(curve, self.low, self.high, count, work)
        self.ts, self.points, measured, self.cut_short = samples
        self.measured = measured
        # The mean sample step
        self.spacing = (self.high - self.low) / (len(self.ts) - 1)
        # Segment k joins sample k and sample k + 1
        self.segments = [k for k in range(len(self.ts) - 1) if _finite(self.points[k]) and _finite(self.points[k + 1])]
        self.lengths: Dict[int, float] = {}
        # Each segment's bounding box (low x, low y, high x, high y)
        self.boxes: Dict[int, Tuple[float, float, float, float]] = {}
        # How far each segment's curve strays from it (its midpoint's distance; the segment's
        # length where that was not measured)
        self.devs: Dict[int, float] = {}
        for k in self.segments:
            (x0, y0), (x1, y1) = self.points[k], self.points[k + 1]
            self.lengths[k] = math.hypot(x1 - x0, y1 - y0)
            self.devs[k] = measured[k] if measured[k] >= 0.0 else self.lengths[k]
            self.boxes[k] = (x0 if x0 < x1 else x1, y0 if y0 < y1 else y1, x1 if x0 < x1 else x0, y1 if y0 < y1 else y0)
        self.tree = _BoxTree(self)

    def speed(self) -> float:
        """Typical speed: the polyline's length per unit of parameter (1 when undefined)."""
        length = sum(self.lengths.values())
        span = self.high - self.low
        return length / span if span > 0.0 and length > 0.0 else 1.0

    def magnitude(self) -> float:
        values = [abs(c) for p in self.points if _finite(p) for c in p]
        return max(values) if values else 1.0


def _adaptive_samples(
    curve: Curve, low: float, high: float, count: int, work: List[int], limit: float = _SAMPLE_SHARE * _WORK_BUDGET
) -> Tuple[List[float], List[Point2], List[float], bool]:
    """``count`` uniform steps, each halved (depth first, so the samples stay in order) while its
    midpoint leaves the chord, up to _REFINE_PASSES times, _MAX_SAMPLES samples in all and the
    ``limit`` of the work (the sampling share). Also returns each interval's measured deviation of its midpoint
    from the chord (-1 where it was not measured) and whether refinement ran into those limits
    of work."""
    ts = [low]
    points = [curve.point(low)]
    devs: List[float] = []
    extra_left = _MAX_SAMPLES - count
    cut_short = False
    for i in range(1, count + 1):
        right_t = high if i == count else low + (high - low) * i / count
        # Pending right ends (parameter, point, depth of the interval ending there)
        stack = [(right_t, curve.point(right_t), 0)]
        left_t, left_p = ts[-1], points[-1]
        while stack:
            end_t, end_p, depth = stack[-1]
            middle = 0.5 * (left_t + end_t)
            dev = -1.0
            # (the checks are inlined: this loop runs once per sample)
            isfinite = math.isfinite
            if (
                depth < _REFINE_PASSES
                and middle != left_t
                and middle != end_t
                and isfinite(left_p[0])
                and isfinite(left_p[1])
                and isfinite(end_p[0])
                and isfinite(end_p[1])
            ):
                if extra_left <= 0 or work[0] > limit:
                    cut_short = True
                else:
                    mid_p = curve.point(middle)
                    off_x = mid_p[0] - 0.5 * (left_p[0] + end_p[0])
                    off_y = mid_p[1] - 0.5 * (left_p[1] + end_p[1])
                    chord_x, chord_y = end_p[0] - left_p[0], end_p[1] - left_p[1]
                    rounding = _ROUNDING_UNITS * _EPS * (abs(mid_p[0]) + abs(mid_p[1]) + 1.0)
                    allowed = _DEVIATION_RATIO * math.sqrt(chord_x * chord_x + chord_y * chord_y) + rounding
                    # (NaN compares false: a midpoint off the curve's domain is not a split)
                    if off_x * off_x + off_y * off_y > allowed * allowed:
                        extra_left -= 1
                        stack[-1] = (end_t, end_p, depth + 1)
                        stack.append((middle, mid_p, depth + 1))
                        continue
                    if off_x * off_x + off_y * off_y <= allowed * allowed:
                        dev = math.sqrt(off_x * off_x + off_y * off_y)
            stack.pop()
            ts.append(end_t)
            points.append(end_p)
            devs.append(dev)
            left_t, left_p = end_t, end_p
    return ts, points, devs, cut_short


class _BoxTree:
    """A binary tree of bounding boxes over runs of a polyline's segments (built bottom up:
    leaves of _LEAF_SIZE consecutive segments, then neighbouring nodes paired, so a node's
    children are in the polyline's order)."""

    def __init__(self, line: _Polyline) -> None:
        self.boxes: List[Tuple[float, float, float, float]] = []
        self.reach: List[float] = []
        self.children: List[Optional[Tuple[int, int]]] = []
        self.spans: List[Tuple[int, int]] = []
        segments, points, lengths = line.segments, line.points, line.lengths
        level: List[int] = []
        for lo in range(0, len(segments), _LEAF_SIZE):
            hi = min(lo + _LEAF_SIZE, len(segments))
            first = points[segments[lo]]
            low_x = high_x = first[0]
            low_y = high_y = first[1]
            longest = 0.0
            for k in segments[lo:hi]:
                for x, y in (points[k], points[k + 1]):
                    if x < low_x:
                        low_x = x
                    elif x > high_x:
                        high_x = x
                    if y < low_y:
                        low_y = y
                    elif y > high_y:
                        high_y = y
                if lengths[k] > longest:
                    longest = lengths[k]
            level.append(self._node((low_x, low_y, high_x, high_y), _NEAR_FACTOR * longest, None, (lo, hi)))
        while len(level) > 1:
            paired: List[int] = []
            for index in range(0, len(level) - 1, 2):
                left, right = level[index], level[index + 1]
                a, b = self.boxes[left], self.boxes[right]
                box = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                reach = max(self.reach[left], self.reach[right])
                paired.append(self._node(box, reach, (left, right), (self.spans[left][0], self.spans[right][1])))
            if len(level) % 2:
                paired.append(level[-1])
            level = paired
        self.root = level[0] if level else -1

    def _node(
        self,
        box: Tuple[float, float, float, float],
        reach: float,
        children: Optional[Tuple[int, int]],
        span: Tuple[int, int],
    ) -> int:
        self.boxes.append(box)
        self.reach.append(reach)
        self.children.append(children)
        self.spans.append(span)
        return len(self.boxes) - 1


class _Pair:
    """Two polyline segments that cross (gap 0) or come near; u, v locate their closest points."""

    def __init__(self, i: int, j: int, gap: float, u: float, v: float, crossing: bool) -> None:
        self.i = i
        self.j = j
        self.gap = gap
        self.u = u
        self.v = v
        self.crossing = crossing


def _segment_pair(p0: Point2, p1: Point2, q0: Point2, q1: Point2) -> Tuple[float, float, float, bool]:
    """(gap, u, v, crossing) of segments p0-p1 and q0-q1, u and v their closest points' parameters."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ex, ey = q1[0] - q0[0], q1[1] - q0[1]
    wx, wy = q0[0] - p0[0], q0[1] - p0[1]
    cross = dx * ey - dy * ex
    if abs(cross) > 1e-12 * math.hypot(dx, dy) * math.hypot(ex, ey):
        u = (wx * ey - wy * ex) / cross
        v = (wx * dy - wy * dx) / cross
        if -_SEGMENT_SLACK <= u <= 1.0 + _SEGMENT_SLACK and -_SEGMENT_SLACK <= v <= 1.0 + _SEGMENT_SLACK:
            return (0.0, min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0), True)
    best = (math.inf, 0.0, 0.0)
    for u_end, point in ((0.0, p0), (1.0, p1)):
        v_at, gap = _foot(point, q0, ex, ey)
        if gap < best[0]:
            best = (gap, u_end, v_at)
    for v_end, point in ((0.0, q0), (1.0, q1)):
        u_at, gap = _foot(point, p0, dx, dy)
        if gap < best[0]:
            best = (gap, u_at, v_end)
    return (best[0], best[1], best[2], False)


def _foot(point: Point2, start: Point2, dx: float, dy: float) -> Tuple[float, float]:
    """Parameter (clamped to [0, 1]) of the closest point of a segment to ``point``, and the distance."""
    length2 = dx * dx + dy * dy
    u = 0.0
    if length2 > 0.0:
        u = min(max(((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length2, 0.0), 1.0)
    return u, math.hypot(start[0] + u * dx - point[0], start[1] + u * dy - point[1])


def _close_pairs(a: _Polyline, b: _Polyline, work: List[int]) -> Tuple[List[_Pair], bool]:
    """Segment pairs that cross or come within reach, from both box trees walked together
    (the first curve's segments in order), and whether the walk finished within its budget."""
    pairs: List[_Pair] = []
    ta, tb = a.tree, b.tree
    if ta.root < 0 or tb.root < 0:
        return pairs, True
    # The hottest loop of the search: plain comparisons, no tuples or builtins per visit
    limit = _PAIR_SHARE * _WORK_BUDGET
    boxes_a, boxes_b, reach_a, reach_b = ta.boxes, tb.boxes, ta.reach, tb.reach
    children_a, children_b, spans_a, spans_b = ta.children, tb.children, ta.spans, tb.spans
    stack_a, stack_b = [ta.root], [tb.root]
    while stack_a:
        if work[0] > limit:
            return pairs, False
        na, nb = stack_a.pop(), stack_b.pop()
        work[0] += 1
        box_a, box_b = boxes_a[na], boxes_b[nb]
        reach = reach_a[na]
        if reach_b[nb] > reach:
            reach = reach_b[nb]
        gap_x = box_a[0] - box_b[2]
        if box_b[0] - box_a[2] > gap_x:
            gap_x = box_b[0] - box_a[2]
        gap_y = box_a[1] - box_b[3]
        if box_b[1] - box_a[3] > gap_y:
            gap_y = box_b[1] - box_a[3]
        if gap_x > reach or gap_y > reach:
            continue
        if gap_x > 0.0 and gap_y > 0.0 and gap_x * gap_x + gap_y * gap_y > reach * reach:
            continue
        ca, cb = children_a[na], children_b[nb]
        if ca is None and cb is None:
            _leaf_pairs(a, b, spans_a[na], spans_b[nb], pairs, work)
        elif cb is None or (ca is not None and spans_a[na][1] - spans_a[na][0] >= spans_b[nb][1] - spans_b[nb][0]):
            assert ca is not None
            stack_a.extend((ca[1], ca[0]))
            stack_b.extend((nb, nb))
        else:
            stack_a.extend((na, na))
            stack_b.extend((cb[1], cb[0]))
    return pairs, True


def _leaf_pairs(
    a: _Polyline,
    b: _Polyline,
    span_a: Tuple[int, int],
    span_b: Tuple[int, int],
    pairs: List[_Pair],
    work: List[int],
) -> None:
    work[0] += (span_a[1] - span_a[0]) * (span_b[1] - span_b[0])
    boxes_a, boxes_b, lengths_a, lengths_b = a.boxes, b.boxes, a.lengths, b.lengths
    segments_b = b.segments[span_b[0] : span_b[1]]
    for i in a.segments[span_a[0] : span_a[1]]:
        box_a, length_a = boxes_a[i], lengths_a[i]
        for j in segments_b:
            box_b, length_b = boxes_b[j], lengths_b[j]
            reach = _NEAR_FACTOR * (length_a if length_a > length_b else length_b)
            if (
                box_b[0] - box_a[2] > reach
                or box_a[0] - box_b[2] > reach
                or box_b[1] - box_a[3] > reach
                or box_a[1] - box_b[3] > reach
            ):
                continue
            gap, u, v, crossing = _segment_pair(a.points[i], a.points[i + 1], b.points[j], b.points[j + 1])
            if crossing or gap <= reach:
                pairs.append(_Pair(i, j, gap, u, v, crossing))
                work[0] += _CROSSING_PAIR_WORK if crossing else _NEAR_PAIR_WORK


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


class _Counted:
    """A curve whose evaluations are counted against the search's work budget."""

    def __init__(self, curve: Curve, counter: List[int]) -> None:
        self.curve = curve
        self.counter = counter
        self.t_min = curve.t_min
        self.t_max = curve.t_max

    def point(self, t: float) -> Point2:
        self.counter[0] += 1
        return self.curve.point(t)

    def derivative(self, t: float) -> Point2:
        self.counter[0] += 2
        return self.curve.derivative(t)

    def samples(self, span: float) -> int:
        return self.curve.samples(span)


class _PairSearch:
    def __init__(self, first: Curve, second: Curve, size: float, extents: Tuple[float, float] = (0.0, 0.0)) -> None:
        self.extents = extents
        self.evaluations = [0]
        self.a = _Counted(first, self.evaluations)
        self.b = _Counted(second, self.evaluations)
        self.size = size
        self.result = CurvePairResult()
        # Foot (on the other curve) of the first sample of each shared stretch
        self.piece_feet: Dict[Tuple[float, float], float] = {}
        # Stretches of the first curve reported as one contact point (searched no further)
        self.contact_zones: List[Tuple[float, float]] = []

    # ------------------- Tolerances and evaluation -------------------

    def gap(self, s: float, t: float) -> Tuple[float, float]:
        (ax, ay), (bx, by) = self.a.point(s), self.b.point(t)
        return (ax - bx, ay - by)

    def tolerance(self, s: float, t: float, near: Optional[Point2] = None) -> float:
        """How far apart the curves may be at (s, t) and still meet: size plus rounding of the
        coordinates (``near``, a point there, when the caller has one) and of the parameters
        (at the curves' typical speeds)."""
        if near is None:
            near = self.a.point(s)
        coordinates = max(abs(near[0]), abs(near[1]))
        parameters = abs(s) * self.speed[0] + abs(t) * self.speed[1]
        if not math.isfinite(coordinates + parameters):
            return _ON_OBJECT_TOLERANCE * self.size
        return _ON_OBJECT_TOLERANCE * self.size + _ROUNDING_UNITS * _EPS * (coordinates + parameters)

    def clamp_s(self, s: float) -> float:
        return min(max(s, self.a.t_min), self.a.t_max)

    def clamp_t(self, t: float) -> float:
        return min(max(t, self.b.t_min), self.b.t_max)

    # ------------------- Driver -------------------

    def run(self) -> CurvePairResult:
        line_a, line_b = _Polyline(self.a, self.evaluations), _Polyline(self.b, self.evaluations)
        line_a, line_b = self._one_period(line_a, 0), self._one_period(line_b, 1)
        self.lines = (line_a, line_b)
        self.result.spacing = (line_a.spacing, line_b.spacing)
        self.max_step = (_STEP_SPACINGS * line_a.spacing, _STEP_SPACINGS * line_b.spacing)
        # Found points are filed in a grid of cells a few typical segments wide
        # (the larger curve's median segment: a finely refined curve must not shrink the cells
        # until a lookup spans too many of them)
        typical = max(_median(list(line_a.lengths.values())), _median(list(line_b.lengths.values())))
        self.cell = max(_GRID_SEGMENTS * typical, _EPS * max(line_a.magnitude(), line_b.magnitude(), 1.0))
        self.grid: Dict[Tuple[int, int], List[CurveHit]] = {}
        self.speed = (line_a.speed(), line_b.speed())
        self.closed = (self._closed(self.a), self._closed(self.b))
        pairs, self.result.complete = _close_pairs(line_a, line_b, self.evaluations)
        if line_a.cut_short or line_b.cut_short:
            # Parts of a curve stayed coarser than its turning asks for
            self.result.complete = False
        if not pairs:
            return self.result
        self._find_overlaps(line_a, line_b, pairs)
        self._short_pieces_to_points(line_a)
        self._find_crossings(line_a, line_b, [p for p in pairs if p.crossing])
        self._find_touches(line_a, line_b, [p for p in pairs if not p.crossing], [p for p in pairs if p.crossing])
        self.result.hits = self._one_per_contact(self.result.hits, line_a.spacing)
        return self.result

    def _one_period(self, line: _Polyline, which: int) -> _Polyline:
        """The curve over its first period when its range retraces it (a circle over t in
        [0, 1000]): the same points, found once."""
        extent = self.extents[which]
        period = _repeat_period(line, extent)
        if period is None:
            return line
        periods = list(self.result.periods)
        periods[which] = period
        self.result.periods = (periods[0], periods[1])
        return _first_period(line, period, self.evaluations)

    def _out_of_budget(self, share: float = 1.0) -> bool:
        """Whether the work since sampling has used up ``share`` of the budget (then the result is
        marked incomplete)."""
        if self.evaluations[0] <= share * _WORK_BUDGET:
            return False
        self.result.complete = False
        return True

    def _one_per_contact(self, hits: List[CurveHit], spacing: float) -> List[CurveHit]:
        """Hits close along the first curve with the curves within tolerance all the way between
        them are one contact (a flat touch such as a cubic's inflection on a line): the tangent
        one is kept."""
        kept: List[CurveHit] = []
        for hit in sorted(hits, key=lambda h: h.s):
            last = kept[-1] if kept else None
            if last is not None and abs(hit.s - last.s) <= 2.0 * spacing and self._joined(last, hit):
                if hit.tangent and not last.tangent:
                    kept[-1] = hit
                continue
            kept.append(hit)
        return kept

    def _joined(self, first: CurveHit, second: CurveHit) -> bool:
        guess = first.t
        for fraction in (0.25, 0.5, 0.75):
            on, guess = self._on_second(first.s + fraction * (second.s - first.s), guess)
            if not on:
                return False
        return True

    # ------------------- Overlaps -------------------

    def _find_overlaps(self, line_a: _Polyline, line_b: _Polyline, pairs: List[_Pair]) -> None:
        guesses_a: Dict[int, List[Tuple[float, float]]] = {}
        guesses_b: Dict[int, List[Tuple[float, float]]] = {}
        for pair in pairs:
            # Only samples lying near the other polyline's segment can lie on the other curve
            _add_guesses(guesses_a, line_a, pair.i, line_b, pair.j)
            _add_guesses(guesses_b, line_b, pair.j, line_a, pair.i)
        self.result.first_pieces = self._walk(line_a, guesses_a, line_b.spacing, self._on_second)
        if self._walk_needed(guesses_b):
            self.result.second_pieces = self._walk(line_b, guesses_b, line_a.spacing, self._on_first_outside_pieces)

    def _closed(self, curve: Curve) -> bool:
        start, end = curve.point(curve.t_min), curve.point(curve.t_max)
        if not (_finite(start) and _finite(end)):
            return False
        limit = _MERGE_TOLERANCE * self.size + _SAME_POINT_ULPS * _EPS * max(abs(c) for c in start + end)
        return _distance(start, end) <= limit

    def _short_pieces_to_points(self, line_a: _Polyline) -> None:
        """A stretch of a few sample steps where the curves stay within tolerance but do not
        coincide to rounding is a high-order contact (y = x^4 on y = 0, a parabola and its
        osculating circle, a cusp on a line), not a shared stretch: one point at its closest
        place, a tangency when the curves stay on one side of each other just past its ends."""
        kept: List[Tuple[float, float]] = []
        contacts: List[Tuple[float, float, Optional[bool]]] = []
        for low, high in self.result.first_pieces:
            contact = self._contact(low, high, line_a.spacing)
            if contact is None:
                kept.append((low, high))
            else:
                contacts.append(contact)
                self.contact_zones.append((low, high))
        self.result.first_pieces = kept
        for s, t, touching in contacts:
            if touching:
                # The closest place is flat to high order: locate it where the curves run parallel
                polished = self._polished_tangency(s, t, line_a.spacing)
                if polished is not None:
                    s, t = polished
            self._add_hit(s, t, line_a, touching)

    def _contact(self, low: float, high: float, spacing: float) -> Optional[Tuple[float, float, Optional[bool]]]:
        """(s, t, touching) of a short stretch that is a contact, None for a shared stretch."""
        if high - low > _SHORT_PIECE_SPACINGS * spacing:
            return None
        guess = self.piece_feet.get((low, high), self.b.t_min)
        apart = 0.0
        for fraction in (0.25, 0.5, 0.75):
            s = low + fraction * (high - low)
            point = self.a.point(s)
            guess, distance = self._project(self.b, point, guess, self.clamp_t, self.max_step[1])
            apart = max(apart, distance / (_NOISE_UNITS * _EPS * (abs(point[0]) + abs(point[1]) + 1.0)))
        if apart <= 1.0:
            return None
        s = self._closest_along(low, high, guess)
        t, _ = self._project(self.b, self.a.point(s), guess, self.clamp_t, self.max_step[1])
        middle = 0.5 * (high - low) / spacing
        return (s, t, self._touching(s, t, True, middle + 0.5))

    def _closest_along(self, low: float, high: float, guess: float) -> float:
        """Golden-section search for the first curve's parameter closest to the second curve."""
        ratio = 0.5 * (math.sqrt(5.0) - 1.0)

        def apart(s: float) -> float:
            return self._project(self.b, self.a.point(s), guess, self.clamp_t, self.max_step[1])[1]

        a, b = low, high
        c, d = b - ratio * (b - a), a + ratio * (b - a)
        fc, fd = apart(c), apart(d)
        for _ in range(_GOLDEN_STEPS):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b - ratio * (b - a)
                fc = apart(c)
            else:
                a, c, fc = c, d, fd
                d = a + ratio * (b - a)
                fd = apart(d)
        return c if fc <= fd else d

    def _walk_needed(self, guesses_b: Dict[int, List[Tuple[float, float]]]) -> bool:
        """The second walk only matters where the first curve's stretches do not already reach."""
        return any(not self._in_first_pieces(guess) for options in guesses_b.values() for _, guess in options)

    def _walk(
        self,
        line: _Polyline,
        guesses: Dict[int, List[Tuple[float, float]]],
        other_spacing: float,
        on_other: Callable[[float, float], Tuple[bool, float]],
    ) -> List[Tuple[float, float]]:
        """Parameter intervals of ``line``'s curve lying on the other curve (runs of on-curve samples).

        Each sample tries the feet of its closest segment pairs (a few distinct ones: on a
        closed curve the foot may lie on either side of its seam).
        """
        flags: Dict[int, Tuple[bool, float]] = {}
        # A stretch needs two neighbouring samples on the other curve: a lone candidate is a crossing
        candidates = [k for k in sorted(guesses) if k - 1 in guesses or k + 1 in guesses]
        # More than one foot is worth trying only across the seam of a closed curve
        other_closed = self.closed[1] if line is self.lines[0] else self.closed[0]
        tries = _WALK_GUESSES if other_closed else 1
        for k in candidates:
            if self._out_of_budget(_WALK_SHARE):
                break
            tried: List[float] = []
            for _, guess in sorted(guesses[k]):
                if len(tried) >= tries or any(abs(guess - g) <= 2.0 * other_spacing for g in tried):
                    continue
                tried.append(guess)
                flag = on_other(line.ts[k], guess)
                if k not in flags or flag[0]:
                    flags[k] = flag
                if flag[0]:
                    break
        valid = set(line.segments)
        pieces: List[Tuple[float, float]] = []
        indices = sorted(k for k, (on, _) in flags.items() if on)
        n = 0
        while n < len(indices):
            m = n
            while m + 1 < len(indices) and indices[m + 1] == indices[m] + 1 and indices[m] in valid:
                m += 1
            first, last = indices[n], indices[m]
            n = m + 1
            if last == first:
                continue
            if last == first + 1:
                middle = 0.5 * (line.ts[first] + line.ts[last])
                if not on_other(middle, flags[first][1])[0]:
                    continue
            start = self._run_end(line, first, first - 1, flags, on_other, valid)
            end = self._run_end(line, last, last + 1, flags, on_other, valid)
            pieces.append((start, end))
            self.piece_feet[(start, end)] = flags[first][1]
        return self._joined_pieces(pieces, line.spacing, on_other)

    def _joined_pieces(
        self,
        pieces: List[Tuple[float, float]],
        spacing: float,
        on_other: Callable[[float, float], Tuple[bool, float]],
    ) -> List[Tuple[float, float]]:
        """Pieces that touch, overlap, or are a short gap apart with the gap's middle on the
        other curve too are one maximal stretch (samples that miss each other split runs)."""
        joined: List[Tuple[float, float]] = []
        for low, high in sorted(pieces):
            if joined:
                last_low, last_high = joined[-1]
                gap = low - last_high
                if gap <= 0.0 or (
                    gap <= _JOIN_SPACINGS * spacing
                    and on_other(0.5 * (low + last_high), self.piece_feet.get((low, high), 0.0))[0]
                ):
                    merged = (last_low, max(last_high, high))
                    self.piece_feet[merged] = self.piece_feet.get((last_low, last_high), 0.0)
                    joined[-1] = merged
                    continue
            joined.append((low, high))
        return joined

    def _run_end(
        self,
        line: _Polyline,
        inside: int,
        outside: int,
        flags: Dict[int, Tuple[bool, float]],
        on_other: Callable[[float, float], Tuple[bool, float]],
        valid: set,
    ) -> float:
        """Bisect between the run's last sample and the next one to where the curves part."""
        if outside < 0 or outside >= len(line.ts) or min(inside, outside) not in valid:
            return line.ts[inside]
        guess = flags[inside][1]
        s_in, s_out = line.ts[inside], line.ts[outside]
        for _ in range(_BISECTION_STEPS):
            middle = 0.5 * (s_in + s_out)
            if middle in (s_in, s_out):
                break
            on, foot = on_other(middle, guess)
            if on:
                s_in, guess = middle, foot
            else:
                s_out = middle
        return s_in

    def _on_second(self, s: float, t_guess: float) -> Tuple[bool, float]:
        """Whether the first curve's point at s lies on the second curve, and the foot's t."""
        point = self.a.point(s)
        if not _finite(point):
            return (False, t_guess)
        limit = self.tolerance(s, t_guess, point)
        t, distance = self._project(self.b, point, t_guess, self.clamp_t, self.max_step[1], limit)
        return (distance <= self.tolerance(s, t, point), t)

    def _on_first_outside_pieces(self, t: float, s_guess: float) -> Tuple[bool, float]:
        point = self.b.point(t)
        if not _finite(point):
            return (False, s_guess)
        limit = self.tolerance(s_guess, t, point)
        s, distance = self._project(self.a, point, s_guess, self.clamp_s, self.max_step[0], limit)
        on = distance <= self.tolerance(s, t, point) and not self._in_first_pieces(s)
        return (on, s)

    def _in_first_pieces(self, s: float, slack: float = 0.0) -> bool:
        return any(low - slack <= s <= high + slack for low, high in self.result.first_pieces)

    def _in_second_pieces(self, t: float) -> bool:
        return any(low <= t <= high for low, high in self.result.second_pieces)

    def _project(
        self,
        curve: Curve,
        point: Point2,
        guess: float,
        clamp: Callable[[float], float],
        max_step: float,
        give_up: float = 0.0,
    ) -> Tuple[float, float]:
        """Gauss-Newton on the distance from ``point`` to the curve near ``guess``: (parameter, distance).

        With ``give_up``, stops once the distance is far above it and has stalled (a minimum
        off the curve: a point on it converges quadratically), for the on-curve tests.
        """
        t = clamp(guess)
        here = curve.point(t)
        distance = _distance(here, point)
        if not math.isfinite(distance):
            return (t, math.inf)
        for _ in range(_PROJECT_STEPS):
            dx, dy = curve.derivative(t)
            speed2 = dx * dx + dy * dy
            if not math.isfinite(speed2) or speed2 == 0.0:
                break
            step = -((here[0] - point[0]) * dx + (here[1] - point[1]) * dy) / speed2
            # A step cut to the limit is still travelling towards the foot
            travelling = abs(step) > max_step
            step = max(-max_step, min(max_step, step))
            if abs(step) <= 4.0 * _EPS * max(1.0, abs(t)):
                # Already at the foot to rounding: no backtracking needed
                break
            improved = False
            for _ in range(_BACKTRACK_STEPS):
                candidate = clamp(t + step)
                there = curve.point(candidate)
                candidate_distance = _distance(there, point)
                if candidate_distance < distance:
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break
            moved = abs(candidate - t)
            previous = distance
            t, here, distance = candidate, there, candidate_distance
            if distance == 0.0 or moved <= 4.0 * _EPS * max(1.0, abs(t)):
                break
            stalled = not travelling and distance > _GIVE_UP_RATIO * previous
            if give_up > 0.0 and distance > _GIVE_UP_FACTOR * give_up and stalled:
                break
        return (t, distance)

    # ------------------- Crossings -------------------

    def _find_crossings(self, line_a: _Polyline, line_b: _Polyline, pairs: List[_Pair]) -> None:
        for pair in sorted(pairs, key=lambda p: (p.i, p.j)):
            if self._out_of_budget():
                return
            self.evaluations[0] += _CHECK_WORK
            s0, s1 = line_a.ts[pair.i], line_a.ts[pair.i + 1]
            t0, t1 = line_b.ts[pair.j], line_b.ts[pair.j + 1]
            if self._covered(line_a, line_b, pair):
                continue
            s, t = s0 + pair.u * (s1 - s0), t0 + pair.v * (t1 - t0)
            solved = self._newton(s, t)
            if solved is None:
                box = self._shrink((s0, s1, t0, t1))
                if box is not None:
                    solved = self._newton(*box) or self._minimise(*box)
            if solved is not None:
                self._add_hit(*solved, line_a)
            else:
                # Last resort: where the first curve changes sides of the second around the pair
                first, last = max(pair.i - 1, 0), min(pair.i + 2, len(line_a.ts) - 1)
                self._scan_stretch(first, last, t, line_a)

    def _covered(self, line_a: _Polyline, line_b: _Polyline, pair: _Pair) -> bool:
        """The pair lies in a shared stretch or a contact already reported as one point. (A
        point found from another pass of a curve does not cover it: on a curve that nearly
        retraces itself the next pass crosses at a distinct point close by.)"""
        s0, s1 = line_a.ts[pair.i], line_a.ts[pair.i + 1]
        if self._in_first_pieces(s0) and self._in_first_pieces(s1):
            return True
        t0, t1 = line_b.ts[pair.j], line_b.ts[pair.j + 1]
        if self._in_second_pieces(t0) and self._in_second_pieces(t1):
            return True
        if any(s1 >= low and s0 <= high for low, high in self.contact_zones):
            return True
        return False

    def _grid_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))

    def _newton(self, s: float, t: float) -> Optional[Tuple[float, float]]:
        """2D Newton on A(s) - B(t) = 0 with a step limit and backtracking; None when it fails."""
        s, t = self.clamp_s(s), self.clamp_t(t)
        fx, fy = self.gap(s, t)
        norm = math.hypot(fx, fy)
        if not math.isfinite(norm):
            return None
        # Converged well inside the tolerance (judged once: it barely moves along the iteration)
        settled = 1e-3 * self.tolerance(s, t)
        for _ in range(_NEWTON_STEPS):
            if norm <= settled or norm == 0.0:
                break
            (ax, ay), (bx, by) = self.a.derivative(s), self.b.derivative(t)
            det = bx * ay - ax * by
            if not math.isfinite(det) or abs(det) <= 1e-14 * _norm((ax, ay)) * _norm((bx, by)):
                break
            ds = (fx * by - bx * fy) / det
            dt = (ay * fx - ax * fy) / det
            scale = max(abs(ds) / self.max_step[0], abs(dt) / self.max_step[1], 1.0)
            ds, dt = ds / scale, dt / scale
            moved = False
            for _ in range(_BACKTRACK_STEPS):
                ns, nt = self.clamp_s(s + ds), self.clamp_t(t + dt)
                gx, gy = self.gap(ns, nt)
                new_norm = math.hypot(gx, gy)
                if new_norm < norm:
                    moved = True
                    break
                ds, dt = 0.5 * ds, 0.5 * dt
            if not moved:
                break
            tiny = abs(ns - s) <= 4.0 * _EPS * max(1.0, abs(s)) and abs(nt - t) <= 4.0 * _EPS * max(1.0, abs(t))
            s, t, fx, fy, norm = ns, nt, gx, gy, new_norm
            if tiny:
                break
        return (s, t) if norm <= self.tolerance(s, t) else None

    def _shrink(self, box: Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
        """Halve both parameter intervals, keeping the halves whose chords still meet (or nearly)."""
        cache: Dict[Tuple[int, float], Point2] = {}

        def at(which: int, value: float) -> Point2:
            key = (which, value)
            if key not in cache:
                cache[key] = (self.a if which == 0 else self.b).point(value)
            return cache[key]

        boxes = [box]
        for _ in range(_SHRINK_LEVELS):
            kept: List[Tuple[float, Tuple[float, float, float, float]]] = []
            for s0, s1, t0, t1 in boxes:
                sm, tm = 0.5 * (s0 + s1), 0.5 * (t0 + t1)
                for sa, sb in ((s0, sm), (sm, s1)):
                    for ta, tb in ((t0, tm), (tm, t1)):
                        p0, p1, q0, q1 = at(0, sa), at(0, sb), at(1, ta), at(1, tb)
                        if not all(_finite(p) for p in (p0, p1, q0, q1)):
                            continue
                        gap = _segment_pair(p0, p1, q0, q1)[0]
                        reach = _NEAR_FACTOR * max(_distance(p0, p1), _distance(q0, q1))
                        if gap <= reach:
                            kept.append((gap, (sa, sb, ta, tb)))
            if not kept:
                return None
            kept.sort(key=lambda item: item[0])
            boxes = [item[1] for item in kept[:_SHRINK_BOXES]]
            s0, s1, t0, t1 = boxes[0]
            if 0.5 * (s0 + s1) in (s0, s1) or 0.5 * (t0 + t1) in (t0, t1):
                break
        s0, s1, t0, t1 = boxes[0]
        return (0.5 * (s0 + s1), 0.5 * (t0 + t1))

    def _minimise(self, s: float, t: float) -> Optional[Tuple[float, float]]:
        """Levenberg-Marquardt on |A(s) - B(t)|^2; the minimum when it is within tolerance."""
        s, t = self.clamp_s(s), self.clamp_t(t)
        fx, fy = self.gap(s, t)
        cost = fx * fx + fy * fy
        if not math.isfinite(cost):
            return None
        damping = 1e-3
        for _ in range(_MINIMISE_STEPS):
            if cost == 0.0:
                break
            (ax, ay), (bx, by) = self.a.derivative(s), self.b.derivative(t)
            # J = [A', -B']: normal equations (J^T J + damping * diag) d = -J^T F
            jaa, jbb, jab = ax * ax + ay * ay, bx * bx + by * by, -(ax * bx + ay * by)
            ga, gb = ax * fx + ay * fy, -(bx * fx + by * fy)
            if not all(math.isfinite(v) for v in (jaa, jbb, jab, ga, gb)) or jaa == 0.0 or jbb == 0.0:
                break
            accepted = False
            for _ in range(_BACKTRACK_STEPS):
                m11, m22 = jaa * (1.0 + damping), jbb * (1.0 + damping)
                det = m11 * m22 - jab * jab
                if det <= 0.0:
                    damping *= 4.0
                    continue
                ds = (-ga * m22 + gb * jab) / det
                dt = (-gb * m11 + ga * jab) / det
                scale = max(abs(ds) / self.max_step[0], abs(dt) / self.max_step[1], 1.0)
                ns, nt = self.clamp_s(s + ds / scale), self.clamp_t(t + dt / scale)
                gx, gy = self.gap(ns, nt)
                new_cost = gx * gx + gy * gy
                if new_cost < cost:
                    accepted = True
                    damping = max(damping / 3.0, 1e-12)
                    break
                damping *= 4.0
            if not accepted:
                break
            tiny = abs(ns - s) <= 4.0 * _EPS * max(1.0, abs(s)) and abs(nt - t) <= 4.0 * _EPS * max(1.0, abs(t))
            s, t, fx, fy, cost = ns, nt, gx, gy, new_cost
            if tiny:
                break
        return (s, t) if math.sqrt(cost) <= self.tolerance(s, t) else None

    # ------------------- Near touches -------------------

    def _find_touches(
        self, line_a: _Polyline, line_b: _Polyline, pairs: List[_Pair], crossing_pairs: List[_Pair]
    ) -> None:
        """Where the polylines come close without crossing (local minima of the gap along the
        first curve) but the curves may still meet (the gap is within twice their strays from
        the segments): minimise the distance (a touch), and scan the side the first curve lies
        on of the second for crossings the polylines hide (two close crossings between samples)."""
        best: Dict[int, _Pair] = {}
        for pair in pairs:
            if pair.i not in best or pair.gap < best[pair.i].gap:
                best[pair.i] = pair
        for pair in crossing_pairs:
            # A crossing is the closest the polylines get: its neighbours are not minima
            best[pair.i] = pair
        starts: List[_Pair] = []
        for i, pair in best.items():
            before, after = best.get(i - 1), best.get(i + 1)
            if pair.crossing:
                continue
            if (before is None or before.gap >= pair.gap) and (after is None or after.gap >= pair.gap):
                stray = line_a.devs[pair.i] + line_b.devs[pair.j]
                if pair.gap <= _MEET_FACTOR * stray + _ON_OBJECT_TOLERANCE * self.size:
                    starts.append(pair)
        # The stretches of the first curve around the starts (a segment either side), merged
        # (first sample, last sample, the closest pair in it)
        stretches: List[Tuple[int, int, _Pair]] = []
        for pair in sorted(starts, key=lambda p: p.i):
            if self._covered(line_a, line_b, pair):
                continue
            first, last = max(pair.i - 1, 0), min(pair.i + 2, len(line_a.ts) - 1)
            if stretches and first <= stretches[-1][1]:
                head, tail, closest = stretches[-1]
                stretches[-1] = (head, max(tail, last), pair if pair.gap < closest.gap else closest)
            else:
                stretches.append((first, last, pair))
        for first, last, closest in stretches:
            if self._out_of_budget():
                return
            s0, s1 = line_a.ts[closest.i], line_a.ts[closest.i + 1]
            t0, t1 = line_b.ts[closest.j], line_b.ts[closest.j + 1]
            guess = t0 + closest.v * (t1 - t0)
            # The closest place may be a touch the side cannot show (at a cusp the distance is
            # not a parabola)
            solved = self._minimise(s0 + closest.u * (s1 - s0), guess)
            if solved is not None:
                self._add_hit(*solved, line_a)
            self._scan_stretch(first, last, guess, line_a)

    def _scan_stretch(self, first: int, last: int, guess: float, line_a: _Polyline) -> None:
        """The side the first curve lies on of the second, at the samples first..last and the
        segments' midpoints: each sign change is bisected to a crossing, and where the side
        dips towards zero between samples without changing sign (a parabola through three
        values reaching the tolerance, or well below them) the distance is minimised (a touch)."""
        run: List[Tuple[float, float, float]] = []
        for k in range(first, last + 1):
            positions = [line_a.ts[k]]
            if k < last:
                positions.append(0.5 * (line_a.ts[k] + line_a.ts[k + 1]))
            for s in positions:
                if self._out_of_budget():
                    return
                side, guess = self._side(s, guess)
                if side is None:
                    self._check_run(run, line_a)
                    run = []
                    continue
                run.append((s, side, guess))
        self._check_run(run, line_a)

    def _check_run(self, run: List[Tuple[float, float, float]], line_a: _Polyline) -> None:
        for index, (s, side, foot) in enumerate(run):
            if side == 0.0:
                solved = self._newton(s, foot) or self._minimise(s, foot)
                if solved is not None:
                    self._add_hit(*solved, line_a)
                continue
            if index == 0:
                continue
            previous = run[index - 1]
            if previous[1] != 0.0 and (side > 0.0) != (previous[1] > 0.0):
                root = self._bisect_side(previous[0], s, previous[1], previous[2])
                if root is not None:
                    solved = self._newton(*root) or (root if self._meets(*root) else None)
                    if solved is not None:
                        self._add_hit(*solved, line_a)
            if index + 1 < len(run):
                following = run[index + 1]
                dip = _dip(previous, (s, side, foot), following)
                if dip is not None:
                    solved = self._minimise(dip, foot)
                    if solved is not None:
                        self._add_hit(*solved, line_a)

    def _meets(self, s: float, t: float) -> bool:
        return _distance(self.a.point(s), self.b.point(t)) <= self.tolerance(s, t)

    def _side(self, s: float, guess: float) -> Tuple[Optional[float], float]:
        """Signed distance of the first curve's point at s from the second curve (at the foot
        of the perpendicular, along the second curve's normal; 0 within rounding), and the
        foot; None where it has no meaning (past the second curve's ends, undefined)."""
        point = self.a.point(s)
        if not _finite(point):
            return (None, guess)
        t, distance = self._project(self.b, point, guess, self.clamp_t, self.max_step[1])
        base, (dx, dy) = self.b.point(t), self.b.derivative(t)
        speed = _norm((dx, dy))
        if not (_finite(base) and math.isfinite(speed)) or speed == 0.0:
            return (None, t)
        side = (dx * (point[1] - base[1]) - dy * (point[0] - base[0])) / speed
        along = abs(dx * (point[0] - base[0]) + dy * (point[1] - base[1])) / speed
        noise = _NOISE_UNITS * _EPS * (abs(point[0]) + abs(point[1]) + 1.0)
        if along > abs(side) + noise:
            # The foot is at an end of the second curve: the point lies past it
            return (None, t)
        return (0.0 if abs(side) <= noise else side, t)

    def _bisect_side(self, low: float, high: float, side_low: float, guess: float) -> Optional[Tuple[float, float]]:
        """Bisect a sign change of the side to where the first curve crosses the second."""
        for _ in range(_BISECTION_STEPS):
            middle = 0.5 * (low + high)
            if middle in (low, high):
                break
            side, guess = self._side(middle, guess)
            if side is None:
                return None
            if side == 0.0:
                low = high = middle
                break
            if (side > 0.0) == (side_low > 0.0):
                low = middle
            else:
                high = middle
        s = 0.5 * (low + high)
        t, _ = self._project(self.b, self.a.point(s), guess, self.clamp_t, self.max_step[1])
        return (s, t)

    # ------------------- Points -------------------

    def _add_hit(self, s: float, t: float, line_a: _Polyline, touching: Optional[bool] = None) -> None:
        """Record a meeting point, polished and flagged when the curves run parallel there.

        Where either curve stops (a cusp: its derivative all but vanishes) the directions say
        nothing, so the tangency is whether the curves stay on one side of each other there.
        ``touching`` gives that answer when the caller already has it.
        """
        if self._in_first_pieces(s) or self._in_second_pieces(t):
            return
        if touching is None and any(low <= s <= high for low, high in self.contact_zones):
            return
        da, db = self.a.derivative(s), self.b.derivative(t)
        if touching is None and (_norm(da) <= _CUSP_RATIO * self.speed[0] or _norm(db) <= _CUSP_RATIO * self.speed[1]):
            probe_first = _norm(da) / self.speed[0] <= _norm(db) / self.speed[1]
            touching = self._touching(s, t, probe_first, 0.5)
        if touching is not None:
            tangent = touching
        else:
            tangent = False
            sine = _sine(da, db)
            if sine <= _POLISH_ANGLE:
                polished = self._polished_tangency(s, t, line_a.spacing)
                if polished is not None:
                    s, t, tangent = polished[0], polished[1], True
                else:
                    tangent = sine <= _TANGENT_ANGLE
        (ax, ay), (bx, by) = self.a.point(s), self.b.point(t)
        hit = CurveHit(s, t, 0.5 * (ax + bx), 0.5 * (ay + by), tangent)
        same = self._same_hit(hit)
        if same is not None:
            same.tangent = same.tangent or hit.tangent
            return
        self.result.hits.append(hit)
        key = self._grid_cell(hit.x, hit.y)
        if key in self.grid:
            self.grid[key].append(hit)
        else:
            self.grid[key] = [hit]

    def _same_hit(self, hit: CurveHit) -> Optional[CurveHit]:
        """A hit already found at this place (within the report's merge distance), if any."""
        limit = _MERGE_TOLERANCE * self.size + _SAME_POINT_ULPS * _EPS * max(abs(hit.x), abs(hit.y))
        gx, gy = self._grid_cell(hit.x, hit.y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in self.grid.get((gx + dx, gy + dy), []):
                    if math.hypot(other.x - hit.x, other.y - hit.y) <= limit:
                        return other
        return None

    def _touching(self, s: float, t: float, probe_first: bool, steps: float) -> Optional[bool]:
        """Whether points of one curve ``steps`` sample steps either side of the meeting point
        lie on the same side of the other curve (None when that cannot be told)."""
        curve, other = (self.a, self.b) if probe_first else (self.b, self.a)
        here, foot = (s, t) if probe_first else (t, s)
        spacing = self.result.spacing[0 if probe_first else 1]
        clamp_other = self.clamp_t if probe_first else self.clamp_s
        step_other = self.max_step[1 if probe_first else 0]
        sides: List[bool] = []
        for direction in (-1.0, 1.0):
            value = self._probe_parameter(curve, here + direction * steps * spacing, probe_first)
            if value is None:
                return None
            point = curve.point(value)
            if not _finite(point):
                return None
            at, _ = self._project(other, point, foot, clamp_other, step_other)
            base, (dx, dy) = other.point(at), other.derivative(at)
            side = dx * (point[1] - base[1]) - dy * (point[0] - base[0])
            noise = _NOISE_UNITS * _EPS * (abs(point[0]) + abs(point[1]) + 1.0) * _norm((dx, dy))
            if not math.isfinite(side) or abs(side) <= noise:
                return None
            sides.append(side > 0.0)
        return sides[0] == sides[1]

    def _probe_parameter(self, curve: Curve, value: float, first: bool) -> Optional[float]:
        """``value`` inside the curve's range, wrapped across the seam of a closed curve."""
        low, high = curve.t_min, curve.t_max
        if low <= value <= high:
            return value
        if not self.closed[0 if first else 1]:
            return None
        span = high - low
        return low + (value - low) % span

    def _polished_tangency(self, s: float, t: float, spacing: float) -> Optional[Tuple[float, float]]:
        """Where the directions are exactly parallel near s (the second curve's point the foot of
        the perpendicular), if the curves still meet there; None when they do not turn parallel."""
        feet: Dict[float, float] = {}

        def cross(value: float) -> float:
            point = self.a.point(value)
            if not _finite(point):
                return math.nan
            foot, _ = self._project(self.b, point, feet.get(value, t), self.clamp_t, self.max_step[1])
            feet[value] = foot
            (ax, ay), (bx, by) = self.a.derivative(value), self.b.derivative(foot)
            return ax * by - ay * bx

        reach = 2.0 * spacing
        low, high = self.clamp_s(s - reach), self.clamp_s(s + reach)
        f_low, f_high = cross(low), cross(high)
        if not (math.isfinite(f_low) and math.isfinite(f_high)):
            return None
        if f_low == 0.0 or f_high == 0.0:
            low = high = low if f_low == 0.0 else high
        elif (f_low < 0.0) == (f_high < 0.0):
            # No turn to parallel inside: a tangency at the end of the curve's range, if any
            clipped = [end for end in (low, high) if end in (self.a.t_min, self.a.t_max)]
            here = abs(cross(s))
            ends = [end for end in clipped if abs(cross(end)) <= here]
            if not ends:
                return None
            low = high = ends[0]
        for _ in range(_BISECTION_STEPS):
            middle = 0.5 * (low + high)
            if middle in (low, high):
                break
            value = cross(middle)
            if not math.isfinite(value):
                return None
            if value == 0.0:
                low = high = middle
                break
            if (value < 0.0) == (f_low < 0.0):
                low, f_low = middle, value
            else:
                high = middle
        polished = 0.5 * (low + high)
        foot, distance = self._project(self.b, self.a.point(polished), t, self.clamp_t, self.max_step[1])
        allowed = _distance(self.a.point(s), self.b.point(t)) + self.tolerance(s, t)
        return (polished, foot) if distance <= allowed else None


def _add_guesses(
    table: Dict[int, List[Tuple[float, float]]], line: _Polyline, segment: int, other: _Polyline, other_segment: int
) -> None:
    """When both ends of ``segment`` lie along ``other_segment`` (within the polylines' own error
    of it, as a shared stretch does), each end's (distance, foot parameter)."""
    start, end = other.points[other_segment], other.points[other_segment + 1]
    p0, p1 = line.points[segment], line.points[segment + 1]
    if _sine((p1[0] - p0[0], p1[1] - p0[1]), (end[0] - start[0], end[1] - start[1])) > _ALONG_SINE:
        return
    length = other.lengths[other_segment]
    if length == 0.0:
        return
    dx, dy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
    feet: List[Tuple[float, float]] = []
    # The two curves' samples need not line up: the segment may be several of the other's long
    own = line.lengths[segment]
    beyond = max(1.0, own / length)
    for k in (segment, segment + 1):
        # Off the chord's line (not its nearest end: the other curve's samples fall anywhere
        # along a shared stretch), with the foot at most a segment beyond either end
        rx, ry = line.points[k][0] - start[0], line.points[k][1] - start[1]
        fraction = (rx * dx + ry * dy) / length
        distance = abs(rx * dy - ry * dx)
        if distance > _ALONG_FACTOR * length or not -beyond <= fraction <= 1.0 + beyond:
            return
        feet.append((min(max(fraction, 0.0), 1.0), distance))
    for k, (fraction, distance) in zip((segment, segment + 1), feet):
        item = (distance, _along(other, other_segment, fraction))
        # Not setdefault: Brython's dict.setdefault turns int keys into strings
        if k in table:
            table[k].append(item)
        else:
            table[k] = [item]


def _along(line: _Polyline, k: int, fraction: float) -> float:
    return line.ts[k] + fraction * (line.ts[k + 1] - line.ts[k])


def _dip(
    before: Tuple[float, float, float], here: Tuple[float, float, float], after: Tuple[float, float, float]
) -> Optional[float]:
    """Where |side| dips between samples without changing sign: the parameter of the vertex of
    the parabola through three (s, |side|) values when ``here`` is the smallest and the vertex
    reaches a quarter of it (a touch, or a near miss to check), else None."""
    if not ((before[1] > 0.0) == (here[1] > 0.0) == (after[1] > 0.0)) or 0.0 in (before[1], after[1]):
        return None
    (x0, y0), (x1, y1), (x2, y2) = (before[0], abs(before[1])), (here[0], abs(here[1])), (after[0], abs(after[1]))
    if not (y1 <= y0 and y1 <= y2):
        return None
    d01, d12 = (y1 - y0) / (x1 - x0), (y2 - y1) / (x2 - x1)
    curvature = (d12 - d01) / (x2 - x0)
    if curvature <= 0.0:
        return None
    vertex = 0.5 * (x0 + x1) - d01 / (2.0 * curvature)
    vertex = min(max(vertex, x0), x2)
    lowest = y1 + d01 * (vertex - x1) + curvature * (vertex - x0) * (vertex - x1)
    return vertex if lowest <= 0.25 * y1 else None


def _median(values: List[float]) -> float:
    return sorted(values)[len(values) // 2] if values else 0.0


def _norm(v: Sequence[float]) -> float:
    return math.hypot(v[0], v[1])


def _distance(p: Point2, q: Point2) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _sine(u: Point2, v: Point2) -> float:
    """|sin| of the angle between two directions (1 when either is undefined)."""
    sizes = _norm(u) * _norm(v)
    if not math.isfinite(sizes) or sizes == 0.0:
        return 1.0
    return abs(u[0] * v[1] - u[1] * v[0]) / sizes
