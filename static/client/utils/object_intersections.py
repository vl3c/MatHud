"""Intersection points of two canvas objects: lines, circles, arcs, ellipses and curves.

This module has no browser/Brython dependencies so it can be validated by the
server-side pytest suites as well as the Brython test runner. (The intersection code in
``utils.geometry_utils`` serves region and area work and imports ``MathUtils``, which needs
the browser; it also reports neither tangency, overlaps nor the objects' parameters.)

Shapes:
    LineShape        a segment or vector, or its whole line (``bounded=False``)
    CircleShape      a circle, or a circle arc (start angle and counter-clockwise sweep)
    EllipseShape     an ellipse, possibly rotated
    FunctionShape    y = f(x) on [left, right] (plain or piecewise functions)
    ParametricShape  (x(t), y(t)) on [t_min, t_max]
    PointShape       a zero-length segment or zero-radius circle

Method:
    - Closed forms for line/line, line/circle, circle/circle and line/ellipse (the line is
      mapped into the frame where the ellipse is the unit circle). A double root (a gap
      within the rounding error of its computation) is one point with ``tangent``.
    - Circle/ellipse and ellipse/ellipse: the roots of the other conic's implicit equation
      along the ellipse's parametrisation, over one full turn.
    - Functions and parametric curves: the roots of the other object's implicit equation
      along the curve (over the other object's x extent for a function).
    The numeric roots come from ``function_features.scan_roots`` (sampling, Brent's method,
    touching roots as tangencies). Values within rounding noise of zero count as zero, so
    curves that coincide over a stretch give an overlap rather than many false roots.
    A numeric crossing where the two curves run parallel (a line through an inflection
    point of a cubic) is also a tangency.

Coinciding objects (collinear overlapping segments, identical circles, arcs of one circle,
a function running along a segment) are reported as overlaps, not as points. Whether two
objects coincide is judged against their size (radius, segment length), never their
distance from the origin. Points are rounded to 10 significant digits, sorted by (x, y)
and capped at ``max_results``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

from utils.function_features import MAX_SAMPLES, scan_roots, sample_count

DEFAULT_MAX_RESULTS = 50

# Rounding units (of every term of a computation) within which a gap is zero: a line or
# circle that close to touching is tangent, and a coordinate that close is the same
_ROUNDING_UNITS = 16.0
# Sine of the angle between two lines below which they are parallel
_PARALLEL_TOLERANCE = 1e-12
# Distance (relative to the objects' size, plus coordinate rounding) within which parallel
# lines are one line, circles are one circle, and a degenerate point lies on an object
_ON_OBJECT_TOLERANCE = 1e-10
# Slack of a segment parameter beyond [0, 1] and of an angle beyond an arc's ends
_T_SLACK = 1e-10
_ANGLE_SLACK = 1e-9
# Sine of the angle between two curves at a numeric crossing below which they are tangent
_TANGENT_ANGLE = 1e-7
# Points closer than this (relative to their size) are one point
_MERGE_TOLERANCE = 1e-9
# A point this close (relative to its size) to an overlap's end is part of the overlap
_ABSORB_TOLERANCE = 1e-7
# Rounding units of the coordinates within which two computed points are one point
_SAME_POINT_ULPS = 4.0
# An implicit value within this many rounding units of its terms counts as zero
_NOISE_FACTOR = 64.0
_EPS = 2.220446049250313e-16
# Samples of one turn of an ellipse, and the minimum / estimate count for parametric curves
_CONIC_SAMPLES = 1024
_MIN_CURVE_SAMPLES = 1000
_CURVE_PROBE_SAMPLES = 256
_SAMPLES_PER_PIXEL = 2
# Padding of a function's search range beyond the other object's x extent
_EXTENT_PADDING = 1e-3
_REPORT_DIGITS = 10
# Most significant digits a coordinate can need (a small object far from the origin)
_MAX_DIGITS = 17
_ZERO_FLOOR = 1e-12
_BISECTION_STEPS = 60
_TWO_PI = 2.0 * math.pi

Point2 = Tuple[float, float]
Params = Dict[str, float]


class _IntersectionPointBase(TypedDict):
    x: float
    y: float


class IntersectionPoint(_IntersectionPointBase, total=False):
    """One intersection point.

    ``params`` maps an object's name to where the point lies on it: ``t`` on a segment or
    vector (0 at its first point, 1 at its second) or a parametric curve, ``angle`` in
    radians in [0, 2*pi) on a circle or arc (polar angle about the centre) or an ellipse
    (parametric angle in the ellipse's own frame), the parameter draw_tangent_line takes.
    ``tangent`` marks a point where the objects touch.
    ``point_name`` is added by callers that place a point there.
    """

    params: Dict[str, Params]
    tangent: bool
    point_name: str


class Overlap(TypedDict, total=False):
    """A stretch shared by both objects: ``kind`` is 'line', 'segment', 'circle', 'arc',
    'ellipse' or 'curve'; ``start`` and ``end`` are its ends when it has any."""

    kind: str
    start: List[float]
    end: List[float]


class IntersectionReport(TypedDict):
    points: List[IntersectionPoint]
    overlaps: List[Overlap]
    total_found: int
    truncated: bool
    notes: List[str]


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


class PointShape:
    """A single point: what a zero-length segment or a zero-radius circle is."""

    rank = 0
    note: Optional[str] = None

    def __init__(self, name: str, x: float, y: float) -> None:
        self.name = name
        self.x = float(x)
        self.y = float(y)

    def implicit(self, x: float, y: float) -> float:
        return math.hypot(x - self.x, y - self.y)

    def noise(self, x: float, y: float) -> float:
        return _coincidence_tolerance(1.0, x, y, self.x, self.y)

    def accepts(self, x: float, y: float) -> Optional[Params]:
        return {}

    def tangent_at(self, x: float, y: float) -> Optional[Point2]:
        return None

    def x_extent(self) -> Optional[Tuple[float, float]]:
        return (self.x, self.x)


class LineShape:
    """A segment or vector from (x1, y1) to (x2, y2); unbounded, the whole line through them."""

    rank = 1
    note: Optional[str] = None

    def __init__(self, name: str, start: Point2, end: Point2, bounded: bool = True) -> None:
        self.name = name
        self.x1, self.y1 = float(start[0]), float(start[1])
        self.x2, self.y2 = float(end[0]), float(end[1])
        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1
        self.bounded = bool(bounded)

    @property
    def length(self) -> float:
        return math.hypot(self.dx, self.dy)

    def scale(self) -> float:
        return max(1.0, abs(self.x1), abs(self.y1), abs(self.x2), abs(self.y2))

    def point_at(self, t: float) -> Point2:
        if t == 1.0:
            return (self.x2, self.y2)
        return (self.x1 + t * self.dx, self.y1 + t * self.dy)

    def parameter_of(self, x: float, y: float) -> float:
        return ((x - self.x1) * self.dx + (y - self.y1) * self.dy) / (self.dx * self.dx + self.dy * self.dy)

    def accepted_t(self, t: float) -> Optional[float]:
        """t itself on a line; on a segment, t clamped to [0, 1] when within slack of it, else None."""
        if not self.bounded:
            return t
        if t < -_T_SLACK or t > 1.0 + _T_SLACK:
            return None
        return min(max(t, 0.0), 1.0)

    def implicit(self, x: float, y: float) -> float:
        """Signed distance from the line."""
        return ((x - self.x1) * self.dy - (y - self.y1) * self.dx) / self.length

    def noise(self, x: float, y: float) -> float:
        return _NOISE_FACTOR * _EPS * (abs(x) + abs(y) + self.scale())

    def params_for_t(self, t: float) -> Params:
        return {"t": t}

    def accepts(self, x: float, y: float) -> Optional[Params]:
        t = self.accepted_t(self.parameter_of(x, y))
        return None if t is None else self.params_for_t(t)

    def coordinates(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def tangent_at(self, x: float, y: float) -> Optional[Point2]:
        return (self.dx, self.dy)

    def x_extent(self) -> Optional[Tuple[float, float]]:
        if self.bounded:
            return (min(self.x1, self.x2), max(self.x1, self.x2))
        if self.dx == 0.0:
            return (self.x1, self.x1)
        return None


class CircleShape:
    """A circle, or the arc of it from ``arc_start`` sweeping ``arc_sweep`` radians counter-clockwise."""

    rank = 2
    note: Optional[str] = None

    def __init__(
        self,
        name: str,
        center: Point2,
        radius: float,
        arc_start: Optional[float] = None,
        arc_sweep: Optional[float] = None,
    ) -> None:
        self.name = name
        self.cx, self.cy = float(center[0]), float(center[1])
        self.r = float(radius)
        is_arc = arc_start is not None and arc_sweep is not None and float(arc_sweep) < _TWO_PI
        self.arc_start = float(arc_start) % _TWO_PI if is_arc and arc_start is not None else None
        self.arc_sweep = float(arc_sweep) if is_arc and arc_sweep is not None else None

    @property
    def is_arc(self) -> bool:
        return self.arc_start is not None

    def scale(self) -> float:
        return max(1.0, abs(self.cx), abs(self.cy), self.r)

    def angle_of(self, x: float, y: float) -> float:
        return math.atan2(y - self.cy, x - self.cx) % _TWO_PI

    def contains_angle(self, angle: float) -> bool:
        if self.arc_start is None or self.arc_sweep is None:
            return True
        offset = (angle - self.arc_start) % _TWO_PI
        return offset <= self.arc_sweep + _ANGLE_SLACK or offset >= _TWO_PI - _ANGLE_SLACK

    def point_at_angle(self, angle: float) -> Point2:
        return (self.cx + self.r * math.cos(angle), self.cy + self.r * math.sin(angle))

    def implicit(self, x: float, y: float) -> float:
        return math.hypot(x - self.cx, y - self.cy) - self.r

    def noise(self, x: float, y: float) -> float:
        return _NOISE_FACTOR * _EPS * (abs(x) + abs(y) + self.scale())

    def accepts(self, x: float, y: float) -> Optional[Params]:
        angle = self.angle_of(x, y)
        return {"angle": angle} if self.contains_angle(angle) else None

    def tangent_at(self, x: float, y: float) -> Optional[Point2]:
        return (-(y - self.cy), x - self.cx)

    def x_extent(self) -> Optional[Tuple[float, float]]:
        return (self.cx - self.r, self.cx + self.r)


class EllipseShape:
    """An ellipse with radii (rx, ry) along its own axes, rotated counter-clockwise by ``rotation`` radians."""

    rank = 3
    note: Optional[str] = None

    def __init__(self, name: str, center: Point2, radius_x: float, radius_y: float, rotation: float = 0.0) -> None:
        self.name = name
        self.cx, self.cy = float(center[0]), float(center[1])
        self.rx = float(radius_x)
        self.ry = float(radius_y)
        self.rotation = float(rotation)
        self.cos = math.cos(self.rotation)
        self.sin = math.sin(self.rotation)

    def scale(self) -> float:
        return max(1.0, abs(self.cx), abs(self.cy), self.rx, self.ry)

    def to_unit(self, x: float, y: float) -> Point2:
        """Coordinates in the frame where the ellipse is the unit circle."""
        dx, dy = x - self.cx, y - self.cy
        return ((self.cos * dx + self.sin * dy) / self.rx, (-self.sin * dx + self.cos * dy) / self.ry)

    def point_at_angle(self, angle: float) -> Point2:
        u, v = self.rx * math.cos(angle), self.ry * math.sin(angle)
        return (self.cx + self.cos * u - self.sin * v, self.cy + self.sin * u + self.cos * v)

    def derivative_at_angle(self, angle: float) -> Point2:
        u, v = -self.rx * math.sin(angle), self.ry * math.cos(angle)
        return (self.cos * u - self.sin * v, self.sin * u + self.cos * v)

    def implicit(self, x: float, y: float) -> float:
        u, v = self.to_unit(x, y)
        return math.hypot(u, v) - 1.0

    def noise(self, x: float, y: float) -> float:
        return _NOISE_FACTOR * _EPS * (1.0 + (abs(x) + abs(y) + self.scale()) / min(self.rx, self.ry))

    def accepts(self, x: float, y: float) -> Optional[Params]:
        u, v = self.to_unit(x, y)
        return {"angle": math.atan2(v, u) % _TWO_PI}

    def tangent_at(self, x: float, y: float) -> Optional[Point2]:
        u, v = self.to_unit(x, y)
        gu, gv = u / self.rx, v / self.ry
        gx, gy = self.cos * gu - self.sin * gv, self.sin * gu + self.cos * gv
        return (-gy, gx)

    def x_extent(self) -> Optional[Tuple[float, float]]:
        half = math.hypot(self.rx * self.cos, self.ry * self.sin)
        return (self.cx - half, self.cx + half)

    def quadratic_form(self) -> Tuple[float, float, float]:
        """(a, b, c) of a*dx^2 + 2*b*dx*dy + c*dy^2 = 1 about the centre."""
        p, q = 1.0 / (self.rx * self.rx), 1.0 / (self.ry * self.ry)
        return (
            p * self.cos * self.cos + q * self.sin * self.sin,
            (p - q) * self.cos * self.sin,
            p * self.sin * self.sin + q * self.cos * self.cos,
        )


class FunctionShape:
    """The graph of y = f(x) for x in [left, right], split at ``breakpoints`` (asymptotes, jumps)."""

    rank = 4
    note: Optional[str] = None

    def __init__(
        self,
        name: str,
        f: Callable[[float], float],
        left: float,
        right: float,
        breakpoints: Sequence[float] = (),
        pixels_per_unit: Optional[float] = None,
        period: Optional[float] = None,
    ) -> None:
        self.name = name
        self.f = f
        self.left = float(left)
        self.right = float(right)
        self.breakpoints = [float(b) for b in breakpoints]
        self.pixels_per_unit = pixels_per_unit
        self.period = period

    def value(self, x: float) -> float:
        if not self.left <= x <= self.right:
            return math.nan
        return _safe_float(self.f, x)

    def point(self, x: float) -> Point2:
        return (x, self.value(x))

    def derivative(self, x: float) -> Point2:
        """(1, f'(x)) by a central difference, one-sided at the ends of the range."""
        step = 1e-5 * max(1.0, abs(x))
        before, here, after = self.value(x - step), self.value(x), self.value(x + step)
        slope = (after - before) / (2.0 * step)
        if not math.isfinite(slope):
            slope = (after - here) / step if math.isfinite(after) else (here - before) / step
        return (1.0, slope) if math.isfinite(slope) else (0.0, 1.0)

    def params(self, x: float) -> Params:
        return {}

    def samples(self, span: float) -> int:
        pixel_span = span * self.pixels_per_unit if self.pixels_per_unit else None
        return int(sample_count(span, pixel_span=pixel_span, period=self.period))

    def implicit(self, x: float, y: float) -> float:
        return y - self.value(x)

    def noise(self, x: float, y: float) -> float:
        value = self.value(x)
        return _NOISE_FACTOR * _EPS * (abs(x) + abs(y) + (abs(value) if math.isfinite(value) else 0.0))

    def accepts(self, x: float, y: float) -> Optional[Params]:
        return {}

    def tangent_at(self, x: float, y: float) -> Optional[Point2]:
        return self.derivative(x)

    def x_extent(self) -> Optional[Tuple[float, float]]:
        return (self.left, self.right)


class ParametricShape:
    """The curve (x(t), y(t)) for t in [t_min, t_max]."""

    rank = 5
    note: Optional[str] = None

    def __init__(
        self,
        name: str,
        x_of_t: Callable[[float], float],
        y_of_t: Callable[[float], float],
        t_min: float,
        t_max: float,
        pixels_per_unit: Optional[float] = None,
    ) -> None:
        self.name = name
        self.x_of_t = x_of_t
        self.y_of_t = y_of_t
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.pixels_per_unit = pixels_per_unit
        self.breakpoints: List[float] = []

    def point(self, t: float) -> Point2:
        return (_safe_float(self.x_of_t, t), _safe_float(self.y_of_t, t))

    def derivative(self, t: float) -> Point2:
        step = 1e-5 * max(1.0, abs(t))
        (x0, y0), (x1, y1) = self.point(t - step), self.point(t + step)
        return ((x1 - x0) / (2.0 * step), (y1 - y0) / (2.0 * step))

    def params(self, t: float) -> Params:
        return {"t": t}

    def extent(self) -> float:
        """Diagonal of the box around a coarse sampling of the curve (its size for tolerances)."""
        points = [self.point(self.t_min + (self.t_max - self.t_min) * i / 64) for i in range(65)]
        xs = [x for x, y in points if math.isfinite(x) and math.isfinite(y)]
        ys = [y for x, y in points if math.isfinite(x) and math.isfinite(y)]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) if xs else 0.0

    def samples(self, span: float) -> int:
        """Two samples per screen pixel of the curve's length (at least _MIN_CURVE_SAMPLES)."""
        if not self.pixels_per_unit:
            return _MIN_CURVE_SAMPLES
        length = 0.0
        previous = self.point(self.t_min)
        for i in range(1, _CURVE_PROBE_SAMPLES + 1):
            current = self.point(self.t_min + span * i / _CURVE_PROBE_SAMPLES)
            step = math.hypot(current[0] - previous[0], current[1] - previous[1])
            if math.isfinite(step):
                length += step
            previous = current
        wanted = math.ceil(length * self.pixels_per_unit * _SAMPLES_PER_PIXEL)
        return int(min(max(_MIN_CURVE_SAMPLES, wanted), MAX_SAMPLES))


class _EllipseAxis(LineShape):
    """An ellipse with one zero radius: the segment along its other axis, still reporting the angle.

    The segment runs from the parametric angle pi to 0 (or 3*pi/2 to pi/2 for an upright
    one), so its t maps to the angle in [0, pi] (or [-pi/2, pi/2]) at the same point.
    """

    def __init__(self, name: str, start: Point2, end: Point2, upright: bool) -> None:
        super().__init__(name, start, end)
        self.upright = upright

    def params_for_t(self, t: float) -> Params:
        along = min(max(2.0 * t - 1.0, -1.0), 1.0)
        return {"angle": math.asin(along) if self.upright else math.acos(along)}


Shape = Union[PointShape, LineShape, CircleShape, EllipseShape, FunctionShape, ParametricShape]
_ImplicitShape = Union[PointShape, LineShape, CircleShape, EllipseShape, FunctionShape]


class _EllipseCurve:
    """An ellipse walked by its parametric angle, as the moving side of a numeric search."""

    def __init__(self, ellipse: EllipseShape) -> None:
        self.ellipse = ellipse
        self.name = ellipse.name
        self.breakpoints: List[float] = []

    def point(self, angle: float) -> Point2:
        return self.ellipse.point_at_angle(angle)

    def derivative(self, angle: float) -> Point2:
        return self.ellipse.derivative_at_angle(angle)

    def params(self, angle: float) -> Params:
        return {"angle": angle % _TWO_PI}

    def samples(self, span: float) -> int:
        return _CONIC_SAMPLES


_Curve = Union[FunctionShape, ParametricShape, _EllipseCurve]


class _Hit:
    """An unrounded intersection point with the objects' parameters there."""

    def __init__(self, x: float, y: float, params: Dict[str, Params], tangent: bool = False) -> None:
        self.x = x
        self.y = y
        self.params = params
        self.tangent = tangent


class _Found:
    """Hits, overlaps and notes collected for one pair of objects."""

    def __init__(self) -> None:
        self.hits: List[_Hit] = []
        self.overlaps: List[Overlap] = []
        self.notes: List[str] = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_object_intersections(
    first: Shape, second: Shape, max_results: int = DEFAULT_MAX_RESULTS
) -> IntersectionReport:
    """Find where two shapes meet: points (with parameters and tangency), overlaps and notes.

    Raises:
        ValueError: For two parametric curves (not supported) or shapes with one name.
    """
    if first.name == second.name:
        raise ValueError(f"Give two different objects; got '{first.name}' twice.")
    low, high = (first, second) if first.rank <= second.rank else (second, first)
    found = _dispatch(low, high)
    found.notes[:0] = [shape.note for shape in (first, second) if shape.note]
    return _report(found, max_results, _pair_size(first, second))


def line_shape(name: str, start: Point2, end: Point2, bounded: bool = True) -> Shape:
    """A segment or vector (its whole line unless ``bounded``); a point when it has zero length."""
    line = LineShape(name, start, end, bounded)
    if line.length <= _EPS * line.scale():
        return _degenerate_point(name, "zero length", (line.x1, line.y1))
    return line


def circle_shape(
    name: str, center: Point2, radius: float, arc_start: Optional[float] = None, arc_sweep: Optional[float] = None
) -> Shape:
    """A circle or arc; a point when its radius is zero."""
    circle = CircleShape(name, center, abs(float(radius)), arc_start, arc_sweep)
    if circle.r <= _EPS * circle.scale():
        return _degenerate_point(name, "a zero radius", (circle.cx, circle.cy))
    return circle


def ellipse_shape(name: str, center: Point2, radius_x: float, radius_y: float, rotation: float = 0.0) -> Shape:
    """An ellipse (rotation in radians); a segment along its axis when one radius is zero, a point when both are."""
    ellipse = EllipseShape(name, center, abs(float(radius_x)), abs(float(radius_y)), rotation)
    tiny = _EPS * ellipse.scale()
    if ellipse.rx <= tiny and ellipse.ry <= tiny:
        return _degenerate_point(name, "zero radii", (ellipse.cx, ellipse.cy))
    if ellipse.rx > tiny and ellipse.ry > tiny:
        return ellipse
    upright = ellipse.rx <= tiny
    angle = 0.5 * math.pi if upright else 0.0
    start, end = ellipse.point_at_angle(angle + math.pi), ellipse.point_at_angle(angle)
    axis = _EllipseAxis(name, start, end, upright)
    axis.note = (
        f"{name} has a zero radius; it was treated as the segment from "
        f"({_round(start[0])}, {_round(start[1])}) to ({_round(end[0])}, {_round(end[1])}), "
        f"its angle still the ellipse's parametric angle."
    )
    return axis


def _degenerate_point(name: str, what: str, point: Point2) -> PointShape:
    shape = PointShape(name, point[0], point[1])
    shape.note = f"{name} has {what}; it was treated as the point ({_round(point[0])}, {_round(point[1])})."
    return shape


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _dispatch(low: Shape, high: Shape) -> _Found:
    """Route a pair (ordered by rank) to its closed form or numeric search."""
    if isinstance(low, PointShape):
        return _point_hits(low, high)
    if isinstance(high, ParametricShape):
        if isinstance(low, ParametricShape):
            raise ValueError("Intersections of two parametric curves are not supported.")
        return _curve_hits(high, low, high.t_min, high.t_max)
    if isinstance(high, FunctionShape) and not isinstance(low, ParametricShape):
        return _function_hits(high, low)
    if isinstance(low, LineShape):
        if isinstance(high, LineShape):
            return _line_line(low, high)
        if isinstance(high, CircleShape):
            return _line_circle(low, high)
        if isinstance(high, EllipseShape):
            return _line_ellipse(low, high)
    if isinstance(low, CircleShape) and isinstance(high, CircleShape):
        return _circle_circle(low, high)
    if isinstance(low, (CircleShape, EllipseShape)) and isinstance(high, EllipseShape):
        return _conic_conic(low, high)
    raise ValueError(f"Cannot intersect {low.name} and {high.name}.")


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------


def _line_line(a: LineShape, b: LineShape) -> _Found:
    found = _Found()
    cross = a.dx * b.dy - a.dy * b.dx
    if abs(cross) <= _PARALLEL_TOLERANCE * a.length * b.length:
        return _parallel_lines(a, b)
    wx, wy = b.x1 - a.x1, b.y1 - a.y1
    t = a.accepted_t((wx * b.dy - wy * b.dx) / cross)
    u = b.accepted_t((wx * a.dy - wy * a.dx) / cross)
    if t is not None and u is not None:
        x, y = a.point_at(t)
        found.hits.append(_Hit(x, y, {a.name: a.params_for_t(t), b.name: b.params_for_t(u)}))
    return found


def _parallel_lines(a: LineShape, b: LineShape) -> _Found:
    """Parallel lines: apart, one line, or segments that overlap (or touch end to end)."""
    found = _Found()
    tolerance = _coincidence_tolerance(max(a.length, b.length), *a.coordinates(), *b.coordinates())
    if abs(a.implicit(b.x1, b.y1)) > tolerance:
        found.notes.append(f"{a.name} and {b.name} are parallel and do not meet.")
        return found
    if not (a.bounded or b.bounded):
        found.overlaps.append({"kind": "line"})
        return found
    if not a.bounded:
        a, b = b, a
    t1, t2 = sorted((a.parameter_of(b.x1, b.y1), a.parameter_of(b.x2, b.y2)))
    if not b.bounded:
        t1, t2 = -math.inf, math.inf
    low, high = max(0.0, t1), min(1.0, t2)
    if high - low > _T_SLACK:
        start, end = a.point_at(low), a.point_at(high)
        found.overlaps.append({"kind": "segment", "start": list(start), "end": list(end)})
    elif high - low >= -_T_SLACK:
        t = min(max(low, 0.0), 1.0)
        x, y = a.point_at(t)
        u = min(max(b.parameter_of(x, y), 0.0), 1.0)
        found.hits.append(_Hit(x, y, {a.name: a.params_for_t(t), b.name: b.params_for_t(u)}))
    else:
        found.notes.append(f"{a.name} and {b.name} lie on one line but do not meet.")
    return found


def _line_circle(line: LineShape, circle: CircleShape) -> _Found:
    found = _Found()
    # Relative to the centre, so rounding scales with the circle and line, not their offset
    rx, ry = line.x1 - circle.cx, line.y1 - circle.cy
    length2 = line.dx * line.dx + line.dy * line.dy
    t0 = -(rx * line.dx + ry * line.dy) / length2
    distance = math.hypot(rx + t0 * line.dx, ry + t0 * line.dy)
    # ...plus the rounding already in the inputs (a tangent built from rounded coordinates)
    inputs = _largest(*line.coordinates()) + _largest(circle.cx, circle.cy)
    terms = abs(rx) + abs(ry) + abs(t0) * math.sqrt(length2) + circle.r + inputs
    for t, tangent in _chord_parameters(t0, distance, circle.r, length2, _ROUNDING_UNITS * _EPS * terms):
        accepted = line.accepted_t(t)
        if accepted is None:
            continue
        x, y = line.point_at(accepted)
        params = circle.accepts(x, y)
        if params is not None:
            found.hits.append(_Hit(x, y, {line.name: line.params_for_t(accepted), circle.name: params}, tangent))
    return found


def _line_ellipse(line: LineShape, ellipse: EllipseShape) -> _Found:
    """The line mapped (affinely, so t is unchanged) into the ellipse's unit-circle frame."""
    found = _Found()
    u1, v1 = ellipse.to_unit(line.x1, line.y1)
    u2, v2 = ellipse.to_unit(line.x2, line.y2)
    du, dv = u2 - u1, v2 - v1
    length2 = du * du + dv * dv
    t0 = -(u1 * du + v1 * dv) / length2
    distance = math.hypot(u1 + t0 * du, v1 + t0 * dv)
    # to_unit subtracts the centre first, so rounding scales with the unit-frame coordinates
    # ...plus the rounding already in the inputs, measured in the unit frame
    inputs = (_largest(*line.coordinates()) + _largest(ellipse.cx, ellipse.cy)) / min(ellipse.rx, ellipse.ry)
    terms = 1.0 + abs(u1) + abs(v1) + abs(t0) * math.sqrt(length2) + inputs
    for t, tangent in _chord_parameters(t0, distance, 1.0, length2, _ROUNDING_UNITS * _EPS * terms):
        accepted = line.accepted_t(t)
        if accepted is None:
            continue
        x, y = line.point_at(accepted)
        params = ellipse.accepts(x, y) or {}
        found.hits.append(_Hit(x, y, {line.name: line.params_for_t(accepted), ellipse.name: params}, tangent))
    return found


def _chord_parameters(
    t0: float, distance: float, radius: float, length2: float, error: float
) -> List[Tuple[float, bool]]:
    """Line parameters where a line at ``distance`` from a circle's centre (foot at t0) meets it.

    A gap within ``error`` (the rounding error of the distance) is a tangency: one point.
    """
    gap = radius - distance
    if gap < -error:
        return []
    if gap <= error:
        return [(t0, True)]
    half = math.sqrt((radius - distance) * (radius + distance) / length2)
    return [(t0 - half, False), (t0 + half, False)]


def _circle_circle(a: CircleShape, b: CircleShape) -> _Found:
    """The radical-line construction; tangency when the centre distance is r1 + r2 or |r1 - r2|."""
    dx, dy = b.cx - a.cx, b.cy - a.cy
    d = math.hypot(dx, dy)
    if d <= _coincidence_tolerance(max(a.r, b.r), a.cx, a.cy, b.cx, b.cy):
        return _concentric_circles(a, b)
    found = _Found()
    outer_gap = a.r + b.r - d
    inner_gap = d - abs(a.r - b.r)
    # d comes from the centres' difference, rounded relative to d itself, plus the rounding
    # already in the centres' coordinates
    error = _ROUNDING_UNITS * _EPS * (a.r + b.r + d + _largest(a.cx, a.cy) + _largest(b.cx, b.cy))
    if outer_gap < -error or inner_gap < -error:
        return found
    ux, uy = dx / d, dy / d
    along = (d * d + a.r * a.r - b.r * b.r) / (2.0 * d)
    if abs(outer_gap) <= error or abs(inner_gap) <= error:
        reach = a.r if along >= 0.0 else -a.r
        candidates = [((a.cx + reach * ux, a.cy + reach * uy), True)]
    else:
        half = math.sqrt(max(0.0, (a.r - along) * (a.r + along)))
        base_x, base_y = a.cx + along * ux, a.cy + along * uy
        candidates = [
            ((base_x - half * uy, base_y + half * ux), False),
            ((base_x + half * uy, base_y - half * ux), False),
        ]
    for (x, y), tangent in candidates:
        params_a, params_b = a.accepts(x, y), b.accepts(x, y)
        if params_a is not None and params_b is not None:
            found.hits.append(_Hit(x, y, {a.name: params_a, b.name: params_b}, tangent))
    return found


def _concentric_circles(a: CircleShape, b: CircleShape) -> _Found:
    found = _Found()
    if abs(a.r - b.r) > _coincidence_tolerance(max(a.r, b.r)):
        found.notes.append(f"{a.name} and {b.name} are concentric with different radii and do not meet.")
        return found
    _shared_arcs(a, b, found)
    return found


def _shared_arcs(a: CircleShape, b: CircleShape, found: _Found) -> None:
    """Overlaps (and single shared end points) of two arcs or circles on one circle."""
    if not a.is_arc and not b.is_arc:
        found.overlaps.append({"kind": "circle"})
        return
    if not a.is_arc or not b.is_arc:
        arc = a if a.is_arc else b
        _append_arc_overlap(found, arc, arc.arc_start or 0.0, arc.arc_sweep or 0.0)
        return
    a_start, a_sweep = a.arc_start or 0.0, a.arc_sweep or 0.0
    b_start = a_start + (((b.arc_start or 0.0) - a_start) % _TWO_PI)
    for shifted in (b_start, b_start - _TWO_PI):
        low = max(a_start, shifted)
        high = min(a_start + a_sweep, shifted + (b.arc_sweep or 0.0))
        if high - low > _ANGLE_SLACK:
            _append_arc_overlap(found, a, low, high - low)
        elif high - low >= -_ANGLE_SLACK:
            x, y = a.point_at_angle(low)
            params = {"angle": low % _TWO_PI}
            found.hits.append(_Hit(x, y, {a.name: params, b.name: dict(params)}))


def _append_arc_overlap(found: _Found, circle: CircleShape, start: float, sweep: float) -> None:
    first, last = circle.point_at_angle(start), circle.point_at_angle(start + sweep)
    found.overlaps.append({"kind": "arc", "start": list(first), "end": list(last)})


# ---------------------------------------------------------------------------
# Numeric searches
# ---------------------------------------------------------------------------


def _conic_conic(low: Union[CircleShape, EllipseShape], ellipse: EllipseShape) -> _Found:
    """Walk an ellipse once around and find where the other conic's implicit value is zero."""
    if _same_conic(low, ellipse):
        return _same_conic_overlap(low, ellipse)
    other: Union[CircleShape, EllipseShape]
    if isinstance(low, CircleShape):
        curve, other = _EllipseCurve(ellipse), low
    else:
        curve, other = _EllipseCurve(low), ellipse
    # Run a little past a full turn so a root or tangency at angle 0 is inside the range
    pad = 4.0 * _TWO_PI / _CONIC_SAMPLES
    # Distinct conics share at most four points: a stretch where they agree to rounding
    # (nearly equal ellipses) is a tangency, not an overlap
    return _curve_hits(curve, other, -pad, _TWO_PI + pad, runs_touch=True)


def _same_conic(a: Union[CircleShape, EllipseShape], b: EllipseShape) -> bool:
    """Same centre and same quadratic form (any rotation that maps the ellipse onto itself)."""
    size = max(_conic_size(a), _conic_size(b))
    if math.hypot(a.cx - b.cx, a.cy - b.cy) > _coincidence_tolerance(size, a.cx, a.cy, b.cx, b.cy):
        return False
    form_a = _circle_form(a) if isinstance(a, CircleShape) else a.quadratic_form()
    form_b = b.quadratic_form()
    size = max(abs(value) for value in form_a + form_b)
    return all(abs(p - q) <= _ON_OBJECT_TOLERANCE * size for p, q in zip(form_a, form_b))


def _conic_size(conic: Union[CircleShape, EllipseShape]) -> float:
    return conic.r if isinstance(conic, CircleShape) else max(conic.rx, conic.ry)


def _largest(*values: float) -> float:
    return max(abs(value) for value in values)


def _coincidence_tolerance(size: float, *coordinates: float) -> float:
    """How far apart two things may be and still be one: 1e-10 of the objects' size, plus the
    rounding of their coordinates (never a fraction of the coordinates themselves)."""
    largest = max((abs(c) for c in coordinates), default=0.0)
    return _ON_OBJECT_TOLERANCE * size + _ROUNDING_UNITS * _EPS * largest


def _circle_form(circle: CircleShape) -> Tuple[float, float, float]:
    inverse = 1.0 / (circle.r * circle.r)
    return (inverse, 0.0, inverse)


def _same_conic_overlap(a: Union[CircleShape, EllipseShape], b: EllipseShape) -> _Found:
    found = _Found()
    if isinstance(a, CircleShape) and a.is_arc:
        _append_arc_overlap(found, a, a.arc_start or 0.0, a.arc_sweep or 0.0)
    else:
        found.overlaps.append({"kind": "ellipse"})
    return found


def _function_hits(function: FunctionShape, other: _ImplicitShape) -> _Found:
    """Search along the function, over the other object's x extent (padded) within its range."""
    left, right = function.left, function.right
    extent = other.x_extent()
    if extent is not None:
        pad = _EXTENT_PADDING * (extent[1] - extent[0]) + 1e-9 * max(1.0, abs(extent[0]), abs(extent[1]))
        left, right = max(left, extent[0] - pad), min(right, extent[1] + pad)
    if not left < right:
        found = _Found()
        found.notes.append(f"{function.name} is not plotted where {other.name} is.")
        return found
    return _curve_hits(function, other, left, right)


def _curve_hits(curve: _Curve, other: _ImplicitShape, low: float, high: float, runs_touch: bool = False) -> _Found:
    """Roots of the other object's implicit value along the curve, filtered to both objects.

    A run where the value is zero is an overlap, or with ``runs_touch`` one tangent point.
    """
    found = _Found()

    def gap(s: float) -> float:
        x, y = curve.point(s)
        if not (math.isfinite(x) and math.isfinite(y)):
            return math.nan
        value = other.implicit(x, y)
        return 0.0 if abs(value) <= other.noise(x, y) else value

    samples = curve.samples(high - low)
    spacing = (high - low) / samples
    breakpoints = list(curve.breakpoints)
    if isinstance(other, FunctionShape):
        breakpoints.extend(other.breakpoints)
    pieces: List[Tuple[float, float]] = []
    for root in scan_roots(gap, low, high, breakpoints=breakpoints, samples=samples):
        touching = root.touching
        s = root.x
        if root.end is not None:
            if not runs_touch:
                pieces.extend(_run_pieces(curve, other, gap, (root.x, root.end), (low, high), spacing))
                continue
            s, touching = 0.5 * (root.x + root.end), True
        if touching:
            s = _polished_tangency(curve, other, s, spacing)
        hit = _curve_hit(curve, other, s, touching)
        if hit is not None:
            found.hits.append(hit)
    size = _pair_size(curve, other)
    for first, last, span in _joined_at_seam(curve, pieces, (low, high), size):
        _add_piece(found, curve, other, (first, last), span >= spacing, size)
    return found


def _polished_tangency(curve: _Curve, other: _ImplicitShape, s: float, spacing: float) -> float:
    """Locate a touching root where the curves run parallel, to full precision.

    The touching root comes from minimising the gap, which (flat to second order, and zero
    within rounding noise) pins it only to about sqrt(eps). The cross product of the two
    directions has a simple root there instead. Kept only if the gap is no worse there.
    """

    def cross(t: float) -> float:
        x, y = curve.point(t)
        direction = other.tangent_at(x, y)
        if direction is None or not (math.isfinite(x) and math.isfinite(y)):
            return math.nan
        dx, dy = curve.derivative(t)
        return dx * direction[1] - dy * direction[0]

    reach = 2.0 * spacing
    low, high = s - reach, s + reach
    f_low, f_high = cross(low), cross(high)
    if not (math.isfinite(f_low) and math.isfinite(f_high)) or (f_low < 0.0) == (f_high < 0.0):
        return s
    polished = _bisect_root(cross, low, high, f_low)
    if polished is None:
        return s
    allowed = abs(_raw_gap(curve, other, s)) + other.noise(*curve.point(s))
    return polished if abs(_raw_gap(curve, other, polished)) <= allowed else s


def _raw_gap(curve: _Curve, other: _ImplicitShape, s: float) -> float:
    x, y = curve.point(s)
    value = other.implicit(x, y) if math.isfinite(x) and math.isfinite(y) else math.nan
    return value if math.isfinite(value) else math.inf


def _bisect_root(f: Callable[[float], float], low: float, high: float, f_low: float) -> Optional[float]:
    """Bisect a sign-change bracket down to adjacent floats; None where f is undefined."""
    for _ in range(2 * _BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if middle in (low, high):
            break
        value = f(middle)
        if not math.isfinite(value):
            return None
        if value == 0.0:
            return middle
        if (value < 0.0) == (f_low < 0.0):
            low, f_low = middle, value
        else:
            high = middle
    return 0.5 * (low + high)


def _curve_hit(curve: _Curve, other: _ImplicitShape, s: float, touching: bool) -> Optional[_Hit]:
    x, y = curve.point(s)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    other_params = other.accepts(x, y)
    if other_params is None:
        return None
    params = {curve.name: curve.params(s), other.name: other_params}
    tangent = touching or _runs_parallel(curve.derivative(s), other.tangent_at(x, y))
    return _Hit(x, y, params, tangent)


def _runs_parallel(first: Point2, second: Optional[Point2]) -> bool:
    if second is None:
        return False
    sizes = math.hypot(*first) * math.hypot(*second)
    if not math.isfinite(sizes) or sizes == 0.0:
        return False
    return abs(first[0] * second[1] - first[1] * second[0]) <= _TANGENT_ANGLE * sizes


def _run_pieces(
    curve: _Curve,
    other: _ImplicitShape,
    gap: Callable[[float], float],
    run: Tuple[float, float],
    domain: Tuple[float, float],
    spacing: float,
) -> List[Tuple[float, float]]:
    """The parameter intervals of a run of zero gaps that both objects share (cut where a segment or arc ends)."""

    def shared(s: float) -> bool:
        x, y = curve.point(s)
        return gap(s) == 0.0 and other.accepts(x, y) is not None

    start, end = max(domain[0], run[0] - spacing), min(domain[1], run[1] + spacing)
    count = max(2, int(math.ceil((end - start) / spacing)))
    grid = [start + (end - start) * i / count for i in range(count)] + [end]
    flags = [shared(s) for s in grid]
    pieces: List[Tuple[float, float]] = []
    i = 0
    while i < len(grid):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(grid) and flags[j + 1]:
            j += 1
        first = grid[i] if i == 0 else _boundary(shared, grid[i], grid[i - 1])
        last = grid[j] if j == len(grid) - 1 else _boundary(shared, grid[j], grid[j + 1])
        pieces.append((first, last))
        i = j + 1
    return pieces


def _joined_at_seam(
    curve: _Curve, pieces: List[Tuple[float, float]], domain: Tuple[float, float], size: float
) -> List[Tuple[float, float, float]]:
    """Pieces as (first, last, parameter span); on a closed curve, the pieces at both ends of
    its parameter range are one piece running across the seam."""
    spans = [(first, last, last - first) for first, last in sorted(pieces)]
    if len(spans) < 2 or not _is_closed(curve, domain, size):
        return spans
    head, tail = spans[0], spans[-1]
    if head[0] != domain[0] or tail[1] != domain[1]:
        return spans
    return [(tail[0], head[1], tail[2] + head[2])] + spans[1:-1]


def _is_closed(curve: _Curve, domain: Tuple[float, float], size: float) -> bool:
    start, end = curve.point(domain[0]), curve.point(domain[1])
    if not all(math.isfinite(value) for value in start + end):
        return False
    return _same_point(start, end, size)


def _add_piece(
    found: _Found,
    curve: _Curve,
    other: _ImplicitShape,
    piece: Tuple[float, float],
    long_run: bool,
    size: float,
) -> None:
    """An overlap from the curve at ``first`` to ``last``; a single point when it has no length.

    A piece longer than a sample step whose ends meet went once round a closed curve: the
    whole circle or ellipse (or closed curve) is shared.
    """
    first, last = piece
    start, end = _snapped_end(other, curve.point(first)), _snapped_end(other, curve.point(last))
    if not _same_point(start, end, size):
        found.overlaps.append({"kind": "curve", "start": list(start), "end": list(end)})
    elif long_run:
        found.overlaps.append({"kind": _closed_overlap_kind(other)})
    else:
        hit = _curve_hit(curve, other, first, False)
        if hit is not None:
            found.hits.append(hit)


def _closed_overlap_kind(other: _ImplicitShape) -> str:
    if isinstance(other, CircleShape) and not other.is_arc:
        return "circle"
    if isinstance(other, EllipseShape):
        return "ellipse"
    return "curve"


def _snapped_end(other: _ImplicitShape, point: Point2) -> Point2:
    """An overlap end that bisection left just past a segment's end or an arc's end, moved onto it."""
    if isinstance(other, LineShape) and other.bounded:
        t = other.parameter_of(*point)
        if t < 0.0 or t > 1.0:
            return other.point_at(min(max(t, 0.0), 1.0))
    if isinstance(other, CircleShape) and other.arc_start is not None and other.arc_sweep is not None:
        offset = (other.angle_of(*point) - other.arc_start) % _TWO_PI
        if offset > other.arc_sweep:
            end = other.arc_sweep if offset - other.arc_sweep < _TWO_PI - offset else 0.0
            return other.point_at_angle(other.arc_start + end)
    return point


def _boundary(inside: Callable[[float], bool], s_in: float, s_out: float) -> float:
    """Bisect between a parameter where ``inside`` holds and one where it does not."""
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (s_in + s_out)
        if middle in (s_in, s_out):
            break
        if inside(middle):
            s_in = middle
        else:
            s_out = middle
    return s_in


def _point_hits(point: PointShape, other: Shape) -> _Found:
    """A degenerate object: the point itself, when it lies on the other object."""
    if isinstance(other, ParametricShape):
        found = _curve_hits(other, point, other.t_min, other.t_max)
        for hit in found.hits:
            # The distance to a point only touches zero (no tangency); report the point itself
            hit.x, hit.y, hit.tangent = point.x, point.y, False
        return found
    found = _Found()
    if isinstance(other, PointShape):
        on_object = other.implicit(point.x, point.y) <= other.noise(point.x, point.y)
        params: Optional[Params] = {} if on_object else None
    else:
        value = other.implicit(point.x, point.y)
        tolerance = _on_object_tolerance(other, point.x, point.y)
        params = other.accepts(point.x, point.y) if abs(value) <= tolerance else None
    if params is not None:
        found.hits.append(_Hit(point.x, point.y, {point.name: {}, other.name: params}))
    return found


def _on_object_tolerance(shape: _ImplicitShape, x: float, y: float) -> float:
    """How far (in the shape's implicit value) a point may be off the shape and still lie on it."""
    if isinstance(shape, LineShape):
        return _coincidence_tolerance(shape.length, x, y, *shape.coordinates())
    if isinstance(shape, CircleShape):
        return _coincidence_tolerance(shape.r, x, y, shape.cx, shape.cy)
    if isinstance(shape, EllipseShape):
        # The implicit value is relative to the radii
        return _coincidence_tolerance(max(shape.rx, shape.ry), x, y, shape.cx, shape.cy) / min(shape.rx, shape.ry)
    if isinstance(shape, FunctionShape):
        return _coincidence_tolerance(max(1.0, abs(y)), x, y)
    return shape.noise(x, y)


# ---------------------------------------------------------------------------
# Merging, rounding and the report
# ---------------------------------------------------------------------------


def _report(found: _Found, max_results: int, size: float) -> IntersectionReport:
    """``size`` is the smaller object's size: it, not the distance from the origin, sets which
    points are one point and how many digits tell the points apart."""
    hits = [hit for hit in _merged_hits(found.hits, size) if not _is_overlap_end(hit, found.overlaps, size)]
    points = sorted((_rounded_point(hit, size) for hit in hits), key=lambda point: (point["x"], point["y"]))
    limit = max(0, int(max_results))
    overlaps = [_rounded_overlap(overlap, size) for overlap in found.overlaps]
    return {
        "points": points[:limit],
        "overlaps": overlaps,
        "total_found": len(points),
        "truncated": len(points) > limit,
        "notes": list(found.notes),
    }


def _merged_hits(hits: List[_Hit], size: float) -> List[_Hit]:
    """One hit per location (a root found from both ends of a closed curve's turn)."""
    merged: List[_Hit] = []
    for hit in hits:
        same = [m for m in merged if _same_point((m.x, m.y), (hit.x, hit.y), size)]
        if same:
            same[0].tangent = same[0].tangent or hit.tangent
            continue
        merged.append(hit)
    return merged


def _is_overlap_end(hit: _Hit, overlaps: List[Overlap], size: float) -> bool:
    """A point at (or within rounding of) the end of a reported overlap is part of it."""
    ends = [overlap["start"] for overlap in overlaps if "start" in overlap]
    ends += [overlap["end"] for overlap in overlaps if "end" in overlap]
    return any(_same_point((end[0], end[1]), (hit.x, hit.y), size, _ABSORB_TOLERANCE) for end in ends)


def _same_point(p: Point2, q: Point2, size: float, tolerance: float = _MERGE_TOLERANCE) -> bool:
    """p and q are one point: within ``tolerance`` of the objects' size, plus coordinate rounding."""
    magnitude = max(abs(p[0]), abs(p[1]), abs(q[0]), abs(q[1]))
    limit = tolerance * size + _SAME_POINT_ULPS * _EPS * magnitude
    return math.hypot(p[0] - q[0], p[1] - q[1]) <= limit


def _shape_size(shape: Union[Shape, _EllipseCurve]) -> float:
    """The object's own size (length, radius, extent), which tolerances are relative to."""
    if isinstance(shape, LineShape):
        return shape.length
    if isinstance(shape, CircleShape):
        return shape.r
    if isinstance(shape, EllipseShape):
        return max(shape.rx, shape.ry)
    if isinstance(shape, _EllipseCurve):
        return max(shape.ellipse.rx, shape.ellipse.ry)
    if isinstance(shape, FunctionShape):
        return shape.right - shape.left
    if isinstance(shape, ParametricShape):
        return shape.extent()
    return 0.0


def _pair_size(a: Union[Shape, _EllipseCurve], b: Union[Shape, _EllipseCurve]) -> float:
    """The smaller positive size of the two objects (1 when neither has one)."""
    sizes = [size for size in (_shape_size(a), _shape_size(b)) if math.isfinite(size) and size > 0.0]
    return min(sizes) if sizes else 1.0


def _rounded_point(hit: _Hit, size: float) -> IntersectionPoint:
    x, y = _round_pair((hit.x, hit.y), size)
    point: IntersectionPoint = {"x": x, "y": y}
    params = {name: _rounded_params(values) for name, values in hit.params.items() if values}
    if params:
        point["params"] = params
    if hit.tangent:
        point["tangent"] = True
    return point


def _rounded_params(values: Params) -> Params:
    rounded: Params = {}
    for key, value in values.items():
        if key == "angle":
            angle = _round(value % _TWO_PI, _TWO_PI)
            rounded[key] = 0.0 if angle >= _round(_TWO_PI) else angle
        else:
            rounded[key] = _round(value)
    return rounded


def _rounded_overlap(overlap: Overlap, size: float) -> Overlap:
    result: Overlap = {"kind": overlap["kind"]}
    if "start" in overlap:
        result["start"] = _round_pair(overlap["start"], size)
    if "end" in overlap:
        result["end"] = _round_pair(overlap["end"], size)
    return result


def _round_pair(pair: Sequence[float], size: float) -> List[float]:
    """Round a point's coordinates to 10 significant digits of the object's size.

    Near the origin that is 10 significant digits; a small object far from it keeps more,
    so points a radius apart stay apart, but never digits below the rounding of the
    coordinates themselves (1000000.000005, not 1000000.0000049999). Values within rounding
    of zero, or within 1e-12 of the size, are 0.
    """
    magnitude = max(1.0, abs(pair[0]), abs(pair[1]))
    digits = _REPORT_DIGITS
    if 0.0 < size < magnitude:
        digits = min(_MAX_DIGITS, _REPORT_DIGITS + int(math.ceil(math.log10(magnitude / size))))
    resolution = _ROUNDING_UNITS * _EPS * magnitude
    floor = max(_ZERO_FLOOR * min(magnitude, size), resolution)
    return [_round_digits(value, digits, floor, resolution) for value in pair[:2]]


def _round_digits(value: float, digits: int, floor: float, resolution: float) -> float:
    """``digits`` significant digits, but no decimal finer than ``resolution``."""
    if not math.isfinite(value):
        return value
    if abs(value) <= floor:
        return 0.0
    significant_decimals = digits - 1 - int(math.floor(math.log10(abs(value))))
    resolved_decimals = -int(math.floor(math.log10(resolution)))
    if 0 <= resolved_decimals < significant_decimals:
        return float(f"{value:.{resolved_decimals}f}") + 0.0
    return float(f"{value:.{digits}g}") + 0.0


def _round(value: float, size: float = 1.0) -> float:
    """10 significant digits; values within 1e-12 of ``size`` of zero (and -0.0) are 0.0."""
    if not math.isfinite(value):
        return value
    if abs(value) <= _ZERO_FLOOR * size:
        return 0.0
    return float(f"{value:.{_REPORT_DIGITS}g}") + 0.0


def _safe_float(f: Callable[[float], float], x: float) -> float:
    """f(x) as a real float; NaN when f fails or is complex."""
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
