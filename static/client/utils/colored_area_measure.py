"""Measuring the area of a coloured area (``calculate_area`` on a coloured area's name).

Areas bounded by functions and segments are measured by numeric integration of
|f1(x) - f2(x)| over the area's x-interval: the interval is cut where the two
bounds cross (sign changes of f1 - f2 on a 512-interval grid, refined by
bisection), and each piece is integrated with Simpson's rule
(``utils.numeric_integration.integrate``), whose error estimate (the difference
between two step counts, over 15) is summed over the pieces. Bounds that are
straight lines (segments, the x-axis, constants) make the integrand piecewise
linear, so those areas come out exact up to rounding. Closed shapes use their
exact formulas: the shoelace formula for polygons and sampled regions, pi*r^2
and pi*a*b for circles and ellipses.

This module has no browser dependencies, so server-side pytest suites can test
it with stand-in drawables.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from utils.numeric_integration import integrate

# Grid on which crossings of the two bounds are looked for.
CROSSING_GRID_INTERVALS = 512
# Bisection steps refining each crossing (an interval of 2^-60 of a grid cell).
CROSSING_BISECTION_STEPS = 60
# Simpson steps over the whole interval, shared among the pieces (each gets at least the minimum).
SIMPSON_STEPS = 200
SIMPSON_MIN_STEPS_PER_PIECE = 16
# Beyond this many crossings the pieces are not split any further (an oscillating difference).
MAX_PIECES = 64

ColoredAreaMeasure = Dict[str, Any]
YFunction = Callable[[float], Optional[float]]


def measure_colored_area(area: Any) -> ColoredAreaMeasure:
    """The area of ``area`` with how it was measured.

    Returns ``{"value", "method", "error_estimate", ...}``: ``bounds`` for the
    integrated kinds, ``crossings`` (x values where the bounds cross) when there
    are any, and ``warning`` when the integrand is suspect near an end.

    Raises:
        ValueError: the area has no finite x-interval, a bound is undefined
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
        )
    if kind == "SegmentsBoundedColoredArea":
        return _between_segments(area.segment1, area.segment2)
    if kind == "ClosedShapeColoredArea":
        return _closed_shape(area)
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
    return _between(_line_through(segment1), lower, left, right, str(getattr(segment1, "name", "segment")), second)


def _between(
    upper: YFunction,
    lower: YFunction,
    left: Optional[float],
    right: Optional[float],
    upper_name: str,
    lower_name: str,
) -> ColoredAreaMeasure:
    """Integral of |upper - lower| over [left, right], cut at the crossings."""
    if left is None or right is None or not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError("The coloured area has no finite x-interval to measure over")
    a, b = float(left), float(right)
    if not a < b:
        raise ValueError(f"The coloured area's x-interval [{a:g}, {b:g}] is empty")

    def difference(x: float) -> float:
        y1, y2 = upper(x), lower(x)
        for name, y in ((upper_name, y1), (lower_name, y2)):
            if y is None or not isinstance(y, (int, float)) or not math.isfinite(y):
                raise ValueError(f"{name} is undefined or not finite at x = {x:.12g}, inside [{a:g}, {b:g}]")
        return float(y1) - float(y2)  # type: ignore[arg-type]

    crossings = _crossings(difference, a, b)
    cuts = [a] + crossings + [b]
    pieces = [(lo, hi) for lo, hi in zip(cuts, cuts[1:]) if hi > lo]
    steps = max(SIMPSON_MIN_STEPS_PER_PIECE, SIMPSON_STEPS // max(len(pieces), 1))
    total = 0.0
    error = 0.0
    warnings: List[str] = []
    for lo, hi in pieces:
        result = integrate(lambda x: abs(difference(x)), lo, hi, method="simpson", steps=steps)
        total += result["value"]
        error += result["error_estimate"]
        if "warning" in result and result["warning"] not in warnings:
            warnings.append(result["warning"])
    split = f", split at {len(crossings)} crossing(s) of the bounds" if crossings else ""
    measure: ColoredAreaMeasure = {
        "value": total,
        "method": (
            f"numeric integration of |{upper_name} - {lower_name}| over [{a:.12g}, {b:.12g}] with Simpson's rule "
            f"({2 * steps} subintervals per piece{split})"
        ),
        "error_estimate": error,
        "bounds": [a, b],
    }
    if crossings:
        measure["crossings"] = crossings
    if warnings:
        measure["warning"] = " ".join(warnings)
    return measure


def _crossings(difference: Callable[[float], float], a: float, b: float) -> List[float]:
    """x values in (a, b) where ``difference`` changes sign, from a grid refined by bisection."""
    step = (b - a) / CROSSING_GRID_INTERVALS
    xs = [a + i * step for i in range(CROSSING_GRID_INTERVALS)] + [b]
    values = [difference(x) for x in xs]
    found: List[float] = []
    for i in range(CROSSING_GRID_INTERVALS):
        if 0 < i and values[i] == 0.0:
            found.append(xs[i])
        elif values[i] * values[i + 1] < 0:
            found.append(_bisect(difference, xs[i], xs[i + 1], values[i]))
        if len(found) >= MAX_PIECES - 1:
            break
    return found


def _bisect(difference: Callable[[float], float], lo: float, hi: float, value_lo: float) -> float:
    for _ in range(CROSSING_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        value = difference(mid)
        if value == 0.0:
            return mid
        if (value < 0) == (value_lo < 0):
            lo, value_lo = mid, value
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# Closed shapes
# ----------------------------------------------------------------------


def _closed_shape(area: Any) -> ColoredAreaMeasure:
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
        return _exact(_shoelace(coords), f"the shoelace formula over its {len(coords)} vertices")
    if shape == "region" and len(getattr(area, "points", None) or []) >= 3:
        points = [(float(x), float(y)) for x, y in area.points]
        return _exact(
            _shoelace(points),
            f"the shoelace formula over the {len(points)} points that outline the region (as drawn)",
        )
    if shape in ("circle_segment", "ellipse_segment"):
        raise ValueError(
            "Cannot measure a coloured circle or ellipse segment by name; use calculate_area with the "
            "shape and its chord segment, e.g. 'C(5) & AB'"
        )
    raise ValueError(f"Cannot measure a coloured closed shape of type '{shape}'")


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
