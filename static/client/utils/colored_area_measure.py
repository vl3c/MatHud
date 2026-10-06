"""Measuring the area of a coloured area (``calculate_area`` on a coloured area's name).

Areas between two bounds (functions, segments, constants or the x-axis) are the
integral of |f1(x) - f2(x)| over the area's x-interval:

1. A vertical asymptote that a bounding function lists inside the interval is an
   error ("the area diverges near x ≈ ..."), as is a value that cannot be computed.
2. f1 - f2 is sampled on a grid of 512 cells, refined (x4, up to 32768 cells) until
   there are at least 8 cells per sign change, so close crossings are not missed.
3. Poles the function does not list are found numerically: a sign change whose
   bracket, narrowed by the Illinois method, ends where |f1 - f2| grows instead of
   vanishing, or a sharp local peak of |f1 - f2| whose refined maximum (ternary
   search) grows without bound. Both are "diverges" errors.
4. With at most 1024 sign changes, each crossing is located by the Illinois method
   and every piece between crossings is integrated with Simpson's rule
   (``utils.numeric_integration.integrate``); the error estimate is the sum of the
   pieces' Richardson estimates (the difference between two step counts, over 15),
   which matches the actual error closely for smooth bounds.
5. With more sign changes, |f1 - f2| is integrated cell by cell as a piecewise linear
   function, cut at each cell's interpolated crossing, on the grid and on every
   other grid point; the error estimate is their difference over 3 (Richardson for
   a second-order rule), and the result carries a warning that accuracy is limited.

Bounds that are straight lines make the integrand piecewise linear, so those areas
come out exact up to rounding. Closed shapes use exact formulas (the shoelace formula
for straight-edged polygons, pi*r^2 and pi*a*b); a region area is measured from its
region expression, with the drawn outline as a check.

This module has no browser dependencies, so server-side pytest suites can test it
with stand-in drawables.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from utils.numeric_integration import integrate

# Initial grid, its refinement factor and limit, and the cells wanted per sign change.
GRID_CELLS = 512
GRID_REFINE_FACTOR = 4
MAX_GRID_CELLS = 32768
CELLS_PER_CROSSING = 8
# Most sign changes located one by one; beyond this the cell-by-cell rule is used.
MAX_LOCATED_CROSSINGS = 1024
# Illinois iterations locating a crossing; and how much |f1 - f2| at the narrowed bracket
# may exceed its value at the cell ends before the "crossing" counts as a pole.
CROSSING_ITERATIONS = 80
POLE_GROWTH = 1e3
# A grid peak sharper than this ratio to a neighbour is refined; a refined maximum this
# many times the grid value is a pole. At most this many peaks are refined.
PEAK_SHARPNESS = 2.0
PEAK_REFINE_STEPS = 60
PEAK_POLE_RATIO = 1e6
MAX_PEAK_PROBES = 200
# Simpson steps over the whole interval, shared among the pieces (each gets at least the minimum).
SIMPSON_STEPS = 200
SIMPSON_MIN_STEPS_PER_PIECE = 16
# A piece whose error estimate exceeds this share of its value is integrated again with
# four times the steps, up to the limit (a narrow peak needs more than the share).
PIECE_RELATIVE_TOLERANCE = 1e-7
MAX_STEPS_PER_PIECE = 16384

ColoredAreaMeasure = Dict[str, Any]
YFunction = Callable[[float], Optional[float]]
ExpressionArea = Callable[[str], float]


class _Undefined(ValueError):
    """f1 - f2 cannot be computed at a point."""


def measure_colored_area(area: Any, expression_area: Optional[ExpressionArea] = None) -> ColoredAreaMeasure:
    """The area of ``area`` with how it was measured.

    Returns ``{"value", "method", "error_estimate", ...}``: ``bounds`` for the
    integrated kinds, ``crossings`` (x values where the bounds cross, when at most
    50) or ``crossing_count``, and ``warning`` when accuracy is limited.
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

    def difference(x: float) -> float:
        y1, y2 = upper(x), lower(x)
        for name, y in ((upper_name, y1), (lower_name, y2)):
            if y is None or not isinstance(y, (int, float)) or not math.isfinite(y):
                raise _Undefined(
                    f"The area diverges or is undefined near x ≈ {x:.6g}: {name} is undefined or not "
                    f"finite at x = {x:.12g}, inside [{a:g}, {b:g}]"
                )
        return float(y1) - float(y2)  # type: ignore[arg-type]

    xs, values = _sample(difference, a, b)
    warnings: List[str] = []
    _check_peaks(difference, xs, values, warnings)
    changes = _sign_change_cells(values)
    zeros = [i for i in range(1, len(xs) - 1) if values[i] == 0.0]
    names = f"|{upper_name} - {lower_name}|"
    if len(changes) + len(zeros) <= MAX_LOCATED_CROSSINGS:
        measure = _integrate_pieces(difference, xs, values, changes, zeros, names, warnings)
    else:
        measure = _integrate_cells(difference, xs, values, changes, names, warnings)
    if len(changes) * CELLS_PER_CROSSING > len(xs) - 1:
        warnings.append(
            f"The bounds cross about {len(changes)} times, too often for the finest grid "
            f"({len(xs) - 1} cells): crossings closer than a cell apart may be missed."
        )
    measure["bounds"] = [a, b]
    if warnings:
        measure["warning"] = " ".join(warnings)
    return measure


def _sign_change_cells(values: Sequence[float]) -> List[int]:
    """Cells whose end values have opposite signs (a zero at a grid point is not a sign-change cell)."""
    return [i for i in range(len(values) - 1) if values[i] * values[i + 1] < 0]


def _sample(difference: Callable[[float], float], a: float, b: float) -> Tuple[List[float], List[float]]:
    """Grid points and values, refined until each sign change has enough cells (or the limit)."""
    cells = GRID_CELLS
    while True:
        step = (b - a) / cells
        xs = [a + i * step for i in range(cells)] + [b]
        values = [difference(x) for x in xs]
        changes = len(_sign_change_cells(values)) + sum(1 for v in values[1:-1] if v == 0.0)
        if changes * CELLS_PER_CROSSING <= cells or cells >= MAX_GRID_CELLS:
            return xs, values
        cells = min(cells * GRID_REFINE_FACTOR, MAX_GRID_CELLS)


def _check_peaks(
    difference: Callable[[float], float], xs: List[float], values: List[float], warnings: List[str]
) -> None:
    """Refine every sharp interior peak of |f1 - f2|; one that grows without bound is a pole."""
    probes = 0
    for i in range(1, len(xs) - 1):
        here, left, right = abs(values[i]), abs(values[i - 1]), abs(values[i + 1])
        if here == 0.0 or here < left or here < right or here <= PEAK_SHARPNESS * min(left, right):
            continue
        if probes >= MAX_PEAK_PROBES:
            warnings.append(f"Only the first {MAX_PEAK_PROBES} sharp peaks were checked for poles.")
            return
        probes += 1
        peak_x, peak = _refine_peak(difference, xs[i - 1], xs[i + 1])
        if peak is None or peak > PEAK_POLE_RATIO * here:
            raise _diverges(peak_x, "the difference of the bounds grows without bound there")


def _refine_peak(difference: Callable[[float], float], lo: float, hi: float) -> Tuple[float, Optional[float]]:
    """Ternary search for the maximum of |f1 - f2| in [lo, hi]: (x, value), value None if not computable."""
    for _ in range(PEAK_REFINE_STEPS):
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        try:
            v1, v2 = abs(difference(m1)), abs(difference(m2))
        except _Undefined:
            return 0.5 * (m1 + m2), None
        if v1 < v2:
            lo = m1
        else:
            hi = m2
    x = 0.5 * (lo + hi)
    try:
        return x, abs(difference(x))
    except _Undefined:
        return x, None


def _locate_crossing(difference: Callable[[float], float], lo: float, hi: float, f_lo: float, f_hi: float) -> float:
    """The root of f1 - f2 in a sign-change cell (Illinois method); a pole there raises."""
    scale = max(abs(f_lo), abs(f_hi))
    side = 0
    for _ in range(CROSSING_ITERATIONS):
        if hi - lo <= 1e-15 * max(1.0, abs(lo), abs(hi)):
            break
        x = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
        if not lo < x < hi:
            x = 0.5 * (lo + hi)
        try:
            f_x = difference(x)
        except _Undefined:
            raise _diverges(x, "the bounds' difference changes sign through a point where it is undefined")
        if f_x == 0.0:
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
        f_x = abs(difference(x))
    except _Undefined:
        raise _diverges(x, "the bounds' difference changes sign through a point where it is undefined")
    if f_x > POLE_GROWTH * scale:
        raise _diverges(x, "the bounds' difference changes sign through a pole there")
    return x


def _integrate_pieces(
    difference: Callable[[float], float],
    xs: List[float],
    values: List[float],
    changes: List[int],
    zeros: List[int],
    names: str,
    warnings: List[str],
) -> ColoredAreaMeasure:
    """Locate each crossing and integrate every piece between them with Simpson's rule.

    A grid point where the bounds meet exactly is a cut too.
    """
    located = [_locate_crossing(difference, xs[i], xs[i + 1], values[i], values[i + 1]) for i in changes]
    crossings = sorted(located + [xs[i] for i in zeros])
    cuts = [xs[0]] + crossings + [xs[-1]]
    pieces = [(lo, hi) for lo, hi in zip(cuts, cuts[1:]) if hi > lo]
    steps = max(SIMPSON_MIN_STEPS_PER_PIECE, SIMPSON_STEPS // max(len(pieces), 1))
    total = 0.0
    error = 0.0
    most_steps = steps
    for lo, hi in pieces:
        result = _integrate_piece(difference, lo, hi, steps)
        most_steps = max(most_steps, result["steps"] // 2)
        total += result["value"]
        error += result["error_estimate"]
        if "warning" in result and result["warning"] not in warnings:
            warnings.append(result["warning"])
    if error > PIECE_RELATIVE_TOLERANCE * abs(total) and most_steps >= MAX_STEPS_PER_PIECE:
        warnings.append("A piece did not converge to the target accuracy (a very narrow feature): see error_estimate.")
    split = f", split at {len(crossings)} crossing(s) of the bounds" if crossings else ""
    refined = f", up to {2 * most_steps} where needed" if most_steps > steps else ""
    measure: ColoredAreaMeasure = {
        "value": total,
        "method": (
            f"numeric integration of {names} over [{xs[0]:.12g}, {xs[-1]:.12g}] with Simpson's rule "
            f"({2 * steps} subintervals per piece{refined}{split})"
        ),
        "error_estimate": error,
    }
    _report_crossings(measure, crossings)
    return measure


def _integrate_piece(difference: Callable[[float], float], lo: float, hi: float, steps: int) -> Dict[str, Any]:
    """Simpson's rule on one piece, with four times the steps while the estimate is too large."""
    while True:
        result = integrate(lambda x: abs(difference(x)), lo, hi, method="simpson", steps=steps)
        target = PIECE_RELATIVE_TOLERANCE * abs(result["value"])
        if result["error_estimate"] <= target or steps >= MAX_STEPS_PER_PIECE:
            return dict(result)
        steps = min(steps * 4, MAX_STEPS_PER_PIECE)


def _integrate_cells(
    difference: Callable[[float], float],
    xs: List[float],
    values: List[float],
    changes: List[int],
    names: str,
    warnings: List[str],
) -> ColoredAreaMeasure:
    """Integrate |f1 - f2| as piecewise linear on the grid and on every other point; Richardson for the error."""
    for i in changes:
        # One evaluation per crossing: f1 - f2 at the interpolated root is small for a root, large at a pole.
        x = (xs[i] * values[i + 1] - xs[i + 1] * values[i]) / (values[i + 1] - values[i])
        try:
            at_root = abs(difference(x))
        except _Undefined:
            raise _diverges(x, "the bounds' difference changes sign through a point where it is undefined")
        if at_root > max(abs(values[i]), abs(values[i + 1])):
            _locate_crossing(difference, xs[i], xs[i + 1], values[i], values[i + 1])
    fine = _linear_abs_integral(xs, values)
    coarse = _linear_abs_integral(xs[::2], values[::2]) if (len(xs) - 1) % 2 == 0 else fine
    error = abs(fine - coarse) / 3.0
    warnings.append(
        f"The bounds cross {len(changes)} times, so the area was integrated cell by cell as a piecewise "
        "linear function: accuracy is limited (see error_estimate)."
    )
    return {
        "value": fine,
        "method": (
            f"numeric integration of {names} over [{xs[0]:.12g}, {xs[-1]:.12g}] as a piecewise linear function "
            f"on {len(xs) - 1} cells, cut at each cell's interpolated crossing ({len(changes)} crossings)"
        ),
        "error_estimate": error,
        "crossing_count": len(changes),
    }


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


def _report_crossings(measure: ColoredAreaMeasure, crossings: List[float]) -> None:
    if not crossings:
        return
    if len(crossings) <= 50:
        measure["crossings"] = crossings
    else:
        measure["crossing_count"] = len(crossings)


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
