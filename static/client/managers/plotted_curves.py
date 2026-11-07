"""Plotted curves as point functions, for tools that measure a curve by name (arc_length).

``resolve_plotted_curve(drawables, name)`` finds a function, piecewise function, parametric
curve, polar curve, circle, circle arc or ellipse by name and returns a NamedCurve: its kind, a point
function of its parameter, its plotted range, the parameters where it may change formula or
have a hole (piecewise boundaries, holes) and its vertical asymptotes.

Pure Python (no browser imports): it reads only the drawables' attributes.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

from managers.implicit_refusals import implicit_refusal
from utils.arc_length import Point
from utils.arc_length_tool import NamedCurve

# Containers searched, in order, and how each kind of drawable becomes a NamedCurve
_KINDS: Sequence[Tuple[str, str]] = (
    ("Functions", "function"),
    ("PiecewiseFunctions", "piecewise"),
    ("ParametricFunctions", "parametric"),
    ("PolarFunctions", "polar"),
    ("Circles", "circle"),
    ("CircleArcs", "circle_arc"),
    ("Ellipses", "ellipse"),
)


def resolve_plotted_curve(drawables: Any, name: str) -> NamedCurve:
    """The plotted curve called name, as a NamedCurve.

    Raises:
        ValueError: If no function, piecewise function, parametric or polar curve, circle,
            circle arc or ellipse has that name.
    """
    for container, kind in _KINDS:
        for drawable in list(getattr(drawables, container, None) or []):
            if getattr(drawable, "name", None) == name:
                return _BUILDERS[kind](drawable)
    refusal = implicit_refusal(drawables, name, "arc_length")
    if refusal:
        raise ValueError(refusal)
    raise ValueError(
        f"No function, parametric or polar curve, circle, circle arc or ellipse named '{name}' is on the canvas. "
        "Give the curve's name as the canvas shows it, or its expression instead."
    )


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _floats(values: Any) -> List[float]:
    return [float(v) for v in (values or [])]


def _function(drawable: Any) -> NamedCurve:
    evaluate: Callable[[float], float] = drawable.function

    def point(x: float) -> Optional[Point]:
        return (x, evaluate(x))

    return NamedCurve(
        "function",
        point,
        (_finite_or_none(drawable.left_bound), _finite_or_none(drawable.right_bound)),
        _floats(getattr(drawable, "point_discontinuities", None))
        + _floats(getattr(drawable, "undefined_at", None))
        + _floats(getattr(drawable, "corner_points", None)),
        _floats(getattr(drawable, "vertical_asymptotes", None)),
        f"{drawable.name}(x) = {drawable.function_string}",
    )


def _piecewise(drawable: Any) -> NamedCurve:
    evaluate: Callable[[float], float] = drawable.function
    boundaries: List[float] = []
    for interval in drawable.intervals:
        for edge in (interval.left, interval.right):
            if edge is not None and math.isfinite(float(edge)):
                boundaries.append(float(edge))
        boundaries.extend(_floats(getattr(interval, "undefined_at", None)))

    def point(x: float) -> Optional[Point]:
        return (x, evaluate(x))

    pieces = ", ".join(str(interval.expression) for interval in drawable.intervals)
    return NamedCurve(
        "function",
        point,
        (_finite_or_none(drawable.left_bound), _finite_or_none(drawable.right_bound)),
        boundaries
        + _floats(getattr(drawable, "point_discontinuities", None))
        + _floats(getattr(drawable, "corner_points", None)),
        _floats(getattr(drawable, "vertical_asymptotes", None)),
        f"piecewise function {drawable.name} ({pieces})",
    )


def _parametric(drawable: Any) -> NamedCurve:
    evaluate: Callable[[float], Point] = drawable.evaluate
    return NamedCurve(
        "parametric",
        evaluate,
        (float(drawable.t_min), float(drawable.t_max)),
        (),
        (),
        f"{drawable.name}: (x(t), y(t)) = ({drawable.x_expression}, {drawable.y_expression})",
    )


def _polar(drawable: Any) -> NamedCurve:
    evaluate: Callable[[float], Point] = drawable.evaluate
    return NamedCurve(
        "polar",
        evaluate,
        (float(drawable.theta_min), float(drawable.theta_max)),
        (),
        (),
        f"{drawable.name}: r = {drawable.r_expression}",
    )


def _circle(drawable: Any) -> NamedCurve:
    cx, cy, radius = float(drawable.center.x), float(drawable.center.y), float(drawable.radius)

    def point(t: float) -> Optional[Point]:
        return (cx + radius * math.cos(t), cy + radius * math.sin(t))

    return NamedCurve(
        "parametric",
        point,
        (0.0, 2 * math.pi),
        (),
        (),
        f"circle {drawable.name}: (x(t), y(t)) = ({cx:g} + {radius:g}*cos(t), {cy:g} + {radius:g}*sin(t))",
    )


def _circle_arc(drawable: Any) -> NamedCurve:
    cx, cy, radius = float(drawable.center_x), float(drawable.center_y), float(drawable.radius)
    start, sweep = drawable.start_and_sweep()
    turn = 1.0 if sweep >= 0 else -1.0

    def point(t: float) -> Optional[Point]:
        angle = start + turn * t
        return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    return NamedCurve(
        "parametric",
        point,
        (0.0, abs(float(sweep))),
        (),
        (),
        f"circle arc {drawable.name} (radius {radius:g}, turning {math.degrees(abs(sweep)):g}°), traced by t "
        "from 0 to its angle in radians",
    )


def _ellipse(drawable: Any) -> NamedCurve:
    cx, cy = float(drawable.center.x), float(drawable.center.y)
    a, b = float(drawable.radius_x), float(drawable.radius_y)
    angle = math.radians(float(getattr(drawable, "rotation_angle", 0.0) or 0.0))
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def point(t: float) -> Optional[Point]:
        u, v = a * math.cos(t), b * math.sin(t)
        return (cx + u * cos_a - v * sin_a, cy + u * sin_a + v * cos_a)

    return NamedCurve(
        "parametric",
        point,
        (0.0, 2 * math.pi),
        (),
        (),
        f"ellipse {drawable.name} (semi-axes {a:g} and {b:g}), traced by t from 0 to 2π",
    )


_BUILDERS = {
    "function": _function,
    "piecewise": _piecewise,
    "parametric": _parametric,
    "polar": _polar,
    "circle": _circle,
    "circle_arc": _circle_arc,
    "ellipse": _ellipse,
}
