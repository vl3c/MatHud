"""
The canvas state as sent with a prompt.

A prompt carries ``Canvas.get_canvas_state()`` plus what the server needs to tell
how big the drawings are on screen, and to add a view note when they are too small
or outside the view (``static/canvas_view_note.py``):

``canvas_size_px``  the canvas size in CSS pixels, ``{"width", "height"}``
``curve_extents``   for each function graph and parametric curve, the box its
                    sampled points span in math units, ``{bucket: {name: {"box":
                    [left, right, bottom, top], "clipped": bool, "turns": bool,
                    "spiky": bool}}}``

A function without both bounds is sampled over the visible x range only
(``clipped``), since its graph runs across the whole view. ``turns`` says the
samples change direction (sin does, a straight line never does); ``spiky`` says the
graph has a vertical asymptote among the samples: a few of them reach far beyond the
rest, or the graph blows up halfway between the samples next to its biggest values. The server
removes both keys before the canvas reaches the model; they are added to the
prompt's copy only, never to saved workspaces or traces.

The cost is bounded: at most ``MAX_MEASURED_CURVES`` curves of ``CURVE_SAMPLES``
evaluations each, shared fairly between graphs and parametric curves, bounded
graphs entirely off screen are skipped, and measuring stops after ``TIME_BUDGET_S``.

Pure Python (no ``browser`` import), tested by ``client_tests/test_prompt_canvas_state.py``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Mirror static/canvas_state_formatter.CANVAS_SIZE_KEY and CURVE_EXTENTS_KEY.
CANVAS_SIZE_KEY = "canvas_size_px"
CURVE_EXTENTS_KEY = "curve_extents"

CURVE_SAMPLES = 64
MAX_MEASURED_CURVES = 20
TIME_BUDGET_S = 0.03
# Share of the samples dropped at each end of a graph's y values, so the steep ends near
# a vertical asymptote (tan, 1/x) do not stretch its box.
_Y_TRIM = 0.02
# Spiky: the untrimmed y range is this many times the trimmed one, or a value halfway between
# two of the samples next to the biggest values is this many times every sample.
_SPIKE_RATIO = 10.0
_PEAKS_CHECKED = 3

# Drawable class -> state bucket.
_GRAPH_CLASSES = (("Function", "Functions"), ("PiecewiseFunction", "PiecewiseFunctions"))
_CURVE_CLASS = ("ParametricFunction", "ParametricFunctions")


def canvas_size_px(canvas: Any) -> Optional[Dict[str, float]]:
    """Return ``{"width", "height"}`` of the canvas in CSS pixels, or None when unknown."""
    try:
        width = float(canvas.width)
        height = float(canvas.height)
    except Exception:
        return None
    if not (math.isfinite(width) and math.isfinite(height)) or width <= 0 or height <= 0:
        return None
    return {"width": round(width, 2), "height": round(height, 2)}


def with_view_info(state: Any, canvas: Any, clock: Callable[[], float] = time.time) -> Any:
    """Return a shallow copy of ``state`` with the canvas size and curve extents added.

    ``state`` itself is unchanged. Returns ``state`` as given when it is not a dict
    or the canvas size is unknown.
    """
    size = canvas_size_px(canvas)
    if size is None or not isinstance(state, dict):
        return state
    result = dict(state)
    result[CANVAS_SIZE_KEY] = size
    extents = curve_extents(canvas, clock)
    if extents:
        result[CURVE_EXTENTS_KEY] = extents
    return result


def curve_extents(canvas: Any, clock: Callable[[], float] = time.time) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """The sampled box of every function graph and parametric curve, by bucket and name."""
    view = _visible_x_range(canvas)
    deadline = clock() + TIME_BUDGET_S
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for bucket, drawable in _fair_order(canvas)[:MAX_MEASURED_CURVES]:
        if clock() > deadline:
            break
        if bucket == _CURVE_CLASS[1]:
            entry = _curve_extent(drawable)
        else:
            entry = _graph_extent(drawable, view)
        name = getattr(drawable, "name", None)
        if entry is not None and isinstance(name, str):
            result.setdefault(bucket, {})[name] = entry
    return result


def _fair_order(canvas: Any) -> List[Tuple[str, Any]]:
    """Graphs and parametric curves taken in turn, so neither kind can use up the cap alone."""
    graphs = [(bucket, d) for class_name, bucket in _GRAPH_CLASSES for d in _drawables(canvas, class_name)]
    curves = [(_CURVE_CLASS[1], d) for d in _drawables(canvas, _CURVE_CLASS[0])]
    ordered: List[Tuple[str, Any]] = []
    for index in range(max(len(graphs), len(curves))):
        ordered.extend(group[index] for group in (graphs, curves) if index < len(group))
    return ordered


def _graph_extent(function: Any, view: Optional[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
    """Sample y = f(x) over its bounds, or over the visible x range where it has no bound."""
    left, right = _bound(function, "left_bound"), _bound(function, "right_bound")
    clipped = left is None or right is None
    if clipped:
        if view is None:
            return None
        left = view[0] if left is None else max(left, view[0])
        right = view[1] if right is None else min(right, view[1])
    elif view is not None and (right < view[0] or left > view[1]):
        return None  # entirely off screen: nothing to tell about how it looks
    if left is None or right is None or right < left:
        return None
    evaluate = getattr(function, "function", None)
    if not callable(evaluate):
        return None
    points = _sample(lambda x: (x, evaluate(x)), left, right)
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ordered = sorted(ys)
    trim = int(len(ordered) * _Y_TRIM)
    kept = ordered[trim : len(ordered) - trim] if len(ordered) > 2 * trim + 1 else ordered
    full_range, kept_range = ordered[-1] - ordered[0], kept[-1] - kept[0]
    return {
        "box": [min(xs), max(xs), kept[0], kept[-1]],
        "clipped": clipped,
        "turns": _turns(ys),
        "spiky": full_range > _SPIKE_RATIO * kept_range or _pole_between_samples(evaluate, points),
    }


def _pole_between_samples(evaluate: Callable[[float], Any], points: List[Tuple[float, float]]) -> bool:
    """Look between the samples next to the biggest values: a pole there (1/x, 1/x^2) blows up.

    A smooth graph stays within its sampled range halfway between two samples; next to a
    vertical asymptote the midpoint fails, is infinite, or is far beyond every sample.
    """
    largest = max(abs(y) for _, y in points)
    peaks = sorted(range(len(points)), key=lambda i: abs(points[i][1]), reverse=True)[:_PEAKS_CHECKED]
    for index in peaks:
        for neighbour in (index - 1, index + 1):
            if not 0 <= neighbour < len(points):
                continue
            middle = (points[index][0] + points[neighbour][0]) / 2.0
            try:
                value = float(evaluate(middle))
            except Exception:
                return True
            if math.isinf(value) or abs(value) > _SPIKE_RATIO * max(largest, 1e-300):
                return True
    return False


def _turns(ys: List[float]) -> bool:
    """True when the samples change direction: rising then falling, or the other way round."""
    span = max(ys) - min(ys)
    if span <= 0:
        return False
    signs = []
    for first, second in zip(ys, ys[1:]):
        step = second - first
        if abs(step) > 1e-9 * span:
            signs.append(step > 0)
    return any(a != b for a, b in zip(signs, signs[1:]))


def _curve_extent(curve: Any) -> Optional[Dict[str, Any]]:
    """Sample (x(t), y(t)) over [t_min, t_max]."""
    t_min, t_max = _bound(curve, "t_min"), _bound(curve, "t_max")
    evaluate = getattr(curve, "evaluate", None)
    if t_min is None or t_max is None or t_max < t_min or not callable(evaluate):
        return None
    points = _sample(evaluate, t_min, t_max)
    if len(points) < 2:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return {"box": [min(xs), max(xs), min(ys), max(ys)], "clipped": False}


def _sample(point_at: Callable[[float], Any], low: float, high: float) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for index in range(CURVE_SAMPLES):
        t = low + (high - low) * index / (CURVE_SAMPLES - 1)
        try:
            x, y = point_at(t)
            x, y = float(x), float(y)
        except Exception:
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    return points


def _bound(drawable: Any, attribute: str) -> Optional[float]:
    value = getattr(drawable, attribute, None)
    try:
        number = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return number if number is not None and math.isfinite(number) else None


def _visible_x_range(canvas: Any) -> Optional[Tuple[float, float]]:
    try:
        mapper = canvas.coordinate_mapper
        left, right = float(mapper.get_visible_left_bound()), float(mapper.get_visible_right_bound())
    except Exception:
        return None
    return (left, right) if math.isfinite(left) and math.isfinite(right) and right > left else None


def _drawables(canvas: Any, class_name: str) -> List[Any]:
    try:
        return list(canvas.get_drawables_by_class_name(class_name))
    except Exception:
        return []
