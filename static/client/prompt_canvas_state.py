"""
The canvas state as sent after a tool batch.

The prompt that returns a batch's results carries ``Canvas.get_canvas_state()`` plus
what the server needs to tell whether something the batch drew can be seen
(``static/canvas_view_note.py``):

``canvas_size_px``  the canvas size in CSS pixels, ``{"width", "height"}``
``curve_extents``   for each function graph and parametric curve the batch created or
                    redefined, the box its samples span in math units, ``{bucket: {name:
                    {"box": [left, right, bottom, top], "clipped": bool, "waves": bool,
                    "spiky": bool}}}``

A function without both bounds is sampled over the visible x range only (``clipped``).
``waves`` says the samples change direction at least ``_MIN_TURNS`` times (sin does; a
line, a parabola or a single bump does not, and zooming in on one cannot show it better).
For a wave, ``resolved`` says whether the samples follow it (halfway between two samples
it is about their average); an aliased wave is not, and ``period`` estimates its period
so the server only suggests a zoom that shows at least one;
``spiky`` says the graph has a vertical asymptote among the samples: a few of them reach
far beyond the rest, or it blows up halfway between the samples next to its biggest
values.

Measuring is deterministic: the new or redefined curves are found by comparing the
batch's canvas with the one before it (bucket, name and definition), at most
``MAX_MEASURED_CURVES`` of them in canvas order are measured, and if measuring runs past
``TIME_LIMIT_S`` no extents are sent at all (no note) rather than a partial set. The
server removes both keys before the canvas reaches the model; they are added to the
prompt's copy only, never to saved workspaces or traces.

Pure Python (no ``browser`` import), tested by ``client_tests/test_prompt_canvas_state.py``.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Mirror static/canvas_state_formatter.CANVAS_SIZE_KEY and CURVE_EXTENTS_KEY.
CANVAS_SIZE_KEY = "canvas_size_px"
CURVE_EXTENTS_KEY = "curve_extents"

CURVE_SAMPLES = 64
MAX_MEASURED_CURVES = 20
TIME_LIMIT_S = 0.1
# Share of the samples dropped at each end of a graph's y values, so the steep ends near
# a vertical asymptote (tan, 1/x) do not stretch its box.
_Y_TRIM = 0.02
# Spiky: the untrimmed y range is this many times the trimmed one, or a value halfway between
# two of the samples next to the biggest values is this many times every sample.
_SPIKE_RATIO = 10.0
_PEAKS_CHECKED = 3
# Waves: an oscillating graph changes direction at least this often among its samples.
_MIN_TURNS = 3
# Args left out when telling a redefined curve from an unchanged one: its colour, and lists
# derived from its definition (comparing them would cost Brython time for nothing).
_IGNORED_ARGS = frozenset({"color", "vertical_asymptotes", "horizontal_asymptotes", "point_discontinuities"})
# Resolved: a value halfway between two samples stays within this share of the samples' range
# of their average. Checked at this many midpoints spread over the samples.
_RESOLVED_TOLERANCE = 0.25
_MIDPOINTS_CHECKED = 16
_CHECK_FRACTIONS = (0.5, 0.31)
# Finding the period of an unresolved wave: halve the step this often at most, and sample
# this many steps each time.
_MAX_HALVINGS = 24
_PERIOD_STEPS = 16

# Drawable class -> state bucket.
_CURVE_CLASSES = (
    ("Function", "Functions"),
    ("PiecewiseFunction", "PiecewiseFunctions"),
    ("ParametricFunction", "ParametricFunctions"),
)


class _TimeUp(Exception):
    """Measuring ran past its limit; the prompt goes without extents."""


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


def with_view_info(state: Any, canvas: Any, previous_state: Any = None, clock: Callable[[], float] = time.time) -> Any:
    """Return a shallow copy of a batch's ``state`` with the canvas size and new curves' extents.

    ``previous_state`` is the canvas before the batch. ``state`` itself is unchanged; it is
    returned as given when it is not a dict or the canvas size is unknown.
    """
    size = canvas_size_px(canvas)
    if size is None or not isinstance(state, dict):
        return state
    result = dict(state)
    result[CANVAS_SIZE_KEY] = size
    extents = curve_extents(canvas, new_curve_names(state, previous_state), clock)
    if extents:
        result[CURVE_EXTENTS_KEY] = extents
    return result


def new_curve_names(state: Dict[str, Any], previous_state: Any) -> Dict[str, List[str]]:
    """The curves of ``state`` that ``previous_state`` lacks or defines differently, by bucket."""
    before = previous_state if isinstance(previous_state, dict) else {}
    names: Dict[str, List[str]] = {}
    for _, bucket in _CURVE_CLASSES:
        old = {item.get("name"): _definition(item) for item in _items(before, bucket)}
        for item in _items(state, bucket):
            name = item.get("name")
            if isinstance(name, str) and old.get(name) != _definition(item):
                names.setdefault(bucket, []).append(name)
    return names


def curve_extents(
    canvas: Any, wanted: Dict[str, List[str]], clock: Callable[[], float] = time.time
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """The sampled boxes of the ``wanted`` curves (at most ``MAX_MEASURED_CURVES``, in canvas order)."""
    view = _visible_x_range(canvas)
    deadline = clock() + TIME_LIMIT_S
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    measured = 0
    try:
        for class_name, bucket in _CURVE_CLASSES:
            names = set(wanted.get(bucket, []))
            for drawable in _drawables(canvas, class_name):
                name = getattr(drawable, "name", None)
                if name not in names or measured >= MAX_MEASURED_CURVES:
                    continue
                measured += 1
                entry = _curve_extent(drawable) if bucket == "ParametricFunctions" else _graph_extent(drawable, view)
                if clock() > deadline:
                    raise _TimeUp()
                if entry is not None:
                    result.setdefault(bucket, {})[name] = entry
    except _TimeUp:
        return {}
    return result


def _graph_extent(function: Any, view: Optional[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
    """Sample y = f(x) over its bounds, or over the visible x range where it has no bound."""
    left, right = _bound(function, "left_bound"), _bound(function, "right_bound")
    clipped = left is None or right is None
    if clipped:
        if view is None:
            return None
        left = view[0] if left is None else max(left, view[0])
        right = view[1] if right is None else min(right, view[1])
    evaluate = getattr(function, "function", None)
    if left is None or right is None or right < left or not callable(evaluate):
        return None
    points = _sample(lambda x: (x, evaluate(x)), left, right)
    if len(points) < 2:
        return None
    ys = [p[1] for p in points]
    ordered = sorted(ys)
    trim = int(len(ordered) * _Y_TRIM)
    kept = ordered[trim : len(ordered) - trim] if len(ordered) > 2 * trim + 1 else ordered
    full_range, kept_range = ordered[-1] - ordered[0], kept[-1] - kept[0]
    entry: Dict[str, Any] = {
        "box": [points[0][0], points[-1][0], kept[0], kept[-1]],
        "clipped": clipped,
        "waves": _turn_count(ys) >= _MIN_TURNS,
        "spiky": full_range > _SPIKE_RATIO * kept_range or _pole_between_samples(evaluate, points),
    }
    if entry["waves"] and not entry["spiky"]:
        entry["resolved"] = _resolved(evaluate, points, _MIDPOINTS_CHECKED)
        if not entry["resolved"]:
            entry["period"] = _period(evaluate, (left + right) / 2.0, (right - left) / (CURVE_SAMPLES - 1))
    return entry


def _resolved(evaluate: Callable[[float], Any], points: List[Tuple[float, float]], checks: int) -> bool:
    """True when the samples follow the graph: halfway between two of them it is about their average.

    An aliased wave (sin sampled every 20 units) fails: its midpoints land anywhere.
    """
    span = max(y for _, y in points) - min(y for _, y in points)
    stride = max(1, (len(points) - 1) // checks)
    for index in range(0, len(points) - 1, stride):
        (x1, y1), (x2, y2) = points[index], points[index + 1]
        # Two offsets, one of them not a power of two, so a step that happens to be a
        # multiple of the period cannot fool the check.
        for fraction in _CHECK_FRACTIONS:
            try:
                value = float(evaluate(x1 + fraction * (x2 - x1)))
            except Exception:
                continue
            expected = y1 + fraction * (y2 - y1)
            if math.isfinite(value) and abs(value - expected) > _RESOLVED_TOLERANCE * span:
                return False
    return True


def _period(evaluate: Callable[[float], Any], center: float, step: float) -> Optional[float]:
    """Estimate an unresolved wave's period: halve the step until short runs of samples resolve it.

    Two turns make a period. A step that happens to be near a multiple of the period can look
    resolved by chance, so the estimate must hold at the next finer step too.
    """
    previous: Optional[float] = None
    for _ in range(_MAX_HALVINGS):
        step /= 2.0
        points = _sample(lambda x: (x, evaluate(x)), center, center + _PERIOD_STEPS * step, _PERIOD_STEPS + 1)
        if len(points) < _PERIOD_STEPS or not _resolved(evaluate, points, _PERIOD_STEPS):
            previous = None
            continue
        turns = _turn_count([y for _, y in points])
        if turns < 2:
            if previous is not None:
                return previous  # the finer run is too short to see two turns of it
            continue
        estimate = 2.0 * _PERIOD_STEPS * step / turns
        if previous is not None and 2.0 / 3.0 <= estimate / previous <= 1.5:
            return (estimate + previous) / 2.0
        previous = estimate
    return None


def _turn_count(ys: List[float]) -> int:
    """How many times the samples change direction (rising to falling or back)."""
    span = max(ys) - min(ys)
    signs = [second > first for first, second in zip(ys, ys[1:]) if abs(second - first) > 1e-9 * span]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b) if span > 0 else 0


def _pole_between_samples(evaluate: Callable[[float], Any], points: List[Tuple[float, float]]) -> bool:
    """Look between the samples next to the biggest values: a pole there (1/x, 1/x^2) blows up."""
    largest = max(abs(y) for _, y in points)
    peaks = sorted(range(len(points)), key=lambda i: abs(points[i][1]), reverse=True)[:_PEAKS_CHECKED]
    for index in peaks:
        for neighbour in (index - 1, index + 1):
            if not 0 <= neighbour < len(points):
                continue
            try:
                value = float(evaluate((points[index][0] + points[neighbour][0]) / 2.0))
            except Exception:
                return True
            if math.isinf(value) or abs(value) > _SPIKE_RATIO * max(largest, 1e-300):
                return True
    return False


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


def _sample(
    point_at: Callable[[float], Any], low: float, high: float, count: int = CURVE_SAMPLES
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for index in range(count):
        t = low + (high - low) * index / (count - 1)
        try:
            x, y = point_at(t)
            x, y = float(x), float(y)
        except Exception:
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    return points


def _definition(item: Dict[str, Any]) -> str:
    args = item.get("args") if isinstance(item.get("args"), dict) else {}
    return json.dumps({k: v for k, v in args.items() if k not in _IGNORED_ARGS}, sort_keys=True, default=str)


def _items(state: Dict[str, Any], bucket: str) -> List[Dict[str, Any]]:
    items = state.get(bucket)
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


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
