"""
View notes: tell the model when the drawings are too small on screen or outside the view.

The app never moves or zooms the view on its own. When the shapes are hard to see,
each canvas the model is shown (a user message's <canvas> block, or the
[canvas changes] after a tool batch) carries one "View note:" line, so the model
can offer the user a zoom; the system prompt tells it to change the view only when
the user asks or agrees. ``view_note(previous, current)`` compares the canvas with
the one the model saw before, so a note is given when a problem appears, not every
time it is still there.

Measurement. The view is a uniform linear map (``CoordinateMapper.math_to_screen``),
so screen sizes follow exactly from the view bounds and the canvas size in CSS
pixels, which the client adds to the prompt's state (``canvas_size_px``). Bounded
objects are measured from the state: points, segments, vectors, circles, ellipses,
arcs (by their whole circle), text labels, bars, bar charts and function-bounded
shaded areas with explicit bounds. Function graphs and curves are measured by the
client (``curve_extents``: each one's sampled box in math units; a function without
both bounds is sampled over the visible x range only and marked ``clipped``).

Problems, each reported only on a transition (see ``_view_note``):

1. outside: less than half of the drawing (by samples along the outlines) is in the view;
2. new or changed objects that lie entirely outside the view;
3. too small: the whole drawing spans fewer than max(40 px, 3% of the canvas's smaller side);
4. too small: new or changed shapes, with the small shapes next to them, span fewer than that,
   while something larger elsewhere keeps the whole drawing big;
5. flat: a function graph varies by fewer than 16 px vertically on screen.

Pure module (no Flask, no I/O); never raises from ``view_note``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from static.canvas_state_formatter import CANVAS_SIZE_KEY, CURVE_EXTENTS_KEY, VIEW_SIGNIFICANT_DIGITS, format_number

_logger = logging.getLogger("mathud")

VIEW_NOTE_PREFIX = "View note:"
# Too small: the larger side on screen is under max(TINY_PX, TINY_CANVAS_FRACTION x the canvas's
# smaller side). Point labels are 14 px text, so below about 40 px the labels of neighbouring
# points cover each other and the shape; the fraction keeps the rule proportional on very large
# canvases (smaller side above about 1330 px).
TINY_PX = 40.0
TINY_CANVAS_FRACTION = 0.03
# Without a canvas size (older clients): under this fraction of the view's smaller side,
# which is 40 px of an 800 px canvas.
TINY_VIEW_FRACTION = 0.05
# Flat: a function graph whose whole vertical variation on screen is under about one label
# height reads as a straight line.
FLAT_PX = 16.0
# The suggested view for a flat graph makes its variation fill this share of the view's height.
FLAT_FILL = 0.25
# Outside: less than this share of the drawing's outline samples lies in the view.
MIN_VISIBLE_FRACTION = 0.5
# A problem the previous canvas already had is reported again only when the affected
# shapes' size on screen changed by more than this factor.
REPEAT_SIZE_RATIO = 2.0
# The suggested view shows the target box enlarged this much around its centre.
SUGGESTED_VIEW_MARGIN = 1.25
# Extents below the grid's finest spacing (Cartesian2Axis.min_tick_spacing) or this fraction of
# the coordinates' magnitude count as a point: nothing to zoom into.
DEFAULT_MIN_EXTENT = 1e-6
_RELATIVE_MIN_EXTENT = 1e-9
# Coordinates this large are printed in scientific notation.
_SCIENTIFIC_ABOVE = 1e15
_MAX_NOTE_NAMES = 3
# Outline samples per segment, circle or box edge, for the visible fraction.
_SEGMENT_SAMPLES = 16
_ROUND_SAMPLES = 32
# Two boxes are the same when no edge moved by more than this fraction of their size.
_SAME_BOX_RELATIVE_TOLERANCE = 1e-6

Point2D = Tuple[float, float]
Key = Tuple[str, str]
JsonDict = Dict[str, Any]


# --------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in math units."""

    left: float
    right: float
    bottom: float
    top: float

    @staticmethod
    def around(x: float, y: float, half_width: float = 0.0, half_height: float = 0.0) -> "Box":
        return Box(x - half_width, x + half_width, y - half_height, y + half_height)

    @staticmethod
    def of_points(points: Sequence[Point2D]) -> "Box":
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        return Box(min(xs), max(xs), min(ys), max(ys))

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def size(self) -> float:
        return max(self.width, self.height)

    @property
    def center(self) -> Point2D:
        return ((self.left + self.right) / 2.0, (self.bottom + self.top) / 2.0)

    @property
    def magnitude(self) -> float:
        return max(abs(self.left), abs(self.right), abs(self.bottom), abs(self.top))

    def union(self, other: "Box") -> "Box":
        return Box(
            min(self.left, other.left),
            max(self.right, other.right),
            min(self.bottom, other.bottom),
            max(self.top, other.top),
        )

    def intersects(self, other: "Box") -> bool:
        return (
            self.left <= other.right
            and other.left <= self.right
            and self.bottom <= other.top
            and other.bottom <= self.top
        )

    def gap_to(self, other: "Box") -> float:
        """The distance between the two boxes (0 when they touch or overlap)."""
        dx = max(0.0, other.left - self.right, self.left - other.right)
        dy = max(0.0, other.bottom - self.top, self.bottom - other.top)
        return math.hypot(dx, dy)

    def contains(self, point: Point2D) -> bool:
        return self.left <= point[0] <= self.right and self.bottom <= point[1] <= self.top

    def same_as(self, other: "Box") -> bool:
        tolerance = _SAME_BOX_RELATIVE_TOLERANCE * max(self.size, other.size)
        return all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=tolerance)
            for a, b in zip(
                (self.left, self.right, self.bottom, self.top), (other.left, other.right, other.bottom, other.top)
            )
        )

    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.left, self.right, self.bottom, self.top))

    def outline(self) -> List[Point2D]:
        corners = [(self.left, self.bottom), (self.right, self.bottom), (self.right, self.top), (self.left, self.top)]
        samples: List[Point2D] = []
        for index, start in enumerate(corners):
            samples.extend(_line_samples(start, corners[(index + 1) % 4], _SEGMENT_SAMPLES // 4, closed=False))
        return samples


def _line_samples(start: Point2D, end: Point2D, count: int, closed: bool = True) -> List[Point2D]:
    steps = max(1, count)
    last = steps if closed else steps - 1
    return [
        (start[0] + (end[0] - start[0]) * i / steps, start[1] + (end[1] - start[1]) * i / steps)
        for i in range(last + 1)
    ]


def _round_samples(cx: float, cy: float, rx: float, ry: float, angle: float = 0.0) -> List[Point2D]:
    cos, sin = math.cos(angle), math.sin(angle)
    samples = []
    for i in range(_ROUND_SAMPLES):
        t = 2.0 * math.pi * i / _ROUND_SAMPLES
        x, y = rx * math.cos(t), ry * math.sin(t)
        samples.append((cx + x * cos - y * sin, cy + x * sin + y * cos))
    return samples


@dataclass(frozen=True)
class Shape:
    """One measured object: its box, a signature to spot changes, and samples along its outline."""

    box: Box
    signature: str
    outline: Tuple[Point2D, ...]
    # A function graph without both bounds, sampled over the view's x range only.
    clipped: bool = False
    # A function graph (flat check).
    graph: bool = False
    # The points it is drawn through (segment ends, polygon vertices, circle centre).
    points: Tuple[str, ...] = ()


# --------------------------------------------------------------------------- measuring a state


@dataclass(frozen=True)
class ViewSummary:
    """Where the measurable shapes are relative to the view."""

    view: Box
    canvas_px: Optional[Tuple[float, float]]
    shapes: Dict[Key, Shape]
    min_tick_spacing: float

    @property
    def content(self) -> Optional[Box]:
        """The bounding box of every shape except graphs clipped to the view."""
        return _union_box(shape.box for shape in self.shapes.values() if not shape.clipped)

    @property
    def pixels_per_unit(self) -> Optional[float]:
        return self.canvas_px[0] / self.view.width if self.canvas_px else None

    def size_px(self, box: Box) -> Optional[Tuple[float, float]]:
        scale = self.pixels_per_unit
        return None if scale is None else (box.width * scale, box.height * scale)

    def is_point_like(self, box: Box) -> bool:
        """Too small to zoom into: below the grid's finest spacing or float resolution."""
        return box.size <= max(self.min_tick_spacing, _RELATIVE_MIN_EXTENT * box.magnitude)

    def tiny_threshold(self) -> Optional[float]:
        if self.canvas_px is None:
            return None
        return max(TINY_PX, TINY_CANVAS_FRACTION * min(self.canvas_px))

    def is_tiny(self, box: Optional[Box]) -> bool:
        """True when ``box`` has a size but it is too small to read on screen."""
        if box is None or self.is_point_like(box):
            return False
        threshold = self.tiny_threshold()
        if threshold is not None and self.pixels_per_unit is not None:
            return box.size * self.pixels_per_unit < threshold
        return box.size < TINY_VIEW_FRACTION * min(self.view.width, self.view.height)

    def screen_size(self, box: Box) -> float:
        """The box's larger side in pixels (in view fractions x 1000 without a canvas size)."""
        scale = self.pixels_per_unit or 1000.0 / min(self.view.width, self.view.height)
        return box.size * scale

    def is_flat(self, shape: Shape) -> bool:
        """A function graph wide enough to see but with a few pixels of vertical variation."""
        if not shape.graph or self.is_point_like(Box(0.0, 0.0, shape.box.bottom, shape.box.top)):
            return False
        scale = self.pixels_per_unit or 1000.0 / min(self.view.width, self.view.height)
        visible_width = min(shape.box.right, self.view.right) - max(shape.box.left, self.view.left)
        return shape.box.height * scale < FLAT_PX and visible_width * scale >= FLAT_PX

    def is_outside(self, shape: Shape) -> bool:
        return not shape.box.intersects(self.view)

    def visible_fraction(self) -> float:
        """Share of the outline samples of every shape (graphs clipped to the view aside) in the view."""
        total = inside = 0
        for shape in self.shapes.values():
            if shape.clipped:
                continue
            total += len(shape.outline)
            inside += sum(1 for point in shape.outline if self.view.contains(point))
        return inside / total if total else 1.0


def summarize_view(state: Any) -> Optional[ViewSummary]:
    """Measure the state's shapes against its view; None without a usable view."""
    if not isinstance(state, Mapping):
        return None
    view = _view_box(state)
    if view is None:
        return None
    spacing = _as_float(state.get("min_tick_spacing"))
    min_extent = spacing if spacing is not None and spacing > 0 else DEFAULT_MIN_EXTENT
    return ViewSummary(view, _canvas_px(state), measure_shapes(state), min_extent)


def measure_shapes(state: Mapping[str, Any]) -> Dict[Key, Shape]:
    """Every measurable object of the state, keyed (bucket, name) like the rendered entries."""
    shapes: Dict[Key, Shape] = {}
    positions: Dict[str, Point2D] = {}
    for key, item in _keyed_items(state, "Points"):
        xy = _xy(_args(item).get("position"))
        if xy is not None:
            shapes[key] = Shape(Box.around(*xy), _signature(item), (xy,))
            positions.setdefault(key[1], xy)
    for key, item in _keyed_items(state, "Labels"):
        xy = _xy(_args(item).get("position"))
        if xy is not None:
            shapes[key] = Shape(Box.around(*xy), _signature(item), (xy,))
    for bucket, measure in _MEASURES:
        for key, item in _keyed_items(state, bucket):
            shape = measure(_args(item), positions, _signature(item))
            if shape is not None and shape.box.is_finite():
                shapes[key] = shape
    for bucket in _POLYGON_BUCKETS:
        for key, item in _keyed_items(state, bucket):
            shape = _polygon_shape(_args(item), positions, _signature(item))
            if shape is not None and shape.box.is_finite():
                shapes[key] = shape
    curves = _curve_shapes(state)
    shapes.update(curves)
    for key, item in _keyed_items(state, "FunctionsBoundedColoredAreas"):
        shape = _function_area_shape(_args(item), curves, _signature(item))
        if shape is not None and shape.box.is_finite():
            shapes[key] = shape
    return shapes


def _segment_shape(first: Any, second: Any) -> Callable[[JsonDict, Mapping[str, Point2D], str], Optional[Shape]]:
    def measure(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
        start, end = positions.get(str(args.get(first))), positions.get(str(args.get(second)))
        if start is None or end is None:
            return None
        outline = tuple(_line_samples(start, end, _SEGMENT_SAMPLES))
        # The signature includes the endpoints, so a segment whose point moved counts as changed.
        members = (str(args.get(first)), str(args.get(second)))
        return Shape(Box.of_points([start, end]), f"{signature}{start}{end}", outline, points=members)

    return measure


_POLYGON_BUCKETS = (
    "Triangles",
    "Rectangles",
    "Quadrilaterals",
    "Pentagons",
    "Hexagons",
    "Heptagons",
    "Octagons",
    "Nonagons",
    "Decagons",
    "GenericPolygons",
)


def _polygon_shape(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    names: List[Any] = []
    for list_key in ("points", "vertices"):
        if isinstance(args.get(list_key), list):
            names = list(args[list_key])
            break
    else:
        numbered = [(int(str(key)[1:]), value) for key, value in args.items() if re.fullmatch(r"p\d+", str(key))]
        names = [value for _, value in sorted(numbered, key=lambda pair: pair[0])]
    vertices = [positions.get(str(name)) for name in names]
    if len(vertices) < 2 or any(v is None for v in vertices):
        return None
    corners = [v for v in vertices if v is not None]
    outline: List[Point2D] = []
    for index, start in enumerate(corners):
        outline.extend(_line_samples(start, corners[(index + 1) % len(corners)], _SEGMENT_SAMPLES, closed=False))
    return Shape(Box.of_points(corners), f"{signature}{corners}", tuple(outline), points=tuple(map(str, names)))


def _circle_shape(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    center = positions.get(str(args.get("center")))
    radius = _as_float(args.get("radius"))
    if center is None or radius is None or radius < 0:
        return None
    outline = tuple(_round_samples(center[0], center[1], radius, radius))
    box = Box.around(center[0], center[1], radius, radius)
    return Shape(box, f"{signature}{center}", outline, points=(str(args.get("center")),))


def _ellipse_shape(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    center = positions.get(str(args.get("center")))
    rx, ry = _as_float(args.get("radius_x")), _as_float(args.get("radius_y"))
    if center is None or rx is None or ry is None:
        return None
    angle = math.radians(_as_float(args.get("rotation_angle")) or 0.0)
    cos, sin = math.cos(angle), math.sin(angle)
    box = Box.around(center[0], center[1], math.hypot(rx * cos, ry * sin), math.hypot(rx * sin, ry * cos))
    return Shape(box, f"{signature}{center}", tuple(_round_samples(center[0], center[1], rx, ry, angle)))


def _arc_shape(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    cx, cy, radius = _as_float(args.get("center_x")), _as_float(args.get("center_y")), _as_float(args.get("radius"))
    if cx is None or cy is None or radius is None or radius < 0:
        return None
    return Shape(Box.around(cx, cy, radius, radius), signature, tuple(_round_samples(cx, cy, radius, radius)))


def _box_shape(box: Optional[Box], signature: str, clipped: bool = False, graph: bool = False) -> Optional[Shape]:
    return None if box is None else Shape(box, signature, tuple(box.outline()), clipped=clipped, graph=graph)


def _bar_shape(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    left, right, bottom, top = (_as_float(args.get(key)) for key in ("x_left", "x_right", "y_bottom", "y_top"))
    if left is None or right is None or bottom is None or top is None:
        return None
    return _box_shape(Box(min(left, right), max(left, right), min(bottom, top), max(bottom, top)), signature)


def _bars_plot_shape(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    """Bars of width w every w + spacing from x_start, each from y_base to y_base + value."""
    values = [v for v in (_as_float(value) for value in _as_list(args.get("values"))) if v is not None]
    if not values:
        return None
    width = _as_float(args.get("bar_width")) or 1.0
    spacing = _as_float(args.get("bar_spacing"))
    spacing = 0.2 if spacing is None else spacing
    x_start = _as_float(args.get("x_start")) or 0.0
    y_base = _as_float(args.get("y_base")) or 0.0
    right = x_start + (len(values) - 1) * (width + spacing) + width
    return _box_shape(Box(x_start, right, y_base + min(0.0, *values), y_base + max(0.0, *values)), signature)


_Measure = Callable[[JsonDict, Mapping[str, Point2D], str], Optional[Shape]]
_MEASURES: Tuple[Tuple[str, _Measure], ...] = (
    ("Segments", _segment_shape("p1", "p2")),
    ("Vectors", _segment_shape("origin", "tip")),
    ("Circles", _circle_shape),
    ("Ellipses", _ellipse_shape),
    ("CircleArcs", _arc_shape),
    ("Bars", _bar_shape),
    ("BarsPlots", _bars_plot_shape),
)
_GRAPH_BUCKETS = ("Functions", "PiecewiseFunctions")
_CURVE_BUCKETS = _GRAPH_BUCKETS + ("ParametricFunctions",)


def _curve_shapes(state: Mapping[str, Any]) -> Dict[Key, Shape]:
    """Function graphs and curves, from the boxes the client sampled (``curve_extents``)."""
    extents = state.get(CURVE_EXTENTS_KEY)
    if not isinstance(extents, Mapping):
        return {}
    shapes: Dict[Key, Shape] = {}
    for bucket in _CURVE_BUCKETS:
        measured = extents.get(bucket)
        if not isinstance(measured, Mapping):
            continue
        for key, item in _keyed_items(state, bucket):
            entry = measured.get(key[1])
            if not isinstance(entry, Mapping):
                continue
            box = _extent_box(entry.get("box"))
            if box is None:
                continue
            clipped = bool(entry.get("clipped"))
            shape = _box_shape(box, _signature(item), clipped=clipped, graph=bucket in _GRAPH_BUCKETS)
            if shape is not None:
                shapes[key] = shape
    return shapes


def _extent_box(raw: Any) -> Optional[Box]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    values = [_as_float(v) for v in raw]
    left, right, bottom, top = values
    if left is None or right is None or bottom is None or top is None or right < left or top < bottom:
        return None
    return Box(left, right, bottom, top)


def _function_area_shape(args: JsonDict, curves: Mapping[Key, Shape], signature: str) -> Optional[Shape]:
    """An area between functions over explicit bounds: those bounds, by the functions' y ranges."""
    left, right = _as_float(args.get("left_bound")), _as_float(args.get("right_bound"))
    if left is None or right is None or right < left:
        return None
    ys: List[float] = []
    for ref in (args.get("func1"), args.get("func2")):
        name = str(ref) if ref is not None else "x_axis"
        if name == "x_axis":
            ys.append(0.0)
            continue
        constant = _as_float(ref) if not isinstance(ref, str) else _as_float(_constant_of(name))
        if constant is not None:
            ys.append(constant)
            continue
        curve = next((curves[(bucket, name)] for bucket in _GRAPH_BUCKETS if (bucket, name) in curves), None)
        if curve is None:
            return None
        ys.extend((curve.box.bottom, curve.box.top))
    return _box_shape(Box(left, right, min(ys), max(ys)), signature)


def _constant_of(name: str) -> Optional[float]:
    """The value of a ``y_<number>`` boundary name (FunctionsBoundedColoredArea._get_function_name)."""
    if not name.startswith("y_"):
        return None
    try:
        return float(name[2:])
    except ValueError:
        return None


# --------------------------------------------------------------------------- the note


@dataclass(frozen=True)
class _Problem:
    text: str
    target: Box
    flat: bool = False


def view_note(previous: Optional[Mapping[str, Any]], current: Mapping[str, Any]) -> Optional[str]:
    """One "View note:" line when the shapes in ``current`` are hard to see, else None.

    ``previous`` is the last canvas the model was shown (None at the start of a
    conversation). A problem is reported when it appears: not when the previous
    canvas already had it at about the same size on screen, and never because of the
    view alone (the user's own pan or zoom). Never raises.
    """
    try:
        note = _view_note(previous, current)
    except Exception:
        _logger.warning("Could not measure the canvas view; sending no view note", exc_info=True)
        return None
    # Object names go into the note; it must stay one line.
    return None if note is None else note.replace("\r", " ").replace("\n", " ")


def _view_note(previous: Optional[Mapping[str, Any]], current: Mapping[str, Any]) -> Optional[str]:
    now = summarize_view(current)
    if now is None or not now.shapes:
        return None
    first = not isinstance(previous, Mapping)
    before_shapes = measure_shapes(previous) if isinstance(previous, Mapping) else {}
    before = summarize_view(previous)
    changed = [
        key
        for key, shape in now.shapes.items()
        if key not in before_shapes
        or before_shapes[key].signature != shape.signature
        or (not shape.clipped and not before_shapes[key].box.same_as(shape.box))
    ]
    for find in (_outside_problem, _changed_outside_problem, _tiny_problem, _tiny_cluster_problem, _flat_problem):
        problem = find(now, before, before_shapes, changed, first)
        if problem is not None:
            return (
                f"{VIEW_NOTE_PREFIX} {problem.text}. Offer to {_suggestion(now, problem)}; "
                "don't change the view unless the user agrees."
            )
    return None


def _drawing_changed(now: ViewSummary, before_shapes: Mapping[Key, Shape]) -> bool:
    content = now.content
    before = _union_box(shape.box for shape in before_shapes.values() if not shape.clipped)
    return content is not None and (before is None or not before.same_as(content))


def _seen_before(now: ViewSummary, before: Optional[ViewSummary], box: Box, before_box: Optional[Box]) -> bool:
    """True when the previous canvas showed ``before_box`` at about the size ``box`` has now."""
    if before is None or before_box is None:
        return False
    ratio = now.screen_size(box) / max(before.screen_size(before_box), 1e-300)
    return 1.0 / REPEAT_SIZE_RATIO <= ratio <= REPEAT_SIZE_RATIO


def _outside_problem(
    now: ViewSummary, before: Optional[ViewSummary], before_shapes: Mapping[Key, Shape], changed: List[Key], first: bool
) -> Optional[_Problem]:
    content = now.content
    if content is None or not _drawing_changed(now, before_shapes):
        return None
    fraction = now.visible_fraction()
    # At the start of a conversation a view zoomed into part of a big drawing is the user's choice.
    if fraction >= MIN_VISIBLE_FRACTION or (first and fraction > 0):
        return None
    if before is not None and before.visible_fraction() < MIN_VISIBLE_FRACTION:
        if _seen_before(now, before, content, before.content):
            return None
    where = _shapes_and_view(now, content)
    if fraction <= 0:
        return _Problem(f"the shapes are entirely outside the view ({where})", content)
    percent = "<1" if fraction < 0.01 else f"~{round(fraction * 100)}"
    return _Problem(f"only {percent}% of the drawing is inside the view ({where})", content)


def _changed_outside_problem(
    now: ViewSummary, before: Optional[ViewSummary], before_shapes: Mapping[Key, Shape], changed: List[Key], first: bool
) -> Optional[_Problem]:
    if first:
        return None
    # A point drawn through by a shape on screen (a segment crossing the view) is part of the visible drawing.
    attached = {name for shape in now.shapes.values() if not now.is_outside(shape) for name in shape.points}
    outside = []
    for key in changed:
        if not now.is_outside(now.shapes[key]) or (key[0] == "Points" and key[1] in attached):
            continue
        # An object that was already outside the view (e.g. one the user scrolled away from) is not news.
        if key in before_shapes and before is not None and before.is_outside(before_shapes[key]):
            continue
        outside.append(key)
    if not outside:
        return None
    verb = "is" if len(outside) == 1 else "are"
    # Suggest a view of the whole drawing; graphs clipped to the view have no box of their own beyond it.
    target = now.content or _union_box(now.shapes[key].box for key in outside)
    assert target is not None
    text = f"new or changed {_names(outside)} {verb} outside the view (view {_view_ranges(now.view)})"
    return _Problem(text, target)


def _tiny_problem(
    now: ViewSummary, before: Optional[ViewSummary], before_shapes: Mapping[Key, Shape], changed: List[Key], first: bool
) -> Optional[_Problem]:
    content = now.content
    if content is None or not now.is_tiny(content) or not _drawing_changed(now, before_shapes):
        return None
    if before is not None and before.is_tiny(before.content):
        if _seen_before(now, before, content, before.content):
            return None
    return _Problem(f"the shapes span only {_extent_text(now, content)} ({_shapes_and_view(now, content)})", content)


def _tiny_cluster_problem(
    now: ViewSummary, before: Optional[ViewSummary], before_shapes: Mapping[Key, Shape], changed: List[Key], first: bool
) -> Optional[_Problem]:
    """New small shapes next to nothing but other small shapes, in a drawing that is large overall."""
    bounded = [key for key in changed if not now.shapes[key].clipped]
    new_box = _union_box(now.shapes[key].box for key in bounded)
    if first or new_box is None or now.content is None or now.is_tiny(now.content):
        return None
    reach = _tiny_reach(now)
    neighbours = [
        key
        for key, shape in now.shapes.items()
        if key not in changed
        and not shape.clipped
        and (now.is_point_like(shape.box) or now.is_tiny(shape.box))
        and shape.box.gap_to(new_box) <= reach
    ]
    cluster = new_box
    for key in neighbours:
        cluster = cluster.union(now.shapes[key].box)
    if not now.is_tiny(cluster):
        return None
    old = _union_box(before_shapes[key].box for key in neighbours if key in before_shapes)
    if before is not None and old is not None and before.is_tiny(old) and _seen_before(now, before, cluster, old):
        return None
    names = _names([key for key in bounded if _NAME_ORDER.get(key[0]) is None] or bounded)
    text = (
        f"the new or changed shapes ({names}) span only {_extent_text(now, cluster)} ({_shapes_and_view(now, cluster)})"
    )
    return _Problem(text, cluster)


def _flat_problem(
    now: ViewSummary, before: Optional[ViewSummary], before_shapes: Mapping[Key, Shape], changed: List[Key], first: bool
) -> Optional[_Problem]:
    candidates = list(now.shapes) if first else changed
    for key in candidates:
        shape = now.shapes[key]
        if not now.is_flat(shape):
            continue
        old = before_shapes.get(key)
        if old is not None and before is not None and before.is_flat(old):
            if _seen_before(now, before, _height_box(shape.box), _height_box(old.box)):
                continue
        size = now.size_px(shape.box)
        variation = f"~{_pixels(size[1])} px" if size is not None else "a sliver"
        box = shape.box
        visible = Box(max(box.left, now.view.left), min(box.right, now.view.right), box.bottom, box.top)
        step = _step_for(Box(0, 0, box.bottom, box.top), now.view)
        text = (
            f"{key[1]} varies only {variation} vertically on screen (y {_span(box.bottom, box.top, step)} "
            f"over x {_number(visible.left, VIEW_SIGNIFICANT_DIGITS)}..{_number(visible.right, VIEW_SIGNIFICANT_DIGITS)}; "
            f"view {_view_ranges(now.view)})"
        )
        return _Problem(text, visible, flat=True)
    return None


def _height_box(box: Box) -> Box:
    return Box(0.0, 0.0, box.bottom, box.top)


def _tiny_reach(summary: ViewSummary) -> float:
    """How close (math units) another small shape must be to count as part of the same cluster."""
    threshold = summary.tiny_threshold()
    if threshold is not None and summary.pixels_per_unit:
        return threshold / summary.pixels_per_unit
    return TINY_VIEW_FRACTION * min(summary.view.width, summary.view.height)


# --------------------------------------------------------------------------- text


# Named first in a note: whole shapes before their edges, edges before points.
_NAME_ORDER = {"Points": 2, "Labels": 2, "Segments": 1, "Vectors": 1}


def _names(keys: Sequence[Key]) -> str:
    keys = sorted(keys, key=lambda key: _NAME_ORDER.get(key[0], 0))
    names = ", ".join(key[1] for key in keys[:_MAX_NOTE_NAMES])
    if len(keys) > _MAX_NOTE_NAMES:
        names += f" (+{len(keys) - _MAX_NOTE_NAMES} more)"
    return names


def _extent_text(summary: ViewSummary, box: Box) -> str:
    size = summary.size_px(box)
    if size is not None:
        return f"~{_pixels(size[0])}x{_pixels(size[1])} px on screen"
    share = max(box.width / summary.view.width, box.height / summary.view.height) * 100
    return f"~{format_number(share, 2)}% of the view"


def _pixels(value: float) -> str:
    return "<1" if value < 1 else str(round(value))


def _shapes_and_view(summary: ViewSummary, box: Box) -> str:
    step = _step_for(box, summary.view)
    if summary.is_point_like(box):
        x, y = box.center
        return f"shapes at ({_number(_snap(x, step))}, {_number(_snap(y, step))}); view {_view_ranges(summary.view)}"
    return (
        f"shapes x {_span(box.left, box.right, step)}, y {_span(box.bottom, box.top, step)}; "
        f"view {_view_ranges(summary.view)}"
    )


def _step_for(box: Box, view: Box) -> float:
    return _coordinate_step(box.size / 2.0) or _coordinate_step(view.width / 2.0)


def _view_ranges(view: Box) -> str:
    """The view bounds rounded like the canvas view line (4 significant digits)."""
    return (
        f"x {_number(view.left, VIEW_SIGNIFICANT_DIGITS)}..{_number(view.right, VIEW_SIGNIFICANT_DIGITS)}, "
        f"y {_number(view.bottom, VIEW_SIGNIFICANT_DIGITS)}..{_number(view.top, VIEW_SIGNIFICANT_DIGITS)}"
    )


def _suggestion(summary: ViewSummary, problem: _Problem) -> str:
    """A view showing ``problem.target``: zoom when it is tiny, flat or too big, else just move the view."""
    view, target = summary.view, problem.target
    aspect = view.height / view.width
    current = view.width / 2.0
    if problem.flat:
        # The graph's vertical variation fills FLAT_FILL of the view's height.
        half = _nice_ceil(target.height / (2.0 * FLAT_FILL) / aspect)
        pan_only = False
    else:
        fit = max(target.width / 2.0, target.height / 2.0 / aspect) * SUGGESTED_VIEW_MARGIN
        point_like = summary.is_point_like(target)
        pan_only = point_like or (fit <= current and not summary.is_tiny(target))
        half = _nice_ceil(current if pan_only else fit)
    step = _coordinate_step(half)
    cx, cy = (_snap(value, step) for value in target.center)
    shown = Box.around(cx, cy, half, half * aspect)
    return (
        f"{'move the view' if pan_only else 'zoom'} to about "
        f"x {_span(shown.left, shown.right, step)}, y {_span(shown.bottom, shown.top, step)} "
        f"(zoom center_x={_number(cx)}, center_y={_number(cy)}, range_val={_number(half)}, range_axis=x)"
    )


def _nice_ceil(value: float) -> float:
    """Round up to two significant digits (4.13 -> 4.2, 0.0123 -> 0.013)."""
    exponent = math.floor(math.log10(value))
    unit = 10.0 ** (exponent - 1)
    return math.ceil(value / unit - 1e-9) * unit


def _coordinate_step(half_size: float) -> float:
    """A rounding step for coordinates a tenth of the magnitude of ``half_size`` (0 for 0)."""
    if not half_size > 0 or not math.isfinite(half_size):
        return 0.0
    return 10.0 ** (math.floor(math.log10(half_size)) - 1)


def _snap(value: float, step: float) -> float:
    return round(value / step) * step if step > 0 else value


def _span(low: float, high: float, step: float) -> str:
    return f"{_number(_snap(low, step))}..{_number(_snap(high, step))}"


def _number(value: float, significant_digits: int = 6) -> str:
    """format_number, but huge values in short scientific notation instead of every digit."""
    if math.isfinite(value) and abs(value) >= _SCIENTIFIC_ABOVE:
        mantissa, exponent = f"{value:.3e}".split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    return str(format_number(value, significant_digits))


# --------------------------------------------------------------------------- state access


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [] if value is None else [value]


def _args(item: Mapping[str, Any]) -> JsonDict:
    args = item.get("args")
    return args if isinstance(args, dict) else {}


def _xy(position: Any) -> Optional[Point2D]:
    if not isinstance(position, Mapping):
        return None
    x, y = _as_float(position.get("x")), _as_float(position.get("y"))
    return (x, y) if x is not None and y is not None else None


def _keyed_items(state: Mapping[str, Any], bucket: str) -> List[Tuple[Key, JsonDict]]:
    items = state.get(bucket)
    if not isinstance(items, list):
        return []
    keyed: List[Tuple[Key, JsonDict]] = []
    occurrences: Dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        occurrence = occurrences.get(name, 0)
        occurrences[name] = occurrence + 1
        keyed.append(((bucket, name if occurrence == 0 else f"{name}#{occurrence + 1}"), item))
    return keyed


def _signature(item: Mapping[str, Any]) -> str:
    try:
        return json.dumps(_args(item), sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(_args(item))


def _union_box(boxes: Iterable[Box]) -> Optional[Box]:
    result: Optional[Box] = None
    for box in boxes:
        result = box if result is None else result.union(box)
    return result


def _view_box(state: Mapping[str, Any]) -> Optional[Box]:
    visibility = state.get("Cartesian_System_Visibility")
    if not isinstance(visibility, dict):
        return None
    left, right, bottom, top = (
        _as_float(visibility.get(key)) for key in ("left_bound", "right_bound", "bottom_bound", "top_bound")
    )
    if left is None or right is None or bottom is None or top is None or right <= left or top <= bottom:
        return None
    return Box(left, right, bottom, top)


def _canvas_px(state: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    size = state.get(CANVAS_SIZE_KEY)
    if not isinstance(size, dict):
        return None
    width, height = _as_float(size.get("width")), _as_float(size.get("height"))
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    return (width, height)
