"""
View notes: tell the model when the drawings are too small on screen or outside the view.

The app never moves or zooms the view on its own. When the shapes are hard to see,
the canvas the model is shown (a user message's <canvas> block, or the
[canvas changes] after a tool batch) carries one "View note:" line, so the model
can offer the user a zoom; the system prompt tells it to change the view only when
the user asks or agrees.

A missed note is far better than a false or useless one, so every rule stays silent
when it is unsure:

1. outside: nothing of the drawing is in the view;
2. new or changed objects lie entirely outside the view;
3. too small: the whole drawing spans fewer than max(40 px, 3% of the canvas's smaller side);
4. too small: new or changed shapes with a real extent (not just points or labels), with
   the small shapes right next to them, span fewer than that inside a larger drawing;
5. flat: a function graph that turns (like sin) varies by fewer than 16 px vertically,
   and has no vertical asymptote or spike that would make it look flat by sampling.

Each problem is reported when it appears: a change of view alone (the user's own pan
or zoom) never brings a note, a problem the previous canvas already had comes back
only when it got worse (the shapes shrank to less than half their size), and a
``ViewNoteMemory`` keeps one conversation from hearing the same note twice (a redo).
At the first message of a conversation the view is the user's choice, so only a
drawing with nothing at all on screen is reported.

Measurement. The view is a uniform linear map (``CoordinateMapper.math_to_screen``),
so screen sizes follow exactly from the view bounds and the canvas size in CSS
pixels, which the client adds to the prompt's state (``canvas_size_px``). Bounded
objects are measured from the state; function graphs and curves come from the boxes
the client samples (``curve_extents``, see static/client/prompt_canvas_state.py).

Pure module (no Flask, no I/O); ``view_note`` never raises.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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
# Flat: a turning graph whose whole vertical variation on screen is under about one label
# height reads as a straight line.
FLAT_PX = 16.0
# The suggested view for a flat graph makes its variation fill this share of the view's height.
FLAT_FILL = 0.25
# A problem the previous canvas already had is reported again only when the shapes shrank
# (in math units) to less than this share of their size: it got worse, not just different.
WORSE_SIZE_RATIO = 0.5
# The suggested view shows the target box enlarged this much around its centre.
SUGGESTED_VIEW_MARGIN = 1.25
# Extents below the grid's finest spacing (Cartesian2Axis.min_tick_spacing) or this fraction of
# the coordinates' magnitude count as a point: nothing to zoom into.
DEFAULT_MIN_EXTENT = 1e-6
_RELATIVE_MIN_EXTENT = 1e-9
# Coordinates this large are printed in scientific notation.
_SCIENTIFIC_ABOVE = 1e15
_MAX_NOTE_NAMES = 3
# Two boxes are the same when no edge moved by more than this fraction of their size.
_SAME_BOX_RELATIVE_TOLERANCE = 1e-6
# Args that change how an object looks, not where it is or how big: a recolour is no change here.
_STYLE_ARGS = frozenset(
    {
        "color",
        "opacity",
        "fill_color",
        "stroke_color",
        "fill_opacity",
        "stroke_width",
        "line_width",
        "visible",
        "label",
        "text",
        "font_size",
        "rotation_degrees",
        "render_mode",
        "labels_above",
        "labels_below",
        "label_text",
        "label_above_text",
        "label_below_text",
    }
)
# Buckets whose objects are drawn at one spot (their size on screen is text, not geometry).
_POINT_BUCKETS = frozenset({"Points", "Labels"})

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


@dataclass(frozen=True)
class Shape:
    """One measured object."""

    box: Box
    # The object's geometry (style args left out), to tell a moved or resized object from an unchanged one.
    signature: str
    # A function graph without both bounds, sampled over the view's x range only.
    clipped: bool = False
    # A function graph whose samples change direction (a turning point), and one with a spike or
    # a vertical asymptote in the sampled range: the flat rule needs the first and not the second.
    graph: bool = False
    turns: bool = False
    spiky: bool = False
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

    def scale(self) -> float:
        """Pixels per math unit (a nominal 1000 px for the view's smaller side without a canvas size)."""
        return self.pixels_per_unit or 1000.0 / min(self.view.width, self.view.height)

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

    def is_flat(self, shape: Shape) -> bool:
        """A turning function graph, wide enough to see, with a few pixels of vertical variation."""
        box = shape.box
        if not shape.graph or not shape.turns or shape.spiky or self.is_point_like(Box(0.0, 0.0, box.bottom, box.top)):
            return False
        visible_width = min(box.right, self.view.right) - max(box.left, self.view.left)
        return box.height * self.scale() < FLAT_PX and visible_width * self.scale() >= FLAT_PX

    def is_outside(self, shape: Shape) -> bool:
        return not shape.box.intersects(self.view)


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
    for bucket in ("Points", "Labels"):
        for key, item in _keyed_items(state, bucket):
            xy = _xy(_args(item).get("position"))
            if xy is not None:
                shapes[key] = Shape(Box.around(*xy), _signature(item))
                if bucket == "Points":
                    positions.setdefault(key[1], xy)
    for bucket, measure in _MEASURES:
        for key, item in _keyed_items(state, bucket):
            _add(shapes, key, measure(_args(item), positions, _signature(item)))
    for bucket in _POLYGON_BUCKETS:
        for key, item in _keyed_items(state, bucket):
            _add(shapes, key, _polygon_shape(_args(item), positions, _signature(item)))
    curves = _curve_shapes(state)
    shapes.update(curves)
    for key, item in _keyed_items(state, "FunctionsBoundedColoredAreas"):
        _add(shapes, key, _function_area_shape(_args(item), curves, _signature(item)))
    return shapes


def _add(shapes: Dict[Key, Shape], key: Key, shape: Optional[Shape]) -> None:
    if shape is not None and shape.box.is_finite():
        shapes[key] = shape


_Measure = Callable[[JsonDict, Mapping[str, Point2D], str], Optional[Shape]]


def _segment_shape(first: str, second: str) -> _Measure:
    def measure(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
        start, end = positions.get(str(args.get(first))), positions.get(str(args.get(second)))
        if start is None or end is None:
            return None
        # The signature includes the endpoints, so a segment whose point moved counts as changed.
        members = (str(args.get(first)), str(args.get(second)))
        return Shape(Box.of_points([start, end]), f"{signature}{start}{end}", points=members)

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
    corners = [v for v in vertices if v is not None]
    if len(corners) < 2 or len(corners) != len(vertices):
        return None
    return Shape(Box.of_points(corners), f"{signature}{corners}", points=tuple(map(str, names)))


def _circle_shape(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    center = positions.get(str(args.get("center")))
    radius = _as_float(args.get("radius"))
    if center is None or radius is None or radius < 0:
        return None
    box = Box.around(center[0], center[1], radius, radius)
    return Shape(box, f"{signature}{center}", points=(str(args.get("center")),))


def _ellipse_shape(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    center = positions.get(str(args.get("center")))
    rx, ry = _as_float(args.get("radius_x")), _as_float(args.get("radius_y"))
    if center is None or rx is None or ry is None:
        return None
    angle = math.radians(_as_float(args.get("rotation_angle")) or 0.0)
    cos, sin = math.cos(angle), math.sin(angle)
    box = Box.around(center[0], center[1], math.hypot(rx * cos, ry * sin), math.hypot(rx * sin, ry * cos))
    return Shape(box, f"{signature}{center}", points=(str(args.get("center")),))


def _arc_shape(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    cx, cy, radius = _as_float(args.get("center_x")), _as_float(args.get("center_y")), _as_float(args.get("radius"))
    if cx is None or cy is None or radius is None or radius < 0:
        return None
    return Shape(Box.around(cx, cy, radius, radius), signature)


def _bar_shape(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    left, right, bottom, top = (_as_float(args.get(key)) for key in ("x_left", "x_right", "y_bottom", "y_top"))
    if left is None or right is None or bottom is None or top is None:
        return None
    return Shape(Box(min(left, right), max(left, right), min(bottom, top), max(bottom, top)), signature)


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
    return Shape(Box(x_start, right, y_base + min(0.0, *values), y_base + max(0.0, *values)), signature)


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
            graph = bucket in _GRAPH_BUCKETS
            asymptote = graph and any(
                box.left <= x <= box.right
                for x in (_as_float(v) for v in _as_list(_args(item).get("vertical_asymptotes")))
                if x is not None
            )
            shapes[key] = Shape(
                box,
                _signature(item),
                clipped=bool(entry.get("clipped")),
                graph=graph,
                turns=bool(entry.get("turns")),
                spiky=bool(entry.get("spiky")) or asymptote,
            )
    return shapes


def _extent_box(raw: Any) -> Optional[Box]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    left, right, bottom, top = (_as_float(v) for v in raw)
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
    return Shape(Box(left, right, min(ys), max(ys)), signature)


def _constant_of(name: str) -> Optional[float]:
    """The value of a ``y_<number>`` boundary name (FunctionsBoundedColoredArea._get_function_name)."""
    if not name.startswith("y_"):
        return None
    try:
        return float(name[2:])
    except ValueError:
        return None


# --------------------------------------------------------------------------- the note


@dataclass
class ViewNoteMemory:
    """What one conversation was already told, so the same note never comes twice (e.g. after a redo)."""

    # (kind, box in math units) of every whole-drawing or new-shapes note.
    boxes: List[Tuple[str, Box]] = field(default_factory=list)
    # (key, signature) of every object named in an outside note, and the keys of flat graphs.
    objects: Set[Tuple[Key, str]] = field(default_factory=set)
    flat: Set[Key] = field(default_factory=set)

    def told(self, kind: str, box: Box) -> bool:
        """True when a note of ``kind`` was given for an overlapping box that was not over twice as big."""
        return any(
            seen_kind == kind and seen.intersects(box) and box.size >= WORSE_SIZE_RATIO * seen.size
            for seen_kind, seen in self.boxes
        )


@dataclass(frozen=True)
class _Problem:
    text: str
    target: Box
    remember: Callable[[ViewNoteMemory], None]
    flat: bool = False


@dataclass(frozen=True)
class _Context:
    now: ViewSummary
    before: Optional[ViewSummary]
    before_shapes: Mapping[Key, Shape]
    changed: List[Key]
    first: bool
    memory: ViewNoteMemory


def view_note(
    previous: Optional[Mapping[str, Any]],
    current: Mapping[str, Any],
    memory: Optional[ViewNoteMemory] = None,
) -> Optional[str]:
    """One "View note:" line when the shapes in ``current`` are hard to see, else None.

    ``previous`` is the last canvas the model was shown (None at the start of a
    conversation); ``memory`` holds what this conversation was already told and is
    updated with the note given. Never raises.
    """
    try:
        note = _view_note(previous, current, memory if memory is not None else ViewNoteMemory())
    except Exception:
        _logger.warning("Could not measure the canvas view; sending no view note", exc_info=True)
        return None
    # Object names go into the note; it must stay one line.
    return None if note is None else note.replace("\r", " ").replace("\n", " ")


def _view_note(
    previous: Optional[Mapping[str, Any]], current: Mapping[str, Any], memory: ViewNoteMemory
) -> Optional[str]:
    now = summarize_view(current)
    if now is None or not now.shapes:
        return None
    before_shapes = measure_shapes(previous) if isinstance(previous, Mapping) else {}
    changed = [
        key
        for key, shape in now.shapes.items()
        if key not in before_shapes
        or before_shapes[key].signature != shape.signature
        # A graph clipped to the view changes its box with the view; only its definition counts.
        or (not shape.clipped and not before_shapes[key].box.same_as(shape.box))
    ]
    context = _Context(now, summarize_view(previous), before_shapes, changed, not isinstance(previous, Mapping), memory)
    for find in (_outside_problem, _changed_outside_problem, _tiny_problem, _tiny_new_shapes_problem, _flat_problem):
        problem = find(context)
        if problem is not None:
            problem.remember(memory)
            return (
                f"{VIEW_NOTE_PREFIX} {problem.text}. Offer to {_suggestion(now, problem)}; "
                "don't change the view unless the user agrees."
            )
    return None


def _drawing_changed(context: _Context) -> bool:
    content = context.now.content
    before = _union_box(shape.box for shape in context.before_shapes.values() if not shape.clipped)
    return content is not None and (before is None or not before.same_as(content))


def _not_worse(box: Box, before_box: Optional[Box]) -> bool:
    """The previous canvas had the same problem and the shapes did not shrink to under half their size."""
    return before_box is not None and box.size >= WORSE_SIZE_RATIO * before_box.size


def _remember_box(kind: str, box: Box) -> Callable[[ViewNoteMemory], None]:
    return lambda memory: memory.boxes.append((kind, box))


def _outside_problem(context: _Context) -> Optional[_Problem]:
    """Nothing of the drawing is in the view."""
    now, before = context.now, context.before
    content = now.content
    bounded = [shape for shape in now.shapes.values() if not shape.clipped]
    if content is None or any(not now.is_outside(shape) for shape in bounded):
        return None
    if not (context.first or _drawing_changed(context)) or context.memory.told("outside", content):
        return None
    shown_before = [shape for shape in context.before_shapes.values() if not shape.clipped]
    if before is not None and shown_before and all(before.is_outside(shape) for shape in shown_before):
        return None  # it was already out of sight
    text = f"the shapes are entirely outside the view ({_shapes_and_view(now, content)})"
    return _Problem(text, content, _remember_box("outside", content))


def _changed_outside_problem(context: _Context) -> Optional[_Problem]:
    """New or changed objects that lie entirely outside the view."""
    if context.first:
        return None
    now, before = context.now, context.before
    # A point drawn through by a shape on screen (a segment crossing the view) is part of the visible drawing.
    attached = {name for shape in now.shapes.values() if not now.is_outside(shape) for name in shape.points}
    outside = []
    for key in context.changed:
        shape = now.shapes[key]
        if not now.is_outside(shape) or (key[0] == "Points" and key[1] in attached):
            continue
        # An object that was already outside the view (e.g. one the user scrolled away from) is not news.
        if key in context.before_shapes and before is not None and before.is_outside(context.before_shapes[key]):
            continue
        if (key, shape.signature) in context.memory.objects:
            continue
        outside.append(key)
    if not outside:
        return None
    target = _union_box(now.shapes[key].box for key in outside)
    assert target is not None
    verb = "is" if len(outside) == 1 else "are"
    text = f"new or changed {_names(outside)} {verb} outside the view (view {_view_ranges(now.view)})"
    remembered = {(key, now.shapes[key].signature) for key in outside}
    return _Problem(text, target, lambda memory: memory.objects.update(remembered))


def _tiny_problem(context: _Context) -> Optional[_Problem]:
    """The whole drawing is too small to read."""
    now, before = context.now, context.before
    content = now.content
    if context.first or not now.is_tiny(content) or not _drawing_changed(context):
        return None
    assert content is not None
    if before is not None and before.is_tiny(before.content) and _not_worse(content, before.content):
        return None
    if context.memory.told("tiny", content):
        return None
    text = f"the shapes span only {_extent_text(now, content)} ({_shapes_and_view(now, content)})"
    return _Problem(text, content, _remember_box("tiny", content))


def _tiny_new_shapes_problem(context: _Context) -> Optional[_Problem]:
    """New or changed shapes with a real extent, too small to read, inside a drawing that is large overall."""
    now, before = context.now, context.before
    if context.first or now.is_tiny(now.content):
        return None
    sized = [
        key
        for key in context.changed
        if key[0] not in _POINT_BUCKETS and not now.shapes[key].clipped and not now.is_point_like(now.shapes[key].box)
    ]
    new_box = _union_box(now.shapes[key].box for key in sized)
    if new_box is None:
        return None  # only points, labels or style changes: nothing with a size of its own
    reach = _tiny_reach(now)
    # Only small shapes with a size of their own extend the cluster; a lone point never sets its size.
    neighbours = [
        key
        for key, shape in now.shapes.items()
        if key not in context.changed
        and key[0] not in _POINT_BUCKETS
        and not shape.clipped
        and now.is_tiny(shape.box)
        and shape.box.gap_to(new_box) <= reach
    ]
    cluster = new_box
    for key in neighbours:
        cluster = cluster.union(now.shapes[key].box)
    if not now.is_tiny(cluster):
        return None
    old = _union_box(context.before_shapes[key].box for key in neighbours if key in context.before_shapes)
    if before is not None and before.is_tiny(old) and _not_worse(cluster, old):
        return None
    if context.memory.told("tiny", cluster):
        return None
    text = (
        f"the new or changed shapes ({_names(sized)}) span only {_extent_text(now, cluster)} "
        f"({_shapes_and_view(now, cluster)})"
    )
    return _Problem(text, cluster, _remember_box("tiny", cluster))


def _flat_problem(context: _Context) -> Optional[_Problem]:
    """A turning function graph with only a few pixels of vertical variation."""
    now = context.now
    for key in context.changed:
        shape = now.shapes[key]
        if key in context.memory.flat or not now.is_flat(shape):
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
        return _Problem(text, visible, _remember_flat(key), flat=True)
    return None


def _remember_flat(key: Key) -> Callable[[ViewNoteMemory], None]:
    return lambda memory: memory.flat.add(key)


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
        pan_only = summary.is_point_like(target) or (fit <= current and not summary.is_tiny(target))
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
    """``low..high`` after rounding, or one value when both round to the same."""
    first, last = _number(_snap(low, step)), _number(_snap(high, step))
    return first if first == last else f"{first}..{last}"


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
    """The object's geometry args as text; style args (colour, label text, ...) are left out."""
    geometry = {key: value for key, value in _args(item).items() if key not in _STYLE_ARGS}
    try:
        return json.dumps(geometry, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(geometry)


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
