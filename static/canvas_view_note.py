"""
View notes: tell the model when something it just drew can't be seen.

The app never moves or zooms the view on its own, and the user's view is the
user's business. After each of the model's tool batches, the [canvas changes]
end with one "View note:" line when an object that batch created or moved
cannot be seen, so the model can offer the user a zoom; the system prompt tells
it to change the view only when the user asks or agrees. A missed note is far
better than a false or useless one, so every rule stays silent when unsure:

1. outside: a new object lies entirely outside the view;
2. tiny: the batch's new shapes with a real extent span fewer than 16 px on screen,
   leaving out small shapes attached to a readable shape (angle arcs, right-angle
   squares, highlight circles, tick marks, markers), and silent when they touch or
   share a point with anything drawn before the batch (the user already sees that
   scene at this scale, or declined to zoom on it);
3. flat: a new function graph that waves (turns at least three times, like sin) varies by
   fewer than 16 px vertically, with no spike or vertical asymptote, and is squeezed on
   screen (the samples miss it and its period is under 16 px), when the suggested zoom
   shows at least one period of it.

"New" means created or geometrically changed by the batch, decided from the
previous canvas's objects by bucket, name and geometry (style changes such as a
recolour are no change). Screen sizes follow from the view bounds and the canvas
size in CSS pixels (the view is a uniform linear map); function graphs and curves
are measured by the client (``curve_extents``, static/client/prompt_canvas_state.py).

Pure module (no Flask, no I/O); ``view_note`` never raises.
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
# Tiny: under this many pixels on screen, a new drawing is a speck (the reported triangle was 6 px).
TINY_PX = 16.0
# Without a canvas size: under this fraction of the view's smaller side.
TINY_VIEW_FRACTION = 0.02
# Flat: a waving graph whose whole vertical variation is under about one label height (14 px
# text) reads as a straight line.
FLAT_PX = 16.0
# The suggested view for a flat graph makes its variation fill this share of the view's height.
FLAT_FILL = 0.25
# The suggested view shows the target box enlarged this much around its centre.
SUGGESTED_VIEW_MARGIN = 1.25
# Extents below the grid's finest spacing (Cartesian2Axis.min_tick_spacing) or this fraction of
# the coordinates' magnitude count as a point: nothing to zoom into.
DEFAULT_MIN_EXTENT = 1e-6
_RELATIVE_MIN_EXTENT = 1e-9
_SCIENTIFIC_ABOVE = 1e15
_MAX_NOTE_NAMES = 3
_SAME_BOX_RELATIVE_TOLERANCE = 1e-6
# Args that change how an object looks, not where it is or how big (a recolour is no change),
# and lists derived from a function's definition.
_IGNORED_ARGS = frozenset(
    {
        "vertical_asymptotes",
        "horizontal_asymptotes",
        "point_discontinuities",
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
# Buckets drawn at one spot: never tiny (their size on screen is text, not geometry).
_POINT_BUCKETS = frozenset({"Points", "Labels"})
# Named first: whole shapes before their edges, edges before points.
_NAME_ORDER = {"Points": 2, "Labels": 2, "Segments": 1, "Vectors": 1}

Point2D = Tuple[float, float]
Key = Tuple[str, str]
JsonDict = Dict[str, Any]


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
    # The object's geometry (style args left out): a moved or resized object gets a new one.
    signature: str
    # The named points it is drawn through, and the corners of its outline (segment ends,
    # polygon vertices); circles keep their centre and radius.
    points: Tuple[str, ...] = ()
    corners: Tuple[Point2D, ...] = ()
    radius: Optional[float] = None
    # A function graph, its sampling flags (see prompt_canvas_state), and whether it was
    # sampled over the view's x range only (no bounds).
    graph: bool = False
    clipped: bool = False
    waves: bool = False
    spiky: bool = False
    # Whether the samples follow the graph (a value halfway between two samples is about their
    # average); an aliased wave is not, and its estimated period then says how far to zoom.
    resolved: bool = True
    period: Optional[float] = None


@dataclass(frozen=True)
class ViewSummary:
    """The view and every measured object of a state."""

    view: Box
    canvas_px: Optional[Tuple[float, float]]
    shapes: Dict[Key, Shape]
    min_tick_spacing: float
    # A graph or curve the client did not measure (only a batch's new curves are): a small new
    # shape could lie on it, so the tiny rule stays silent.
    unmeasured_curves: bool = False

    def scale(self) -> float:
        """Pixels per math unit (a nominal 1000 px for the view's smaller side without a canvas size)."""
        if self.canvas_px:
            return self.canvas_px[0] / self.view.width
        return 1000.0 / min(self.view.width, self.view.height)

    def is_point_like(self, box: Box) -> bool:
        """Too small to zoom into: below the grid's finest spacing or float resolution."""
        return box.size <= max(self.min_tick_spacing, _RELATIVE_MIN_EXTENT * box.magnitude)

    def is_tiny(self, box: Box) -> bool:
        if self.is_point_like(box):
            return False
        if self.canvas_px:
            return box.size * self.scale() < TINY_PX
        return box.size < TINY_VIEW_FRACTION * min(self.view.width, self.view.height)

    def is_flat(self, shape: Shape) -> bool:
        """An unresolved waving graph, wide enough to see, with a few pixels of vertical variation.

        A wave the samples already follow is drawn as it is: zooming keeps the aspect ratio,
        so it would only show a straight piece of it.
        """
        box = shape.box
        if not shape.graph or not shape.waves or shape.spiky or shape.resolved:
            return False
        if self.is_point_like(Box(0.0, 0.0, box.bottom, box.top)):
            return False
        visible_width = min(box.right, self.view.right) - max(box.left, self.view.left)
        return box.height * self.scale() < FLAT_PX and visible_width * self.scale() >= FLAT_PX

    def is_outside(self, shape: Shape) -> bool:
        return not shape.box.intersects(self.view)


def summarize_view(state: Any) -> Optional[ViewSummary]:
    """Measure the state's objects against its view; None without a usable view."""
    if not isinstance(state, Mapping):
        return None
    view = _view_box(state)
    if view is None:
        return None
    spacing = _as_float(state.get("min_tick_spacing"))
    min_extent = spacing if spacing is not None and spacing > 0 else DEFAULT_MIN_EXTENT
    shapes = measure_shapes(state)
    unmeasured = any(key not in shapes for bucket in _CURVE_BUCKETS for key, _ in _keyed_items(state, bucket))
    return ViewSummary(view, _canvas_px(state), shapes, min_extent, unmeasured)


# --------------------------------------------------------------------------- measuring


def measure_shapes(state: Mapping[str, Any]) -> Dict[Key, Shape]:
    """Every measurable object of the state, keyed (bucket, name) like the rendered entries."""
    shapes: Dict[Key, Shape] = {}
    positions: Dict[str, Point2D] = {}
    for bucket in ("Points", "Labels"):
        for key, item in _keyed_items(state, bucket):
            xy = _xy(_args(item).get("position"))
            if xy is not None:
                shapes[key] = Shape(Box.around(*xy), _signature(item), corners=(xy,))
                if bucket == "Points":
                    positions.setdefault(key[1], xy)
    for bucket, measure in _MEASURES:
        for key, item in _keyed_items(state, bucket):
            shape = measure(_args(item), positions, _signature(item))
            if shape is not None and shape.box.is_finite():
                shapes[key] = shape
    shapes.update(_curve_shapes(state))
    return shapes


_Measure = Callable[[JsonDict, Mapping[str, Point2D], str], Optional[Shape]]


def _through_points(names: Sequence[Any], positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    """A segment, vector or polygon: the box of its points, which are part of its geometry."""
    corners = [positions.get(str(name)) for name in names]
    found = [c for c in corners if c is not None]
    if len(found) < 2 or len(found) != len(corners):
        return None
    return Shape(Box.of_points(found), f"{signature}{found}", tuple(map(str, names)), tuple(found))


def _segment(first: str, second: str) -> _Measure:
    return lambda args, positions, signature: _through_points([args.get(first), args.get(second)], positions, signature)


def _polygon(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    for list_key in ("points", "vertices"):
        if isinstance(args.get(list_key), list):
            return _through_points(args[list_key], positions, signature)
    numbered = [(int(str(key)[1:]), value) for key, value in args.items() if re.fullmatch(r"p\d+", str(key))]
    return _through_points([value for _, value in sorted(numbered, key=lambda pair: pair[0])], positions, signature)


def _round(
    cx: Optional[float], cy: Optional[float], rx: Optional[float], ry: Optional[float], signature: str, center: str = ""
) -> Optional[Shape]:
    if cx is None or cy is None or rx is None or ry is None or rx < 0 or ry < 0:
        return None
    radius = max(rx, ry)
    return Shape(Box.around(cx, cy, rx, ry), f"{signature}{cx},{cy}", (center,) if center else (), ((cx, cy),), radius)


def _circle(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    center = positions.get(str(args.get("center")))
    radius = _as_float(args.get("radius"))
    if center is None:
        return None
    return _round(center[0], center[1], radius, radius, signature, str(args.get("center")))


def _ellipse(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    center = positions.get(str(args.get("center")))
    rx, ry = _as_float(args.get("radius_x")), _as_float(args.get("radius_y"))
    if center is None or rx is None or ry is None:
        return None
    angle = math.radians(_as_float(args.get("rotation_angle")) or 0.0)
    half_w, half_h = (
        math.hypot(rx * math.cos(angle), ry * math.sin(angle)),
        math.hypot(rx * math.sin(angle), ry * math.cos(angle)),
    )
    return _round(center[0], center[1], half_w, half_h, signature, str(args.get("center")))


def _arc(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    radius = _as_float(args.get("radius"))
    return _round(_as_float(args.get("center_x")), _as_float(args.get("center_y")), radius, radius, signature)


def _bar(args: JsonDict, positions: Mapping[str, Point2D], signature: str) -> Optional[Shape]:
    left, right, bottom, top = (_as_float(args.get(key)) for key in ("x_left", "x_right", "y_bottom", "y_top"))
    if left is None or right is None or bottom is None or top is None:
        return None
    return Shape(Box(min(left, right), max(left, right), min(bottom, top), max(bottom, top)), signature)


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
_MEASURES: Tuple[Tuple[str, _Measure], ...] = (
    ("Segments", _segment("p1", "p2")),
    ("Vectors", _segment("origin", "tip")),
    ("Circles", _circle),
    ("Ellipses", _ellipse),
    ("CircleArcs", _arc),
    ("Bars", _bar),
) + tuple((bucket, _polygon) for bucket in _POLYGON_BUCKETS)
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
            box = _extent_box(entry.get("box")) if isinstance(entry, Mapping) else None
            if box is None or not isinstance(entry, Mapping):
                continue
            graph = bucket in _GRAPH_BUCKETS
            asymptotes = [_as_float(x) for x in _as_list(_args(item).get("vertical_asymptotes"))]
            asymptote = any(x is not None and box.left <= x <= box.right for x in asymptotes)
            shapes[key] = Shape(
                box,
                _signature(item),
                graph=graph,
                clipped=bool(entry.get("clipped")),
                waves=bool(entry.get("waves")),
                spiky=bool(entry.get("spiky")) or (graph and asymptote),
                resolved=entry.get("resolved") is not False,
                period=_as_float(entry.get("period")),
            )
    return shapes


def _extent_box(raw: Any) -> Optional[Box]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    left, right, bottom, top = (_as_float(v) for v in raw)
    if left is None or right is None or bottom is None or top is None or right < left or top < bottom:
        return None
    return Box(left, right, bottom, top)


# --------------------------------------------------------------------------- the note


def view_note(previous: Optional[Mapping[str, Any]], current: Mapping[str, Any]) -> Optional[str]:
    """One "View note:" line about the objects new or changed since ``previous``, else None.

    ``previous`` is the canvas the model saw before its tool batch, ``current`` the
    canvas after it. Never raises.
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
    if now is None or not isinstance(previous, Mapping):
        return None
    before = {key: _signature(item) for key, item in _all_items(previous)}
    signatures = {key: _signature(item) for key, item in _all_items(current)}
    new = [key for key in now.shapes if before.get(key) != signatures.get(key)]
    for find in (_outside, _tiny, _flat):
        found = find(now, new)
        if found is not None:
            keys, template, target, flat = found
            names, named = _names(keys)
            # The verb agrees with the names as printed: "the new ABC spans", "the new P, Q are".
            verbs = _SINGULAR if len(named) == 1 else _PLURAL
            created = all(key not in before for key in named)
            subject = f"the {'new' if created else 'new or changed'} {names}"
            return (
                f"{VIEW_NOTE_PREFIX} {subject} {template.format(**verbs)}. "
                f"Offer to {_suggestion(now, target, flat)}; don't change the view unless the user agrees."
            )
    return None


_SINGULAR = {"is": "is", "spans": "spans", "varies": "varies"}
_PLURAL = {"is": "are", "spans": "span", "varies": "vary"}


_Found = Tuple[List[Key], str, Box, bool]


def _outside(now: ViewSummary, new: List[Key]) -> Optional[_Found]:
    """New objects entirely outside the view (a point that a segment on screen is drawn through is on screen)."""
    attached = {name for shape in now.shapes.values() if not now.is_outside(shape) for name in shape.points}
    outside = [
        key for key in new if now.is_outside(now.shapes[key]) and not (key[0] == "Points" and key[1] in attached)
    ]
    if not outside:
        return None
    # When some of the batch's objects are on screen, the suggested view keeps them in it too.
    target = _union(now.shapes[key].box for key in new if not now.shapes[key].clipped)
    target = target or _union(now.shapes[key].box for key in outside)
    assert target is not None
    return outside, f"{{is}} outside the view (view {_view_ranges(now.view)})", target, False


def _tiny(now: ViewSummary, new: List[Key]) -> Optional[_Found]:
    """The new shapes with a real extent span only a few pixels, markers on readable shapes aside."""
    if now.unmeasured_curves:
        return None  # a tangent or marker on a graph the client did not measure
    sized = [
        key
        for key in new
        if key[0] not in _POINT_BUCKETS
        and not now.shapes[key].clipped
        and not now.is_point_like(now.shapes[key].box)
        and not _part_of_readable_shape(now, key)
    ]
    # Edges of a new polygon are named by the polygon.
    sized = [key for key in sized if not _is_edge_of(now, key, sized)]
    if _builds_on_earlier_shapes(now, sized, new):
        return None
    target = _union(now.shapes[key].box for key in sized)
    if target is None or not now.is_tiny(target):
        return None
    size = (target.width * now.scale(), target.height * now.scale())
    extent = f"~{_pixels(size[0])}x{_pixels(size[1])} px on screen" if now.canvas_px else "a sliver of the view"
    return sized, f"{{spans}} only {extent} ({_where(now, target)})", target, False


def _builds_on_earlier_shapes(now: ViewSummary, sized: List[Key], new: List[Key]) -> bool:
    """True when a new small shape touches or shares a point with anything drawn before the batch.

    Then the batch builds on a scene the user already sees at this scale (a circumcircle after
    a declined tiny triangle, a diagonal after the user zoomed out, a ring around a point).
    """
    fresh = set(new)
    earlier = [(key, shape) for key, shape in now.shapes.items() if key not in fresh]
    for key in sized:
        shape = now.shapes[key]
        for old_key, old in earlier:
            if old_key[0] == "Points":
                if old_key[1] in shape.points:
                    return True
            elif shape.box.intersects(old.box) or set(shape.points) & set(old.points):
                return True
    return False


def _flat(now: ViewSummary, new: List[Key]) -> Optional[_Found]:
    """A new aliased waving graph with only a few pixels of vertical variation, when a zoom would show it."""
    for key in new:
        shape = now.shapes[key]
        if not now.is_flat(shape) or not _zoom_shows_a_period(now, shape):
            continue
        box = shape.box
        visible = Box(max(box.left, now.view.left), min(box.right, now.view.right), box.bottom, box.top)
        variation = f"~{_pixels(box.height * now.scale())} px" if now.canvas_px else "a sliver"
        step = _coordinate_step(box.height / 2.0) or _coordinate_step(now.view.width / 2.0)
        text = (
            f"{{varies}} only {variation} vertically on screen (y {_span(box.bottom, box.top, step)} over x "
            f"{_number(visible.left, VIEW_SIGNIFICANT_DIGITS)}..{_number(visible.right, VIEW_SIGNIFICANT_DIGITS)}; "
            f"view {_view_ranges(now.view)})"
        )
        return [key], text, visible, True
    return None


def _flat_zoom_half_width(summary: ViewSummary, box: Box) -> float:
    """Half the width of the view in which the graph's variation fills FLAT_FILL of the height."""
    aspect = summary.view.height / summary.view.width
    return _nice_ceil(box.height / (2.0 * FLAT_FILL) / aspect)


def _zoom_shows_a_period(summary: ViewSummary, shape: Shape) -> bool:
    """A wave squeezed on screen (a period of under 16 px) that the suggested zoom shows at least once.

    A wave whose period already spans more pixels is drawn as a wave, just a low one: zooming
    keeps the aspect ratio, so it would not help.
    """
    if shape.period is None or shape.period * summary.scale() >= FLAT_PX:
        return False
    return 2.0 * _flat_zoom_half_width(summary, shape.box) >= shape.period


def _part_of_readable_shape(now: ViewSummary, key: Key) -> bool:
    """A small shape drawn on a readable one: centred within its own size of one of the other's
    corners (which covers sharing a vertex with it), or lying on its outline (angle arcs,
    right-angle squares, highlight circles, ticks)."""
    shape = now.shapes[key]
    reach = shape.box.size
    center = shape.box.center
    for other_key, other in now.shapes.items():
        if other_key == key or other_key[0] in _POINT_BUCKETS or now.is_tiny(other.box) or now.is_point_like(other.box):
            continue
        if other.box.size <= shape.box.size:
            continue
        if any(math.dist(center, corner) <= reach for corner in other.corners):
            return True
        if other.radius is not None and other.corners:
            if abs(math.dist(center, other.corners[0]) - other.radius) <= reach:
                return True
        elif other.graph and shape.box.intersects(other.box):
            return True
        elif _box_meets_outline(shape.box, other.corners):
            return True
    return False


def _box_meets_outline(box: Box, corners: Sequence[Point2D]) -> bool:
    """True when ``box`` touches any edge of the closed or open path through ``corners``."""
    edges = list(zip(corners, corners[1:]))
    if len(corners) > 2:
        edges.append((corners[-1], corners[0]))
    return any(_segment_meets_box(a, b, box) for a, b in edges)


def _segment_meets_box(a: Point2D, b: Point2D, box: Box) -> bool:
    """Liang-Barsky clipping: does the segment a-b pass through the box?"""
    t0, t1 = 0.0, 1.0
    dx, dy = b[0] - a[0], b[1] - a[1]
    for p, q in ((-dx, a[0] - box.left), (dx, box.right - a[0]), (-dy, a[1] - box.bottom), (dy, box.top - a[1])):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _is_edge_of(now: ViewSummary, key: Key, keys: List[Key]) -> bool:
    if key[0] not in ("Segments", "Vectors"):
        return False
    ends = set(now.shapes[key].points)
    return any(other[0] in _POLYGON_BUCKETS and ends <= set(now.shapes[other].points) for other in keys)


# --------------------------------------------------------------------------- text


def _names(keys: Sequence[Key]) -> Tuple[str, List[Key]]:
    """The most whole objects only (a triangle without its edges and vertices), and their keys."""
    top = min(_NAME_ORDER.get(key[0], 0) for key in keys)
    named = [key for key in keys if _NAME_ORDER.get(key[0], 0) == top]
    names = ", ".join(key[1] for key in named[:_MAX_NOTE_NAMES])
    if len(named) > _MAX_NOTE_NAMES:
        names += f" (+{len(named) - _MAX_NOTE_NAMES} more)"
    return names, named


def _pixels(value: float) -> str:
    return "<1" if value < 1 else str(round(value))


def _where(summary: ViewSummary, box: Box) -> str:
    step = _coordinate_step(box.size / 2.0) or _coordinate_step(summary.view.width / 2.0)
    return (
        f"x {_span(box.left, box.right, step)}, y {_span(box.bottom, box.top, step)}; view {_view_ranges(summary.view)}"
    )


def _view_ranges(view: Box) -> str:
    """The view bounds rounded like the canvas view line (4 significant digits)."""
    return (
        f"x {_number(view.left, VIEW_SIGNIFICANT_DIGITS)}..{_number(view.right, VIEW_SIGNIFICANT_DIGITS)}, "
        f"y {_number(view.bottom, VIEW_SIGNIFICANT_DIGITS)}..{_number(view.top, VIEW_SIGNIFICANT_DIGITS)}"
    )


def _suggestion(summary: ViewSummary, target: Box, flat: bool) -> str:
    """A view showing ``target``: zoom when it is tiny, flat or too big, else just move the view."""
    view = summary.view
    aspect = view.height / view.width
    current = view.width / 2.0
    if flat:
        half = _flat_zoom_half_width(summary, target)
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
    unit = 10.0 ** (math.floor(math.log10(value)) - 1)
    return math.ceil(value / unit - 1e-9) * unit


def _coordinate_step(half_size: float) -> float:
    """A rounding step a tenth of the magnitude of ``half_size`` (0 for 0)."""
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


def _all_items(state: Mapping[str, Any]) -> List[Tuple[Key, JsonDict]]:
    """Every object of every bucket, with the coordinates of the points it is drawn through.

    A segment's signature includes its endpoints' positions, so moving a point changes the
    segments and polygons through it too.
    """
    positions = {key[1]: _signature(item) for key, item in _keyed_items(state, "Points")}
    keyed: List[Tuple[Key, JsonDict]] = []
    for bucket, items in state.items():
        if not isinstance(items, list):
            continue
        for key, item in _keyed_items(state, bucket):
            args = _args(item)
            through = sorted(positions[str(v)] for v in args.values() if isinstance(v, str) and str(v) in positions)
            keyed.append((key, {"args": dict(args, _through=through)}))
    return keyed


def _signature(item: Mapping[str, Any]) -> str:
    """The object's geometry args as text; style args (colour, label text, ...) are left out."""
    geometry = {key: value for key, value in _args(item).items() if key not in _IGNORED_ARGS}
    try:
        return json.dumps(geometry, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(geometry)


def _union(boxes: Iterable[Box]) -> Optional[Box]:
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
