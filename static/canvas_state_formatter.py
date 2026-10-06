"""
Canvas state rendering for LLM prompts.

The client serializes the canvas with ``Canvas.get_canvas_state()``: a dict of
drawable buckets (``Points``, ``Segments``, ``Triangles``, ...) plus viewport
metadata. Sent verbatim, that JSON is expensive for the model to read: float
noise such as ``199.20000000000002`` costs one token per digit on local models,
render-only fields (``_p1_coords``, ``circle_formula``) repeat information, and
facts the model is asked about (lengths, areas, angle sizes) are not present.

This module turns the state into prompt text. It is pure (no Flask, no I/O):

``render_text``      one object per line, math notation first, with facts
                     computed from the coordinates (``AB = Segment(A, B)  len 4``)
``render_min_json``  the JSON state with noise removed and numbers rounded
``render_delta``     the changes between two states, one line per object
``render_state``     dispatch on a ``CanvasFormat``
``render_update``    what to tell the model after a tool batch changed the canvas
                     (the changes are text lines in every format; see render_delta)
``view_note``        a one-line hint when the drawings are too small on screen or
                     outside the view (see the "view note" section below)

Budget: ``render_text`` and ``render_min_json`` accept ``budget_tokens``. In
text, large scenes first pack points several per line, then drop the least
important objects; min_json keeps the same fraction of every bucket. Either
way a note tells the model to call ``get_current_canvas_state`` for the rest.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Mapping, Optional, Sequence, Set, Tuple

from static.token_estimation import estimate_tokens_from_text

_logger = logging.getLogger("mathud")

CanvasFormat = Literal["json", "min_json", "text"]
CANVAS_FORMATS: Tuple[CanvasFormat, ...] = ("json", "min_json", "text")

SIGNIFICANT_DIGITS = 6
VIEW_SIGNIFICANT_DIGITS = 4

# Values closer to zero than this are float noise (e.g. cos(pi/2) * r).
_ZERO_EPSILON = 1e-11
# Values this close (relative) to an integer are printed as that integer.
_INTEGER_RELATIVE_EPSILON = 1e-9
# Relative tolerance for "point lies on circle".
_ON_CIRCLE_RELATIVE_TOLERANCE = 1e-6
# "passes through" is checked for every circle/point pair; skip it on scenes larger than this.
_MAX_ON_CIRCLE_CHECKS = 20000

# Longest list shown inline for asymptotes, discontinuities and similar lists.
_MAX_INLINE_LIST = 8
# Longest computation result shown before it is cut.
_MAX_COMPUTATION_RESULT_CHARS = 200
# Points are packed several per line when there are more than this many.
_PACK_POINTS_THRESHOLD = 8
_POINTS_PER_PACKED_ROW = 6

# The canvas size in CSS pixels, {"width": w, "height": h}: the client adds it to the
# prompt's canvas_state (never to saved workspaces) so the view note can measure pixels.
CANVAS_SIZE_KEY = "canvas_size_px"

# Top-level keys describing the viewport rather than drawables.
_VIEW_KEYS = frozenset(
    {
        "Cartesian_System_Visibility",
        "current_tick_spacing",
        "default_tick_spacing",
        "current_tick_spacing_repr",
        "min_tick_spacing",
        "visible",
        "coordinate_system",
        CANVAS_SIZE_KEY,
    }
)
_COMPUTATIONS_KEY = "computations"

# Colors that are not worth showing: the drawable default (black), and blue for angles, whose default it is.
_DEFAULT_COLORS = frozenset({"", "black"})
_DEFAULT_ANGLE_COLORS = _DEFAULT_COLORS | {"blue"}
_DEFAULT_AREA_COLOR = "lightblue"
_DEFAULT_AREA_OPACITY = 0.3
_DEFAULT_LABEL_FONT_SIZE = 14.0

# Fields that only exist for rendering/persistence and never reach the model.
_RENDER_ONLY_FIELDS = frozenset(
    {
        "_p1_coords",
        "_p2_coords",
        "_origin_coords",
        "_tip_coords",
        "circle_formula",
        "ellipse_formula",
        "num_sample_points",
        "reference_scale_factor",
        "geometry_snapshot",
        "resolution",
        # Graph ownership bookkeeping: which reused drawables delete_graph keeps.
        "preexisting_points",
        "preexisting_edges",
        "edge_label_records",
    }
)

OMITTED_NOTE = "call get_current_canvas_state with object_names to see them"

JsonDict = Dict[str, Any]
Point2D = Tuple[float, float]


# --------------------------------------------------------------------------- numbers


def format_number(value: Any, significant_digits: int = SIGNIFICANT_DIGITS) -> str:
    """Format a number compactly: snap float noise, keep at most N significant digits.

    Non-numbers are returned via ``str`` so callers can pass raw state values.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    try:
        number = float(value)
    except OverflowError:  # an int too large for a float
        return str(value)
    if not math.isfinite(number):
        return str(value)
    if abs(number) < _ZERO_EPSILON:
        return "0"
    nearest = round(number)
    if abs(number - nearest) <= _INTEGER_RELATIVE_EPSILON * max(1.0, abs(number)):
        return str(int(nearest))
    text = f"{number:.{significant_digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        if "." in mantissa:
            mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{int(exponent)}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def round_numbers(value: Any) -> Any:
    """Recursively round floats the way ``format_number`` prints them."""
    if isinstance(value, float):
        text = format_number(value)
        try:
            number = float(text)
        except ValueError:
            return value
        return int(number) if number.is_integer() else number
    if isinstance(value, dict):
        return {key: round_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_numbers(item) for item in value]
    return value


_LONG_DECIMAL = re.compile(r"(?<![A-Za-z_\d.])(\d+\.\d{7,})")


def format_expression(expression: Any) -> str:
    """Shorten over-long decimal literals inside an expression (e.g. regression fits)."""
    if not isinstance(expression, str):
        return str(expression)
    return _LONG_DECIMAL.sub(lambda match: format_number(float(match.group(1))), expression)


def _point_text(x: Any, y: Any) -> str:
    return f"({format_number(x)}, {format_number(y)})"


def _compact_json(value: Any) -> str:
    return json.dumps(round_numbers(value), separators=(",", ":"), ensure_ascii=False)


def _as_float(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float, or None for anything that is not a real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_number(value: Any) -> bool:
    return _as_float(value) is not None


def _differs(value: Any, default: float) -> bool:
    """True when ``value`` is a number other than ``default``."""
    number = _as_float(value)
    return number is not None and abs(number - default) > 1e-9


def _position(args: Mapping[str, Any]) -> JsonDict:
    position = args.get("position")
    return position if isinstance(position, dict) else {}


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, (str, list, dict)) and len(value) == 0)


def _as_list(value: Any) -> List[Any]:
    """Return a list-typed field as a list: None/empty gives [], a lone scalar gives [value]."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if _is_empty(value) or value is False:
        return []
    return [value]


def _single_line_strings(value: Any) -> Any:
    """Escape line breaks in every string (names, labels, expressions, keys).

    The text format is one object per line inside a <canvas> block, so a name
    or label must not be able to start a new line (or close the block early).
    """
    if isinstance(value, str):
        return value.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    if isinstance(value, dict):
        return {_single_line_strings(key): _single_line_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_single_line_strings(item) for item in value]
    return value


def _inline_list(values: Sequence[Any]) -> str:
    shown = ", ".join(format_number(v) for v in values[:_MAX_INLINE_LIST])
    if len(values) > _MAX_INLINE_LIST:
        shown += f" (+{len(values) - _MAX_INLINE_LIST} more)"
    return shown


# --------------------------------------------------------------------------- scene index


class _Scene:
    """Name lookups over a raw state so renderers can compute facts from coordinates."""

    def __init__(self, state: Mapping[str, Any]) -> None:
        self.state = state
        # Facts are never computed from ambiguous (duplicated) point names.
        duplicate_points = _duplicate_names(state).get("Points", set())
        self.points: Dict[str, Point2D] = {}
        for item in _items(state, "Points"):
            name = str(item.get("name", ""))
            position = _args(item).get("position")
            if name in duplicate_points or not isinstance(position, dict):
                continue
            x, y = _as_float(position.get("x")), _as_float(position.get("y"))
            if x is not None and y is not None:
                self.points[name] = (x, y)
        self.segments: Dict[str, Tuple[str, str]] = {}
        self.segment_items: Dict[str, JsonDict] = {}
        for item in _items(state, "Segments"):
            args = _args(item)
            name = str(item.get("name", ""))
            self.segments[name] = (str(args.get("p1", "")), str(args.get("p2", "")))
            self.segment_items[name] = item
        self.vector_items: Dict[str, JsonDict] = {str(v.get("name", "")): v for v in _items(state, "Vectors")}
        self.check_points_on_circles = len(self.points) * len(_items(state, "Circles")) <= _MAX_ON_CIRCLE_CHECKS
        self.graph_members: Set[str] = set()
        for bucket in _GRAPH_BUCKETS:
            for graph in _items(state, bucket):
                args = _args(graph)
                self.graph_members.update(str(name) for name in _as_list(args.get("segments")))
                self.graph_members.update(str(name) for name in _as_list(args.get("vectors")))

    def xy(self, name: Any) -> Optional[Point2D]:
        return self.points.get(str(name)) if name else None

    def distance(self, first: Any, second: Any) -> Optional[float]:
        a, b = self.xy(first), self.xy(second)
        if a is None or b is None:
            return None
        return math.dist(a, b)


def _items(state: Mapping[str, Any], bucket: str) -> List[JsonDict]:
    items = state.get(bucket)
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _args(item: Mapping[str, Any]) -> JsonDict:
    args = item.get("args")
    return args if isinstance(args, dict) else {}


def _duplicate_names(state: Mapping[str, Any]) -> Dict[str, Set[str]]:
    duplicates: Dict[str, Set[str]] = {}
    for bucket, items in state.items():
        # Computations have no names; tools never address them by name.
        if bucket == _COMPUTATIONS_KEY or not isinstance(items, list):
            continue
        seen: Set[str] = set()
        for item in items:
            if not isinstance(item, dict) or _is_empty(item.get("name")):
                continue
            name = str(item["name"])
            if name in seen:
                duplicates.setdefault(bucket, set()).add(name)
            seen.add(name)
    return duplicates


# --------------------------------------------------------------------------- shared line pieces


def _color_suffix(args: Mapping[str, Any], defaults: FrozenSet[str] = _DEFAULT_COLORS) -> str:
    color = args.get("color")
    if _is_empty(color) or (isinstance(color, str) and color in defaults):
        return ""
    return f" color {color}"


def _segment_label_suffix(args: Mapping[str, Any]) -> str:
    label = args.get("label")
    if not isinstance(label, dict) or not label.get("text"):
        return ""
    hidden = "" if label.get("visible", True) else " (hidden)"
    return f' label "{label["text"]}"{hidden}'


def _extras_suffix(args: Mapping[str, Any], handled: Set[str]) -> str:
    """Show args a renderer does not know about so nothing is silently dropped."""
    extras = [
        f"{key} {_compact_json(value)}"
        for key, value in args.items()
        if key not in handled and key not in _RENDER_ONLY_FIELDS and not _is_empty(value)
    ]
    return f"  [{'; '.join(extras)}]" if extras else ""


def _range_text(low: Any, high: Any) -> str:
    low_text = format_number(low) if low is not None else "-inf"
    high_text = format_number(high) if high is not None else "inf"
    return f"[{low_text}, {high_text}]"


def _polygon_area(coords: Sequence[Point2D]) -> float:
    total = 0.0
    for index, (x1, y1) in enumerate(coords):
        x2, y2 = coords[(index + 1) % len(coords)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


# --------------------------------------------------------------------------- per-bucket line renderers
# Each renderer returns the text for one object ("name = Definition  facts"),
# or None when the object is shown elsewhere (graph edges inside their graph).

Renderer = Callable[[_Scene, JsonDict], Optional[str]]


def _render_point(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    position = _position(args)
    line = f"{item.get('name', '?')} = {_point_text(position.get('x'), position.get('y'))}"
    return line + _color_suffix(args) + _extras_suffix(args, {"position", "color"})


def _render_segment(scene: _Scene, item: JsonDict) -> Optional[str]:
    name = str(item.get("name", "?"))
    if name in scene.graph_members:
        return None
    args = _args(item)
    p1, p2 = args.get("p1"), args.get("p2")
    line = f"{name} = Segment({p1}, {p2})"
    length = scene.distance(p1, p2)
    if length is not None:
        line += f"  len {format_number(length)}"
    return (
        line + _segment_label_suffix(args) + _color_suffix(args) + _extras_suffix(args, {"p1", "p2", "label", "color"})
    )


def _render_vector(scene: _Scene, item: JsonDict) -> Optional[str]:
    name = str(item.get("name", "?"))
    if name in scene.graph_members:
        return None
    args = _args(item)
    origin, tip = args.get("origin"), args.get("tip")
    line = f"{name} = Vector({origin} -> {tip})"
    start, end = scene.xy(origin), scene.xy(tip)
    if start is not None and end is not None:
        dx, dy = end[0] - start[0], end[1] - start[1]
        line += f"  <{format_number(dx)}, {format_number(dy)}> len {format_number(math.hypot(dx, dy))}"
    return (
        line
        + _segment_label_suffix(args)
        + _color_suffix(args)
        + _extras_suffix(args, {"origin", "tip", "label", "color"})
    )


_VERTEX_KEY = re.compile(r"^p(\d+)$")


def _polygon_vertices(args: Mapping[str, Any]) -> List[str]:
    for list_key in ("points", "vertices"):
        if isinstance(args.get(list_key), list):
            return [str(v) for v in args[list_key]]
    numbered = [(int(m.group(1)), str(value)) for key, value in args.items() if (m := _VERTEX_KEY.match(key))]
    return [value for _, value in sorted(numbered)]


def _cyclic_order(vertices: List[str], scene: _Scene) -> List[str]:
    """Order vertices around their centroid (Rectangle state lists them alphabetically)."""
    coords = [scene.xy(v) for v in vertices]
    if not coords or any(c is None for c in coords):
        return vertices
    points = [c for c in coords if c is not None]
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    angles = {v: math.atan2(p[1] - cy, p[0] - cx) for v, p in zip(vertices, points)}
    ordered = sorted(vertices, key=lambda v: angles[v])
    start = ordered.index(vertices[0])
    return ordered[start:] + ordered[:start]


def _polygon_renderer(kind: str, reorder: bool = False) -> Renderer:
    def render(scene: _Scene, item: JsonDict) -> str:
        args = _args(item)
        vertices = _polygon_vertices(args)
        if reorder:
            vertices = _cyclic_order(vertices, scene)
        line = f"{item.get('name', '?')} = {kind}({', '.join(vertices)})"
        types = [str(t) for t in _as_list(item.get("types")) if str(t).lower() != kind.lower()]
        if types:
            line += "  " + " ".join(types)
        coords = [scene.xy(v) for v in vertices]
        if len(coords) >= 3 and all(c is not None for c in coords):
            points = [c for c in coords if c is not None]
            sides = []
            for index, vertex in enumerate(vertices):
                following = vertices[(index + 1) % len(vertices)]
                length = math.dist(points[index], points[(index + 1) % len(points)])
                sides.append(f"{vertex}{following}={format_number(length)}")
            line += f"; sides {' '.join(sides)}; area {format_number(_polygon_area(points))}"
        handled = {"points", "vertices", "color"} | {key for key in args if _VERTEX_KEY.match(key)}
        return line + _color_suffix(args) + _extras_suffix(args, handled)

    return render


def _render_circle(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    center, radius = args.get("center"), args.get("radius")
    line = f"{item.get('name', '?')} = Circle(center {center}, r {format_number(radius)})"
    r = _as_float(radius)
    if r is not None:
        line += f"  area {format_number(math.pi * r * r)}"
        center_xy = scene.xy(center)
        if center_xy is not None and r > 0 and scene.check_points_on_circles:
            on_circle = [
                name
                for name, point in scene.points.items()
                if name != center and abs(math.dist(point, center_xy) - r) <= _ON_CIRCLE_RELATIVE_TOLERANCE * r
            ]
            if on_circle:
                line += "; passes through " + ", ".join(on_circle)
    return line + _color_suffix(args) + _extras_suffix(args, {"center", "radius", "color"})


def _render_ellipse(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    rx, ry = args.get("radius_x"), args.get("radius_y")
    rotation = args.get("rotation_angle")
    line = (
        f"{item.get('name', '?')} = Ellipse(center {args.get('center')}, rx {format_number(rx)}, ry {format_number(ry)}"
    )
    if _differs(rotation, 0.0):
        line += f", rotation {format_number(rotation)} deg"
    line += ")"
    radius_x, radius_y = _as_float(rx), _as_float(ry)
    if radius_x is not None and radius_y is not None:
        line += f"  area {format_number(math.pi * radius_x * radius_y)}"
    handled = {"center", "radius_x", "radius_y", "rotation_angle", "color"}
    return line + _color_suffix(args) + _extras_suffix(args, handled)


def _render_arc(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    p1, p2 = args.get("point1_name"), args.get("point2_name")
    cx, cy, radius = args.get("center_x"), args.get("center_y"), args.get("radius")
    major = bool(args.get("use_major_arc"))
    parts = [f"{p1} to {p2}"]
    if args.get("circle_name"):
        parts.append(f"on {args['circle_name']}")
    parts.append(f"center {_point_text(cx, cy)}, r {format_number(radius)}")
    line = f"{item.get('name', '?')} = Arc({', '.join(parts)}){' major' if major else ''}"
    start, end = scene.xy(p1), scene.xy(p2)
    center_x, center_y, r = _as_float(cx), _as_float(cy), _as_float(radius)
    if start is not None and end is not None and center_x is not None and center_y is not None and r is not None:
        a1 = math.atan2(start[1] - center_y, start[0] - center_x)
        a2 = math.atan2(end[1] - center_y, end[0] - center_x)
        minor = abs(math.degrees(a2 - a1)) % 360.0
        minor = min(minor, 360.0 - minor)
        sweep = 360.0 - minor if major else minor
        line += f"  sweep {format_number(sweep)} deg, length {format_number(math.radians(sweep) * r)}"
    handled = {"point1_name", "point2_name", "center_x", "center_y", "radius", "circle_name", "use_major_arc", "color"}
    return line + _color_suffix(args) + _extras_suffix(args, handled)


def _angle_degrees(scene: _Scene, args: Mapping[str, Any]) -> Optional[Tuple[str, float]]:
    first = scene.segments.get(str(args.get("segment1_name")))
    second = scene.segments.get(str(args.get("segment2_name")))
    if not first or not second:
        return None
    shared = set(first) & set(second)
    if len(shared) != 1:
        return None
    vertex = shared.pop()
    arm1 = first[0] if first[1] == vertex else first[1]
    arm2 = second[0] if second[1] == vertex else second[1]
    v, p, q = scene.xy(vertex), scene.xy(arm1), scene.xy(arm2)
    if v is None or p is None or q is None or p == v or q == v:
        return None
    raw = abs(math.degrees(math.atan2(q[1] - v[1], q[0] - v[0]) - math.atan2(p[1] - v[1], p[0] - v[0]))) % 360.0
    small = min(raw, 360.0 - raw)
    return vertex, (360.0 - small if args.get("is_reflex") else small)


def _render_angle(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    line = f"{item.get('name', '?')} = Angle({args.get('segment1_name')}, {args.get('segment2_name')})"
    if args.get("is_reflex"):
        line += " reflex"
    measured = _angle_degrees(scene, args)
    if measured is not None:
        line += f"  vertex {measured[0]}, {format_number(measured[1])} deg"
    handled = {"segment1_name", "segment2_name", "is_reflex", "color"}
    return line + _color_suffix(args, _DEFAULT_ANGLE_COLORS) + _extras_suffix(args, handled)


def _function_features(args: Mapping[str, Any]) -> str:
    features = []
    for key, label in (
        ("vertical_asymptotes", "vertical asymptotes x ="),
        ("horizontal_asymptotes", "horizontal asymptotes y ="),
        ("point_discontinuities", "discontinuities at x ="),
        ("undefined_at", "undefined at x ="),
    ):
        values = args.get(key)
        if isinstance(values, list) and values:
            features.append(f"{label} {_inline_list(values)}")
    return ("; " + "; ".join(features)) if features else ""


_FUNCTION_FEATURE_KEYS = {"vertical_asymptotes", "horizontal_asymptotes", "point_discontinuities", "undefined_at"}


def _render_function(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    line = f"{item.get('name', '?')}(x) = {format_expression(args.get('function_string'))}"
    left, right = args.get("left_bound"), args.get("right_bound")
    if left is not None or right is not None:
        line += f"  on {_range_text(left, right)}"
    line += _function_features(args)
    handled = {"function_string", "left_bound", "right_bound", "color"} | _FUNCTION_FEATURE_KEYS
    return line + _color_suffix(args) + _extras_suffix(args, handled)


def _render_piecewise(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    pieces = []
    for piece in _as_list(args.get("pieces")):
        if not isinstance(piece, dict):
            continue
        open_bracket = "[" if piece.get("left_inclusive", True) and piece.get("left") is not None else "("
        close_bracket = "]" if piece.get("right_inclusive", True) and piece.get("right") is not None else ")"
        low = format_number(piece["left"]) if piece.get("left") is not None else "-inf"
        high = format_number(piece["right"]) if piece.get("right") is not None else "inf"
        text = f"{format_expression(piece.get('expression'))} on {open_bracket}{low}, {high}{close_bracket}"
        undefined_at = _as_list(piece.get("undefined_at"))
        if undefined_at:
            text += f" (undefined at {_inline_list(undefined_at)})"
        pieces.append(text)
    line = f"{item.get('name', '?')}(x) = piecewise {{ {'; '.join(pieces)} }}" + _function_features(args)
    return line + _color_suffix(args) + _extras_suffix(args, {"pieces", "color"} | _FUNCTION_FEATURE_KEYS)


def _render_parametric(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    t_min, t_max = args.get("t_min"), args.get("t_max")
    t_range = f"t in {_range_text(t_min, t_max)}"
    if not _differs(t_max, 2 * math.pi) and _is_number(t_max) and not _differs(t_min or 0, 0.0):
        t_range = "t in [0, 2pi]"
    line = (
        f"{item.get('name', '?')} = Curve(x(t) = {format_expression(args.get('x_expression'))}, "
        f"y(t) = {format_expression(args.get('y_expression'))}, {t_range})"
    )
    handled = {"x_expression", "y_expression", "t_min", "t_max", "color"}
    return line + _color_suffix(args) + _extras_suffix(args, handled)


def _area_style(args: Mapping[str, Any]) -> str:
    style = []
    color = args.get("color")
    if color and color != _DEFAULT_AREA_COLOR:
        style.append(f"color {color}")
    opacity = args.get("opacity")
    if _differs(opacity, _DEFAULT_AREA_OPACITY):
        style.append(f"opacity {format_number(opacity)}")
    return ("  " + " ".join(style)) if style else ""


def _render_functions_area(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    refs = [str(args[key]) for key in ("func1", "func2") if args.get(key) is not None]
    line = f"{item.get('name', '?')} = AreaBetween({', '.join(refs)}"
    if args.get("left_bound") is not None or args.get("right_bound") is not None:
        line += f", x in {_range_text(args.get('left_bound'), args.get('right_bound'))}"
    line += ")"
    handled = {"func1", "func2", "left_bound", "right_bound", "color", "opacity"}
    return line + _area_style(args) + _extras_suffix(args, handled)


def _render_function_segment_area(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    line = f"{item.get('name', '?')} = AreaBetween({args.get('func')}, segment {args.get('segment')})"
    return line + _area_style(args) + _extras_suffix(args, {"func", "segment", "color", "opacity"})


def _render_segments_area(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    line = f"{item.get('name', '?')} = AreaBetween(segment {args.get('segment1')}, {args.get('segment2')})"
    return line + _area_style(args) + _extras_suffix(args, {"segment1", "segment2", "color", "opacity"})


def _render_closed_shape_area(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    parts = [str(args.get("shape_type") or "region")]
    if args.get("segments"):
        parts.append("segments " + " ".join(str(s) for s in _as_list(args["segments"])))
    for key in ("circle", "ellipse", "chord_segment"):
        if args.get(key):
            parts.append(f"{key.replace('_', ' ')} {args[key]}")
    if args.get("expression"):
        parts.append(f'expression "{args["expression"]}"')
    if args.get("arc_clockwise"):
        parts.append("clockwise arc")
    line = f"{item.get('name', '?')} = ShadedRegion({'; '.join(parts)})"
    handled = {
        "shape_type",
        "segments",
        "circle",
        "ellipse",
        "chord_segment",
        "expression",
        "arc_clockwise",
        "points",
        "color",
        "opacity",
    }
    return line + _area_style(args) + _extras_suffix(args, handled)


_GRAPH_BUCKETS: Tuple[str, ...] = ("UndirectedGraphs", "DirectedGraphs", "Trees", "Graphs")
_GRAPH_KINDS = {"UndirectedGraphs": "undirected", "DirectedGraphs": "directed", "Trees": "tree", "Graphs": "graph"}


def _edge_weight(item: Optional[JsonDict]) -> str:
    label = _args(item or {}).get("label")
    return str(label.get("text") or "").strip() if isinstance(label, dict) else ""


def _graph_renderer(bucket: str) -> Renderer:
    def render(scene: _Scene, item: JsonDict) -> str:
        args = _args(item)
        edges: List[str] = []
        vertices: List[str] = []
        weighted = False

        def add_edge(start: Any, end: Any, arrow: str, weight: str) -> None:
            edges.append(f"{start}{arrow}{end}" + (f" {weight}" if weight else ""))
            for vertex in (start, end):
                if vertex and str(vertex) not in vertices:
                    vertices.append(str(vertex))

        for name in _as_list(args.get("segments")):
            segment = scene.segment_items.get(str(name))
            seg_args = _args(segment or {})
            weight = _edge_weight(segment)
            weighted = weighted or bool(weight)
            add_edge(seg_args.get("p1", "?"), seg_args.get("p2", "?"), "-", weight)
        for name in _as_list(args.get("vectors")):
            vector = scene.vector_items.get(str(name))
            vec_args = _args(vector or {})
            weight = _edge_weight(vector)
            weighted = weighted or bool(weight)
            add_edge(vec_args.get("origin", "?"), vec_args.get("tip", "?"), "->", weight)
        for isolated in _as_list(args.get("isolated_points")):
            if str(isolated) not in vertices:
                vertices.append(str(isolated))

        properties = [_GRAPH_KINDS.get(bucket, "graph")] + (["weighted"] if weighted else [])
        if args.get("root"):
            properties.append(f"root {args['root']}")
        header = f"{item.get('name', '?')} = Graph({' '.join(properties)}; vertices {' '.join(vertices) or '(none)'})"
        header += _extras_suffix(args, {"segments", "vectors", "isolated_points", "root"})
        return header + f"\n  edges ({len(edges)}): " + (", ".join(edges) if edges else "(none)")

    return render


_DEFAULT_BAR_OPTIONS: Dict[str, float] = {"bar_spacing": 0.2, "bar_width": 1, "x_start": 0, "y_base": 0}


def _render_bars_plot(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    values = _as_list(args.get("values"))
    labels = _as_list(args.get("labels_below")) or [str(index) for index in range(len(values))]
    pairs = ", ".join(f"{label} {format_number(value)}" for label, value in zip(labels, values))
    options = [
        f"{key} {format_number(args[key])}"
        for key, default in _DEFAULT_BAR_OPTIONS.items()
        if _differs(args.get(key), default)
    ]
    if args.get("labels_above"):
        options.append("labels above " + "/".join(str(label) for label in _as_list(args["labels_above"])))
    if args.get("fill_color") not in (None, _DEFAULT_AREA_COLOR):
        options.append(f"fill {args['fill_color']}")
    if args.get("stroke_color"):
        options.append(f"stroke {args['stroke_color']}")
    line = f"{item.get('name', '?')} = BarChart({pairs})" + (("  " + " ".join(options)) if options else "")
    handled = {"plot_type", "values", "labels_below", "labels_above", "fill_color", "stroke_color", "fill_opacity"}
    return line + _extras_suffix(args, handled | set(_DEFAULT_BAR_OPTIONS))


def _distribution_head(args: Mapping[str, Any]) -> str:
    params = args.get("distribution_params") if isinstance(args.get("distribution_params"), dict) else {}
    parts = [str(args.get("distribution_type") or "distribution")]
    if params:
        parts.append(", ".join(f"{key} {format_number(value)}" for key, value in params.items()))
    bounds = args.get("bounds")
    if isinstance(bounds, dict) and (bounds.get("left") is not None or bounds.get("right") is not None):
        parts.append(f"on {_range_text(bounds.get('left'), bounds.get('right'))}")
    return "; ".join(parts)


def _render_continuous_plot(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    line = f"{item.get('name', '?')} = Distribution({_distribution_head(args)})"
    refs = []
    if args.get("function_name"):
        refs.append(f"curve {args['function_name']}")
    if args.get("fill_area_name"):
        refs.append(f"fill {args['fill_area_name']}")
    if refs:
        line += "  " + ", ".join(refs)
    handled = {"plot_type", "distribution_type", "distribution_params", "bounds", "function_name", "fill_area_name"}
    return line + _extras_suffix(args, handled | {"metadata"})


def _render_discrete_plot(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    line = f"{item.get('name', '?')} = DiscreteDistribution({_distribution_head(args)})"
    if _is_number(args.get("bar_count")):
        line += f"  {format_number(args['bar_count'])} bars"
    if args.get("bar_labels"):
        line += " labelled " + "/".join(str(label) for label in _as_list(args["bar_labels"]))
    handled = {
        "plot_type",
        "distribution_type",
        "distribution_params",
        "bounds",
        "metadata",
        "bar_count",
        "bar_labels",
        "curve_color",
        "fill_color",
        "fill_opacity",
        "rectangle_names",
        "fill_area_names",
    }
    return line + _extras_suffix(args, handled)


def _render_plot(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    line = f"{item.get('name', '?')} = Plot({args.get('plot_type') or 'plot'}; {_distribution_head(args)})"
    return line + _extras_suffix(args, {"plot_type", "distribution_type", "distribution_params", "bounds", "metadata"})


def _render_bar(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    line = (
        f"{item.get('name', '?')} = Bar(x {_range_text(args.get('x_left'), args.get('x_right'))}, "
        f"y {_range_text(args.get('y_bottom'), args.get('y_top'))})"
    )
    texts = [str(args[key]) for key in ("label_text", "label_above_text", "label_below_text") if args.get(key)]
    if texts:
        line += "  labels " + "/".join(f'"{text}"' for text in texts)
    handled = {
        "x_left",
        "x_right",
        "y_bottom",
        "y_top",
        "label_text",
        "label_above_text",
        "label_below_text",
        "stroke_color",
        "fill_color",
        "fill_opacity",
    }
    return line + _extras_suffix(args, handled)


def _render_label(scene: _Scene, item: JsonDict) -> str:
    args = _args(item)
    position = _position(args)
    text = str(args.get("text", "")).replace("\n", "\\n")
    line = f'{item.get("name", "?")} = Text("{text}" at {_point_text(position.get("x"), position.get("y"))})'
    options = []
    if _differs(args.get("rotation_degrees"), 0.0):
        options.append(f"rotation {format_number(args['rotation_degrees'])} deg")
    if _differs(args.get("font_size"), _DEFAULT_LABEL_FONT_SIZE):
        options.append(f"font {format_number(args['font_size'])}")
    if args.get("visible") is False:
        options.append("hidden")
    render_mode = args.get("render_mode")
    if isinstance(render_mode, dict) and render_mode.get("kind") not in (None, "world"):
        options.append(f"render_mode {_compact_json(render_mode)}")
    line += (" " + " ".join(options)) if options else ""
    handled = {"position", "text", "color", "font_size", "rotation_degrees", "visible", "render_mode"}
    return line + _color_suffix(args) + _extras_suffix(args, handled)


def _generic_renderer(bucket: str) -> Renderer:
    kind = bucket[:-1] if bucket.endswith("s") else bucket

    def render(scene: _Scene, item: JsonDict) -> str:
        details = {key: value for key, value in item.items() if key != "name" and key not in _RENDER_ONLY_FIELDS}
        if isinstance(details.get("args"), dict):
            details["args"] = {k: v for k, v in details["args"].items() if k not in _RENDER_ONLY_FIELDS}
        return f"{item.get('name', '?')} = {kind}{_compact_json(details)}"

    return render


@dataclass(frozen=True)
class _BucketSpec:
    bucket: str
    group: str
    render: Renderer


# Output order: dependencies before the objects that reference them.
_BUCKET_SPECS: Tuple[_BucketSpec, ...] = (
    _BucketSpec("Points", "points", _render_point),
    _BucketSpec("Segments", "segments", _render_segment),
    _BucketSpec("Vectors", "vectors", _render_vector),
    _BucketSpec("Triangles", "polygons", _polygon_renderer("Triangle")),
    _BucketSpec("Rectangles", "polygons", _polygon_renderer("Rectangle", reorder=True)),
    _BucketSpec("Quadrilaterals", "polygons", _polygon_renderer("Quadrilateral")),
    _BucketSpec("Pentagons", "polygons", _polygon_renderer("Pentagon")),
    _BucketSpec("Hexagons", "polygons", _polygon_renderer("Hexagon")),
    _BucketSpec("Heptagons", "polygons", _polygon_renderer("Heptagon")),
    _BucketSpec("Octagons", "polygons", _polygon_renderer("Octagon")),
    _BucketSpec("Nonagons", "polygons", _polygon_renderer("Nonagon")),
    _BucketSpec("Decagons", "polygons", _polygon_renderer("Decagon")),
    _BucketSpec("GenericPolygons", "polygons", _polygon_renderer("Polygon")),
    _BucketSpec("Circles", "circles and arcs", _render_circle),
    _BucketSpec("Ellipses", "circles and arcs", _render_ellipse),
    _BucketSpec("CircleArcs", "circles and arcs", _render_arc),
    _BucketSpec("Angles", "angles", _render_angle),
    _BucketSpec("Functions", "functions", _render_function),
    _BucketSpec("PiecewiseFunctions", "functions", _render_piecewise),
    _BucketSpec("ParametricFunctions", "functions", _render_parametric),
    _BucketSpec("FunctionsBoundedColoredAreas", "shaded areas", _render_functions_area),
    _BucketSpec("FunctionSegmentBoundedColoredAreas", "shaded areas", _render_function_segment_area),
    _BucketSpec("SegmentsBoundedColoredAreas", "shaded areas", _render_segments_area),
    _BucketSpec("ClosedShapeColoredAreas", "shaded areas", _render_closed_shape_area),
    _BucketSpec("UndirectedGraphs", "graphs", _graph_renderer("UndirectedGraphs")),
    _BucketSpec("DirectedGraphs", "graphs", _graph_renderer("DirectedGraphs")),
    _BucketSpec("Trees", "graphs", _graph_renderer("Trees")),
    _BucketSpec("Graphs", "graphs", _graph_renderer("Graphs")),
    _BucketSpec("BarsPlots", "plots", _render_bars_plot),
    _BucketSpec("ContinuousPlots", "plots", _render_continuous_plot),
    _BucketSpec("DiscretePlots", "plots", _render_discrete_plot),
    _BucketSpec("Plots", "plots", _render_plot),
    _BucketSpec("Bars", "plots", _render_bar),
    _BucketSpec("Labels", "text labels", _render_label),
)
_KNOWN_BUCKETS = frozenset(spec.bucket for spec in _BUCKET_SPECS)

# Budget degradation shrinks groups tier by tier; functions, graphs and polygons go last.
# Groups in one tier shrink together (same fraction kept) so no single kind vanishes first.
_TRUNCATION_TIERS: Tuple[Tuple[str, ...], ...] = (
    ("text labels", "computations", "other objects"),
    ("points", "segments", "vectors", "angles"),
    ("shaded areas", "plots", "circles and arcs"),
    ("polygons", "functions", "graphs"),
)
# Steps of the kept-fraction search per tier.
_FRACTION_SEARCH_STEPS = 10


# --------------------------------------------------------------------------- object entries


@dataclass
class _Entry:
    key: Tuple[str, str]
    text: str


@dataclass
class _Group:
    name: str
    entries: List[_Entry] = field(default_factory=list)
    shown: Optional[int] = None  # None == all entries
    packed: bool = False

    def visible_entries(self) -> List[_Entry]:
        return self.entries if self.shown is None else self.entries[: self.shown]


def _entry_key(bucket: str, name: str, occurrence: int) -> Tuple[str, str]:
    return (bucket, name if occurrence == 0 else f"{name}#{occurrence + 1}")


def _bucket_entries(scene: _Scene, bucket: str, render: Renderer) -> List[_Entry]:
    entries: List[_Entry] = []
    occurrences: Dict[str, int] = {}
    for item in _items(scene.state, bucket):
        name = str(item.get("name", ""))
        occurrence = occurrences.get(name, 0)
        occurrences[name] = occurrence + 1
        text = render(scene, item)
        if text is not None:
            entries.append(_Entry(_entry_key(bucket, name, occurrence), text))
    return entries


def _duplicate_warning(state: Mapping[str, Any]) -> Optional[str]:
    """One line naming duplicated object names (tools address objects by name)."""
    counts: List[str] = []
    for bucket, names in _duplicate_names(state).items():
        items = _items(state, bucket)
        for name in sorted(names):
            total = sum(1 for item in items if str(item.get("name", "")) == name)
            counts.append(f"{name} x{total} ({_kind_of((bucket, name))}s)")
    if not counts:
        return None
    return "! duplicate names, tools cannot tell these apart: " + ", ".join(counts)


def _computation_entries(state: Mapping[str, Any]) -> List[_Entry]:
    entries: List[_Entry] = []
    computations = state.get(_COMPUTATIONS_KEY)
    if not isinstance(computations, list):
        return entries
    for index, computation in enumerate(computations):
        if not isinstance(computation, dict):
            continue
        expression = str(computation.get("expression", ""))
        result = computation.get("result")
        result_text = (
            format_number(result)
            if _is_number(result)
            else (result if isinstance(result, str) else _compact_json(result))
        )
        if len(result_text) > _MAX_COMPUTATION_RESULT_CHARS:
            result_text = result_text[:_MAX_COMPUTATION_RESULT_CHARS] + "..."
        entries.append(_Entry((_COMPUTATIONS_KEY, expression or str(index)), f"calc {expression} = {result_text}"))
    return entries


def _collect_groups(state: Mapping[str, Any]) -> List[_Group]:
    """Render every object in the state, grouped, in output order."""
    scene = _Scene(state)
    groups: Dict[str, _Group] = {}
    for spec in _BUCKET_SPECS:
        entries = _bucket_entries(scene, spec.bucket, spec.render)
        if entries:
            groups.setdefault(spec.group, _Group(spec.group)).entries.extend(entries)
    for bucket, items in state.items():
        if bucket in _KNOWN_BUCKETS or bucket in _VIEW_KEYS or bucket == _COMPUTATIONS_KEY:
            continue
        if isinstance(items, list):
            entries = _bucket_entries(scene, bucket, _generic_renderer(bucket))
        elif not _is_empty(items):
            entries = [_Entry(("", bucket), f"{bucket}: {_compact_json(items)}")]
        else:
            entries = []
        if entries:
            groups.setdefault("other objects", _Group("other objects")).entries.extend(entries)
    computations = _computation_entries(state)
    if computations:
        groups["computations"] = _Group("computations", computations)
    return list(groups.values())


def _view_line(state: Mapping[str, Any]) -> Optional[str]:
    visibility = state.get("Cartesian_System_Visibility")
    if not isinstance(visibility, dict):
        return None
    x_range = _range_text_sig(visibility.get("left_bound"), visibility.get("right_bound"))
    y_range = _range_text_sig(visibility.get("bottom_bound"), visibility.get("top_bound"))
    parts = [f"view x {x_range} y {y_range}"]
    if _is_number(state.get("current_tick_spacing")):
        parts.append(f"grid {format_number(state['current_tick_spacing'])}")
    coordinate_system = state.get("coordinate_system")
    mode = coordinate_system.get("mode") if isinstance(coordinate_system, dict) else None
    if mode and mode != "cartesian":
        parts.append(f"{mode} coordinates")
    if state.get("visible") is False:
        parts.append("axes hidden")
    if mode == "polar" and coordinate_system.get("polar_grid_visible") is False:
        parts.append("polar grid hidden")
    return "; ".join(parts)


def _range_text_sig(low: Any, high: Any) -> str:
    return f"[{format_number(low, VIEW_SIGNIFICANT_DIGITS)}, {format_number(high, VIEW_SIGNIFICANT_DIGITS)}]"


# --------------------------------------------------------------------------- text format


def render_text(
    state: Mapping[str, Any],
    budget_tokens: Optional[int] = None,
    count_tokens: Callable[[str], int] = estimate_tokens_from_text,
    view_note: Optional[str] = None,
) -> str:
    """Render the state one object per line with engine-computed facts.

    Args:
        state: A ``get_canvas_state()`` dict (possibly filtered).
        budget_tokens: Optional token budget; larger scenes are packed and truncated.
        count_tokens: Token counter used for the budget (heuristic by default).
        view_note: Optional ``view_note`` line, placed under the view line; header
            lines are never trimmed, so it survives the budget.
    """
    state = _single_line_strings(state)
    return _render_text_groups(state, _collect_groups(state), budget_tokens, count_tokens, view_note)


def _render_text_groups(
    state: Mapping[str, Any],
    groups: List[_Group],
    budget_tokens: Optional[int],
    count_tokens: Callable[[str], int] = estimate_tokens_from_text,
    view_note: Optional[str] = None,
) -> str:
    """Assemble already-rendered groups of a single-line state, fitting the budget."""
    header = [line for line in (_view_line(state), view_note, _duplicate_warning(state)) if line]
    text = _assemble(header, groups)
    if budget_tokens is None or budget_tokens <= 0 or count_tokens(text) <= budget_tokens:
        return text
    return _fit_to_budget(header, groups, budget_tokens, count_tokens)


def _assemble(header: Sequence[str], groups: Sequence[_Group]) -> str:
    lines = list(header)
    for group in groups:
        entries = group.visible_entries()
        if group.packed:
            packed = [entry.text.replace(" = ", "=", 1) for entry in entries]
            lines.extend(
                "; ".join(packed[i : i + _POINTS_PER_PACKED_ROW]) for i in range(0, len(packed), _POINTS_PER_PACKED_ROW)
            )
        else:
            lines.extend(entry.text for entry in entries)
        omitted = len(group.entries) - len(entries)
        if omitted > 0:
            lines.append(f"... {omitted} more {group.name} omitted; {OMITTED_NOTE}")
    if not groups:
        lines.append("(empty canvas)")
    return "\n".join(lines)


def _fit_to_budget(
    header: Sequence[str],
    groups: List[_Group],
    budget_tokens: int,
    count_tokens: Callable[[str], int],
) -> str:
    """Pack points, then shrink groups tier by tier (least important first) until the text fits."""

    def fits() -> bool:
        return count_tokens(_assemble(header, groups)) <= budget_tokens

    for group in groups:
        if group.name == "points" and len(group.entries) > _PACK_POINTS_THRESHOLD:
            group.packed = True
    if fits():
        return _assemble(header, groups)

    for tier in _TRUNCATION_TIERS:
        members = [group for group in groups if group.name in tier]
        if not members:
            continue
        # Largest kept fraction that fits (fitting is monotone in the fraction).
        low, high = 0.0, 1.0
        for _ in range(_FRACTION_SEARCH_STEPS):
            middle = (low + high) / 2.0
            _keep_fraction(members, middle)
            if fits():
                low = middle
            else:
                high = middle
        _keep_fraction(members, low)
        if fits():
            break
    return _assemble(header, groups)


def _keep_fraction(groups: Sequence[_Group], fraction: float) -> None:
    for group in groups:
        group.shown = math.ceil(len(group.entries) * fraction)


# --------------------------------------------------------------------------- compact JSON format


def render_min_json(
    state: Mapping[str, Any],
    budget_tokens: Optional[int] = None,
    count_tokens: Callable[[str], int] = estimate_tokens_from_text,
    view_note: Optional[str] = None,
) -> str:
    """Return the state as minified JSON without render-only fields, defaults or float noise.

    Over ``budget_tokens``, every object list is cut to the same kept fraction and
    ``"omitted"`` (counts per bucket) plus ``"note"`` say how to get the rest.
    A ``view_note`` goes into a ``"view_note"`` key, which trimming never removes.
    """
    output = _min_json_output(state)
    if view_note:
        output["view_note"] = view_note
    text = _dump_min_json(output)
    if budget_tokens is None or budget_tokens <= 0 or count_tokens(text) <= budget_tokens:
        return text
    return _fit_min_json_to_budget(output, budget_tokens, count_tokens)


def _dump_min_json(output: Mapping[str, Any]) -> str:
    return json.dumps(round_numbers(output), separators=(",", ":"), ensure_ascii=False)


def _fit_min_json_to_budget(output: JsonDict, budget_tokens: int, count_tokens: Callable[[str], int]) -> str:
    lists = {key: value for key, value in output.items() if isinstance(value, list) and value and key != "view"}

    def build(fraction: float) -> str:
        trimmed = dict(output)
        omitted: Dict[str, int] = {}
        for key, items in lists.items():
            kept = math.ceil(len(items) * fraction)
            trimmed[key] = items[:kept]
            if kept < len(items):
                omitted[key] = len(items) - kept
        if omitted:
            trimmed["omitted"] = omitted
            trimmed["note"] = OMITTED_NOTE
        return _dump_min_json(trimmed)

    low, high = 0.0, 1.0
    for _ in range(_FRACTION_SEARCH_STEPS):
        middle = (low + high) / 2.0
        if count_tokens(build(middle)) <= budget_tokens:
            low = middle
        else:
            high = middle
    return build(low)


def _min_json_output(state: Mapping[str, Any]) -> JsonDict:
    output: JsonDict = {}
    visibility = state.get("Cartesian_System_Visibility")
    if isinstance(visibility, dict):
        output["view"] = [visibility.get(k) for k in ("left_bound", "right_bound", "bottom_bound", "top_bound")]
    if "current_tick_spacing" in state:
        output["grid"] = state["current_tick_spacing"]
    coordinate_system = state.get("coordinate_system")
    if isinstance(coordinate_system, dict) and coordinate_system.get("mode") not in (None, "cartesian"):
        output["coords"] = coordinate_system["mode"]
    if state.get("visible") is False:
        output["axes_hidden"] = True
    if output.get("coords") == "polar" and coordinate_system.get("polar_grid_visible") is False:
        output["polar_grid_hidden"] = True
    for bucket, items in state.items():
        if bucket in _VIEW_KEYS or _is_empty(items):
            continue
        if bucket == _COMPUTATIONS_KEY or not isinstance(items, list):
            output[bucket] = items
            continue
        output[bucket] = [_min_json_item(item, bucket) for item in items if isinstance(item, dict)]
    return output


def _min_json_item(item: Mapping[str, Any], bucket: str = "") -> JsonDict:
    args = {k: v for k, v in _args(item).items() if k not in _RENDER_ONLY_FIELDS and not _is_empty(v)}
    default_colors = _DEFAULT_ANGLE_COLORS if bucket == "Angles" else _DEFAULT_COLORS
    if isinstance(args.get("color"), str) and args["color"] in default_colors:
        args.pop("color", None)
    label = args.get("label")
    if isinstance(label, dict):
        if not label.get("text"):
            args.pop("label")
        elif label.get("visible", True):
            args["label"] = label["text"]
    if isinstance(args.get("position"), dict):
        args["position"] = [args["position"].get("x"), args["position"].get("y")]
    if args.get("font_size") == _DEFAULT_LABEL_FONT_SIZE:
        args.pop("font_size")
    if args.get("rotation_degrees") == 0:
        args.pop("rotation_degrees")
    if args.get("visible") is True:
        args.pop("visible")
    if isinstance(args.get("render_mode"), dict) and args["render_mode"].get("kind") == "world":
        args.pop("render_mode")
    entry: JsonDict = {"name": item.get("name")}
    entry.update(args)
    for key, value in item.items():
        if key not in ("name", "args", "type") and key not in _RENDER_ONLY_FIELDS and not _is_empty(value):
            entry[key] = value
    return entry


# --------------------------------------------------------------------------- deltas


def _object_lines(groups: Sequence[_Group]) -> Dict[Tuple[str, str], str]:
    lines: Dict[Tuple[str, str], str] = {}
    for group in groups:
        for entry in group.entries:
            lines[entry.key] = entry.text.replace("\n  ", "; ")
    return lines


def _kind_of(key: Tuple[str, str]) -> str:
    bucket = key[0]
    if bucket == _COMPUTATIONS_KEY:
        return "computation"
    kind = bucket[:-1] if bucket.endswith("s") else bucket
    return re.sub(r"(?<!^)(?=[A-Z])", " ", kind).lower() or "object"


def render_delta(previous: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    """List what changed between two states, one line per object; empty when nothing changed.

    ``+`` added, ``~`` changed (old line, then the new definition), ``-`` removed.
    Objects are compared by their rendered line, so a moved point also reports
    the segments, polygons and angles whose lengths/areas/sizes changed with it.
    """
    previous, current = _single_line_strings(previous), _single_line_strings(current)
    return _delta_text(previous, current, _collect_groups(previous), _collect_groups(current))


def _delta_text(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    previous_groups: Sequence[_Group],
    current_groups: Sequence[_Group],
) -> str:
    before, after = _object_lines(previous_groups), _object_lines(current_groups)
    added = [f"+ {text}" for key, text in after.items() if key not in before]
    changed = [
        f"~ {before[key]}  ->  {text.split(' = ', 1)[-1]}"
        for key, text in after.items()
        if key in before and before[key] != text
    ]
    removed = [f"- {key[1]} ({_kind_of(key)}) removed" for key in before if key not in after]
    view_before, view_after = _view_line(previous), _view_line(current)
    if view_after and view_before != view_after:
        changed.append(f"~ {view_after}")
    return "\n".join(added + changed + removed)


# --------------------------------------------------------------------------- view note
# The app never moves or zooms the view on its own. When the shapes are too small on
# screen, or (partly) outside the view, the model gets one "View note:" line so it can
# offer the user a zoom; the system prompt tells it to change the view only when the
# user asks or agrees. The view is a uniform linear map (CoordinateMapper.math_to_screen),
# so pixel sizes follow exactly from the view bounds and the canvas size in CSS pixels.
#
# Only bounded objects are measured: points (so every segment, vector, polygon, angle
# and graph through them), circles, ellipses, arcs (by their whole circle), text labels,
# bars and bar charts. Function graphs, curves, shaded areas and distribution plots
# have no bounded extent in the state and are left out.
#
# Repeats: the whole-scene problems (too small, mostly outside the view) are reported
# only when the shapes' bounding box differs from the one in the last canvas the model
# was shown, and "changed objects outside the view" only for objects that are new or
# changed since then. So a declined offer is not repeated when the user replies, pans
# or zooms; it comes back only when the drawing itself grows, shrinks or moves.

VIEW_NOTE_PREFIX = "View note:"
# "Too small": the shapes' larger side on screen is under max(TINY_PX, TINY_CANVAS_FRACTION
# x the canvas's smaller side). Point labels are 14 px text, so below about 40 px the labels
# of neighbouring points cover each other and the shape; the fraction keeps the rule
# proportional on very large canvases (smaller side above about 1330 px).
TINY_PX = 40.0
TINY_CANVAS_FRACTION = 0.03
# Without a canvas size (older clients): under this fraction of the view's smaller side,
# which is 40 px of an 800 px canvas.
TINY_VIEW_FRACTION = 0.05
# "Outside the view": less than this fraction of the shapes' bounding box is visible.
MIN_VISIBLE_FRACTION = 0.5
# The suggested view shows the shapes' bounding box enlarged this much around its centre.
SUGGESTED_VIEW_MARGIN = 1.25
_MAX_NOTE_NAMES = 3
# Two boxes are the same when no edge moved by more than this fraction of their size.
_SAME_BOX_RELATIVE_TOLERANCE = 1e-6


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

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def center(self) -> Point2D:
        return ((self.left + self.right) / 2.0, (self.bottom + self.top) / 2.0)

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

    def fraction_inside(self, view: "Box") -> float:
        """Share of this box inside ``view`` (of its length when it is flat, 0 or 1 for a point)."""
        return _axis_fraction(self.left, self.right, view.left, view.right) * _axis_fraction(
            self.bottom, self.top, view.bottom, view.top
        )

    def same_as(self, other: "Box") -> bool:
        scale = max(self.width, self.height, other.width, other.height)
        tolerance = _SAME_BOX_RELATIVE_TOLERANCE * scale
        return all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=tolerance)
            for a, b in zip(
                (self.left, self.right, self.bottom, self.top), (other.left, other.right, other.bottom, other.top)
            )
        )


def _axis_fraction(low: float, high: float, view_low: float, view_high: float) -> float:
    if high <= low:
        return 1.0 if view_low <= low <= view_high else 0.0
    return max(0.0, min(high, view_high) - max(low, view_low)) / (high - low)


@dataclass(frozen=True)
class ViewSummary:
    """Where the measurable shapes are relative to the view (see ``summarize_view``)."""

    view: Box
    canvas_px: Optional[Tuple[float, float]]
    objects: Dict[Tuple[str, str], Box]
    content: Optional[Box]

    @property
    def pixels_per_unit(self) -> Optional[float]:
        return self.canvas_px[0] / self.view.width if self.canvas_px else None

    def content_px(self) -> Optional[Tuple[float, float]]:
        """The shapes' bounding box size in screen pixels, when the canvas size is known."""
        scale = self.pixels_per_unit
        if scale is None or self.content is None:
            return None
        return (self.content.width * scale, self.content.height * scale)

    def visible_fraction(self) -> float:
        return self.content.fraction_inside(self.view) if self.content is not None else 1.0

    def is_tiny(self) -> bool:
        """True when the shapes have a size but it is too small to read on screen."""
        if self.content is None:
            return False
        size = max(self.content.width, self.content.height)
        if size <= 0:
            return False  # one point (or several on one spot) is readable at any zoom
        if self.canvas_px is not None and self.pixels_per_unit is not None:
            threshold = max(TINY_PX, TINY_CANVAS_FRACTION * min(self.canvas_px))
            return size * self.pixels_per_unit < threshold
        return size < TINY_VIEW_FRACTION * min(self.view.width, self.view.height)


def summarize_view(state: Mapping[str, Any]) -> Optional[ViewSummary]:
    """Measure the state's bounded shapes against its view; None without a usable view."""
    view = _view_box(state)
    if view is None:
        return None
    objects = _object_extents(state)
    return ViewSummary(view, _canvas_px(state), objects, _union_box(objects.values()))


def _view_box(state: Mapping[str, Any]) -> Optional[Box]:
    visibility = state.get("Cartesian_System_Visibility")
    if not isinstance(visibility, dict):
        return None
    bounds = [_as_float(visibility.get(key)) for key in ("left_bound", "right_bound", "bottom_bound", "top_bound")]
    left, right, bottom, top = bounds
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


def _union_box(boxes: Any) -> Optional[Box]:
    result: Optional[Box] = None
    for box in boxes:
        result = box if result is None else result.union(box)
    return result


def _keyed_items(state: Mapping[str, Any], bucket: str) -> List[Tuple[Tuple[str, str], JsonDict]]:
    keyed: List[Tuple[Tuple[str, str], JsonDict]] = []
    occurrences: Dict[str, int] = {}
    for item in _items(state, bucket):
        name = str(item.get("name", ""))
        occurrence = occurrences.get(name, 0)
        occurrences[name] = occurrence + 1
        keyed.append((_entry_key(bucket, name, occurrence), item))
    return keyed


def _position_xy(args: Mapping[str, Any]) -> Optional[Point2D]:
    position = _position(args)
    x, y = _as_float(position.get("x")), _as_float(position.get("y"))
    return (x, y) if x is not None and y is not None else None


def _object_extents(state: Mapping[str, Any]) -> Dict[Tuple[str, str], Box]:
    """The math-unit bounding box of every bounded object, keyed like the rendered entries."""
    extents: Dict[Tuple[str, str], Box] = {}
    positions: Dict[str, Point2D] = {}
    for bucket in ("Points", "Labels"):
        for key, item in _keyed_items(state, bucket):
            xy = _position_xy(_args(item))
            if xy is None:
                continue
            extents[key] = Box.around(*xy)
            if bucket == "Points":
                positions.setdefault(str(item.get("name", "")), xy)
    for bucket, measure in _EXTENT_MEASURES:
        for key, item in _keyed_items(state, bucket):
            box = measure(_args(item), positions)
            if box is not None and all(math.isfinite(v) for v in (box.left, box.right, box.bottom, box.top)):
                extents[key] = box
    return extents


def _circle_extent(args: Mapping[str, Any], positions: Mapping[str, Point2D]) -> Optional[Box]:
    center = positions.get(str(args.get("center")))
    radius = _as_float(args.get("radius"))
    if center is None or radius is None or radius < 0:
        return None
    return Box.around(center[0], center[1], radius, radius)


def _ellipse_extent(args: Mapping[str, Any], positions: Mapping[str, Point2D]) -> Optional[Box]:
    center = positions.get(str(args.get("center")))
    rx, ry = _as_float(args.get("radius_x")), _as_float(args.get("radius_y"))
    if center is None or rx is None or ry is None:
        return None
    angle = math.radians(_as_float(args.get("rotation_angle")) or 0.0)
    cos, sin = math.cos(angle), math.sin(angle)
    half_width = math.hypot(rx * cos, ry * sin)
    half_height = math.hypot(rx * sin, ry * cos)
    return Box.around(center[0], center[1], half_width, half_height)


def _arc_extent(args: Mapping[str, Any], positions: Mapping[str, Point2D]) -> Optional[Box]:
    cx, cy, radius = _as_float(args.get("center_x")), _as_float(args.get("center_y")), _as_float(args.get("radius"))
    if cx is None or cy is None or radius is None or radius < 0:
        return None
    return Box.around(cx, cy, radius, radius)


def _bar_extent(args: Mapping[str, Any], positions: Mapping[str, Point2D]) -> Optional[Box]:
    values = [_as_float(args.get(key)) for key in ("x_left", "x_right", "y_bottom", "y_top")]
    left, right, bottom, top = values
    if left is None or right is None or bottom is None or top is None:
        return None
    return Box(min(left, right), max(left, right), min(bottom, top), max(bottom, top))


def _bars_plot_extent(args: Mapping[str, Any], positions: Mapping[str, Point2D]) -> Optional[Box]:
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
    return Box(x_start, right, y_base + min(0.0, *values), y_base + max(0.0, *values))


_ExtentMeasure = Callable[[Mapping[str, Any], Mapping[str, Point2D]], Optional[Box]]
_EXTENT_MEASURES: Tuple[Tuple[str, _ExtentMeasure], ...] = (
    ("Circles", _circle_extent),
    ("Ellipses", _ellipse_extent),
    ("CircleArcs", _arc_extent),
    ("Bars", _bar_extent),
    ("BarsPlots", _bars_plot_extent),
)


def view_note(previous: Optional[Mapping[str, Any]], current: Mapping[str, Any]) -> Optional[str]:
    """One "View note:" line when the shapes in ``current`` are hard to see, else None.

    ``previous`` is the last canvas state the model was shown (None at the start of a
    conversation); it decides what counts as new, so a note is not repeated for a
    drawing that did not change (see the comment above ``VIEW_NOTE_PREFIX``). Never raises.
    """
    try:
        note = _view_note(previous, current)
        # Object names go into the note; it must stay one line.
        return _single_line_strings(note) if note is not None else None
    except Exception:
        _logger.warning("Could not measure the canvas view; sending no view note", exc_info=True)
        return None


def _view_note(previous: Optional[Mapping[str, Any]], current: Mapping[str, Any]) -> Optional[str]:
    summary = summarize_view(current)
    if summary is None or summary.content is None:
        return None
    before = _object_extents(previous) if isinstance(previous, Mapping) else None
    before_content = _union_box(before.values()) if before is not None else None
    drawing_changed = before_content is None or not before_content.same_as(summary.content)
    problem: Optional[str] = None
    if drawing_changed and summary.visible_fraction() < MIN_VISIBLE_FRACTION:
        problem = _outside_problem(summary)
    elif before is not None:
        problem = _changed_objects_outside_problem(summary, before)
    if problem is None and drawing_changed and summary.is_tiny():
        problem = _tiny_problem(summary)
    if problem is None:
        return None
    return (
        f"{VIEW_NOTE_PREFIX} {problem}. Offer to {_suggestion(summary)}; don't change the view unless the user agrees."
    )


def _outside_problem(summary: ViewSummary) -> str:
    fraction = summary.visible_fraction()
    where = _content_and_view(summary)
    if fraction <= 0:
        return f"the shapes are entirely outside the view ({where})"
    percent = "<1" if fraction < 0.01 else f"~{round(fraction * 100)}"
    return f"only {percent}% of the shapes' extent is inside the view ({where})"


def _changed_objects_outside_problem(summary: ViewSummary, before: Mapping[Tuple[str, str], Box]) -> Optional[str]:
    outside = [
        key[1]
        for key, box in summary.objects.items()
        if (key not in before or not box.same_as(before[key])) and not box.intersects(summary.view)
    ]
    if not outside:
        return None
    names = ", ".join(outside[:_MAX_NOTE_NAMES])
    if len(outside) > _MAX_NOTE_NAMES:
        names += f" (+{len(outside) - _MAX_NOTE_NAMES} more)"
    verb = "is" if len(outside) == 1 else "are"
    return f"new or changed {names} {verb} outside the view ({_view_ranges(summary.view)})"


def _tiny_problem(summary: ViewSummary) -> str:
    size = summary.content_px()
    content = summary.content
    assert content is not None
    if size is not None:
        extent = f"~{_pixels(size[0])}x{_pixels(size[1])} px on screen"
    else:
        share = max(content.width / summary.view.width, content.height / summary.view.height) * 100
        extent = f"~{format_number(share, 2)}% of the view"
    return f"the shapes span only {extent} ({_content_and_view(summary)})"


def _pixels(value: float) -> str:
    return "<1" if value < 1 else str(round(value))


def _content_and_view(summary: ViewSummary) -> str:
    content = summary.content
    assert content is not None
    step = _coordinate_step(max(content.width, content.height) / 2.0) or _coordinate_step(summary.view.width / 2.0)
    shapes = f"x {_span(content.left, content.right, step)}, y {_span(content.bottom, content.top, step)}"
    return f"shapes {shapes}; view {_view_ranges(summary.view)}"


def _view_ranges(view: Box) -> str:
    step = _coordinate_step(view.width / 2.0)
    return f"x {_span(view.left, view.right, step)}, y {_span(view.bottom, view.top, step)}"


def _suggestion(summary: ViewSummary) -> str:
    """A view showing every shape: zoom when they are tiny or too big, else just move the view."""
    content, view = summary.content, summary.view
    assert content is not None
    aspect = view.height / view.width
    fit = max(content.width / 2.0, content.height / 2.0 / aspect) * SUGGESTED_VIEW_MARGIN
    current = view.width / 2.0
    pan_only = fit <= 0 or (fit <= current and not summary.is_tiny())
    half = _nice_ceil(current if pan_only else fit)
    step = _coordinate_step(half)
    cx, cy = (_snap(value, step) for value in content.center)
    shown = Box.around(cx, cy, half, half * aspect)
    return (
        f"{'move the view' if pan_only else 'zoom'} to about "
        f"x {_span(shown.left, shown.right, step)}, y {_span(shown.bottom, shown.top, step)} "
        f"(zoom center_x={format_number(cx)}, center_y={format_number(cy)}, "
        f"range_val={format_number(half)}, range_axis=x)"
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
    return f"{format_number(_snap(low, step))}..{format_number(_snap(high, step))}"


# --------------------------------------------------------------------------- dispatch


def render_state(
    state: Mapping[str, Any],
    fmt: CanvasFormat,
    budget_tokens: Optional[int] = None,
    view_note: Optional[str] = None,
) -> str:
    """Render a state in the requested format (``json`` is the raw state, unchanged).

    ``view_note`` (see ``view_note``) becomes a line under the view in text and a
    ``"view_note"`` key in min_json and json. Never raises: a state the renderers
    cannot handle is sent as compact JSON instead, so one malformed object cannot
    fail the whole request.
    """
    try:
        if fmt == "text":
            return render_text(state, budget_tokens=budget_tokens, view_note=view_note)
        if fmt == "min_json":
            return render_min_json(state, budget_tokens=budget_tokens, view_note=view_note)
        return json.dumps(dict(state, view_note=view_note) if view_note else state)
    except Exception:
        _logger.warning("Could not render the canvas state as %s; sending compact JSON", fmt, exc_info=True)
        return _fallback_json(state)


def _fallback_json(state: Any) -> str:
    try:
        return json.dumps(state, separators=(",", ":"), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(state)


CHANGES_HEADER = "[canvas changes]"
CURRENT_HEADER = "[canvas now]"


def render_update(
    previous: Optional[Mapping[str, Any]],
    current: Mapping[str, Any],
    fmt: CanvasFormat,
    budget_tokens: Optional[int] = None,
    view_note: Optional[str] = None,
) -> str:
    """Describe the canvas after a tool batch; empty string when nothing changed.

    Sends the delta when it is smaller than the full rendering, else the full
    state (e.g. after clear_canvas or when no previous state was shown). The
    delta is the text of ``render_delta`` in every format (for min_json too:
    one changed object per line reads better than a JSON diff); a ``view_note``
    is its last line. Never raises: if the states cannot be compared, the
    current state is sent as JSON.
    """
    try:
        return _render_update(previous, current, fmt, budget_tokens, view_note)
    except Exception:
        _logger.warning("Could not describe the canvas changes; sending the state as JSON", exc_info=True)
        return f"{CURRENT_HEADER}\n{_fallback_json(current)}"


def _render_update(
    previous: Optional[Mapping[str, Any]],
    current: Mapping[str, Any],
    fmt: CanvasFormat,
    budget_tokens: Optional[int],
    view_note: Optional[str] = None,
) -> str:
    if previous is None:
        return f"{CURRENT_HEADER}\n{render_state(current, fmt, budget_tokens, view_note)}"
    # Each state is rendered once; the current groups serve both the delta and the full text.
    previous, current = _single_line_strings(previous), _single_line_strings(current)
    current_groups = _collect_groups(current)
    delta = _delta_text(previous, current, _collect_groups(previous), current_groups)
    if not delta:
        return f"{CHANGES_HEADER}\n{view_note}" if view_note else ""
    if fmt == "text":
        full = _render_text_groups(current, current_groups, budget_tokens, view_note=view_note)
    else:
        full = render_state(current, fmt, budget_tokens, view_note)
    if view_note:
        delta = f"{delta}\n{view_note}"
    if estimate_tokens_from_text(delta) >= estimate_tokens_from_text(full):
        return f"{CURRENT_HEADER}\n{full}"
    return f"{CHANGES_HEADER}\n{delta}"


def parse_canvas_format(raw: Optional[str]) -> Optional[CanvasFormat]:
    """Return the canvas format named by ``raw`` (case-insensitive), or None if unknown."""
    value = (raw or "").strip().lower()
    for fmt in CANVAS_FORMATS:
        if value == fmt:
            return fmt
    return None
