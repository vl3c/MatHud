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

Budget: ``render_text`` accepts ``budget_tokens``; large scenes first pack
points several per line, then drop the least important objects with a note
telling the model to call ``get_current_canvas_state`` for the rest.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple

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

# Longest list shown inline for asymptotes, discontinuities and similar lists.
_MAX_INLINE_LIST = 8
# Longest computation result shown before it is cut.
_MAX_COMPUTATION_RESULT_CHARS = 200
# Points are packed several per line when there are more than this many.
_PACK_POINTS_THRESHOLD = 8
_POINTS_PER_PACKED_ROW = 6

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
    }
)
_COMPUTATIONS_KEY = "computations"

_DEFAULT_COLORS = frozenset({"", "black", "blue"})
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
        if not isinstance(items, list):
            continue
        seen: Set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            if name in seen:
                duplicates.setdefault(bucket, set()).add(name)
            seen.add(name)
    return duplicates


# --------------------------------------------------------------------------- shared line pieces


def _color_suffix(args: Mapping[str, Any]) -> str:
    color = args.get("color")
    if _is_empty(color) or (isinstance(color, str) and color in _DEFAULT_COLORS):
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
        if center_xy is not None and r > 0:
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
    return line + _color_suffix(args) + _extras_suffix(args, {"segment1_name", "segment2_name", "is_reflex", "color"})


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
    return "; ".join(parts)


def _range_text_sig(low: Any, high: Any) -> str:
    return f"[{format_number(low, VIEW_SIGNIFICANT_DIGITS)}, {format_number(high, VIEW_SIGNIFICANT_DIGITS)}]"


# --------------------------------------------------------------------------- text format


def render_text(
    state: Mapping[str, Any],
    budget_tokens: Optional[int] = None,
    count_tokens: Callable[[str], int] = estimate_tokens_from_text,
) -> str:
    """Render the state one object per line with engine-computed facts.

    Args:
        state: A ``get_canvas_state()`` dict (possibly filtered).
        budget_tokens: Optional token budget; larger scenes are packed and truncated.
        count_tokens: Token counter used for the budget (heuristic by default).
    """
    header = [line for line in (_view_line(state), _duplicate_warning(state)) if line]
    groups = _collect_groups(state)
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


def render_min_json(state: Mapping[str, Any]) -> str:
    """Return the state as minified JSON without render-only fields, defaults or float noise."""
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
    for bucket, items in state.items():
        if bucket in _VIEW_KEYS or _is_empty(items):
            continue
        if bucket == _COMPUTATIONS_KEY or not isinstance(items, list):
            output[bucket] = items
            continue
        output[bucket] = [_min_json_item(item) for item in items if isinstance(item, dict)]
    return json.dumps(round_numbers(output), separators=(",", ":"), ensure_ascii=False)


def _min_json_item(item: Mapping[str, Any]) -> JsonDict:
    args = {k: v for k, v in _args(item).items() if k not in _RENDER_ONLY_FIELDS and not _is_empty(v)}
    if isinstance(args.get("color"), str) and args["color"] in _DEFAULT_COLORS:
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


def _object_lines(state: Mapping[str, Any]) -> Dict[Tuple[str, str], str]:
    lines: Dict[Tuple[str, str], str] = {}
    for group in _collect_groups(state):
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
    before, after = _object_lines(previous), _object_lines(current)
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


# --------------------------------------------------------------------------- dispatch


def render_state(state: Mapping[str, Any], fmt: CanvasFormat, budget_tokens: Optional[int] = None) -> str:
    """Render a state in the requested format (``json`` is the raw state, unchanged).

    Never raises: a state the renderers cannot handle is sent as compact JSON instead,
    so one malformed object cannot fail the whole request.
    """
    try:
        if fmt == "text":
            return render_text(state, budget_tokens=budget_tokens)
        if fmt == "min_json":
            return render_min_json(state)
        return json.dumps(state)
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
) -> str:
    """Describe the canvas after a tool batch; empty string when nothing changed.

    Sends the delta when it is smaller than the full rendering, else the full
    state (e.g. after clear_canvas or when no previous state was shown). Never
    raises: if the states cannot be compared, the current state is sent as JSON.
    """
    try:
        return _render_update(previous, current, fmt, budget_tokens)
    except Exception:
        _logger.warning("Could not describe the canvas changes; sending the state as JSON", exc_info=True)
        return f"{CURRENT_HEADER}\n{_fallback_json(current)}"


def _render_update(
    previous: Optional[Mapping[str, Any]],
    current: Mapping[str, Any],
    fmt: CanvasFormat,
    budget_tokens: Optional[int],
) -> str:
    full = render_state(current, fmt, budget_tokens)
    if previous is None:
        return f"{CURRENT_HEADER}\n{full}"
    delta = render_delta(previous, current)
    if not delta:
        return ""
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
