from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from rendering import shared_drawable_renderers as shared
from rendering.shared_drawable_renderers import RendererPrimitives

_STYLE_CLASSES = (
    shared.StrokeStyle,
    shared.FillStyle,
    shared.FontStyle,
    shared.TextAlignment,
)

Number = float | int
MapState = Dict[str, float]


PrimitiveArgs = Tuple[Any, ...]
PrimitiveKwargs = Dict[str, Any]


def _style_signature(style: Any) -> Optional[Tuple[Any, ...]]:
    if style is None:
        return None
    attrs = []
    for attr in ("color", "width", "line_join", "line_cap", "opacity", "family", "size", "weight"):
        if hasattr(style, attr):
            attrs.append(getattr(style, attr))
    return tuple(attrs) if attrs else None


def _geometry_signature(values: Iterable[Any]) -> Tuple[Any, ...]:
    signature: List[Any] = []
    for value in values:
        if isinstance(value, (tuple, list)):
            signature.append(tuple(value))
        else:
            signature.append(value)
    return tuple(signature)


def _quantize_number(value: Any, *, decimals: int = 4) -> Any:
    if isinstance(value, float):
        if value == 0.0:
            return 0.0
        try:
            return round(value, decimals)
        except Exception:
            return value
    return value


def _quantize_geometry(value: Any, *, decimals: int = 4) -> Any:
    if isinstance(value, (tuple, list)):
        return tuple(_quantize_geometry(v, decimals=decimals) for v in value)
    return _quantize_number(value, decimals=decimals)


def _capture_map_state(mapper: Any) -> MapState:
    origin = getattr(mapper, "origin", None)
    offset = getattr(mapper, "offset", None)
    return {
        "scale": float(getattr(mapper, "scale_factor", 1.0)),
        "offset_x": float(getattr(offset, "x", 0.0)) if offset is not None else 0.0,
        "offset_y": float(getattr(offset, "y", 0.0)) if offset is not None else 0.0,
        "origin_x": float(getattr(origin, "x", 0.0)) if origin is not None else 0.0,
        "origin_y": float(getattr(origin, "y", 0.0)) if origin is not None else 0.0,
    }


def _map_state_equal(left: MapState, right: MapState, *, epsilon: float = 1e-6) -> bool:
    if left is right:
        return True
    for key in ("scale", "offset_x", "offset_y", "origin_x", "origin_y"):
        lv = float(left.get(key, 0.0))
        rv = float(right.get(key, 0.0))
        if abs(lv - rv) > epsilon:
            return False
    return True


def _math_to_screen_point(math_point: Tuple[float, float], state: MapState) -> Tuple[float, float]:
    mx, my = math_point
    sx = state["origin_x"] + mx * state["scale"] + state["offset_x"]
    sy = state["origin_y"] - my * state["scale"] + state["offset_y"]
    return (sx, sy)


def _screen_to_math_point(screen_point: Tuple[float, float], state: MapState) -> Tuple[float, float]:
    sx, sy = screen_point
    scale = state["scale"] if state["scale"] else 1.0
    mx = (sx - state["offset_x"] - state["origin_x"]) / scale
    my = (state["origin_y"] + state["offset_y"] - sy) / scale
    return (mx, my)


def _reproject_points(points: Iterable[Tuple[float, float]], old: MapState, new: MapState) -> Tuple[Tuple[float, float], ...]:
    return tuple(_math_to_screen_point(_screen_to_math_point(point, old), new) for point in points)


def _reproject_radius(radius: float, old: MapState, new: MapState) -> float:
    scale_old = old["scale"] if old["scale"] else 1.0
    math_radius = radius / scale_old
    return math_radius * new["scale"]


def _reproject_command(command: PrimitiveCommand, old_state: MapState, new_state: MapState) -> None:
    op = command.op
    if not op:
        return

    if op == "stroke_line":
        start, end, stroke = command.args
        new_start = _math_to_screen_point(_screen_to_math_point(start, old_state), new_state)
        new_end = _math_to_screen_point(_screen_to_math_point(end, old_state), new_state)
        command.args = (new_start, new_end, stroke)
        command.meta["geometry"] = _quantize_geometry((new_start, new_end))
        return

    if op == "stroke_polyline":
        points, stroke = command.args
        new_points = _reproject_points(points, old_state, new_state)
        command.args = (new_points, stroke)
        command.meta["geometry"] = _quantize_geometry(new_points)
        return

    if op == "stroke_circle":
        center, radius, stroke = command.args
        new_center = _math_to_screen_point(_screen_to_math_point(center, old_state), new_state)
        new_radius = _reproject_radius(float(radius), old_state, new_state)
        command.args = (new_center, new_radius, stroke)
        command.meta["geometry"] = _quantize_geometry((new_center, new_radius))
        return

    if op == "fill_circle":
        center, radius, fill, stroke = command.args
        new_center = _math_to_screen_point(_screen_to_math_point(center, old_state), new_state)
        new_radius = _reproject_radius(float(radius), old_state, new_state)
        command.args = (new_center, new_radius, fill, stroke)
        command.meta["geometry"] = _quantize_geometry((new_center, new_radius))
        return

    if op == "stroke_ellipse":
        center, radius_x, radius_y, rotation, stroke = command.args
        new_center = _math_to_screen_point(_screen_to_math_point(center, old_state), new_state)
        new_rx = _reproject_radius(float(radius_x), old_state, new_state)
        new_ry = _reproject_radius(float(radius_y), old_state, new_state)
        command.args = (new_center, new_rx, new_ry, rotation, stroke)
        command.meta["geometry"] = _quantize_geometry((new_center, new_rx, new_ry, rotation))
        return

    if op == "fill_polygon":
        points, fill, stroke = command.args
        new_points = _reproject_points(points, old_state, new_state)
        command.args = (new_points, fill, stroke)
        command.meta["geometry"] = _quantize_geometry(new_points)
        return

    if op == "fill_joined_area":
        forward, reverse, fill = command.args
        new_forward = _reproject_points(forward, old_state, new_state)
        new_reverse = _reproject_points(reverse, old_state, new_state)
        command.args = (new_forward, new_reverse, fill)
        command.meta["geometry"] = _quantize_geometry(new_forward + new_reverse)
        return

    if op == "stroke_arc":
        center, radius, start_angle, end_angle, sweep_clockwise, stroke = command.args
        new_center = _math_to_screen_point(_screen_to_math_point(center, old_state), new_state)
        new_radius = _reproject_radius(float(radius), old_state, new_state)
        command.args = (new_center, new_radius, start_angle, end_angle, sweep_clockwise, stroke)
        command.meta["geometry"] = _quantize_geometry((new_center, new_radius))
        return

    if op == "draw_text":
        text, position, font, color, alignment = command.args
        new_position = _math_to_screen_point(_screen_to_math_point(position, old_state), new_state)
        command.args = (text, new_position, font, color, alignment)
        command.meta["geometry"] = _quantize_geometry((new_position,))
        return

    if op == "stroke_vector":
        # Not currently emitted, guard for completeness
        return

    if op == "stroke_line_with_arrow":
        # Not currently emitted
        return


def _drawable_key(drawable: Any, fallback: str) -> str:
    if drawable is None:
        return fallback
    name = getattr(drawable, "name", None)
    if isinstance(name, str) and name:
        return name
    identifier = getattr(drawable, "id", None)
    if isinstance(identifier, str) and identifier:
        return identifier
    return f"{fallback}:{id(drawable)}"


class _CachedCoordinateMapper:
    """Lightweight wrapper that caches math_to_screen and scale_value lookups."""

    def __init__(self, mapper: Any) -> None:
        self._mapper = mapper
        self._point_cache: Dict[Tuple[float, float], Tuple[float, float]] = {}
        self._scale_cache: Dict[Any, Any] = {}

    def math_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        key = (float(x), float(y))
        if key not in self._point_cache:
            self._point_cache[key] = self._mapper.math_to_screen(x, y)
        return self._point_cache[key]

    def scale_value(self, value: Any) -> Any:
        try:
            key = float(value)
        except Exception:
            key = value
        if key not in self._scale_cache:
            self._scale_cache[key] = self._mapper.scale_value(value)
        return self._scale_cache[key]

    def __getattr__(self, item: str) -> Any:
        return getattr(self._mapper, item)


class PrimitiveCommand:
    __slots__ = ("op", "args", "kwargs", "key", "meta")

    def __init__(
        self,
        op: str,
        args: PrimitiveArgs,
        kwargs: PrimitiveKwargs,
        key: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.op = op
        self.args = args
        self.kwargs = kwargs
        self.key = key
        self.meta = meta or {}


class OptimizedPrimitivePlan:
    __slots__ = ("drawable", "commands", "plan_key", "metadata", "_map_state")

    def __init__(
        self,
        *,
        drawable: Any,
        commands: List[PrimitiveCommand],
        plan_key: str,
        metadata: Dict[str, Any],
    ) -> None:
        self.drawable = drawable
        self.commands = commands
        self.plan_key = plan_key
        self.metadata = metadata
        self._map_state: Optional[MapState] = dict(metadata.get("map_state", {}))

    def update_map_state(self, new_state: MapState) -> None:
        if not new_state:
            return
        current_state = self._map_state or {}
        if _map_state_equal(current_state, new_state):
            self._map_state = dict(new_state)
            self.metadata["map_state"] = dict(new_state)
            return
        if not current_state:
            self._map_state = dict(new_state)
            self.metadata["map_state"] = dict(new_state)
            return
        for command in self.commands:
            _reproject_command(command, current_state, new_state)
        self._map_state = dict(new_state)
        self.metadata["map_state"] = dict(new_state)

    def apply(self, primitives: RendererPrimitives) -> None:
        if not self.commands:
            return
        primitives.begin_batch(self)
        try:
            for command in self.commands:
                primitives.execute_optimized(command)
        finally:
            primitives.end_batch(self)


class _RecordingPrimitives(shared.RendererPrimitives):
    def __init__(self, drawable_key: str) -> None:
        self.commands: List[PrimitiveCommand] = []
        self._drawable_key = drawable_key
        self._counter = 0
        self._style_pool: Dict[Tuple[Any, ...], Any] = {}

    def _style_signature(self, style: Any) -> Optional[Tuple[Any, ...]]:
        if style is None:
            return None
        if not isinstance(style, _STYLE_CLASSES):
            return None
        signature: List[Any] = [style.__class__.__name__]
        for attr in getattr(style, "__slots__", ()):
            signature.append(getattr(style, attr, None))
        return tuple(signature)

    def _pool_style(self, style: Any) -> Any:
        signature = self._style_signature(style)
        if signature is None:
            return style
        pooled = self._style_pool.get(signature)
        if pooled is None:
            self._style_pool[signature] = style
            return style
        return pooled

    def _pool_styles(self, value: Any) -> Any:
        if isinstance(value, _STYLE_CLASSES):
            return self._pool_style(value)
        if isinstance(value, tuple):
            return tuple(self._pool_styles(item) for item in value)
        if isinstance(value, list):
            return [self._pool_styles(item) for item in value]
        if isinstance(value, dict):
            return {key: self._pool_styles(item) for key, item in value.items()}
        return value

    def _record(self, op: str, args: PrimitiveArgs, kwargs: PrimitiveKwargs, *, style: Any = None, geometry: Iterable[Any] = ()) -> None:
        command_key = f"{self._drawable_key}:{op}:{self._counter}"
        self._counter += 1
        meta: Dict[str, Any] = {}
        style_sig = _style_signature(style)
        if style_sig is not None:
            meta["style"] = style_sig
        geometry_sig = _geometry_signature(geometry)
        if geometry_sig:
            meta["geometry"] = _quantize_geometry(geometry_sig)
        pooled_args = self._pool_styles(args)
        pooled_kwargs = self._pool_styles(kwargs)
        self.commands.append(PrimitiveCommand(op, pooled_args, pooled_kwargs, command_key, meta))

    def stroke_line(self, start, end, stroke, *, include_width=True):
        self._record("stroke_line", (start, end, stroke), {"include_width": include_width}, style=stroke, geometry=(start, end))

    def stroke_polyline(self, points, stroke):
        self._record("stroke_polyline", (tuple(points), stroke), {}, style=stroke, geometry=points)

    def stroke_circle(self, center, radius, stroke):
        self._record("stroke_circle", (center, radius, stroke), {}, style=stroke, geometry=(center, radius))

    def fill_circle(self, center, radius, fill, stroke=None):
        self._record("fill_circle", (center, radius, fill, stroke), {}, style=fill, geometry=(center, radius))

    def stroke_ellipse(self, center, radius_x, radius_y, rotation_rad, stroke):
        self._record("stroke_ellipse", (center, radius_x, radius_y, rotation_rad, stroke), {}, style=stroke, geometry=(center, radius_x, radius_y, rotation_rad))

    def fill_polygon(self, points, fill, stroke=None):
        self._record("fill_polygon", (tuple(points), fill, stroke), {}, style=fill, geometry=points)

    def fill_joined_area(self, forward, reverse, fill):
        self._record("fill_joined_area", (tuple(forward), tuple(reverse), fill), {}, style=fill, geometry=list(forward) + list(reverse))

    def stroke_arc(self, center, radius, start_angle_rad, end_angle_rad, sweep_clockwise, stroke, css_class=None):
        self._record(
            "stroke_arc",
            (center, radius, start_angle_rad, end_angle_rad, sweep_clockwise, stroke),
            {"css_class": css_class},
            style=stroke,
            geometry=(center, radius, start_angle_rad, end_angle_rad, sweep_clockwise),
        )

    def draw_text(self, text, position, font, color, alignment, style_overrides=None):
        self._record("draw_text", (text, position, font, color, alignment), {"style_overrides": style_overrides or {}}, style=font, geometry=(position,))

    def clear_surface(self):
        return None

    def resize_surface(self, width, height):
        return None


_HELPERS: Dict[str, Any] = {
    "Point": shared.render_point_helper,
    "Segment": shared.render_segment_helper,
    "Circle": shared.render_circle_helper,
    "Ellipse": shared.render_ellipse_helper,
    "Vector": shared.render_vector_helper,
    "Angle": shared.render_angle_helper,
    "Triangle": shared.render_triangle_helper,
    "Rectangle": shared.render_rectangle_helper,
    "Function": shared.render_function_helper,
    "FunctionsBoundedColoredArea": shared.render_functions_bounded_area_helper,
    "FunctionSegmentBoundedColoredArea": shared.render_function_segment_area_helper,
    "SegmentsBoundedColoredArea": shared.render_segments_bounded_area_helper,
}


def build_plan_for_drawable(drawable: Any, coordinate_mapper: Any, style: Dict[str, Any]) -> Optional[OptimizedPrimitivePlan]:
    class_name = getattr(drawable, "get_class_name", None)
    if callable(class_name):
        class_name = class_name()
    elif class_name is None:
        class_name = drawable.__class__.__name__

    helper = _HELPERS.get(class_name)
    if helper is None:
        return None

    drawable_key = _drawable_key(drawable, class_name.lower())
    recorder = _RecordingPrimitives(drawable_key)
    cached_mapper = _CachedCoordinateMapper(coordinate_mapper)
    helper(recorder, drawable, cached_mapper, style)
    map_state = _capture_map_state(coordinate_mapper)
    plan = OptimizedPrimitivePlan(
        drawable=drawable,
        commands=list(recorder.commands),
        plan_key=drawable_key,
        metadata={"class_name": class_name, "map_state": map_state},
    )
    plan.update_map_state(map_state)
    return plan


def build_plan_for_cartesian(cartesian: Any, coordinate_mapper: Any, style: Dict[str, Any]) -> OptimizedPrimitivePlan:
    key = _drawable_key(cartesian, "cartesian")
    recorder = _RecordingPrimitives(key)
    cached_mapper = _CachedCoordinateMapper(coordinate_mapper)
    shared.render_cartesian_helper(recorder, cartesian, cached_mapper, style)
    map_state = _capture_map_state(coordinate_mapper)
    plan = OptimizedPrimitivePlan(
        drawable=cartesian,
        commands=list(recorder.commands),
        plan_key=key,
        metadata={"class_name": "Cartesian2Axis", "map_state": map_state},
    )
    plan.update_map_state(map_state)
    return plan

