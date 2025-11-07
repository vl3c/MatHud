from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from rendering import shared_drawable_renderers as shared
from rendering.shared_drawable_renderers import RendererPrimitives


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
    __slots__ = ("drawable", "commands", "plan_key", "metadata")

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

    def _record(self, op: str, args: PrimitiveArgs, kwargs: PrimitiveKwargs, *, style: Any = None, geometry: Iterable[Any] = ()) -> None:
        command_key = f"{self._drawable_key}:{op}:{self._counter}"
        self._counter += 1
        meta: Dict[str, Any] = {}
        style_sig = _style_signature(style)
        if style_sig is not None:
            meta["style"] = style_sig
        geometry_sig = _geometry_signature(geometry)
        if geometry_sig:
            meta["geometry"] = geometry_sig
        self.commands.append(PrimitiveCommand(op, args, kwargs, command_key, meta))

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
    return OptimizedPrimitivePlan(drawable=drawable, commands=list(recorder.commands), plan_key=drawable_key, metadata={"class_name": class_name})


def build_plan_for_cartesian(cartesian: Any, coordinate_mapper: Any, style: Dict[str, Any]) -> OptimizedPrimitivePlan:
    key = _drawable_key(cartesian, "cartesian")
    recorder = _RecordingPrimitives(key)
    cached_mapper = _CachedCoordinateMapper(coordinate_mapper)
    shared.render_cartesian_helper(recorder, cartesian, cached_mapper, style)
    return OptimizedPrimitivePlan(drawable=cartesian, commands=list(recorder.commands), plan_key=key, metadata={"class_name": "Cartesian2Axis"})

