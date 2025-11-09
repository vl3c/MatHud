from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from browser import document, svg

from rendering.shared_drawable_renderers import (
    FillStyle,
    FontStyle,
    Point2D,
    RendererPrimitives,
    StrokeStyle,
    TextAlignment,
)


class SvgPrimitiveAdapter(RendererPrimitives):
    """RendererPrimitives implementation backed by the SVG surface."""

    def __init__(self, surface_id: str = "math-svg", *, telemetry: Optional[Any] = None) -> None:
        self.surface_id = surface_id
        self._pool: Dict[str, List[Any]] = {}
        self._active_indices: Dict[str, int] = {}
        self._batch_stack: List[str] = []
        self._group_stack: List[Tuple[str, Any]] = []
        self._groups: Dict[str, Any] = {}
        self._staged_fragment: Optional[Any] = None
        self._telemetry: Optional[Any] = telemetry

    @property
    def _surface(self) -> Any:
        return document[self.surface_id]

    def set_telemetry(self, telemetry: Any) -> None:
        self._telemetry = telemetry

    def _record_adapter_event(self, name: str, amount: int = 1) -> None:
        telemetry = self._telemetry
        if telemetry is None:
            return
        try:
            telemetry.record_adapter_event(name, amount)
        except Exception:
            pass

    def _track_batch_depth(self) -> None:
        telemetry = self._telemetry
        if telemetry is None:
            return
        try:
            telemetry.track_batch_depth(len(self._batch_stack))
        except Exception:
            pass

    def _create_fragment(self) -> Optional[Any]:
        try:
            fragment = document.createDocumentFragment()
        except Exception:
            fragment = None
        return fragment

    def _detach_element(self, elem: Any) -> None:
        try:
            elem.remove()
        except Exception:
            try:
                parent = getattr(elem, "parentNode", None)
                if parent is not None:
                    parent.removeChild(elem)
            except Exception:
                pass

    def _ensure_parent(self, elem: Any, target: Any) -> None:
        parent = getattr(elem, "parentNode", None)
        if parent is target:
            return
        if parent is not None:
            self._detach_element(elem)
        try:
            target <= elem
        except Exception:
            pass

    def _ensure_group(self, plan_key: str) -> Any:
        group = self._groups.get(plan_key)
        if group is None:
            group = svg.g()
            self._groups[plan_key] = group
            try:
                self._surface <= group
            except Exception:
                pass
        else:
            current_parent = getattr(group, "parentNode", None)
            if current_parent is not self._surface:
                self._detach_element(group)
                try:
                    self._surface <= group
                except Exception:
                    pass
        return group

    def begin_frame(self) -> None:
        for key in self._pool:
            self._active_indices[key] = 0
        self._staged_fragment = self._create_fragment()
        self._record_adapter_event("frame_begin")

    def end_frame(self) -> None:
        for key, pool in self._pool.items():
            active = self._active_indices.get(key, 0)
            for idx in range(active, len(pool)):
                elem = pool[idx]
                if elem.parentNode is self._surface:
                    self._surface.removeChild(elem)
        fragment = self._staged_fragment
        if fragment is not None:
            try:
                self._surface <= fragment
            except Exception:
                try:
                    # Fallback: append children individually if fragment append fails
                    for child in list(getattr(fragment, "children", [])):
                        self._surface <= child
                except Exception:
                    pass
        self._staged_fragment = None
        self._record_adapter_event("frame_end")

    def reserve_usage_counts(self, usage_counts: Dict[str, int], *, trim_excess: bool = False) -> None:
        if not usage_counts:
            return
        for kind, required in usage_counts.items():
            if required <= 0:
                continue
            pool = self._pool.get(kind)
            if not pool:
                continue
            current = self._active_indices.get(kind, 0)
            if required > len(pool):
                required = len(pool)
            if required > current:
                self._active_indices[kind] = required
                target = self._surface
                if self._group_stack:
                    _, group = self._group_stack[-1]
                    if group is not None:
                        target = group
                for idx in range(required):
                    elem = pool[idx]
                    self._ensure_parent(elem, target)
                    try:
                        elem.style["display"] = ""
                    except Exception:
                        pass
            if trim_excess and len(pool) > required:
                for idx in range(required, len(pool)):
                    elem = pool[idx]
                    parent = getattr(elem, "parentNode", None)
                    if parent is self._surface:
                        self._detach_element(elem)
                self._active_indices[kind] = min(self._active_indices.get(kind, 0), len(pool))
                self._record_adapter_event(f"pool_trim_{kind}")

    def begin_batch(self, plan: Any = None) -> None:
        key = getattr(plan, "plan_key", "")
        plan_key = str(key)
        self._batch_stack.append(plan_key)
        group = None
        if plan_key:
            group = self._ensure_group(plan_key)
        self._group_stack.append((plan_key, group))
        self._track_batch_depth()

    def end_batch(self, plan: Any = None) -> None:
        if self._batch_stack:
            self._batch_stack.pop()
        if self._group_stack:
            self._group_stack.pop()
        self._track_batch_depth()

    def execute_optimized(self, command: Any) -> None:
        handler_name = f"_optimized_{getattr(command, 'op', '')}"
        handler = getattr(self, handler_name, None)
        if callable(handler):
            handler(command)
            return
        super().execute_optimized(command)

    def _acquire_element(self, kind: str, factory: Callable[[], Any]) -> Tuple[Any, Dict[str, Any]]:
        pool = self._pool.setdefault(kind, [])
        index = self._active_indices.get(kind, 0)
        target = self._surface
        if self._group_stack:
            _, group = self._group_stack[-1]
            if group is not None:
                target = group
        if index < len(pool):
            elem = pool[index]
            self._ensure_parent(elem, target)
        else:
            elem = factory()
            pool.append(elem)
            self._attach_new_element(elem, target)
        self._active_indices[kind] = index + 1
        try:
            elem.style["display"] = ""
        except Exception:
            pass
        cache = getattr(elem, "_mathud_cache", None)
        if cache is None:
            cache = {}
            setattr(elem, "_mathud_cache", cache)
        return elem, cache

    def _attach_new_element(self, elem: Any, target: Any) -> None:
        if target is not self._surface:
            try:
                target <= elem
            except Exception:
                return
            self._record_adapter_event("direct_append")
            self._record_adapter_event("new_elements")
            return
        if elem.parentNode is not None and elem.parentNode is not target:
            self._detach_element(elem)
        fragment = self._staged_fragment
        appended = False
        if fragment is not None:
            try:
                fragment <= elem
            except Exception:
                appended = False
            else:
                self._record_adapter_event("fragment_append")
                appended = True
        if not appended:
            try:
                target <= elem
            except Exception:
                pass
            else:
                self._record_adapter_event("direct_append")
                appended = True
        if appended:
            self._record_adapter_event("new_elements")

    def set_group_transform(self, plan_key: str, transform: Optional[str]) -> None:
        group = self._groups.get(plan_key)
        if group is None:
            return
        if transform:
            try:
                group.setAttribute("transform", transform)
            except Exception:
                pass
        else:
            try:
                group.removeAttribute("transform")
            except Exception:
                try:
                    del group.attrs["transform"]
                except Exception:
                    pass

    def drop_group(self, plan_key: str) -> None:
        group = self._groups.pop(plan_key, None)
        if group is None:
            return
        try:
            group.remove()
        except Exception:
            try:
                self._surface.removeChild(group)
            except Exception:
                pass

    def clear_group(self, plan_key: str) -> None:
        group = self._groups.get(plan_key)
        if group is None:
            return
        try:
            while group.firstChild is not None:
                child = group.firstChild
                try:
                    child.remove()
                except Exception:
                    try:
                        group.removeChild(child)
                    except Exception:
                        break
        except Exception:
            pass

    def _set_attribute(self, elem: Any, cache: Dict[str, Any], name: str, value: Optional[str]) -> None:
        attrs = cache.setdefault("attrs", {})
        if attrs.get(name) == value:
            return
        if value is None:
            attrs.pop(name, None)
            try:
                del elem.attrs[name]
            except Exception:
                try:
                    elem.removeAttribute(name)
                except Exception:
                    pass
            return
        elem.setAttribute(name, value)
        attrs[name] = value

    def _format_number(self, value: Any) -> str:
        try:
            num = float(value)
        except Exception:
            return str(value)
        if math.isfinite(num) and num.is_integer():
            return str(int(num))
        return str(num)

    def _ensure_stroke_attrs(self, elem: Any, cache: Dict[str, Any], stroke: StrokeStyle, *, include_width: bool = True) -> None:
        stroke_cache = cache.setdefault("stroke", {})
        if stroke_cache.get("color") != stroke.color:
            self._set_attribute(elem, cache, "stroke", stroke.color)
            stroke_cache["color"] = stroke.color
        if include_width:
            width_value = self._format_number(stroke.width)
            if stroke_cache.get("width") != width_value:
                self._set_attribute(elem, cache, "stroke-width", width_value)
                stroke_cache["width"] = width_value
        if stroke.line_join != stroke_cache.get("line_join"):
            if stroke.line_join:
                self._set_attribute(elem, cache, "stroke-linejoin", stroke.line_join)
            else:
                self._set_attribute(elem, cache, "stroke-linejoin", None)
            stroke_cache["line_join"] = stroke.line_join
        if stroke.line_cap != stroke_cache.get("line_cap"):
            if stroke.line_cap:
                self._set_attribute(elem, cache, "stroke-linecap", stroke.line_cap)
            else:
                self._set_attribute(elem, cache, "stroke-linecap", None)
            stroke_cache["line_cap"] = stroke.line_cap

    def _ensure_fill_attrs(self, elem: Any, cache: Dict[str, Any], fill: FillStyle) -> None:
        fill_cache = cache.setdefault("fill", {})
        if fill_cache.get("color") != fill.color:
            self._set_attribute(elem, cache, "fill", fill.color)
            fill_cache["color"] = fill.color
        if fill.opacity is None:
            if fill_cache.get("opacity") is not None:
                self._set_attribute(elem, cache, "fill-opacity", None)
                fill_cache["opacity"] = None
        else:
            opacity_value = self._format_number(fill.opacity)
            if fill_cache.get("opacity") != opacity_value:
                self._set_attribute(elem, cache, "fill-opacity", opacity_value)
                fill_cache["opacity"] = opacity_value

    def _apply_style_overrides(self, elem: Any, cache: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> None:
        style_cache = cache.setdefault("style_overrides", {})
        keys_to_remove = [key for key in style_cache.keys() if not overrides or key not in overrides]
        for key in keys_to_remove:
            try:
                del elem.style[key]
            except Exception:
                pass
            style_cache.pop(key, None)
        if not overrides:
            return
        for key, value in overrides.items():
            if style_cache.get(key) != value:
                elem.style[key] = value
                style_cache[key] = value

    # ------------------------------------------------------------------
    # Optimized command handlers
    # ------------------------------------------------------------------

    def _optimized_stroke_line(self, command: Any) -> None:
        start, end, stroke = command.args
        include_width = command.kwargs.get("include_width", True)
        elem, cache = self._acquire_element("stroke_line", lambda: svg.line())
        self._ensure_stroke_attrs(elem, cache, stroke, include_width=include_width)
        self._set_attribute(elem, cache, "fill", "none")
        self._set_attribute(elem, cache, "x1", self._format_number(start[0]))
        self._set_attribute(elem, cache, "y1", self._format_number(start[1]))
        self._set_attribute(elem, cache, "x2", self._format_number(end[0]))
        self._set_attribute(elem, cache, "y2", self._format_number(end[1]))

    def _optimized_stroke_polyline(self, command: Any) -> None:
        points, stroke = command.args
        elem, cache = self._acquire_element("stroke_polyline", lambda: svg.path())
        segments = [f"{self._format_number(x)} {self._format_number(y)}" for x, y in points]
        d_value = "M " + " L ".join(segments)
        self._set_attribute(elem, cache, "d", d_value)
        self._set_attribute(elem, cache, "fill", "none")
        self._ensure_stroke_attrs(elem, cache, stroke)

    def _optimized_stroke_circle(self, command: Any) -> None:
        center, radius, stroke = command.args
        elem, cache = self._acquire_element("stroke_circle", lambda: svg.circle())
        self._ensure_stroke_attrs(elem, cache, stroke)
        self._set_attribute(elem, cache, "cx", self._format_number(center[0]))
        self._set_attribute(elem, cache, "cy", self._format_number(center[1]))
        self._set_attribute(elem, cache, "r", self._format_number(radius))
        self._set_attribute(elem, cache, "fill", "none")

    def _optimized_fill_circle(self, command: Any) -> None:
        center, radius, fill, stroke = command.args
        screen_space = command.kwargs.get("screen_space")
        elem, cache = self._acquire_element("fill_circle", lambda: svg.circle())
        self._set_attribute(elem, cache, "cx", self._format_number(center[0]))
        self._set_attribute(elem, cache, "cy", self._format_number(center[1]))
        self._set_attribute(elem, cache, "r", self._format_number(radius))
        self._ensure_fill_attrs(elem, cache, fill)
        if stroke:
            self._ensure_stroke_attrs(elem, cache, stroke)

    def _optimized_stroke_ellipse(self, command: Any) -> None:
        center, radius_x, radius_y, rotation_rad, stroke = command.args
        elem, cache = self._acquire_element("stroke_ellipse", lambda: svg.ellipse())
        self._ensure_stroke_attrs(elem, cache, stroke)
        self._set_attribute(elem, cache, "cx", self._format_number(center[0]))
        self._set_attribute(elem, cache, "cy", self._format_number(center[1]))
        self._set_attribute(elem, cache, "rx", self._format_number(radius_x))
        self._set_attribute(elem, cache, "ry", self._format_number(radius_y))
        if rotation_rad % (2 * math.pi) != 0:
            transform = f"rotate({math.degrees(rotation_rad)} {self._format_number(center[0])} {self._format_number(center[1])})"
            self._set_attribute(elem, cache, "transform", transform)
        else:
            self._set_attribute(elem, cache, "transform", None)
        self._set_attribute(elem, cache, "fill", "none")

    def _optimized_fill_polygon(self, command: Any) -> None:
        points, fill, stroke = command.args
        elem, cache = self._acquire_element("fill_polygon", lambda: svg.polygon())
        points_str = " ".join(f"{self._format_number(x)},{self._format_number(y)}" for x, y in points)
        self._set_attribute(elem, cache, "points", points_str)
        self._ensure_fill_attrs(elem, cache, fill)
        if stroke:
            self._ensure_stroke_attrs(elem, cache, stroke)

    def _optimized_fill_joined_area(self, command: Any) -> None:
        forward, reverse, fill = command.args
        elem, cache = self._acquire_element("fill_joined_area", lambda: svg.path())
        d_parts = [f"M {self._format_number(forward[0][0])} {self._format_number(forward[0][1])}"]
        d_parts.extend(f"L {self._format_number(x)} {self._format_number(y)}" for x, y in forward[1:])
        d_parts.extend(f"L {self._format_number(x)} {self._format_number(y)}" for x, y in reverse)
        d_parts.append("Z")
        self._set_attribute(elem, cache, "d", " ".join(d_parts))
        self._ensure_fill_attrs(elem, cache, fill)
        self._set_attribute(elem, cache, "stroke", "none")

    def _optimized_stroke_arc(self, command: Any) -> None:
        center, radius, start_angle_rad, end_angle_rad, sweep_clockwise, stroke = command.args
        css_class = command.kwargs.get("css_class")
        elem, cache = self._acquire_element("stroke_arc", lambda: svg.path())
        start_x = center[0] + radius * math.cos(start_angle_rad)
        start_y = center[1] + radius * math.sin(start_angle_rad)
        end_x = center[0] + radius * math.cos(end_angle_rad)
        end_y = center[1] + radius * math.sin(end_angle_rad)
        large_arc_flag = "1" if abs(end_angle_rad - start_angle_rad) > math.pi else "0"
        sweep_flag = "1" if sweep_clockwise else "0"
        radius_str = self._format_number(radius)
        d_value = (
            f"M {self._format_number(start_x)} {self._format_number(start_y)} "
            f"A {radius_str} {radius_str} 0 {large_arc_flag} {sweep_flag} "
            f"{self._format_number(end_x)} {self._format_number(end_y)}"
        )
        self._set_attribute(elem, cache, "d", d_value)
        if css_class:
            self._set_attribute(elem, cache, "class", css_class)
        else:
            self._set_attribute(elem, cache, "class", None)
        self._set_attribute(elem, cache, "fill", "none")
        self._ensure_stroke_attrs(elem, cache, stroke)

    def _optimized_draw_text(self, command: Any) -> None:
        text, position, font, color, alignment = command.args
        style_overrides = command.kwargs.get("style_overrides")
        elem, cache = self._acquire_element("draw_text", lambda: svg.text())
        if cache.get("text") != text:
            elem.text = text
            cache["text"] = text
        self._set_attribute(elem, cache, "x", self._format_number(position[0]))
        self._set_attribute(elem, cache, "y", self._format_number(position[1]))
        self._ensure_fill_attrs(elem, cache, FillStyle(color=color))
        font_cache = cache.setdefault("font", {})
        font_size_str = self._format_number(font.size)
        if font_cache.get("size") != font_size_str:
            elem.setAttribute("font-size", font_size_str)
            font_cache["size"] = font_size_str
        if font.family and font_cache.get("family") != font.family:
            elem.setAttribute("font-family", font.family)
            font_cache["family"] = font.family
        if font.weight:
            if font_cache.get("weight") != font.weight:
                elem.setAttribute("font-weight", font.weight)
                font_cache["weight"] = font.weight
        elif font_cache.get("weight"):
            self._set_attribute(elem, cache, "font-weight", None)
            font_cache["weight"] = None
        horizontal = alignment.horizontal or "left"
        anchor_map = {"left": "start", "center": "middle", "right": "end"}
        svg_anchor = anchor_map.get(horizontal, horizontal)
        self._set_attribute(elem, cache, "text-anchor", svg_anchor)
        vertical = alignment.vertical or "alphabetic"
        self._set_attribute(elem, cache, "dominant-baseline", vertical)
        self._apply_style_overrides(elem, cache, style_overrides)
        metadata = command.kwargs.get("metadata")
        if isinstance(metadata, dict):
            label_meta = metadata.get("label")
            if isinstance(label_meta, dict):
                try:
                    rotation_deg = float(label_meta.get("rotation_degrees", 0.0))
                except Exception:
                    rotation_deg = 0.0
                if math.isfinite(rotation_deg) and rotation_deg != 0.0:
                    transform_value = f"rotate({-rotation_deg} {self._format_number(position[0])} {self._format_number(position[1])})"
                    self._set_attribute(elem, cache, "transform", transform_value)
                else:
                    self._set_attribute(elem, cache, "transform", None)
            else:
                self._set_attribute(elem, cache, "transform", None)
        else:
            self._set_attribute(elem, cache, "transform", None)

    def _stroke_kwargs(self, stroke: StrokeStyle, include_width: bool = True) -> dict[str, str]:
        kwargs: dict[str, str] = {"stroke": stroke.color}
        if include_width and stroke.width:
            try:
                width_float = float(stroke.width)
                if width_float.is_integer():
                    kwargs["stroke-width"] = str(int(width_float))
                else:
                    kwargs["stroke-width"] = str(stroke.width)
            except (ValueError, TypeError, AttributeError):
                kwargs["stroke-width"] = str(stroke.width)
        if stroke.line_join:
            kwargs["stroke-linejoin"] = stroke.line_join
        if stroke.line_cap:
            kwargs["stroke-linecap"] = stroke.line_cap
        return kwargs

    def stroke_line(self, start: Point2D, end: Point2D, stroke: StrokeStyle, *, include_width: bool = True) -> None:
        kwargs = self._stroke_kwargs(stroke, include_width=include_width)
        elem = svg.line(x1=str(start[0]), y1=str(start[1]), x2=str(end[0]), y2=str(end[1]), **kwargs)
        self._surface <= elem

    def stroke_polyline(self, points: List[Point2D], stroke: StrokeStyle) -> None:
        if len(points) < 2:
            return
        segments = [f"{x},{y}" for x, y in points]
        d = "M" + " L".join(segments)
        kwargs = self._stroke_kwargs(stroke)
        kwargs["fill"] = "none"
        elem = svg.path(d=d, **kwargs)
        self._surface <= elem

    def stroke_circle(self, center: Point2D, radius: float, stroke: StrokeStyle) -> None:
        kwargs = self._stroke_kwargs(stroke)
        kwargs["fill"] = "none"
        elem = svg.circle(cx=str(center[0]), cy=str(center[1]), r=str(radius), **kwargs)
        self._surface <= elem

    def fill_circle(
        self,
        center: Point2D,
        radius: float,
        fill: FillStyle,
        stroke: Optional[StrokeStyle] = None,
        *,
        screen_space: bool = False,
    ) -> None:
        attrs: Dict[str, Any] = {"cx": str(center[0]), "cy": str(center[1]), "r": str(radius), "fill": fill.color}
        if fill.opacity is not None:
            attrs["fill-opacity"] = str(fill.opacity)
        elem = svg.circle(**attrs)
        if stroke:
            for key, val in self._stroke_kwargs(stroke).items():
                elem.setAttribute(key, val)
        self._surface <= elem

    def stroke_ellipse(
        self,
        center: Point2D,
        radius_x: float,
        radius_y: float,
        rotation_rad: float,
        stroke: StrokeStyle,
    ) -> None:
        transform = None
        if rotation_rad % (2 * math.pi) != 0:
            transform = f"rotate({math.degrees(rotation_rad)} {center[0]} {center[1]})"
        kwargs = self._stroke_kwargs(stroke)
        kwargs["fill"] = "none"
        elem = svg.ellipse(cx=str(center[0]), cy=str(center[1]), rx=str(radius_x), ry=str(radius_y), **kwargs)
        if transform:
            elem.setAttribute("transform", transform)
        self._surface <= elem

    def fill_polygon(
        self,
        points: List[Point2D],
        fill: FillStyle,
        stroke: Optional[StrokeStyle] = None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(points) < 3:
            return
        points_str = " ".join(f"{x},{y}" for x, y in points)
        elem = svg.polygon(points=points_str, fill=fill.color)
        if fill.opacity is not None:
            elem.setAttribute("fill-opacity", str(fill.opacity))
        if stroke:
            elem.setAttribute("stroke", stroke.color)
        self._surface <= elem

    def fill_joined_area(
        self,
        forward: List[Point2D],
        reverse: List[Point2D],
        fill: FillStyle,
    ) -> None:
        if len(forward) < 2 or not reverse:
            return
        d = [f"M {forward[0][0]} {forward[0][1]}"]
        d.extend(f"L {x} {y}" for x, y in forward[1:])
        d.extend(f"L {x} {y}" for x, y in reverse)
        d.append("Z")
        elem = svg.path(d=" ".join(d), fill=fill.color, stroke="none")
        if fill.opacity is not None:
            elem.setAttribute("fill-opacity", str(fill.opacity))
        self._surface <= elem

    def stroke_arc(
        self,
        center: Point2D,
        radius: float,
        start_angle_rad: float,
        end_angle_rad: float,
        sweep_clockwise: bool,
        stroke: StrokeStyle,
        css_class: str = None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        start_x = center[0] + radius * math.cos(start_angle_rad)
        start_y = center[1] + radius * math.sin(start_angle_rad)
        end_x = center[0] + radius * math.cos(end_angle_rad)
        end_y = center[1] + radius * math.sin(end_angle_rad)
        large_arc_flag = "1" if abs(end_angle_rad - start_angle_rad) > math.pi else "0"
        sweep_flag = "1" if sweep_clockwise else "0"
        radius_str = str(int(radius)) if isinstance(radius, (int, float)) and float(radius).is_integer() else str(radius)
        d = f"M {start_x} {start_y} A {radius_str} {radius_str} 0 {large_arc_flag} {sweep_flag} {end_x} {end_y}"
        kwargs = self._stroke_kwargs(stroke)
        kwargs["fill"] = "none"
        if css_class:
            kwargs["class"] = css_class
        elem = svg.path(d=d, **kwargs)
        self._surface <= elem

    def draw_text(
        self,
        text: str,
        position: Point2D,
        font: FontStyle,
        color: str,
        alignment: TextAlignment,
        style_overrides: Optional[Dict[str, Any]] = None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        elem = svg.text(text, x=str(position[0]), y=str(position[1]), fill=color)
        elem.setAttribute("font-size", str(font.size))
        elem.setAttribute("font-family", font.family)
        if font.weight:
            elem.setAttribute("font-weight", font.weight)
        
        horizontal_anchor = alignment.horizontal or "left"
        anchor_map = {"left": "start", "center": "middle", "right": "end"}
        svg_anchor = anchor_map.get(horizontal_anchor, horizontal_anchor)
        elem.setAttribute("text-anchor", svg_anchor)

        vertical_anchor = alignment.vertical or "alphabetic"
        elem.setAttribute("dominant-baseline", vertical_anchor)

        if style_overrides:
            for key, value in style_overrides.items():
                elem.style[key] = value
        
        transform_value = None
        if isinstance(metadata, dict):
            label_meta = metadata.get("label")
            if isinstance(label_meta, dict):
                try:
                    rotation_deg = float(label_meta.get("rotation_degrees", 0.0))
                except Exception:
                    rotation_deg = 0.0
                if math.isfinite(rotation_deg) and rotation_deg != 0.0:
                    transform_value = f"rotate({-rotation_deg} {position[0]} {position[1]})"
        if transform_value:
            elem.setAttribute("transform", transform_value)
        else:
            try:
                elem.removeAttribute("transform")
            except Exception:
                pass

        self._surface <= elem

    def clear_surface(self) -> None:
        self._surface.clear()
        self._groups.clear()
        self._group_stack.clear()
        self._pool.clear()
        self._active_indices.clear()
        self._staged_fragment = None

    def resize_surface(self, width: float, height: float) -> None:
        surface = self._surface
        surface.setAttribute("width", str(width))
        surface.setAttribute("height", str(height))

