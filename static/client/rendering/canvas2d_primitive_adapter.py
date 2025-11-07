from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from rendering.shared_drawable_renderers import (
    FillStyle,
    FontStyle,
    Point2D,
    RendererPrimitives,
    StrokeStyle,
    TextAlignment,
)


class Canvas2DPrimitiveAdapter(RendererPrimitives):
    """RendererPrimitives implementation for Canvas 2D."""

    def __init__(self, canvas_el: Any) -> None:
        self.canvas_el = canvas_el
        self.ctx = canvas_el.getContext("2d")
        self._stroke_state: Dict[str, Any] = {"color": None, "width": None, "line_join": None, "line_cap": None}
        self._fill_state: Dict[str, Any] = {"color": None}
        self._global_alpha: float = 1.0
        self._pending_alpha_reset: bool = False
        self._font_cache: Dict[Tuple[Any, ...], str] = {}
        self._font_state: Dict[str, Any] = {"font": None}
        self._text_color: Optional[str] = None
        self._text_align: Optional[str] = None
        self._text_baseline: Optional[str] = None
        self._shape_depth: int = 0
        self._batch_depth: int = 0

    def _coerce_number(self, value: Any) -> Any:
        try:
            num = float(value)
        except Exception:
            return value
        if math.isfinite(num) and num.is_integer():
            return int(num)
        return num

    def _format_px(self, size: Any) -> str:
        if isinstance(size, str):
            stripped = size.strip()
            if stripped.endswith("px"):
                stripped = stripped[:-2]
            if not stripped:
                return "0px"
            try:
                num = float(stripped)
            except Exception:
                return stripped + "px"
        else:
            try:
                num = float(size)
            except Exception:
                return f"{size}px"
        try:
            if math.isfinite(num) and num.is_integer():
                return f"{int(num)}px"
            return f"{num}px"
        except Exception:
            return f"{num}px"

    def begin_shape(self) -> None:
        self._shape_depth += 1

    def end_shape(self) -> None:
        if self._shape_depth:
            self._shape_depth -= 1
        self._reset_alpha_if_needed()

    def _apply_stroke_style(self, stroke: StrokeStyle, *, include_width: bool = True) -> None:
        if stroke.color != self._stroke_state["color"]:
            self.ctx.strokeStyle = stroke.color
            self._stroke_state["color"] = stroke.color
        if include_width:
            normalized_width = self._coerce_number(stroke.width)
            if normalized_width != self._stroke_state["width"]:
                self.ctx.lineWidth = normalized_width
                self._stroke_state["width"] = normalized_width
        if stroke.line_join != self._stroke_state["line_join"]:
            if stroke.line_join:
                self.ctx.lineJoin = stroke.line_join
            self._stroke_state["line_join"] = stroke.line_join
        if stroke.line_cap != self._stroke_state["line_cap"]:
            if stroke.line_cap:
                self.ctx.lineCap = stroke.line_cap
            self._stroke_state["line_cap"] = stroke.line_cap

    def _apply_fill_style(self, fill: FillStyle) -> None:
        if fill.color != self._fill_state["color"]:
            self.ctx.fillStyle = fill.color
            self._fill_state["color"] = fill.color
        if fill.opacity is None:
            if self._global_alpha != 1.0:
                self.ctx.globalAlpha = 1.0
                self._global_alpha = 1.0
            self._pending_alpha_reset = False
        else:
            opacity = float(fill.opacity)
            if opacity != self._global_alpha:
                self.ctx.globalAlpha = opacity
                self._global_alpha = opacity
            self._pending_alpha_reset = True

    def stroke_line(self, start: Point2D, end: Point2D, stroke: StrokeStyle, *, include_width: bool = True) -> None:
        self._apply_stroke_style(stroke, include_width=include_width)
        self.ctx.beginPath()
        self.ctx.moveTo(start[0], start[1])
        self.ctx.lineTo(end[0], end[1])
        self.ctx.stroke()

    def stroke_polyline(self, points: List[Point2D], stroke: StrokeStyle) -> None:
        if len(points) < 2:
            return
        self._apply_stroke_style(stroke)
        self.ctx.beginPath()
        self.ctx.moveTo(points[0][0], points[0][1])
        for x, y in points[1:]:
            self.ctx.lineTo(x, y)
        self.ctx.stroke()

    def stroke_circle(self, center: Point2D, radius: float, stroke: StrokeStyle) -> None:
        self._apply_stroke_style(stroke)
        self.ctx.beginPath()
        coerced_radius = self._coerce_number(radius)
        self.ctx.arc(center[0], center[1], coerced_radius, 0, 2 * math.pi)
        self.ctx.stroke()

    def fill_circle(
        self,
        center: Point2D,
        radius: float,
        fill: FillStyle,
        stroke: Optional[StrokeStyle] = None,
    ) -> None:
        self._apply_fill_style(fill)
        self.ctx.beginPath()
        coerced_radius = self._coerce_number(radius)
        self.ctx.arc(center[0], center[1], coerced_radius, 0, 2 * math.pi)
        self.ctx.fill()
        if stroke:
            self._apply_stroke_style(stroke)
            self.ctx.stroke()
        self._reset_alpha_if_needed()

    def stroke_ellipse(
        self,
        center: Point2D,
        radius_x: float,
        radius_y: float,
        rotation_rad: float,
        stroke: StrokeStyle,
    ) -> None:
        self._apply_stroke_style(stroke)
        self.ctx.beginPath()
        rx = self._coerce_number(radius_x)
        ry = self._coerce_number(radius_y)
        self.ctx.ellipse(center[0], center[1], rx, ry, rotation_rad, 0, 2 * math.pi)
        self.ctx.stroke()

    def fill_polygon(
        self,
        points: List[Point2D],
        fill: FillStyle,
        stroke: Optional[StrokeStyle] = None,
    ) -> None:
        if len(points) < 3:
            return
        self._apply_fill_style(fill)
        self.ctx.beginPath()
        self.ctx.moveTo(points[0][0], points[0][1])
        for x, y in points[1:]:
            self.ctx.lineTo(x, y)
        self.ctx.closePath()
        self.ctx.fill()
        self._reset_alpha_if_needed()

    def fill_joined_area(
        self,
        forward: List[Point2D],
        reverse: List[Point2D],
        fill: FillStyle,
    ) -> None:
        if len(forward) < 2 or not reverse:
            return
        self._apply_fill_style(fill)
        self.ctx.beginPath()
        self.ctx.moveTo(forward[0][0], forward[0][1])
        for x, y in forward[1:]:
            self.ctx.lineTo(x, y)
        for x, y in reverse:
            self.ctx.lineTo(x, y)
        self.ctx.closePath()
        self.ctx.fill()
        self._reset_alpha_if_needed()

    def stroke_arc(
        self,
        center: Point2D,
        radius: float,
        start_angle_rad: float,
        end_angle_rad: float,
        sweep_clockwise: bool,
        stroke: StrokeStyle,
        css_class: str = None,
    ) -> None:
        self._apply_stroke_style(stroke)
        self.ctx.beginPath()
        coerced_radius = self._coerce_number(radius)
        self.ctx.arc(center[0], center[1], coerced_radius, start_angle_rad, end_angle_rad, not sweep_clockwise)
        self.ctx.stroke()

    def draw_text(
        self,
        text: str,
        position: Point2D,
        font: FontStyle,
        color: str,
        alignment: TextAlignment,
        style_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        if color != self._text_color:
            self.ctx.fillStyle = color
            self._text_color = color
        font_string = self._resolve_font_string(font)
        if font_string != self._font_state["font"]:
            self.ctx.font = font_string
            self._font_state["font"] = font_string
        if alignment.horizontal and alignment.horizontal != self._text_align:
            self.ctx.textAlign = alignment.horizontal
            self._text_align = alignment.horizontal
        if alignment.vertical and alignment.vertical != self._text_baseline:
            self.ctx.textBaseline = alignment.vertical
            self._text_baseline = alignment.vertical
        self.ctx.fillText(text, position[0], position[1])

    def _resolve_font_string(self, font: FontStyle) -> str:
        key = (font.weight or "", font.size, font.family or "")
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached
        size_component = self._format_px(font.size)
        parts: List[str] = []
        if font.weight:
            parts.append(str(font.weight))
        parts.append(size_component)
        if font.family:
            parts.append(font.family)
        font_string = " ".join(parts)
        self._font_cache[key] = font_string
        return font_string

    def _reset_alpha_if_needed(self, *, force: bool = False) -> None:
        if not (self._pending_alpha_reset or force):
            return
        if not force and (self._shape_depth > 0 or self._batch_depth > 0):
            return
        if self._global_alpha != 1.0:
            self.ctx.globalAlpha = 1.0
            self._global_alpha = 1.0
        self._pending_alpha_reset = False

    def begin_frame(self) -> None:
        self._shape_depth = 0
        self._batch_depth = 0
        self._pending_alpha_reset = False

    def end_frame(self) -> None:
        self._reset_alpha_if_needed(force=True)

    def begin_batch(self, plan: Any = None) -> None:
        self._batch_depth += 1

    def end_batch(self, plan: Any = None) -> None:
        if self._batch_depth:
            self._batch_depth -= 1
        self._reset_alpha_if_needed()

    def clear_surface(self) -> None:
        self.ctx.clearRect(0, 0, self.canvas_el.width, self.canvas_el.height)

    def resize_surface(self, width: float, height: float) -> None:
        self.canvas_el.width = width
        self.canvas_el.height = height
        self.canvas_el.attrs["width"] = str(width)
        self.canvas_el.attrs["height"] = str(height)
        self.canvas_el.style.width = f"{int(width)}px"
        self.canvas_el.style.height = f"{int(height)}px"

