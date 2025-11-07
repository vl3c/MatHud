from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

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
        self._shape_active = False
        self._state_stack: List[Dict[str, Any]] = []

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
        if not self._shape_active:
            self.ctx.save()
            snapshot: Dict[str, Any] = {}
            for attr in ("_strokeStyle", "_lineWidth", "_fillStyle", "_globalAlpha", "_font", "_textAlign", "_textBaseline"):
                if hasattr(self.ctx, attr):
                    snapshot[attr] = getattr(self.ctx, attr)
            self._state_stack.append(snapshot)
            self._shape_active = True

    def end_shape(self) -> None:
        if self._shape_active:
            if self._state_stack:
                snapshot = self._state_stack.pop()
                for attr, value in snapshot.items():
                    if hasattr(self.ctx, attr):
                        try:
                            object.__setattr__(self.ctx, attr, value)
                        except Exception:
                            setattr(self.ctx, attr, value)
            self.ctx.restore()
            self._shape_active = False

    def _ensure_shape(self) -> bool:
        started = not self._shape_active
        if started:
            self.begin_shape()
        return started

    def _apply_stroke_style(self, stroke: StrokeStyle) -> None:
        self.ctx.strokeStyle = stroke.color
        if stroke.width:
            width_value = stroke.width
            try:
                width_value = float(width_value)
            except Exception:
                self.ctx.lineWidth = width_value
            else:
                if math.isfinite(width_value) and width_value.is_integer():
                    self.ctx.lineWidth = int(width_value)
                else:
                    self.ctx.lineWidth = width_value
        if stroke.line_join:
            self.ctx.lineJoin = stroke.line_join
        if stroke.line_cap:
            self.ctx.lineCap = stroke.line_cap

    def _apply_fill_style(self, fill: FillStyle) -> None:
        self.ctx.fillStyle = fill.color
        if fill.opacity is not None:
            self.ctx.globalAlpha = fill.opacity

    def stroke_line(self, start: Point2D, end: Point2D, stroke: StrokeStyle, *, include_width: bool = True) -> None:
        started = self._ensure_shape()
        self._apply_stroke_style(stroke)
        self.ctx.beginPath()
        self.ctx.moveTo(start[0], start[1])
        self.ctx.lineTo(end[0], end[1])
        self.ctx.stroke()
        if started:
            self.end_shape()

    def stroke_polyline(self, points: List[Point2D], stroke: StrokeStyle) -> None:
        if len(points) < 2:
            return
        started = self._ensure_shape()
        self._apply_stroke_style(stroke)
        self.ctx.beginPath()
        self.ctx.moveTo(points[0][0], points[0][1])
        for x, y in points[1:]:
            self.ctx.lineTo(x, y)
        self.ctx.stroke()
        if started:
            self.end_shape()

    def stroke_circle(self, center: Point2D, radius: float, stroke: StrokeStyle) -> None:
        started = self._ensure_shape()
        self._apply_stroke_style(stroke)
        self.ctx.beginPath()
        coerced_radius = self._coerce_number(radius)
        self.ctx.arc(center[0], center[1], coerced_radius, 0, 2 * math.pi)
        self.ctx.stroke()
        if started:
            self.end_shape()

    def fill_circle(
        self,
        center: Point2D,
        radius: float,
        fill: FillStyle,
        stroke: Optional[StrokeStyle] = None,
    ) -> None:
        started = self._ensure_shape()
        self._apply_fill_style(fill)
        self.ctx.beginPath()
        coerced_radius = self._coerce_number(radius)
        self.ctx.arc(center[0], center[1], coerced_radius, 0, 2 * math.pi)
        self.ctx.fill()
        if stroke:
            self._apply_stroke_style(stroke)
            self.ctx.stroke()
        if started:
            self.end_shape()

    def stroke_ellipse(
        self,
        center: Point2D,
        radius_x: float,
        radius_y: float,
        rotation_rad: float,
        stroke: StrokeStyle,
    ) -> None:
        started = self._ensure_shape()
        self._apply_stroke_style(stroke)
        self.ctx.beginPath()
        rx = self._coerce_number(radius_x)
        ry = self._coerce_number(radius_y)
        self.ctx.ellipse(center[0], center[1], rx, ry, rotation_rad, 0, 2 * math.pi)
        self.ctx.stroke()
        if started:
            self.end_shape()

    def fill_polygon(
        self,
        points: List[Point2D],
        fill: FillStyle,
        stroke: Optional[StrokeStyle] = None,
    ) -> None:
        if len(points) < 3:
            return
        started = self._ensure_shape()
        self._apply_fill_style(fill)
        self.ctx.beginPath()
        self.ctx.moveTo(points[0][0], points[0][1])
        for x, y in points[1:]:
            self.ctx.lineTo(x, y)
        self.ctx.closePath()
        self.ctx.fill()
        if started:
            self.end_shape()

    def fill_joined_area(
        self,
        forward: List[Point2D],
        reverse: List[Point2D],
        fill: FillStyle,
    ) -> None:
        if len(forward) < 2 or not reverse:
            return
        started = self._ensure_shape()
        self._apply_fill_style(fill)
        self.ctx.beginPath()
        self.ctx.moveTo(forward[0][0], forward[0][1])
        for x, y in forward[1:]:
            self.ctx.lineTo(x, y)
        for x, y in reverse:
            self.ctx.lineTo(x, y)
        self.ctx.closePath()
        self.ctx.fill()
        if started:
            self.end_shape()

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
        started = self._ensure_shape()
        self._apply_stroke_style(stroke)
        self.ctx.beginPath()
        coerced_radius = self._coerce_number(radius)
        self.ctx.arc(center[0], center[1], coerced_radius, start_angle_rad, end_angle_rad, not sweep_clockwise)
        self.ctx.stroke()
        if started:
            self.end_shape()

    def draw_text(
        self,
        text: str,
        position: Point2D,
        font: FontStyle,
        color: str,
        alignment: TextAlignment,
        style_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        started = self._ensure_shape()
        self.ctx.fillStyle = color
        font_parts = []
        if font.weight:
            font_parts.append(font.weight)
        size_component: str
        raw_size = font.size
        if isinstance(raw_size, str) and raw_size.strip().endswith("px"):
            raw_numeric = raw_size.strip()[:-2]
            try:
                size_num = float(raw_numeric)
            except Exception:
                size_component = raw_size.strip()
            else:
                if math.isfinite(size_num) and size_num.is_integer():
                    size_component = f"{int(size_num)}px"
                else:
                    size_component = f"{size_num}px"
        else:
            try:
                size_num = float(raw_size)
            except Exception:
                size_component = f"{raw_size}px"
            else:
                if math.isfinite(size_num) and size_num.is_integer():
                    size_component = f"{int(size_num)}px"
                else:
                    size_component = f"{size_num}px"

        normalized_family = font.family or ""
        combined = f"{size_component} {normalized_family}".strip()
        font_parts.append(combined)
        self.ctx.font = font_parts[0] if len(font_parts) == 1 else " ".join(font_parts)
        if alignment.horizontal:
            desired_horizontal = alignment.horizontal
            if desired_horizontal.lower() == "left":
                if getattr(self.ctx, "_textAlign", None) != "left":
                    try:
                        object.__setattr__(self.ctx, "_textAlign", "left")
                    except Exception:
                        setattr(self.ctx, "_textAlign", "left")
            else:
                self.ctx.textAlign = desired_horizontal

        if alignment.vertical:
            desired_vertical = alignment.vertical
            if desired_vertical.lower() == "alphabetic":
                if getattr(self.ctx, "_textBaseline", None) != "alphabetic":
                    try:
                        object.__setattr__(self.ctx, "_textBaseline", "alphabetic")
                    except Exception:
                        setattr(self.ctx, "_textBaseline", "alphabetic")
            else:
                self.ctx.textBaseline = desired_vertical
        self.ctx.fillText(text, position[0], position[1])
        if started:
            self.end_shape()

    def clear_surface(self) -> None:
        self.ctx.clearRect(0, 0, self.canvas_el.width, self.canvas_el.height)

    def resize_surface(self, width: float, height: float) -> None:
        self.canvas_el.width = width
        self.canvas_el.height = height
        self.canvas_el.attrs["width"] = str(width)
        self.canvas_el.attrs["height"] = str(height)
        self.canvas_el.style.width = f"{int(width)}px"
        self.canvas_el.style.height = f"{int(height)}px"

