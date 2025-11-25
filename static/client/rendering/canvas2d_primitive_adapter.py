from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from rendering.renderer_primitives import (
    FillStyle,
    FontStyle,
    RendererPrimitives,
    StrokeStyle,
    TextAlignment,
)
from rendering.shared_drawable_renderers import Point2D


class Canvas2DPrimitiveAdapter(RendererPrimitives):
    """RendererPrimitives implementation for Canvas 2D."""

    def __init__(self, canvas_el: Any, *, telemetry: Optional[Any] = None) -> None:
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
        self._telemetry = telemetry
        self._line_batch: Optional[Dict[str, Any]] = None

    def set_telemetry(self, telemetry: Any) -> None:
        self._telemetry = telemetry

    def _record_event(self, name: str, amount: int = 1) -> None:
        telemetry = self._telemetry
        if telemetry is None:
            return
        try:
            telemetry.record_adapter_event(name, amount)
        except Exception:
            pass

    def _record_batch_depth(self) -> None:
        telemetry = self._telemetry
        if telemetry is None:
            return
        try:
            telemetry.track_batch_depth(self._batch_depth)
        except Exception:
            pass

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

    def _prepare_stroke(self, stroke: StrokeStyle, *, include_width: bool = True) -> None:
        self._flush_polygon_batch()
        self._apply_stroke_style(stroke, include_width=include_width)

    def _prepare_fill(self, fill: FillStyle) -> None:
        self._flush_polygon_batch()
        self._apply_fill_style(fill)

    def _begin_path(self) -> None:
        self.ctx.beginPath()
        self._record_event("begin_path_calls")

    def _stroke_path(self) -> None:
        self.ctx.stroke()
        self._record_event("stroke_calls")

    def _fill_path(self) -> None:
        self.ctx.fill()
        self._record_event("fill_calls")

    def _ensure_text_brush(self, color: str, font: FontStyle, alignment: TextAlignment) -> None:
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

    def begin_shape(self) -> None:
        self._shape_depth += 1
        self._record_event("begin_shape_calls")

    def end_shape(self) -> None:
        if self._shape_depth:
            self._shape_depth -= 1
        self._reset_alpha_if_needed()
        self._record_event("end_shape_calls")

    def _apply_stroke_style(self, stroke: StrokeStyle, *, include_width: bool = True) -> None:
        if stroke.color != self._stroke_state["color"]:
            self.ctx.strokeStyle = stroke.color
            self._stroke_state["color"] = stroke.color
            self._record_event("stroke_color_changes")
        if include_width:
            normalized_width = self._coerce_number(stroke.width)
            if normalized_width != self._stroke_state["width"]:
                self.ctx.lineWidth = normalized_width
                self._stroke_state["width"] = normalized_width
                self._record_event("stroke_width_changes")
        if stroke.line_join != self._stroke_state["line_join"]:
            if stroke.line_join:
                self.ctx.lineJoin = stroke.line_join
            self._stroke_state["line_join"] = stroke.line_join
            self._record_event("stroke_join_changes")
        if stroke.line_cap != self._stroke_state["line_cap"]:
            if stroke.line_cap:
                self.ctx.lineCap = stroke.line_cap
            self._stroke_state["line_cap"] = stroke.line_cap
            self._record_event("stroke_cap_changes")

    def _apply_fill_style(self, fill: FillStyle) -> None:
        if fill.color != self._fill_state["color"]:
            self.ctx.fillStyle = fill.color
            self._fill_state["color"] = fill.color
            self._record_event("fill_color_changes")
        if fill.opacity is None:
            if self._global_alpha != 1.0:
                self.ctx.globalAlpha = 1.0
                self._global_alpha = 1.0
                self._record_event("alpha_sets")
            self._pending_alpha_reset = False
        else:
            opacity = float(fill.opacity)
            if opacity != self._global_alpha:
                self.ctx.globalAlpha = opacity
                self._global_alpha = opacity
                self._record_event("alpha_sets")
            self._pending_alpha_reset = True

    def stroke_line(self, start: Point2D, end: Point2D, stroke: StrokeStyle, *, include_width: bool = True) -> None:
        self._prepare_stroke(stroke, include_width=include_width)
        self._begin_path()
        self.ctx.moveTo(start[0], start[1])
        self.ctx.lineTo(end[0], end[1])
        self._stroke_path()

    def stroke_polyline(self, points: List[Point2D], stroke: StrokeStyle) -> None:
        if len(points) < 2:
            return
        self._prepare_stroke(stroke)
        self._begin_path()
        self.ctx.moveTo(points[0][0], points[0][1])
        for x, y in points[1:]:
            self.ctx.lineTo(x, y)
        self._stroke_path()

    def stroke_circle(self, center: Point2D, radius: float, stroke: StrokeStyle) -> None:
        self._prepare_stroke(stroke)
        self._begin_path()
        coerced_radius = self._coerce_number(radius)
        self.ctx.arc(center[0], center[1], coerced_radius, 0, 2 * math.pi)
        self._stroke_path()

    def fill_circle(
        self,
        center: Point2D,
        radius: float,
        fill: FillStyle,
        stroke: Optional[StrokeStyle] = None,
        *,
        screen_space: bool = False,
    ) -> None:
        self._prepare_fill(fill)
        self._begin_path()
        coerced_radius = self._coerce_number(radius)
        self.ctx.arc(center[0], center[1], coerced_radius, 0, 2 * math.pi)
        self._fill_path()
        if stroke:
            self._apply_stroke_style(stroke)
            self._stroke_path()
        self._reset_alpha_if_needed()

    def stroke_ellipse(
        self,
        center: Point2D,
        radius_x: float,
        radius_y: float,
        rotation_rad: float,
        stroke: StrokeStyle,
    ) -> None:
        self._prepare_stroke(stroke)
        self._begin_path()
        rx = self._coerce_number(radius_x)
        ry = self._coerce_number(radius_y)
        self.ctx.ellipse(center[0], center[1], rx, ry, rotation_rad, 0, 2 * math.pi)
        self._stroke_path()

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
        self._batch_fill_polygon(points, fill, stroke)

    def fill_joined_area(
        self,
        forward: List[Point2D],
        reverse: List[Point2D],
        fill: FillStyle,
    ) -> None:
        if len(forward) < 2 or not reverse:
            return
        points: List[Point2D] = list(forward) + list(reverse)
        self._batch_fill_polygon(points, fill, None, is_joined_area=True)

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
        self._apply_stroke_style(stroke)
        self._begin_path()
        coerced_radius = self._coerce_number(radius)
        self.ctx.arc(center[0], center[1], coerced_radius, start_angle_rad, end_angle_rad, not sweep_clockwise)
        self._stroke_path()

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
        self._flush_polygon_batch()
        self._ensure_text_brush(color, font, alignment)
        rotation_rad = 0.0
        if isinstance(metadata, dict):
            label_meta = metadata.get("label")
            if isinstance(label_meta, dict):
                try:
                    rotation_deg = float(label_meta.get("rotation_degrees", 0.0))
                except Exception:
                    rotation_deg = 0.0
                if math.isfinite(rotation_deg) and rotation_deg != 0.0:
                    rotation_rad = math.radians(rotation_deg)
        if rotation_rad:
            self.ctx.save()
            self.ctx.translate(position[0], position[1])
            self.ctx.rotate(-rotation_rad)
            self.ctx.fillText(text, 0, 0)
            self.ctx.restore()
        else:
            self.ctx.fillText(text, position[0], position[1])
        self._record_event("text_draw_calls")

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
        did_reset = False
        was_pending = self._pending_alpha_reset
        if self._global_alpha != 1.0:
            self.ctx.globalAlpha = 1.0
            self._global_alpha = 1.0
            did_reset = True
        self._pending_alpha_reset = False
        if did_reset or was_pending:
            self._record_event("alpha_resets")

    def begin_frame(self) -> None:
        self._shape_depth = 0
        self._batch_depth = 0
        self._pending_alpha_reset = False
        self._record_event("frame_begin")

    def end_frame(self) -> None:
        self._flush_polygon_batch()
        self._flush_line_batch()
        self._reset_alpha_if_needed(force=True)
        self._record_event("frame_end")

    def begin_batch(self, plan: Any = None) -> None:
        self._batch_depth += 1
        self._record_event("begin_batch_calls")
        self._record_batch_depth()

    def end_batch(self, plan: Any = None) -> None:
        self._flush_polygon_batch()
        if self._batch_depth:
            self._batch_depth -= 1
        self._reset_alpha_if_needed()
        self._record_event("end_batch_calls")
        self._record_batch_depth()

    def clear_surface(self) -> None:
        self._flush_polygon_batch()
        self._flush_line_batch()
        self.ctx.clearRect(0, 0, self.canvas_el.width, self.canvas_el.height)
        self._record_event("clear_surface_calls")

    def fill_background(self, color: Optional[str]) -> None:
        if not color:
            return
        try:
            self.ctx.save()
            self.ctx.setTransform(1, 0, 0, 1, 0, 0)
            self.ctx.fillStyle = color
            self.ctx.fillRect(0, 0, self.canvas_el.width, self.canvas_el.height)
        except Exception:
            pass
        finally:
            try:
                self.ctx.restore()
            except Exception:
                pass

    def resize_surface(self, width: float, height: float) -> None:
        self._flush_polygon_batch()
        self._flush_line_batch()
        self.canvas_el.width = width
        self.canvas_el.height = height
        self.canvas_el.attrs["width"] = str(width)
        self.canvas_el.attrs["height"] = str(height)
        self.canvas_el.style.width = f"{int(width)}px"
        self.canvas_el.style.height = f"{int(height)}px"
        self._record_event("resize_surface_calls")

    def execute_optimized(self, command: Any) -> None:
        op = getattr(command, "op", "")
        if op == "stroke_line":
            self._batch_stroke_line(command)
            return
        if op == "stroke_polyline":
            self._batch_polyline(command)
            return
        if op in {"fill_polygon", "fill_joined_area"}:
            self._batch_fill_polygon_from_command(command)
            return
        self._flush_line_batch()
        self._flush_polygon_batch()
        handler = getattr(self, op, None)
        if callable(handler):
            handler(*getattr(command, "args", ()), **getattr(command, "kwargs", {}))

    # ------------------------------------------------------------------
    # Line batching helpers
    # ------------------------------------------------------------------

    def _batch_stroke_line(self, command: Any) -> None:
        args = getattr(command, "args", ())
        kwargs = getattr(command, "kwargs", {})
        if len(args) < 3:
            return
        start, end, stroke = args[:3]
        include_width = bool(kwargs.get("include_width", True))
        self._queue_line_segment(start, end, stroke, include_width)

    def _batch_polyline(self, command: Any) -> None:
        args = getattr(command, "args", ())
        if len(args) < 2:
            return
        points, stroke = args[:2]
        if not isinstance(points, (list, tuple)) or len(points) < 2:
            return
        prev = points[0]
        for current in points[1:]:
            self._queue_line_segment(prev, current, stroke, True)
            prev = current

    def _queue_line_segment(self, start: Point2D, end: Point2D, stroke: StrokeStyle, include_width: bool) -> None:
        signature = (
            getattr(stroke, "color", None),
            getattr(stroke, "line_join", None),
            getattr(stroke, "line_cap", None),
            getattr(stroke, "width", None) if include_width else None,
            include_width,
        )
        batch = self._line_batch
        if batch is None or batch["signature"] != signature:
            self._flush_line_batch()
            self._line_batch = {
                "stroke": stroke,
                "include_width": include_width,
                "segments": [],
                "signature": signature,
            }
            batch = self._line_batch
        batch["segments"].append((start, end))
        self._record_event("line_batch_segments")

    def _flush_line_batch(self) -> None:
        batch = self._line_batch
        if not batch:
            return
        stroke = batch["stroke"]
        include_width = batch["include_width"]
        segments = batch["segments"]
        if not segments:
            self._line_batch = None
            return
        self._apply_stroke_style(stroke, include_width=include_width)
        self.ctx.beginPath()
        self._record_event("begin_path_calls")
        for start, end in segments:
            self.ctx.moveTo(start[0], start[1])
            self.ctx.lineTo(end[0], end[1])
        self.ctx.stroke()
        self._record_event("stroke_calls")
        self._line_batch = None

    # ------------------------------------------------------------------
    # Polygon batching helpers
    # ------------------------------------------------------------------

    def _batch_fill_polygon_from_command(self, command: Any) -> None:
        args = getattr(command, "args", ())
        if not args:
            return
        if command.op == "fill_polygon":
            points, fill, stroke = args[:3]
            self._batch_fill_polygon(list(points), fill, stroke)
        else:
            forward, reverse, fill = args[:3]
            combined = list(forward) + list(reverse)
            self._batch_fill_polygon(combined, fill, None, is_joined_area=True)

    def _batch_fill_polygon(
        self,
        points: List[Point2D],
        fill: FillStyle,
        stroke: Optional[StrokeStyle],
        *,
        is_joined_area: bool = False,
    ) -> None:
        if len(points) < 3:
            return
        self._flush_line_batch()
        signature = (
            getattr(fill, "color", None),
            getattr(fill, "opacity", None),
            getattr(stroke, "color", None) if stroke else None,
            getattr(stroke, "width", None) if stroke else None,
            is_joined_area,
        )
        batch = getattr(self, "_polygon_batch", None)
        if (
            batch is None
            or batch["signature"] != signature
        ):
            self._flush_polygon_batch()
            self._polygon_batch = {
                "fill": fill,
                "stroke": stroke,
                "polygons": [],
                "signature": signature,
                "joined": is_joined_area,
            }
            batch = self._polygon_batch
        batch["polygons"].append(tuple(points))
        self._record_event("polygon_batch_polygons")

    def _flush_polygon_batch(self) -> None:
        batch = getattr(self, "_polygon_batch", None)
        if not batch:
            return
        fill = batch["fill"]
        stroke = batch["stroke"]
        polygons = batch["polygons"]
        if not polygons:
            self._polygon_batch = None
            return
        self._apply_fill_style(fill)
        for polygon in polygons:
            if len(polygon) < 3:
                continue
            self.ctx.beginPath()
            self._record_event("begin_path_calls")
            self.ctx.moveTo(polygon[0][0], polygon[0][1])
            for x, y in polygon[1:]:
                self.ctx.lineTo(x, y)
            self.ctx.closePath()
            self.ctx.fill()
            self._record_event("fill_calls")
            if stroke:
                self._apply_stroke_style(stroke)
                self.ctx.stroke()
                self._record_event("stroke_calls")
        self._polygon_batch = None
        self._reset_alpha_if_needed()

