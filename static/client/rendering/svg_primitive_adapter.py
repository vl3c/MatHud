from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

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

    def __init__(self, surface_id: str = "math-svg") -> None:
        self.surface_id = surface_id

    @property
    def _surface(self) -> Any:
        return document[self.surface_id]

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
    ) -> None:
        elem = svg.text(text, x=str(position[0]), y=str(position[1]), fill=color)
        elem.setAttribute("font-size", str(font.size))
        if font.weight:
            elem.setAttribute("font-weight", font.weight)
        
        horizontal_anchor = alignment.horizontal
        if horizontal_anchor == "center":
            horizontal_anchor = "middle"
        
        if horizontal_anchor and horizontal_anchor != "left":
            elem.style["text-anchor"] = horizontal_anchor
        if alignment.vertical and alignment.vertical != "alphabetic":
            elem.style["dominant-baseline"] = alignment.vertical

        if style_overrides:
            for key, value in style_overrides.items():
                elem.style[key] = value
        
        self._surface <= elem

    def clear_surface(self) -> None:
        self._surface.clear()

    def resize_surface(self, width: float, height: float) -> None:
        surface = self._surface
        surface.setAttribute("width", str(width))
        surface.setAttribute("height", str(height))

