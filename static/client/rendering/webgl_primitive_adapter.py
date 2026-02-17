"""WebGL primitive adapter for GPU-accelerated drawing (experimental).

This module provides the WebGLPrimitiveAdapter class that implements the
RendererPrimitives interface using WebGL shader programs. Support is limited
to basic shapes like lines, circles, and polygons.

Key Features:
    - GPU-accelerated point and line rendering
    - Circle and ellipse sampling for stroke rendering
    - Basic polygon outline support
    - Integration with WebGLRenderer shader programs

Note:
    This adapter has limited support compared to Canvas2D and SVG adapters.
    Text rendering and advanced features are not implemented.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Dict

from rendering.primitives import (
    FillStyle,
    FontStyle,
    RendererPrimitives,
    StrokeStyle,
    TextAlignment,
)
from rendering.shared_drawable_renderers import Point2D


class WebGLPrimitiveAdapter(RendererPrimitives):
    """RendererPrimitives implementation for WebGL with limited support.

    Translates drawing commands to WebGL API calls via the parent renderer.
    Only supports basic shapes; text and advanced features are not available.

    Attributes:
        renderer: The WebGLRenderer instance with GL context and programs.
    """

    def __init__(self, renderer: Any) -> None:
        """Initialize the WebGL primitive adapter.

        Args:
            renderer: The WebGLRenderer that provides the GL context.
        """
        self.renderer = renderer

    def _draw_line_strip_with_stroke(self, points: List[Point2D], stroke: StrokeStyle) -> None:
        color = self.renderer._parse_color(stroke.color)
        self.renderer._draw_line_strip(points, color)

    def stroke_line(self, start: Point2D, end: Point2D, stroke: StrokeStyle, *, include_width: bool = True) -> None:
        self.renderer._draw_lines([start, end], self.renderer._parse_color(stroke.color))

    def stroke_polyline(self, points: List[Point2D], stroke: StrokeStyle) -> None:
        self._draw_line_strip_with_stroke(points, stroke)

    def stroke_circle(self, center: Point2D, radius: float, stroke: StrokeStyle) -> None:
        samples = self._sample_circle(center, radius)
        self._draw_line_strip_with_stroke(samples, stroke)

    def fill_circle(
        self,
        center: Point2D,
        radius: float,
        fill: FillStyle,
        stroke: Optional[StrokeStyle] = None,
        *,
        screen_space: bool = False,
    ) -> None:
        try:
            size = float(radius) * 2.0
        except Exception:
            size = 0.0
        if size <= 0.0:
            size = 1.0
        color = self.renderer._parse_color(fill.color)
        self.renderer._draw_points([center], color, size)
        if stroke:
            self.stroke_circle(center, radius, stroke)

    def stroke_ellipse(
        self,
        center: Point2D,
        radius_x: float,
        radius_y: float,
        rotation_rad: float,
        stroke: StrokeStyle,
    ) -> None:
        samples = self._sample_ellipse(center, radius_x, radius_y, rotation_rad)
        self._draw_line_strip_with_stroke(samples, stroke)

    def fill_polygon(
        self,
        points: List[Point2D],
        fill: FillStyle,
        stroke: Optional[StrokeStyle] = None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(points) < 2:
            return
        color_source = stroke.color if stroke else fill.color
        color = self.renderer._parse_color(color_source)
        path = list(points)
        if path[0] != path[-1]:
            path = path + [path[0]]
        self.renderer._draw_line_strip(path, color)

    def fill_joined_area(
        self,
        forward: List[Point2D],
        reverse: List[Point2D],
        fill: FillStyle,
    ) -> None:
        if len(forward) < 2 or not reverse:
            return
        outline = list(forward) + list(reverse)
        if outline[0] != outline[-1]:
            outline.append(outline[0])
        color = self.renderer._parse_color(fill.color)
        self.renderer._draw_line_strip(outline, color)

    def stroke_arc(
        self,
        center: Point2D,
        radius: float,
        start_angle_rad: float,
        end_angle_rad: float,
        sweep_clockwise: bool,
        stroke: StrokeStyle,
        css_class: Optional[str] = None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        samples = self._sample_arc(center, radius, start_angle_rad, end_angle_rad, sweep_clockwise)
        self._draw_line_strip_with_stroke(samples, stroke)

    def draw_text(
        self,
        text: str,
        position: Point2D,
        font: FontStyle,
        color: str,
        alignment: TextAlignment,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Text rendering is not supported in the current WebGL pipeline; no-op.
        return None

    def clear_surface(self) -> None:
        self.renderer.clear()

    def resize_surface(self, width: float, height: float) -> None:
        self.renderer._resize_viewport()

    def _sample_circle(self, center: Point2D, radius: float, segments: int = 64) -> list[Point2D]:
        return [
            (
                center[0] + radius * math.cos(2 * math.pi * i / segments),
                center[1] + radius * math.sin(2 * math.pi * i / segments),
            )
            for i in range(segments)
        ] + [
            (
                center[0] + radius * math.cos(0),
                center[1] + radius * math.sin(0),
            )
        ]

    def _sample_ellipse(
        self,
        center: Point2D,
        radius_x: float,
        radius_y: float,
        rotation_rad: float,
        segments: int = 64,
    ) -> list[Point2D]:
        cos_r = math.cos(rotation_rad)
        sin_r = math.sin(rotation_rad)
        samples: list[Point2D] = []
        for i in range(segments + 1):
            theta = 2 * math.pi * i / segments
            x = radius_x * math.cos(theta)
            y = radius_y * math.sin(theta)
            rotated_x = x * cos_r - y * sin_r + center[0]
            rotated_y = x * sin_r + y * cos_r + center[1]
            samples.append((rotated_x, rotated_y))
        return samples

    def _sample_arc(
        self,
        center: Point2D,
        radius: float,
        start_angle_rad: float,
        end_angle_rad: float,
        sweep_clockwise: bool,
        segments: int = 32,
    ) -> list[Point2D]:
        total_angle = end_angle_rad - start_angle_rad
        if sweep_clockwise and total_angle > 0:
            total_angle -= 2 * math.pi
        elif not sweep_clockwise and total_angle < 0:
            total_angle += 2 * math.pi
        step = total_angle / max(segments, 1)
        samples = []
        for i in range(segments + 1):
            theta = start_angle_rad + step * i
            samples.append((center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)))
        return samples
