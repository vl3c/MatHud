from __future__ import annotations

import math
from typing import Any, List, Optional

from rendering.shared_drawable_renderers import (
    FillStyle,
    FontStyle,
    Point2D,
    RendererPrimitives,
    StrokeStyle,
    TextAlignment,
)


class WebGLPrimitiveAdapter(RendererPrimitives):
    """RendererPrimitives implementation for WebGL (limited support)."""

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer

    def stroke_line(self, start: Point2D, end: Point2D, stroke: StrokeStyle, *, include_width: bool = True) -> None:
        self.renderer._draw_lines([start, end], self.renderer._parse_color(stroke.color))

    def stroke_polyline(self, points: List[Point2D], stroke: StrokeStyle) -> None:
        self.renderer._draw_line_strip(points, self.renderer._parse_color(stroke.color))

    def stroke_circle(self, center: Point2D, radius: float, stroke: StrokeStyle) -> None:
        samples = self._sample_circle(center, radius)
        self.stroke_polyline(samples, stroke)

    def fill_circle(
        self,
        center: Point2D,
        radius: float,
        fill: FillStyle,
        stroke: Optional[StrokeStyle] = None,
    ) -> None:
        raise NotImplementedError

    def stroke_ellipse(
        self,
        center: Point2D,
        radius_x: float,
        radius_y: float,
        rotation_rad: float,
        stroke: StrokeStyle,
    ) -> None:
        samples = self._sample_ellipse(center, radius_x, radius_y, rotation_rad)
        self.stroke_polyline(samples, stroke)

    def fill_polygon(
        self,
        points: List[Point2D],
        fill: FillStyle,
        stroke: Optional[StrokeStyle] = None,
    ) -> None:
        raise NotImplementedError

    def fill_joined_area(
        self,
        forward: List[Point2D],
        reverse: List[Point2D],
        fill: FillStyle,
    ) -> None:
        raise NotImplementedError

    def stroke_arc(
        self,
        center: Point2D,
        radius: float,
        start_angle_rad: float,
        end_angle_rad: float,
        sweep_clockwise: bool,
        stroke: StrokeStyle,
    ) -> None:
        samples = self._sample_arc(center, radius, start_angle_rad, end_angle_rad, sweep_clockwise)
        self.stroke_polyline(samples, stroke)

    def draw_text(
        self,
        text: str,
        position: Point2D,
        font: FontStyle,
        color: str,
        alignment: TextAlignment,
    ) -> None:
        raise NotImplementedError

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

