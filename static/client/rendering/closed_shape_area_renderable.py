"""
Renderable for ClosedShapeColoredArea producing screen-space ClosedArea instructions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rendering.primitives import ClosedArea
from utils.geometry_utils import GeometryUtils
from utils.math_utils import MathUtils


class ClosedShapeAreaRenderable:
    def __init__(self, area_model: Any, coordinate_mapper: Any) -> None:
        self.area = area_model
        self.mapper = coordinate_mapper

    def build_screen_area(self) -> Optional[ClosedArea]:
        if not hasattr(self.area, "get_geometry_spec"):
            return None
        spec: Dict[str, Any] = self.area.get_geometry_spec()
        shape_type: str = spec.get("shape_type", "")

        builders = {
            "polygon": self._build_polygon_area,
            "circle": self._build_circle_area,
            "ellipse": self._build_ellipse_area,
            "circle_segment": self._build_circle_segment_area,
            "ellipse_segment": self._build_ellipse_segment_area,
        }

        builder = builders.get(shape_type)
        if not builder:
            return None

        geometry = builder(spec)
        if geometry is None:
            return None

        forward_points, reverse_points = geometry
        if len(forward_points) < 3 or len(reverse_points) < 2:
            return None

        return ClosedArea(
            forward_points,
            reverse_points,
            is_screen=False,
            color=getattr(self.area, "color", None),
            opacity=getattr(self.area, "opacity", None),
        )

    def _build_polygon_area(
        self, spec: Dict[str, Any]
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        segments = spec.get("segments", [])
        if not segments:
            return None
        coords = GeometryUtils.polygon_math_coordinates_from_segments(segments)
        if not coords or len(coords) < 3:
            return None
        forward = list(coords)
        reverse = list(reversed(coords))
        return forward, reverse

    def _build_circle_area(
        self, spec: Dict[str, Any]
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        circle = spec.get("circle")
        if not circle or not hasattr(circle, "center"):
            return None
        points = MathUtils.sample_circle_arc(
            circle.center.x,
            circle.center.y,
            circle.radius,
            0.0,
            0.0,
            num_samples=spec.get("resolution", 96),
        )
        if len(points) < 3:
            return None
        return points, list(reversed(points))

    def _build_ellipse_area(
        self, spec: Dict[str, Any]
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        ellipse = spec.get("ellipse")
        if not ellipse or not hasattr(ellipse, "center"):
            return None
        points = MathUtils.sample_ellipse_arc(
            ellipse.center.x,
            ellipse.center.y,
            ellipse.radius_x,
            ellipse.radius_y,
            0.0,
            0.0,
            rotation_degrees=getattr(ellipse, "rotation_angle", 0.0),
            num_samples=spec.get("resolution", 96),
        )
        if len(points) < 3:
            return None
        return points, list(reversed(points))

    def _build_circle_segment_area(
        self, spec: Dict[str, Any]
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        circle = spec.get("circle")
        chord_segment = spec.get("chord_segment")
        if not circle or not chord_segment:
            return None
        intersections = MathUtils.circle_segment_intersections(
            circle.center.x,
            circle.center.y,
            circle.radius,
            chord_segment,
        )
        if len(intersections) < 2:
            return None
        start = intersections[0]
        end = intersections[1]
        points = MathUtils.sample_circle_arc(
            circle.center.x,
            circle.center.y,
            circle.radius,
            start["angle"],
            end["angle"],
            num_samples=spec.get("resolution", 64),
            clockwise=spec.get("arc_clockwise", False),
        )
        if not points:
            return None
        points[0] = (start["x"], start["y"])
        points[-1] = (end["x"], end["y"])
        reverse = [(end["x"], end["y"]), (start["x"], start["y"])]
        return points, reverse

    def _build_ellipse_segment_area(
        self, spec: Dict[str, Any]
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        ellipse = spec.get("ellipse")
        chord_segment = spec.get("chord_segment")
        if not ellipse or not chord_segment:
            return None
        intersections = MathUtils.ellipse_segment_intersections(
            ellipse.center.x,
            ellipse.center.y,
            ellipse.radius_x,
            ellipse.radius_y,
            getattr(ellipse, "rotation_angle", 0.0),
            chord_segment,
        )
        if len(intersections) < 2:
            return None
        start = intersections[0]
        end = intersections[1]
        points = MathUtils.sample_ellipse_arc(
            ellipse.center.x,
            ellipse.center.y,
            ellipse.radius_x,
            ellipse.radius_y,
            start["angle"],
            end["angle"],
            rotation_degrees=getattr(ellipse, "rotation_angle", 0.0),
            num_samples=spec.get("resolution", 64),
            clockwise=spec.get("arc_clockwise", False),
        )
        if not points:
            return None
        points[0] = (start["x"], start["y"])
        points[-1] = (end["x"], end["y"])
        reverse = [(end["x"], end["y"]), (start["x"], start["y"])]
        return points, reverse


