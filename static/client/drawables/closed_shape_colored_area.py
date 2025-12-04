"""
MatHud Closed Shape Colored Area

Fills the interior of polygons, circles, ellipses, simple round-shape segments,
or arbitrary regions defined by boolean expressions.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from constants import default_area_fill_color, default_area_opacity, default_closed_shape_resolution, closed_shape_resolution_minimum
from drawables.circle import Circle
from drawables.colored_area import ColoredArea
from drawables.ellipse import Ellipse
from drawables.segment import Segment
from utils.geometry_utils import GeometryUtils


class ClosedShapeColoredArea(ColoredArea):
    """
    Represents a colored area bounded by closed geometric figures or region expressions.
    """

    SUPPORTED_SHAPES = {
        "polygon",
        "circle",
        "ellipse",
        "circle_segment",
        "ellipse_segment",
        "region",
    }

    def __init__(
        self,
        *,
        shape_type: str,
        segments: Optional[List[Segment]] = None,
        circle: Optional[Circle] = None,
        ellipse: Optional[Ellipse] = None,
        chord_segment: Optional[Segment] = None,
        arc_clockwise: bool = False,
        resolution: int = default_closed_shape_resolution,
        color: str = default_area_fill_color,
        opacity: float = default_area_opacity,
        name: Optional[str] = None,
        expression: Optional[str] = None,
        points: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        if shape_type not in self.SUPPORTED_SHAPES:
            raise ValueError(f"Unsupported closed shape type '{shape_type}'")

        self.shape_type: str = shape_type
        self.segments: List[Segment] = list(segments or [])
        self.circle: Optional[Circle] = circle
        self.ellipse: Optional[Ellipse] = ellipse
        self.chord_segment: Optional[Segment] = chord_segment
        if self.chord_segment:
            chord_name = getattr(self.chord_segment, "name", None)
            if all(getattr(segment, "name", None) != chord_name for segment in self.segments):
                self.segments.append(self.chord_segment)
        self.arc_clockwise: bool = bool(arc_clockwise)
        self.resolution: int = max(closed_shape_resolution_minimum, int(resolution))
        self.expression: Optional[str] = expression
        self.points: List[Tuple[float, float]] = list(points or [])

        generated_name = name or self._generate_name()
        super().__init__(name=generated_name, color=color, opacity=opacity)

    def _generate_name(self) -> str:
        if self.shape_type == "polygon" and self.segments:
            seg_names = "_".join(seg.name for seg in self.segments[:3])
            return f"closed_polygon_{seg_names}"
        if self.shape_type.startswith("circle") and self.circle:
            return f"closed_{self.circle.name}"
        if self.shape_type.startswith("ellipse") and self.ellipse:
            return f"closed_{self.ellipse.name}"
        if self.shape_type == "region" and self.expression:
            safe_expr = self.expression.replace(" ", "").replace("&", "_and_").replace("|", "_or_")[:30]
            return f"region_{safe_expr}"
        return f"closed_shape_{self.shape_type}"

    def get_class_name(self) -> str:
        return "ClosedShapeColoredArea"

    def uses_segment(self, segment: Segment) -> bool:
        return any(seg.name == segment.name for seg in self.segments)

    def uses_circle(self, circle: Circle) -> bool:
        return bool(self.circle and self.circle.name == circle.name)

    def uses_ellipse(self, ellipse: Ellipse) -> bool:
        return bool(self.ellipse and self.ellipse.name == ellipse.name)

    def get_geometry_spec(self) -> Dict[str, Any]:
        """
        Provide the raw geometry references for renderables.
        """
        spec: Dict[str, Any] = {
            "shape_type": self.shape_type,
            "segments": self.segments,
            "circle": self.circle,
            "ellipse": self.ellipse,
            "chord_segment": self.chord_segment,
            "arc_clockwise": self.arc_clockwise,
            "resolution": self.resolution,
        }
        if self.shape_type == "region":
            spec["expression"] = self.expression
            spec["points"] = self.points
        return spec

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["args"].update(
            {
                "shape_type": self.shape_type,
                "segments": [segment.name for segment in self.segments],
                "circle": self.circle.name if self.circle else None,
                "ellipse": self.ellipse.name if self.ellipse else None,
                "chord_segment": self.chord_segment.name if self.chord_segment else None,
                "arc_clockwise": self.arc_clockwise,
                "resolution": self.resolution,
                "expression": self.expression,
                "points": [[x, y] for x, y in self.points] if self.points else None,
                "geometry_snapshot": self._snapshot_geometry(),
            }
        )
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ClosedShapeColoredArea":
        if id(self) in memo:
            return memo[id(self)]

        new_segments = [copy.deepcopy(segment, memo) for segment in self.segments]
        new_circle = copy.deepcopy(self.circle, memo) if self.circle else None
        new_ellipse = copy.deepcopy(self.ellipse, memo) if self.ellipse else None
        new_chord = copy.deepcopy(self.chord_segment, memo) if self.chord_segment else None

        copied = ClosedShapeColoredArea(
            shape_type=self.shape_type,
            segments=new_segments,
            circle=new_circle,
            ellipse=new_ellipse,
            chord_segment=new_chord,
            arc_clockwise=self.arc_clockwise,
            resolution=self.resolution,
            color=self.color,
            opacity=self.opacity,
            name=self.name,
            expression=self.expression,
            points=list(self.points) if self.points else None,
        )
        memo[id(self)] = copied
        return copied

    def _snapshot_geometry(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}

        if self.shape_type == "polygon":
            coords = GeometryUtils.polygon_math_coordinates_from_segments(self.segments) or []
            snapshot["polygon_coords"] = [[x, y] for x, y in coords]
        if self.shape_type == "region" and self.points:
            snapshot["region_points"] = [[x, y] for x, y in self.points]
            snapshot["expression"] = self.expression
        if self.circle and hasattr(self.circle, "center"):
            center = getattr(self.circle, "center", None)
            if center and hasattr(center, "x") and hasattr(center, "y"):
                snapshot["circle"] = {
                    "center": [float(center.x), float(center.y)],
                    "radius": float(getattr(self.circle, "radius", 0.0)),
                }
        if self.ellipse and hasattr(self.ellipse, "center"):
            center = getattr(self.ellipse, "center", None)
            if center and hasattr(center, "x") and hasattr(center, "y"):
                snapshot["ellipse"] = {
                    "center": [float(center.x), float(center.y)],
                    "radius_x": float(getattr(self.ellipse, "radius_x", 0.0)),
                    "radius_y": float(getattr(self.ellipse, "radius_y", 0.0)),
                    "rotation": float(getattr(self.ellipse, "rotation_angle", 0.0)),
                }
        if self.chord_segment and hasattr(self.chord_segment, "point1") and hasattr(self.chord_segment, "point2"):
            snapshot["chord_endpoints"] = [
                [float(self.chord_segment.point1.x), float(self.chord_segment.point1.y)],
                [float(self.chord_segment.point2.x), float(self.chord_segment.point2.y)],
            ]
        if self.segments:
            snapshot["segments"] = [
                [
                    float(getattr(segment.point1, "x", 0.0)),
                    float(getattr(segment.point1, "y", 0.0)),
                    float(getattr(segment.point2, "x", 0.0)),
                    float(getattr(segment.point2, "y", 0.0)),
                ]
                for segment in self.segments
            ]

        return snapshot


