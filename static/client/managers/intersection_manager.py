"""
Intersection Manager for MatHud

Finds where two canvas objects meet (the find_intersections tool): segments, vectors,
circles, circle arcs, ellipses, functions, piecewise functions and parametric curves, in any
mix. The geometry lives in ``utils.object_intersections``; this manager looks the objects
up by name, describes them as shapes, words the result for the AI and optionally places a
point at each intersection (one undo step).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Sequence, Tuple, cast

from managers.point_placement import place_points_at
from utils.function_features import round_report_value
from utils.object_intersections import (
    FunctionShape,
    IntersectionReport,
    ParametricShape,
    Shape,
    circle_shape,
    ellipse_shape,
    find_object_intersections,
    line_shape,
)

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.point_manager import PointManager

# Container property and reported type of every kind of object that can be intersected,
# in lookup order
_OBJECT_KINDS: Tuple[Tuple[str, str], ...] = (
    ("Segments", "segment"),
    ("Vectors", "vector"),
    ("Circles", "circle"),
    ("CircleArcs", "circle_arc"),
    ("Ellipses", "ellipse"),
    ("Functions", "function"),
    ("PiecewiseFunctions", "piecewise_function"),
    ("ParametricFunctions", "parametric_function"),
)
_LINE_KINDS = ("segment", "vector")
_FUNCTION_KINDS = ("function", "piecewise_function")


class IntersectionManager:
    """Finds the intersection points of two named canvas objects.

    Attributes:
        canvas: The Canvas (coordinate mapper for the visible range, undo manager)
        drawables: Container of all drawables, searched by name
        point_manager: Creates or reuses the points placed at intersections
    """

    def __init__(self, canvas: "Canvas", drawables: "DrawablesContainer", point_manager: "PointManager") -> None:
        self.canvas = canvas
        self.drawables = drawables
        self.point_manager = point_manager

    def find_intersections(
        self,
        object_names: Sequence[str],
        extend_lines: Optional[bool] = None,
        place_points: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Find where two objects meet.

        Segments and vectors count only between their end points unless ``extend_lines``
        makes them whole lines. A function is searched where it is plotted: its own bounds,
        else the visible x range. With ``place_points`` a point is created at every
        intersection (an existing point there is reused); together they are one undo step.

        Returns:
            Dict with the object names and types, the intersections (x, y, params per
            object, tangent), their count, a truncated flag, overlaps (shared stretches),
            x_range when a function took part, a note when useful, and the placed point
            names when points are placed.
        """
        found = self._lookup(object_names)
        shapes = [self._shape(drawable, kind, bool(extend_lines)) for drawable, kind in found]
        report = find_object_intersections(shapes[0], shapes[1])
        result = self._result(found, shapes, report)
        notes = list(report["notes"]) + self._result_notes(found, report, bool(extend_lines))
        if place_points and report["points"]:
            placed = place_points_at(
                self.point_manager,
                self.canvas.undo_redo_manager,
                cast(List[MutableMapping[str, Any]], report["points"]),
                "intersection",
            )
            reuse_note = placed.pop("note", None)
            result.update(placed)
            if reuse_note:
                notes.append(str(reuse_note))
            if report["overlaps"]:
                notes.append("Overlaps are not marked with points.")
        if notes:
            result["note"] = " ".join(notes)
        return result

    # ------------------- Lookup -------------------

    def _lookup(self, object_names: Sequence[str]) -> List[Tuple[Any, str]]:
        """The two named objects with their kinds; raises ValueError for anything else."""
        names = [str(name).strip() for name in (object_names or []) if str(name).strip()]
        if len(names) != 2:
            raise ValueError(f"Give exactly two object names to intersect; got {len(names)}.")
        if names[0] == names[1]:
            raise ValueError(f"Give two different objects to intersect; got '{names[0]}' twice.")
        found: List[Tuple[Any, str]] = []
        for name in names:
            match = self._find_object(name)
            if match is None:
                raise ValueError(
                    f"No segment, vector, circle, arc, ellipse, function or parametric curve named '{name}'. "
                    "Points and polygons cannot be intersected; use a polygon's edge segments instead."
                )
            found.append(match)
        return found

    def _find_object(self, name: str) -> Optional[Tuple[Any, str]]:
        for attribute, kind in _OBJECT_KINDS:
            for drawable in getattr(self.drawables, attribute, None) or []:
                if getattr(drawable, "name", None) == name:
                    return drawable, kind
        return None

    # ------------------- Shapes -------------------

    def _shape(self, drawable: Any, kind: str, extend_lines: bool) -> Shape:
        name = str(drawable.name)
        if kind == "segment":
            return line_shape(name, _xy(drawable.point1), _xy(drawable.point2), bounded=not extend_lines)
        if kind == "vector":
            return line_shape(name, _xy(drawable.origin), _xy(drawable.tip), bounded=not extend_lines)
        if kind == "circle":
            return circle_shape(name, _xy(drawable.center), float(drawable.radius))
        if kind == "circle_arc":
            return self._arc_shape(drawable)
        if kind == "ellipse":
            rotation = math.radians(float(getattr(drawable, "rotation_angle", 0.0) or 0.0))
            radii = float(drawable.radius_x), float(drawable.radius_y)
            return ellipse_shape(name, _xy(drawable.center), radii[0], radii[1], rotation)
        if kind in _FUNCTION_KINDS:
            return self._function_shape(drawable)
        return ParametricShape(
            name,
            drawable.evaluate_x,
            drawable.evaluate_y,
            float(drawable.t_min),
            float(drawable.t_max),
            pixels_per_unit=self._pixels_per_unit(),
        )

    @staticmethod
    def _arc_shape(arc: Any) -> Shape:
        """The arc from its end points: the shorter way round, or the longer for a major arc."""
        # Follow a parent circle that moved, as the renderer does
        arc.sync_with_circle()
        center = (float(arc.center_x), float(arc.center_y))
        first = math.atan2(float(arc.point1.y) - center[1], float(arc.point1.x) - center[0])
        second = math.atan2(float(arc.point2.y) - center[1], float(arc.point2.x) - center[0])
        counter_clockwise = (second - first) % (2.0 * math.pi)
        minor_is_counter_clockwise = counter_clockwise <= math.pi
        if minor_is_counter_clockwise != bool(arc.use_major_arc):
            start, sweep = first, counter_clockwise
        else:
            start, sweep = second, 2.0 * math.pi - counter_clockwise
        return circle_shape(str(arc.name), center, float(arc.radius), arc_start=start, arc_sweep=sweep)

    def _function_shape(self, curve: Any) -> FunctionShape:
        """The function where it is plotted: its own bounds, else the visible x range."""
        left, right = self._function_range(curve)
        breakpoints: List[float] = []
        for attribute in ("vertical_asymptotes", "point_discontinuities", "undefined_at"):
            breakpoints.extend(float(x) for x in (getattr(curve, attribute, None) or []))
        period = None
        if getattr(curve, "is_periodic", False) and getattr(curve, "estimated_period", None):
            period = float(curve.estimated_period)
        return FunctionShape(
            str(curve.name),
            curve.function,
            left,
            right,
            breakpoints=breakpoints,
            pixels_per_unit=self._pixels_per_unit(),
            period=period,
        )

    def _function_range(self, curve: Any) -> Tuple[float, float]:
        mapper = self.canvas.coordinate_mapper
        own_left = getattr(curve, "left_bound", None)
        own_right = getattr(curve, "right_bound", None)
        left = float(own_left) if own_left is not None else float(mapper.get_visible_left_bound())
        right = float(own_right) if own_right is not None else float(mapper.get_visible_right_bound())
        return (min(left, right), max(left, right))

    def _pixels_per_unit(self) -> Optional[float]:
        scale = float(getattr(self.canvas.coordinate_mapper, "scale_factor", 0.0) or 0.0)
        return scale if scale > 0.0 else None

    # ------------------- Result -------------------

    @staticmethod
    def _result(found: List[Tuple[Any, str]], shapes: List[Shape], report: IntersectionReport) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "object_names": [str(drawable.name) for drawable, _ in found],
            "object_types": [kind for _, kind in found],
            "intersections": report["points"],
            "count": report["total_found"],
            "truncated": report["truncated"],
            "overlaps": report["overlaps"],
        }
        functions = [shape for shape in shapes if isinstance(shape, FunctionShape)]
        if functions:
            left = max(shape.left for shape in functions)
            right = min(shape.right for shape in functions)
            result["x_range"] = [round_report_value(left), round_report_value(right)]
        return result

    @staticmethod
    def _result_notes(found: List[Tuple[Any, str]], report: IntersectionReport, extend_lines: bool) -> List[str]:
        names = " and ".join(str(drawable.name) for drawable, _ in found)
        kinds = [kind for _, kind in found]
        notes: List[str] = []
        if report["overlaps"]:
            notes.append(f"{names} overlap (see overlaps); shared stretches are not listed as intersection points.")
        elif not report["points"] and not report["notes"]:
            notes.append(f"{names} do not intersect.")
        if not report["points"] and not extend_lines and any(kind in _LINE_KINDS for kind in kinds):
            notes.append("Segments and vectors were not extended; set extend_lines to treat them as whole lines.")
        if any(kind in _FUNCTION_KINDS for kind in kinds) and not report["points"]:
            notes.append("A function is searched only where it is plotted (its bounds, else the visible x range).")
        if report["truncated"]:
            notes.append(f"Showing the first {len(report['points'])} of {report['total_found']} intersections by x.")
        return notes


def _xy(point: Any) -> Tuple[float, float]:
    return (float(point.x), float(point.y))
