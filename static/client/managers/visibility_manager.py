"""
MatHud Visibility Manager

Provides viewport culling logic to determine whether drawable objects fall within
the visible canvas area.  Extracted from Canvas to isolate geometric visibility
checks from the central coordinator.

Dependencies are injected as lightweight values (coordinate_mapper, width, height)
so the manager never holds a back-reference to Canvas.
"""

from __future__ import annotations

from typing import Any, Tuple, TYPE_CHECKING

from utils.math_utils import MathUtils

if TYPE_CHECKING:
    from coordinate_mapper import CoordinateMapper
    from drawables.drawable import Drawable


class VisibilityManager:
    """Viewport culling for drawable objects.

    Determines whether points, segments, vectors, and other drawables intersect
    the visible canvas rectangle.  Types without a specialised check default to
    visible so that complex shapes (circles, functions, etc.) are never
    incorrectly culled.

    Attributes:
        _coordinate_mapper: Coordinate transformation service for math-to-screen conversion.
        _width: Canvas viewport width in pixels.
        _height: Canvas viewport height in pixels.
    """

    def __init__(self, coordinate_mapper: "CoordinateMapper", width: float, height: float) -> None:
        """Initialise the visibility manager.

        Args:
            coordinate_mapper: Provides math-to-screen coordinate conversion.
            width: Canvas viewport width in pixels.
            height: Canvas viewport height in pixels.
        """
        self._coordinate_mapper: "CoordinateMapper" = coordinate_mapper
        self._width: float = width
        self._height: float = height

    # ------------------------------------------------------------------
    # Top-level drawable visibility
    # ------------------------------------------------------------------

    def is_drawable_visible(self, drawable: "Drawable") -> bool:
        """Best-effort visibility check to avoid rendering off-canvas objects.

        Mirrors prior behaviour for segments and points; other types default to
        visible because they manage their own bounds or are inexpensive.
        """
        class_name = self._safe_drawable_class_name(drawable)
        try:
            if class_name == "Point":
                return self._is_point_drawable_visible(drawable)

            if class_name == "Segment":
                return self._is_segment_drawable_visible(drawable)

            if class_name == "Vector":
                return self._is_vector_drawable_visible(drawable)

            # Default: visible
            return True
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Public primitive checks
    # ------------------------------------------------------------------

    def is_point_within_canvas_visible_area(self, x: float, y: float) -> bool:
        """Check if a screen-coordinate point is within the visible canvas area."""
        return (0 <= x <= self._width) and (0 <= y <= self._height)

    def any_segment_part_visible_in_canvas_area(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> bool:
        """Check if any part of a segment (in screen coordinates) is visible."""
        intersect_top = MathUtils.segments_intersect(x1, y1, x2, y2, 0, 0, self._width, 0)
        intersect_right = MathUtils.segments_intersect(
            x1, y1, x2, y2, self._width, 0, self._width, self._height
        )
        intersect_bottom = MathUtils.segments_intersect(
            x1, y1, x2, y2, self._width, self._height, 0, self._height
        )
        intersect_left = MathUtils.segments_intersect(x1, y1, x2, y2, 0, self._height, 0, 0)
        point1_visible: bool = self.is_point_within_canvas_visible_area(x1, y1)
        point2_visible: bool = self.is_point_within_canvas_visible_area(x2, y2)
        return bool(
            intersect_top
            or intersect_right
            or intersect_bottom
            or intersect_left
            or point1_visible
            or point2_visible
        )

    # ------------------------------------------------------------------
    # Drawable-type specific checks (private)
    # ------------------------------------------------------------------

    def _is_point_drawable_visible(self, drawable: Any) -> bool:
        """Check visibility for a Point drawable."""
        x, y = self._coordinate_mapper.math_to_screen(drawable.x, drawable.y)
        return self.is_point_within_canvas_visible_area(x, y)

    def _is_segment_drawable_visible(self, drawable: Any) -> bool:
        """Check visibility for a Segment drawable."""
        return self._is_math_segment_visible(drawable.point1, drawable.point2)

    def _is_vector_drawable_visible(self, drawable: Any) -> bool:
        """Check visibility for a Vector drawable."""
        seg = getattr(drawable, "segment", None)
        if seg is None:
            return True
        return self._is_math_segment_visible(seg.point1, seg.point2)

    # ------------------------------------------------------------------
    # Segment helpers (private)
    # ------------------------------------------------------------------

    def _is_math_segment_visible(self, p1: Any, p2: Any) -> bool:
        """Check whether a math-coordinate segment is visible on screen."""
        x1, y1, x2, y2 = self._segment_screen_coordinates(p1, p2)
        return self._is_screen_segment_visible(x1, y1, x2, y2)

    def _segment_screen_coordinates(self, p1: Any, p2: Any) -> Tuple[float, float, float, float]:
        """Convert two math-coordinate points to screen coordinates."""
        x1, y1 = self._coordinate_mapper.math_to_screen(p1.x, p1.y)
        x2, y2 = self._coordinate_mapper.math_to_screen(p2.x, p2.y)
        return x1, y1, x2, y2

    def _is_screen_segment_visible(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> bool:
        """Check whether a screen-coordinate segment is visible."""
        return (
            self.is_point_within_canvas_visible_area(x1, y1)
            or self.is_point_within_canvas_visible_area(x2, y2)
            or self.any_segment_part_visible_in_canvas_area(x1, y1, x2, y2)
        )

    # ------------------------------------------------------------------
    # Utilities (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_drawable_class_name(drawable: Any) -> str:
        """Return the class name of a drawable, falling back to __class__.__name__."""
        try:
            return str(
                drawable.get_class_name()
                if hasattr(drawable, "get_class_name")
                else drawable.__class__.__name__
            )
        except Exception:
            return str(drawable.__class__.__name__)
