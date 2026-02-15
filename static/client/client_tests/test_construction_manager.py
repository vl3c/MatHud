"""Tests for ConstructionManager - geometric constructions."""

from __future__ import annotations

import math
import unittest
from typing import List, Optional

from canvas import Canvas
from drawables.point import Point
from drawables.segment import Segment
from utils.math_utils import MathUtils


class TestConstructionManager(unittest.TestCase):
    """Base test class for ConstructionManager tests."""

    def setUp(self) -> None:
        """Set up test canvas with draw disabled."""
        self.canvas = Canvas(500, 500, draw_enabled=False)

    def _get_points(self) -> List[Point]:
        return [d for d in self.canvas.get_drawables_by_class_name("Point")]

    def _get_segments(self) -> List[Segment]:
        return [d for d in self.canvas.get_drawables_by_class_name("Segment")]

    def _point_count(self) -> int:
        return len(self._get_points())

    def _segment_count(self) -> int:
        return len(self._get_segments())


class TestConstructMidpoint(TestConstructionManager):
    """Tests for midpoint construction."""

    def test_midpoint_of_horizontal_segment(self) -> None:
        """Midpoint of (0,0)-(4,0) should be (2,0)."""
        self.canvas.create_point(0, 0, name="A")
        self.canvas.create_point(4, 0, name="B")
        self.canvas.create_segment(0, 0, 4, 0, name="AB")
        pt = self.canvas.create_midpoint(segment_name="AB", name="M")
        self.assertAlmostEqual(pt.x, 2.0, places=5)
        self.assertAlmostEqual(pt.y, 0.0, places=5)

    def test_midpoint_of_vertical_segment(self) -> None:
        """Midpoint of (0,0)-(0,6) should be (0,3)."""
        self.canvas.create_point(0, 0, name="A")
        self.canvas.create_point(0, 6, name="B")
        self.canvas.create_segment(0, 0, 0, 6, name="AB")
        pt = self.canvas.create_midpoint(segment_name="AB", name="M")
        self.assertAlmostEqual(pt.x, 0.0, places=5)
        self.assertAlmostEqual(pt.y, 3.0, places=5)

    def test_midpoint_of_diagonal_segment(self) -> None:
        """Midpoint of (1,1)-(5,7) should be (3,4)."""
        self.canvas.create_point(1, 1, name="A")
        self.canvas.create_point(5, 7, name="B")
        self.canvas.create_segment(1, 1, 5, 7, name="AB")
        pt = self.canvas.create_midpoint(segment_name="AB", name="M")
        self.assertAlmostEqual(pt.x, 3.0, places=5)
        self.assertAlmostEqual(pt.y, 4.0, places=5)

    def test_midpoint_by_point_names(self) -> None:
        """Midpoint using p1_name/p2_name instead of segment_name."""
        self.canvas.create_point(0, 0, name="A")
        self.canvas.create_point(10, 0, name="B")
        pt = self.canvas.create_midpoint(p1_name="A", p2_name="B", name="M")
        self.assertAlmostEqual(pt.x, 5.0, places=5)
        self.assertAlmostEqual(pt.y, 0.0, places=5)

    def test_midpoint_with_negative_coords(self) -> None:
        """Midpoint of (-4,-2)-(6,8) should be (1,3)."""
        self.canvas.create_point(-4, -2, name="A")
        self.canvas.create_point(6, 8, name="B")
        pt = self.canvas.create_midpoint(p1_name="A", p2_name="B", name="M")
        self.assertAlmostEqual(pt.x, 1.0, places=5)
        self.assertAlmostEqual(pt.y, 3.0, places=5)

    def test_midpoint_nonexistent_segment_raises(self) -> None:
        """Midpoint of nonexistent segment raises ValueError."""
        with self.assertRaises(ValueError):
            self.canvas.create_midpoint(segment_name="nonexistent")

    def test_midpoint_nonexistent_point_raises(self) -> None:
        """Midpoint with nonexistent point names raises ValueError."""
        self.canvas.create_point(0, 0, name="A")
        with self.assertRaises(ValueError):
            self.canvas.create_midpoint(p1_name="A", p2_name="Z")

    def test_midpoint_no_args_raises(self) -> None:
        """Midpoint with no arguments raises ValueError."""
        with self.assertRaises(ValueError):
            self.canvas.create_midpoint()

    def test_midpoint_undo(self) -> None:
        """Undo should remove the midpoint."""
        self.canvas.create_point(0, 0, name="A")
        self.canvas.create_point(4, 0, name="B")
        self.canvas.create_segment(0, 0, 4, 0, name="AB")
        count_before = self._point_count()
        self.canvas.create_midpoint(segment_name="AB", name="M")
        self.assertEqual(self._point_count(), count_before + 1)
        self.canvas.undo()
        self.assertEqual(self._point_count(), count_before)


class TestConstructPerpendicularBisector(TestConstructionManager):
    """Tests for perpendicular bisector construction."""

    def test_bisector_passes_through_midpoint(self) -> None:
        """Perpendicular bisector should pass through the midpoint."""
        self.canvas.create_segment(0, 0, 4, 0, name="AB")
        bisector = self.canvas.create_perpendicular_bisector("AB")
        # Midpoint of (0,0)-(4,0) is (2,0)
        # Bisector is vertical (perp to horizontal), so both endpoints have x=2
        mid_x = (bisector.point1.x + bisector.point2.x) / 2
        mid_y = (bisector.point1.y + bisector.point2.y) / 2
        self.assertAlmostEqual(mid_x, 2.0, places=5)
        self.assertAlmostEqual(mid_y, 0.0, places=5)

    def test_bisector_is_perpendicular(self) -> None:
        """Perpendicular bisector should be perpendicular to the original (dot product ~ 0)."""
        self.canvas.create_segment(0, 0, 4, 2, name="AB")
        bisector = self.canvas.create_perpendicular_bisector("AB")
        # Original direction vector
        orig_dx = 4.0 - 0.0
        orig_dy = 2.0 - 0.0
        # Bisector direction vector
        bis_dx = bisector.point2.x - bisector.point1.x
        bis_dy = bisector.point2.y - bisector.point1.y
        dot = orig_dx * bis_dx + orig_dy * bis_dy
        self.assertAlmostEqual(dot, 0.0, places=3)

    def test_bisector_of_vertical_segment(self) -> None:
        """Perpendicular bisector of vertical segment should be horizontal."""
        self.canvas.create_segment(0, 0, 0, 6, name="AB")
        bisector = self.canvas.create_perpendicular_bisector("AB")
        # Bisector should be horizontal (same y for both endpoints)
        self.assertAlmostEqual(bisector.point1.y, bisector.point2.y, places=5)
        # And pass through midpoint (0, 3)
        self.assertAlmostEqual(bisector.point1.y, 3.0, places=5)

    def test_bisector_custom_length(self) -> None:
        """Perpendicular bisector should have the specified length."""
        self.canvas.create_segment(0, 0, 4, 0, name="AB")
        bisector = self.canvas.create_perpendicular_bisector("AB", length=10.0)
        dx = bisector.point2.x - bisector.point1.x
        dy = bisector.point2.y - bisector.point1.y
        actual_length = math.sqrt(dx**2 + dy**2)
        self.assertAlmostEqual(actual_length, 10.0, places=3)

    def test_bisector_nonexistent_segment_raises(self) -> None:
        """Perpendicular bisector of nonexistent segment raises ValueError."""
        with self.assertRaises(ValueError):
            self.canvas.create_perpendicular_bisector("nonexistent")

    def test_bisector_undo(self) -> None:
        """Undo should remove the bisector segment."""
        self.canvas.create_segment(0, 0, 4, 0, name="AB")
        count_before = self._segment_count()
        self.canvas.create_perpendicular_bisector("AB")
        self.assertEqual(self._segment_count(), count_before + 1)
        self.canvas.undo()
        self.assertEqual(self._segment_count(), count_before)


class TestConstructPerpendicularFromPoint(TestConstructionManager):
    """Tests for perpendicular from point to line construction."""

    def test_perpendicular_foot_on_horizontal_segment(self) -> None:
        """Foot of perpendicular from (3,5) to x-axis segment should be (3,0)."""
        self.canvas.create_point(3, 5, name="P")
        self.canvas.create_segment(0, 0, 6, 0, name="AB")
        result = self.canvas.create_perpendicular_from_point("P", "AB")
        foot = result["foot"]
        self.assertAlmostEqual(foot.x, 3.0, places=5)
        self.assertAlmostEqual(foot.y, 0.0, places=5)

    def test_perpendicular_foot_on_vertical_segment(self) -> None:
        """Foot of perpendicular from (5,3) to y-axis segment should be (0,3)."""
        self.canvas.create_point(5, 3, name="P")
        self.canvas.create_segment(0, 0, 0, 6, name="AB")
        result = self.canvas.create_perpendicular_from_point("P", "AB")
        foot = result["foot"]
        self.assertAlmostEqual(foot.x, 0.0, places=5)
        self.assertAlmostEqual(foot.y, 3.0, places=5)

    def test_perpendicular_foot_on_diagonal(self) -> None:
        """Foot of perpendicular from (0,4) to y=x should be (2,2)."""
        self.canvas.create_point(0, 4, name="P")
        self.canvas.create_segment(0, 0, 4, 4, name="AB")
        result = self.canvas.create_perpendicular_from_point("P", "AB")
        foot = result["foot"]
        self.assertAlmostEqual(foot.x, 2.0, places=5)
        self.assertAlmostEqual(foot.y, 2.0, places=5)

    def test_perpendicular_creates_segment(self) -> None:
        """Construction should create a segment from point to foot."""
        self.canvas.create_point(3, 5, name="P")
        self.canvas.create_segment(0, 0, 6, 0, name="AB")
        result = self.canvas.create_perpendicular_from_point("P", "AB")
        seg = result["segment"]
        # Segment endpoints should be at (3,5) and (3,0)
        xs = sorted([seg.point1.x, seg.point2.x])
        ys = sorted([seg.point1.y, seg.point2.y])
        self.assertAlmostEqual(xs[0], 3.0, places=5)
        self.assertAlmostEqual(xs[1], 3.0, places=5)
        self.assertAlmostEqual(ys[0], 0.0, places=5)
        self.assertAlmostEqual(ys[1], 5.0, places=5)

    def test_perpendicular_single_undo(self) -> None:
        """Composite construction (point + segment) should undo in one step."""
        self.canvas.create_point(3, 5, name="P")
        self.canvas.create_segment(0, 0, 6, 0, name="AB")
        pts_before = self._point_count()
        segs_before = self._segment_count()
        self.canvas.create_perpendicular_from_point("P", "AB")
        # Should have added 1 point (foot) and 1 segment
        self.assertEqual(self._point_count(), pts_before + 1)
        self.assertEqual(self._segment_count(), segs_before + 1)
        # Single undo should remove both
        self.canvas.undo()
        self.assertEqual(self._point_count(), pts_before)
        self.assertEqual(self._segment_count(), segs_before)

    def test_perpendicular_nonexistent_point_raises(self) -> None:
        """Nonexistent point name raises ValueError."""
        self.canvas.create_segment(0, 0, 6, 0, name="AB")
        with self.assertRaises(ValueError):
            self.canvas.create_perpendicular_from_point("Z", "AB")

    def test_perpendicular_nonexistent_segment_raises(self) -> None:
        """Nonexistent segment name raises ValueError."""
        self.canvas.create_point(3, 5, name="P")
        with self.assertRaises(ValueError):
            self.canvas.create_perpendicular_from_point("P", "nonexistent")


class TestConstructAngleBisector(TestConstructionManager):
    """Tests for angle bisector construction."""

    def test_bisector_of_right_angle(self) -> None:
        """Bisector of 90-degree angle at origin should point at 45 degrees."""
        self.canvas.create_point(0, 0, name="V")
        self.canvas.create_point(4, 0, name="A")
        self.canvas.create_point(0, 4, name="B")
        bisector = self.canvas.create_angle_bisector("V", "A", "B")
        # Direction from vertex should be (1/sqrt2, 1/sqrt2)
        dx = bisector.point2.x - bisector.point1.x
        dy = bisector.point2.y - bisector.point1.y
        length = math.sqrt(dx**2 + dy**2)
        if length > 1e-10:
            angle = math.atan2(dy, dx)
            self.assertAlmostEqual(angle, math.pi / 4, places=3)

    def test_bisector_of_60_degree_angle(self) -> None:
        """Bisector of 60-degree angle should point at 30 degrees."""
        self.canvas.create_point(0, 0, name="V")
        self.canvas.create_point(4, 0, name="A")
        self.canvas.create_point(2, 2 * math.sqrt(3), name="B")  # 60 degrees
        bisector = self.canvas.create_angle_bisector("V", "A", "B")
        dx = bisector.point2.x - bisector.point1.x
        dy = bisector.point2.y - bisector.point1.y
        angle = math.atan2(dy, dx)
        self.assertAlmostEqual(angle, math.pi / 6, places=2)

    def test_bisector_starts_at_vertex(self) -> None:
        """Bisector segment should start at the vertex."""
        self.canvas.create_point(1, 1, name="V")
        self.canvas.create_point(5, 1, name="A")
        self.canvas.create_point(1, 5, name="B")
        bisector = self.canvas.create_angle_bisector("V", "A", "B")
        self.assertAlmostEqual(bisector.point1.x, 1.0, places=5)
        self.assertAlmostEqual(bisector.point1.y, 1.0, places=5)

    def test_bisector_custom_length(self) -> None:
        """Bisector should have the specified length."""
        self.canvas.create_point(0, 0, name="V")
        self.canvas.create_point(4, 0, name="A")
        self.canvas.create_point(0, 4, name="B")
        bisector = self.canvas.create_angle_bisector("V", "A", "B", length=8.0)
        dx = bisector.point2.x - bisector.point1.x
        dy = bisector.point2.y - bisector.point1.y
        actual_length = math.sqrt(dx**2 + dy**2)
        self.assertAlmostEqual(actual_length, 8.0, places=3)

    def test_bisector_collinear_raises(self) -> None:
        """Bisector of 180-degree angle (collinear arms) should raise ValueError."""
        self.canvas.create_point(0, 0, name="V")
        self.canvas.create_point(4, 0, name="A")
        self.canvas.create_point(-4, 0, name="B")
        with self.assertRaises(ValueError):
            self.canvas.create_angle_bisector("V", "A", "B")

    def test_bisector_zero_length_arm_raises(self) -> None:
        """Bisector with coincident vertex and arm point should raise."""
        self.canvas.create_point(0, 0, name="V")
        self.canvas.create_point(0, 0, name="A")
        self.canvas.create_point(0, 4, name="B")
        with self.assertRaises(ValueError):
            self.canvas.create_angle_bisector("V", "A", "B")

    def test_bisector_nonexistent_point_raises(self) -> None:
        """Bisector with nonexistent point raises ValueError."""
        self.canvas.create_point(0, 0, name="V")
        self.canvas.create_point(4, 0, name="A")
        with self.assertRaises(ValueError):
            self.canvas.create_angle_bisector("V", "A", "Z")

    def test_bisector_no_args_raises(self) -> None:
        """Bisector with neither points nor angle raises ValueError."""
        with self.assertRaises(ValueError):
            self.canvas.create_angle_bisector()

    def test_bisector_undo(self) -> None:
        """Undo should remove the bisector segment."""
        self.canvas.create_point(0, 0, name="V")
        self.canvas.create_point(4, 0, name="A")
        self.canvas.create_point(0, 4, name="B")
        count_before = self._segment_count()
        self.canvas.create_angle_bisector("V", "A", "B")
        self.assertEqual(self._segment_count(), count_before + 1)
        self.canvas.undo()
        self.assertEqual(self._segment_count(), count_before)

    def test_bisector_reflex_angle_by_name(self) -> None:
        """Bisector of a reflex angle should point into the reflex arc."""
        # Create a 90-degree angle at origin: arms along +x and +y
        self.canvas.create_point(0, 0, name="V")
        self.canvas.create_point(4, 0, name="A")
        self.canvas.create_point(0, 4, name="B")
        self.canvas.create_segment(0, 0, 4, 0)
        self.canvas.create_segment(0, 0, 0, 4)
        # Create the reflex (270-degree) angle
        reflex = self.canvas.create_angle(0, 0, 4, 0, 0, 4, is_reflex=True)
        bisector = self.canvas.create_angle_bisector(angle_name=reflex.name)
        # The reflex bisector should point into the third quadrant (negative x, negative y)
        dx = bisector.point2.x - bisector.point1.x
        dy = bisector.point2.y - bisector.point1.y
        # For a 90-degree angle along +x and +y, the internal bisector is at +45 degrees.
        # The reflex bisector should be at +45+180 = 225 degrees (third quadrant).
        self.assertLess(dx, 0, "Reflex bisector dx should be negative")
        self.assertLess(dy, 0, "Reflex bisector dy should be negative")


class TestConstructParallelLine(TestConstructionManager):
    """Tests for parallel line construction."""

    def test_parallel_to_horizontal(self) -> None:
        """Parallel to horizontal segment through (0,5) should be horizontal at y=5."""
        self.canvas.create_segment(0, 0, 4, 0, name="AB")
        self.canvas.create_point(0, 5, name="P")
        parallel = self.canvas.create_parallel_line("AB", "P")
        # Both endpoints should have y=5
        self.assertAlmostEqual(parallel.point1.y, 5.0, places=5)
        self.assertAlmostEqual(parallel.point2.y, 5.0, places=5)

    def test_parallel_to_vertical(self) -> None:
        """Parallel to vertical segment through (5,0) should be vertical at x=5."""
        self.canvas.create_segment(0, 0, 0, 4, name="AB")
        self.canvas.create_point(5, 0, name="P")
        parallel = self.canvas.create_parallel_line("AB", "P")
        # Both endpoints should have x=5
        self.assertAlmostEqual(parallel.point1.x, 5.0, places=5)
        self.assertAlmostEqual(parallel.point2.x, 5.0, places=5)

    def test_parallel_same_slope(self) -> None:
        """Parallel line should have the same slope as the original."""
        self.canvas.create_segment(0, 0, 4, 2, name="AB")
        self.canvas.create_point(0, 5, name="P")
        parallel = self.canvas.create_parallel_line("AB", "P")
        # Original slope = 2/4 = 0.5
        orig_slope = 2.0 / 4.0
        par_dx = parallel.point2.x - parallel.point1.x
        par_dy = parallel.point2.y - parallel.point1.y
        if abs(par_dx) > 1e-10:
            par_slope = par_dy / par_dx
            self.assertAlmostEqual(par_slope, orig_slope, places=3)

    def test_parallel_centered_on_point(self) -> None:
        """Parallel line should be centered on the specified point."""
        self.canvas.create_segment(0, 0, 4, 0, name="AB")
        self.canvas.create_point(2, 3, name="P")
        parallel = self.canvas.create_parallel_line("AB", "P")
        mid_x = (parallel.point1.x + parallel.point2.x) / 2
        mid_y = (parallel.point1.y + parallel.point2.y) / 2
        self.assertAlmostEqual(mid_x, 2.0, places=5)
        self.assertAlmostEqual(mid_y, 3.0, places=5)

    def test_parallel_custom_length(self) -> None:
        """Parallel line should have the specified length."""
        self.canvas.create_segment(0, 0, 4, 0, name="AB")
        self.canvas.create_point(0, 5, name="P")
        parallel = self.canvas.create_parallel_line("AB", "P", length=12.0)
        dx = parallel.point2.x - parallel.point1.x
        dy = parallel.point2.y - parallel.point1.y
        actual_length = math.sqrt(dx**2 + dy**2)
        self.assertAlmostEqual(actual_length, 12.0, places=3)

    def test_parallel_nonexistent_segment_raises(self) -> None:
        """Parallel to nonexistent segment raises ValueError."""
        self.canvas.create_point(0, 5, name="P")
        with self.assertRaises(ValueError):
            self.canvas.create_parallel_line("nonexistent", "P")

    def test_parallel_nonexistent_point_raises(self) -> None:
        """Parallel through nonexistent point raises ValueError."""
        self.canvas.create_segment(0, 0, 4, 0, name="AB")
        with self.assertRaises(ValueError):
            self.canvas.create_parallel_line("AB", "Z")

    def test_parallel_undo(self) -> None:
        """Undo should remove the parallel segment."""
        self.canvas.create_segment(0, 0, 4, 0, name="AB")
        self.canvas.create_point(0, 5, name="P")
        count_before = self._segment_count()
        self.canvas.create_parallel_line("AB", "P")
        self.assertEqual(self._segment_count(), count_before + 1)
        self.canvas.undo()
        self.assertEqual(self._segment_count(), count_before)


class TestMathUtilsConstructionFunctions(TestConstructionManager):
    """Tests for the low-level math utility functions used by constructions."""

    def test_perpendicular_foot_point_on_line(self) -> None:
        """When point is already on the line, foot should be the point itself."""
        fx, fy = MathUtils.perpendicular_foot(2, 0, 0, 0, 4, 0)
        self.assertAlmostEqual(fx, 2.0, places=5)
        self.assertAlmostEqual(fy, 0.0, places=5)

    def test_perpendicular_foot_degenerate_raises(self) -> None:
        """Degenerate segment (same endpoints) should raise ValueError."""
        with self.assertRaises(ValueError):
            MathUtils.perpendicular_foot(1, 1, 3, 3, 3, 3)

    def test_angle_bisector_direction_right_angle(self) -> None:
        """Bisector of right angle should point at 45 degrees."""
        dx, dy = MathUtils.angle_bisector_direction(0, 0, 1, 0, 0, 1)
        angle = math.atan2(dy, dx)
        self.assertAlmostEqual(angle, math.pi / 4, places=5)
        # Should be unit vector
        length = math.sqrt(dx**2 + dy**2)
        self.assertAlmostEqual(length, 1.0, places=5)

    def test_angle_bisector_direction_collinear_raises(self) -> None:
        """Collinear arms (180 degrees) should raise ValueError."""
        with self.assertRaises(ValueError):
            MathUtils.angle_bisector_direction(0, 0, 1, 0, -1, 0)

    def test_angle_bisector_direction_zero_arm_raises(self) -> None:
        """Zero-length arm should raise ValueError."""
        with self.assertRaises(ValueError):
            MathUtils.angle_bisector_direction(0, 0, 0, 0, 1, 0)

    def test_circumcenter_right_triangle(self) -> None:
        """Circumcenter of right triangle at origin should be midpoint of hypotenuse."""
        cx, cy, r = MathUtils.circumcenter(0, 0, 4, 0, 0, 3)
        self.assertAlmostEqual(cx, 2.0, places=5)
        self.assertAlmostEqual(cy, 1.5, places=5)
        self.assertAlmostEqual(r, 2.5, places=5)

    def test_circumcenter_equilateral(self) -> None:
        """Circumcenter of equilateral triangle should be at centroid."""
        s = 2.0
        x1, y1 = 0.0, 0.0
        x2, y2 = s, 0.0
        x3, y3 = s / 2, s * math.sqrt(3) / 2
        cx, cy, r = MathUtils.circumcenter(x1, y1, x2, y2, x3, y3)
        # Centroid
        self.assertAlmostEqual(cx, s / 2, places=5)
        self.assertAlmostEqual(cy, s * math.sqrt(3) / 6, places=4)
        # Circumradius = s / sqrt(3)
        self.assertAlmostEqual(r, s / math.sqrt(3), places=4)

    def test_circumcenter_collinear_raises(self) -> None:
        """Collinear points should raise ValueError."""
        with self.assertRaises(ValueError):
            MathUtils.circumcenter(0, 0, 1, 0, 2, 0)

    def test_incenter_equilateral(self) -> None:
        """Incircle of equilateral triangle: inradius = s*sqrt(3)/6."""
        s = 6.0
        x1, y1 = 0.0, 0.0
        x2, y2 = s, 0.0
        x3, y3 = s / 2, s * math.sqrt(3) / 2
        cx, cy, r = MathUtils.incenter_and_inradius(x1, y1, x2, y2, x3, y3)
        expected_r = s * math.sqrt(3) / 6
        self.assertAlmostEqual(r, expected_r, places=4)
        # Incenter at centroid for equilateral
        self.assertAlmostEqual(cx, s / 2, places=4)
        self.assertAlmostEqual(cy, s * math.sqrt(3) / 6, places=4)

    def test_incenter_degenerate_raises(self) -> None:
        """Degenerate triangle (collinear) should raise ValueError."""
        with self.assertRaises(ValueError):
            MathUtils.incenter_and_inradius(0, 0, 1, 0, 2, 0)


class TestConstructCircumcircle(TestConstructionManager):
    """Tests for circumcircle construction."""

    def _create_triangle(self, name_suffix: str = "") -> None:
        """Create a right triangle at origin: (0,0), (4,0), (0,3)."""
        self.canvas.create_point(0, 0, name=f"A{name_suffix}")
        self.canvas.create_point(4, 0, name=f"B{name_suffix}")
        self.canvas.create_point(0, 3, name=f"C{name_suffix}")
        self.canvas.create_segment(0, 0, 4, 0)
        self.canvas.create_segment(4, 0, 0, 3)
        self.canvas.create_segment(0, 3, 0, 0)

    def _circle_count(self) -> int:
        return len(self.canvas.get_drawables_by_class_name("Circle"))

    def test_circumcircle_by_triangle_name(self) -> None:
        """Circumcircle of right triangle has center at midpoint of hypotenuse."""
        self._create_triangle()
        triangles = self.canvas.get_drawables_by_class_name("Triangle")
        self.assertTrue(len(triangles) > 0)
        tri_name = triangles[0].name
        circle = self.canvas.create_circumcircle(triangle_name=tri_name)
        self.assertAlmostEqual(circle.radius, 2.5, places=3)

    def test_circumcircle_by_three_points(self) -> None:
        """Circumcircle by 3 point names should produce correct radius."""
        self.canvas.create_point(0, 0, name="P")
        self.canvas.create_point(4, 0, name="Q")
        self.canvas.create_point(0, 3, name="R")
        circle = self.canvas.create_circumcircle(p1_name="P", p2_name="Q", p3_name="R")
        self.assertAlmostEqual(circle.radius, 2.5, places=3)

    def test_circumcircle_collinear_raises(self) -> None:
        """Collinear points should raise ValueError."""
        self.canvas.create_point(0, 0, name="P")
        self.canvas.create_point(1, 0, name="Q")
        self.canvas.create_point(2, 0, name="R")
        with self.assertRaises(ValueError):
            self.canvas.create_circumcircle(p1_name="P", p2_name="Q", p3_name="R")

    def test_circumcircle_no_args_raises(self) -> None:
        """No arguments should raise ValueError."""
        with self.assertRaises(ValueError):
            self.canvas.create_circumcircle()

    def test_circumcircle_undo(self) -> None:
        """Undo should remove the circumcircle."""
        self.canvas.create_point(0, 0, name="P")
        self.canvas.create_point(4, 0, name="Q")
        self.canvas.create_point(0, 3, name="R")
        count_before = self._circle_count()
        self.canvas.create_circumcircle(p1_name="P", p2_name="Q", p3_name="R")
        self.assertEqual(self._circle_count(), count_before + 1)
        self.canvas.undo()
        self.assertEqual(self._circle_count(), count_before)


class TestConstructIncircle(TestConstructionManager):
    """Tests for incircle construction."""

    def _create_triangle(self) -> str:
        """Create a right triangle and return its name."""
        self.canvas.create_point(0, 0, name="A")
        self.canvas.create_point(4, 0, name="B")
        self.canvas.create_point(0, 3, name="C")
        self.canvas.create_segment(0, 0, 4, 0)
        self.canvas.create_segment(4, 0, 0, 3)
        self.canvas.create_segment(0, 3, 0, 0)
        triangles = self.canvas.get_drawables_by_class_name("Triangle")
        self.assertTrue(len(triangles) > 0)
        return triangles[0].name

    def _circle_count(self) -> int:
        return len(self.canvas.get_drawables_by_class_name("Circle"))

    def test_incircle_right_triangle(self) -> None:
        """Incircle of 3-4-5 right triangle: inradius = (3+4-5)/2 = 1."""
        tri_name = self._create_triangle()
        circle = self.canvas.create_incircle(tri_name)
        self.assertAlmostEqual(circle.radius, 1.0, places=3)

    def test_incircle_nonexistent_raises(self) -> None:
        """Nonexistent triangle name should raise ValueError."""
        with self.assertRaises(ValueError):
            self.canvas.create_incircle("nonexistent")

    def test_incircle_undo(self) -> None:
        """Undo should remove the incircle."""
        tri_name = self._create_triangle()
        count_before = self._circle_count()
        self.canvas.create_incircle(tri_name)
        self.assertEqual(self._circle_count(), count_before + 1)
        self.canvas.undo()
        self.assertEqual(self._circle_count(), count_before)
