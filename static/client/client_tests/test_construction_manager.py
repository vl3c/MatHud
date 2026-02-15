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
