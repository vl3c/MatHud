"""Tests for TangentManager - tangent and normal line creation."""

from __future__ import annotations

import math
import unittest
from typing import List, Optional

from canvas import Canvas
from drawables.segment import Segment
from utils.math_utils import MathUtils


class TestTangentManager(unittest.TestCase):
    """Test suite for TangentManager functionality."""

    def setUp(self) -> None:
        """Set up test canvas with draw disabled."""
        self.canvas = Canvas(500, 500, draw_enabled=False)

    def _get_segment_names(self) -> List[str]:
        """Get names of all segments on canvas."""
        return [d.name for d in self.canvas.get_drawables_by_class_name("Segment")]

    def _get_segment_by_name(self, name: str) -> Optional[Segment]:
        """Get a segment by name."""
        return self.canvas.drawable_manager.segment_manager.get_segment_by_name(name)


class TestTangentToFunction(TestTangentManager):
    """Tests for tangent lines to functions y=f(x)."""

    def test_tangent_to_parabola_at_origin(self) -> None:
        """Tangent to y=x^2 at x=0 should be horizontal (slope=0)."""
        self.canvas.draw_function("x^2", "f1")
        segment = self.canvas.create_tangent_line("f1", 0)

        # Tangent at x=0 has slope 0 (y'=2x, y'(0)=0)
        # Point is (0, 0)
        # Segment should be horizontal with y=0 for both endpoints
        self.assertAlmostEqual(segment.point1.y, 0.0, places=5)
        self.assertAlmostEqual(segment.point2.y, 0.0, places=5)

    def test_tangent_to_parabola_at_x_equals_1(self) -> None:
        """Tangent to y=x^2 at x=1 should have slope 2."""
        self.canvas.draw_function("x^2", "f1")
        segment = self.canvas.create_tangent_line("f1", 1, length=2.0)

        # Tangent at x=1 has slope 2 (y'=2x, y'(1)=2)
        # Point is (1, 1)
        # Calculate expected slope from segment endpoints
        dx = segment.point2.x - segment.point1.x
        dy = segment.point2.y - segment.point1.y
        if abs(dx) > 1e-10:
            actual_slope = dy / dx
            self.assertAlmostEqual(actual_slope, 2.0, places=3)

    def test_tangent_to_sine_at_origin(self) -> None:
        """Tangent to y=sin(x) at x=0 should have slope 1."""
        self.canvas.draw_function("sin(x)", "f1")
        segment = self.canvas.create_tangent_line("f1", 0)

        # y' = cos(x), y'(0) = 1
        # Point is (0, 0)
        dx = segment.point2.x - segment.point1.x
        dy = segment.point2.y - segment.point1.y
        if abs(dx) > 1e-10:
            actual_slope = dy / dx
            self.assertAlmostEqual(actual_slope, 1.0, places=3)

    def test_tangent_with_custom_length(self) -> None:
        """Test that custom length is respected."""
        self.canvas.draw_function("x", "f1")
        segment = self.canvas.create_tangent_line("f1", 0, length=10.0)

        # Calculate actual length
        dx = segment.point2.x - segment.point1.x
        dy = segment.point2.y - segment.point1.y
        actual_length = math.sqrt(dx**2 + dy**2)
        self.assertAlmostEqual(actual_length, 10.0, places=3)

    def test_tangent_with_custom_color(self) -> None:
        """Test that custom color is applied."""
        self.canvas.draw_function("x", "f1")
        segment = self.canvas.create_tangent_line("f1", 0, color="red")
        self.assertEqual(segment.color, "red")

    def test_tangent_inherits_curve_color(self) -> None:
        """Test that tangent inherits curve color by default."""
        self.canvas.draw_function("x", "f1", color="blue")
        segment = self.canvas.create_tangent_line("f1", 0)
        self.assertEqual(segment.color, "blue")

    def test_tangent_to_nonexistent_curve_raises(self) -> None:
        """Test that error is raised for nonexistent curve."""
        with self.assertRaises(ValueError):
            self.canvas.create_tangent_line("nonexistent", 0)


class TestNormalToFunction(TestTangentManager):
    """Tests for normal lines to functions y=f(x)."""

    def test_normal_to_parabola_at_origin(self) -> None:
        """Normal to y=x^2 at x=0 should be vertical."""
        self.canvas.draw_function("x^2", "f1")
        segment = self.canvas.create_normal_line("f1", 0)

        # Tangent at x=0 is horizontal (slope=0)
        # Normal should be vertical (both points have same x)
        self.assertAlmostEqual(segment.point1.x, segment.point2.x, places=5)

    def test_normal_to_parabola_at_x_equals_1(self) -> None:
        """Normal to y=x^2 at x=1 should have slope -0.5."""
        self.canvas.draw_function("x^2", "f1")
        segment = self.canvas.create_normal_line("f1", 1, length=2.0)

        # Tangent slope = 2, so normal slope = -1/2
        dx = segment.point2.x - segment.point1.x
        dy = segment.point2.y - segment.point1.y
        if abs(dx) > 1e-10:
            actual_slope = dy / dx
            self.assertAlmostEqual(actual_slope, -0.5, places=3)

    def test_normal_perpendicular_to_tangent(self) -> None:
        """Test that normal is perpendicular to tangent."""
        self.canvas.draw_function("x^3", "f1")

        tangent = self.canvas.create_tangent_line("f1", 1, length=4.0)
        normal = self.canvas.create_normal_line("f1", 1, length=4.0)

        # Calculate slopes
        t_dx = tangent.point2.x - tangent.point1.x
        t_dy = tangent.point2.y - tangent.point1.y
        n_dx = normal.point2.x - normal.point1.x
        n_dy = normal.point2.y - normal.point1.y

        # Dot product of perpendicular vectors should be ~0
        dot_product = t_dx * n_dx + t_dy * n_dy
        self.assertAlmostEqual(dot_product, 0.0, places=3)


class TestTangentToCircle(TestTangentManager):
    """Tests for tangent lines to circles."""

    def test_tangent_to_circle_at_angle_0(self) -> None:
        """Tangent to circle at angle 0 (rightmost point) should be vertical."""
        circle = self.canvas.create_circle(0, 0, 3)
        segment = self.canvas.create_tangent_line(circle.name, 0)

        # At angle 0, point is (3, 0), tangent is vertical
        self.assertAlmostEqual(segment.point1.x, segment.point2.x, places=5)
        self.assertAlmostEqual(segment.point1.x, 3.0, places=5)

    def test_tangent_to_circle_at_angle_pi_over_2(self) -> None:
        """Tangent to circle at angle pi/2 (top) should be horizontal."""
        circle = self.canvas.create_circle(0, 0, 3)
        segment = self.canvas.create_tangent_line(circle.name, math.pi / 2)

        # At angle pi/2, point is (0, 3), tangent is horizontal
        self.assertAlmostEqual(segment.point1.y, segment.point2.y, places=5)
        self.assertAlmostEqual(segment.point1.y, 3.0, places=5)

    def test_normal_to_circle_passes_through_center(self) -> None:
        """Normal to circle should pass through center."""
        cx, cy = 2.0, 3.0
        circle = self.canvas.create_circle(cx, cy, 5)
        segment = self.canvas.create_normal_line(circle.name, math.pi / 4, length=10.0)

        # Check that the line (extended) passes through center
        # Using the line equation: (y - y1) / (y2 - y1) = (x - x1) / (x2 - x1)
        x1, y1 = segment.point1.x, segment.point1.y
        x2, y2 = segment.point2.x, segment.point2.y

        # Calculate parameter t for center point
        if abs(x2 - x1) > 1e-10:
            t = (cx - x1) / (x2 - x1)
            expected_y = y1 + t * (y2 - y1)
            self.assertAlmostEqual(expected_y, cy, places=3)
        else:
            # Vertical line case
            self.assertAlmostEqual(x1, cx, places=3)


class TestTangentToEllipse(TestTangentManager):
    """Tests for tangent lines to ellipses."""

    def test_tangent_to_ellipse_at_angle_0(self) -> None:
        """Tangent to ellipse at angle 0 should be vertical."""
        ellipse = self.canvas.create_ellipse(0, 0, 4, 2, 0)
        segment = self.canvas.create_tangent_line(ellipse.name, 0)

        # At angle 0, point is (4, 0), tangent is vertical
        self.assertAlmostEqual(segment.point1.x, segment.point2.x, places=5)
        self.assertAlmostEqual(segment.point1.x, 4.0, places=5)

    def test_tangent_to_ellipse_at_angle_pi_over_2(self) -> None:
        """Tangent to ellipse at angle pi/2 should be horizontal."""
        ellipse = self.canvas.create_ellipse(0, 0, 4, 2, 0)
        segment = self.canvas.create_tangent_line(ellipse.name, math.pi / 2)

        # At angle pi/2, point is (0, 2), tangent is horizontal
        self.assertAlmostEqual(segment.point1.y, segment.point2.y, places=5)
        self.assertAlmostEqual(segment.point1.y, 2.0, places=5)


class TestTangentToParametricFunction(TestTangentManager):
    """Tests for tangent lines to parametric functions."""

    def test_tangent_to_parametric_circle_at_t_0(self) -> None:
        """Tangent to parametric circle at t=0 should be vertical."""
        self.canvas.draw_parametric_function("cos(t)", "sin(t)", "p1")
        segment = self.canvas.create_tangent_line("p1", 0)

        # At t=0: x=cos(0)=1, y=sin(0)=0
        # dx/dt=-sin(t), dy/dt=cos(t)
        # At t=0: dx/dt=0, dy/dt=1 -> vertical tangent
        self.assertAlmostEqual(segment.point1.x, segment.point2.x, places=5)
        self.assertAlmostEqual(segment.point1.x, 1.0, places=5)

    def test_tangent_to_parametric_circle_at_t_pi_over_2(self) -> None:
        """Tangent to parametric circle at t=pi/2 should be horizontal."""
        self.canvas.draw_parametric_function("cos(t)", "sin(t)", "p1")
        segment = self.canvas.create_tangent_line("p1", math.pi / 2)

        # At t=pi/2: x=0, y=1
        # dx/dt=-1, dy/dt=0 -> horizontal tangent
        self.assertAlmostEqual(segment.point1.y, segment.point2.y, places=5)
        self.assertAlmostEqual(segment.point1.y, 1.0, places=5)

    def test_tangent_to_linear_parametric(self) -> None:
        """Tangent to linear parametric curve t, 2*t should have slope 2."""
        self.canvas.draw_parametric_function("t", "2*t", "p1", t_min=0, t_max=10)
        segment = self.canvas.create_tangent_line("p1", 5)

        # dx/dt=1, dy/dt=2, slope=2
        dx = segment.point2.x - segment.point1.x
        dy = segment.point2.y - segment.point1.y
        if abs(dx) > 1e-10:
            actual_slope = dy / dx
            self.assertAlmostEqual(actual_slope, 2.0, places=3)


class TestMathUtilsTangentFunctions(unittest.TestCase):
    """Tests for MathUtils tangent/normal helper functions."""

    def test_numerical_derivative_at_polynomial(self) -> None:
        """Test numerical derivative of x^2 at x=3."""
        # y = x^2, y' = 2x, y'(3) = 6
        def func(x):
            return x**2
        deriv = MathUtils.numerical_derivative_at(func, 3.0)
        self.assertIsNotNone(deriv)
        self.assertAlmostEqual(deriv, 6.0, places=4)

    def test_numerical_derivative_at_trig(self) -> None:
        """Test numerical derivative of sin(x) at x=0."""
        # y = sin(x), y' = cos(x), y'(0) = 1
        deriv = MathUtils.numerical_derivative_at(math.sin, 0.0)
        self.assertIsNotNone(deriv)
        self.assertAlmostEqual(deriv, 1.0, places=4)

    def test_tangent_line_endpoints_horizontal(self) -> None:
        """Test endpoints for horizontal tangent (slope=0)."""
        point = (5.0, 3.0)
        length = 4.0
        p1, p2 = MathUtils.tangent_line_endpoints(0.0, point, length)

        # Horizontal line: same y, different x
        self.assertAlmostEqual(p1[1], 3.0, places=5)
        self.assertAlmostEqual(p2[1], 3.0, places=5)
        self.assertAlmostEqual(abs(p2[0] - p1[0]), 4.0, places=5)

    def test_tangent_line_endpoints_vertical(self) -> None:
        """Test endpoints for vertical tangent (slope=None)."""
        point = (5.0, 3.0)
        length = 4.0
        p1, p2 = MathUtils.tangent_line_endpoints(None, point, length)

        # Vertical line: same x, different y
        self.assertAlmostEqual(p1[0], 5.0, places=5)
        self.assertAlmostEqual(p2[0], 5.0, places=5)
        self.assertAlmostEqual(abs(p2[1] - p1[1]), 4.0, places=5)

    def test_tangent_line_endpoints_length(self) -> None:
        """Test that segment has correct length."""
        point = (0.0, 0.0)
        length = 6.0
        slope = 2.0
        p1, p2 = MathUtils.tangent_line_endpoints(slope, point, length)

        actual_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        self.assertAlmostEqual(actual_length, 6.0, places=5)

    def test_normal_slope_from_horizontal(self) -> None:
        """Normal to horizontal tangent should be vertical."""
        normal = MathUtils.normal_slope(0.0)
        self.assertIsNone(normal)

    def test_normal_slope_from_vertical(self) -> None:
        """Normal to vertical tangent should be horizontal."""
        normal = MathUtils.normal_slope(None)
        self.assertAlmostEqual(normal, 0.0, places=5)

    def test_normal_slope_perpendicular(self) -> None:
        """Normal slope should be -1/tangent_slope."""
        tangent_slope = 2.0
        normal = MathUtils.normal_slope(tangent_slope)
        self.assertAlmostEqual(normal, -0.5, places=5)

    def test_circle_tangent_at_angle_0(self) -> None:
        """Circle tangent at angle 0 should be vertical."""
        point, slope = MathUtils.circle_tangent_slope_at_angle(0, 0, 1, 0)
        self.assertAlmostEqual(point[0], 1.0, places=5)
        self.assertAlmostEqual(point[1], 0.0, places=5)
        self.assertIsNone(slope)  # Vertical

    def test_circle_tangent_at_angle_pi_over_2(self) -> None:
        """Circle tangent at angle pi/2 should be horizontal."""
        point, slope = MathUtils.circle_tangent_slope_at_angle(0, 0, 1, math.pi / 2)
        self.assertAlmostEqual(point[0], 0.0, places=5)
        self.assertAlmostEqual(point[1], 1.0, places=5)
        self.assertAlmostEqual(slope, 0.0, places=5)  # Horizontal

    def test_ellipse_tangent_at_angle_0(self) -> None:
        """Ellipse tangent at angle 0 should be vertical."""
        point, slope = MathUtils.ellipse_tangent_slope_at_angle(0, 0, 2, 1, 0)
        self.assertAlmostEqual(point[0], 2.0, places=5)
        self.assertAlmostEqual(point[1], 0.0, places=5)
        self.assertIsNone(slope)  # Vertical


class TestUndoRedo(TestTangentManager):
    """Tests for undo/redo support."""

    def test_tangent_line_undo(self) -> None:
        """Test that creating a tangent line can be undone."""
        self.canvas.draw_function("x^2", "f1")
        initial_count = len(self._get_segment_names())

        self.canvas.create_tangent_line("f1", 1)
        self.assertEqual(len(self._get_segment_names()), initial_count + 1)

        self.canvas.undo()
        self.assertEqual(len(self._get_segment_names()), initial_count)

    def test_normal_line_undo(self) -> None:
        """Test that creating a normal line can be undone."""
        self.canvas.draw_function("x^2", "f1")
        initial_count = len(self._get_segment_names())

        self.canvas.create_normal_line("f1", 1)
        self.assertEqual(len(self._get_segment_names()), initial_count + 1)

        self.canvas.undo()
        self.assertEqual(len(self._get_segment_names()), initial_count)

    def test_tangent_line_redo(self) -> None:
        """Test that undo can be redone."""
        self.canvas.draw_function("x^2", "f1")
        initial_count = len(self._get_segment_names())

        self.canvas.create_tangent_line("f1", 1)
        self.canvas.undo()
        self.canvas.redo()

        self.assertEqual(len(self._get_segment_names()), initial_count + 1)
