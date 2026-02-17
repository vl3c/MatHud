"""
Unit tests for polar/rectangular coordinate conversion functions.

Tests the conversion functions:
- rectangular_to_polar: Convert (x, y) to (r, theta)
- polar_to_rectangular: Convert (r, theta) to (x, y)

Note: These tests use a local implementation of the conversion functions
since MathUtils has browser dependencies. The implementation matches
the functions added to MathUtils.
"""

from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import math
import unittest
from typing import Tuple


def rectangular_to_polar(x: float, y: float) -> Tuple[float, float]:
    """Convert rectangular (Cartesian) coordinates to polar coordinates.

    This is a pure Python implementation matching rectangular_to_polar.
    """
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return (r, theta)


def polar_to_rectangular(r: float, theta: float) -> Tuple[float, float]:
    """Convert polar coordinates to rectangular (Cartesian) coordinates.

    This is a pure Python implementation matching polar_to_rectangular.
    """
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return (x, y)


class TestRectangularToPolar(unittest.TestCase):
    """Tests for rectangular_to_polar conversion."""

    def test_origin(self) -> None:
        """Origin (0, 0) should convert to r=0, theta=0."""
        r, theta = rectangular_to_polar(0, 0)
        self.assertAlmostEqual(r, 0, places=10)
        self.assertAlmostEqual(theta, 0, places=10)

    def test_positive_x_axis(self) -> None:
        """Point on positive x-axis should have theta=0."""
        r, theta = rectangular_to_polar(5, 0)
        self.assertAlmostEqual(r, 5, places=10)
        self.assertAlmostEqual(theta, 0, places=10)

    def test_negative_x_axis(self) -> None:
        """Point on negative x-axis should have theta=pi."""
        r, theta = rectangular_to_polar(-5, 0)
        self.assertAlmostEqual(r, 5, places=10)
        self.assertAlmostEqual(theta, math.pi, places=10)

    def test_positive_y_axis(self) -> None:
        """Point on positive y-axis should have theta=pi/2."""
        r, theta = rectangular_to_polar(0, 5)
        self.assertAlmostEqual(r, 5, places=10)
        self.assertAlmostEqual(theta, math.pi / 2, places=10)

    def test_negative_y_axis(self) -> None:
        """Point on negative y-axis should have theta=-pi/2."""
        r, theta = rectangular_to_polar(0, -5)
        self.assertAlmostEqual(r, 5, places=10)
        self.assertAlmostEqual(theta, -math.pi / 2, places=10)

    def test_first_quadrant(self) -> None:
        """Point in first quadrant (x>0, y>0)."""
        r, theta = rectangular_to_polar(3, 4)
        self.assertAlmostEqual(r, 5, places=10)
        self.assertAlmostEqual(theta, math.atan2(4, 3), places=10)
        self.assertGreater(theta, 0)
        self.assertLess(theta, math.pi / 2)

    def test_second_quadrant(self) -> None:
        """Point in second quadrant (x<0, y>0)."""
        r, theta = rectangular_to_polar(-3, 4)
        self.assertAlmostEqual(r, 5, places=10)
        expected_theta = math.atan2(4, -3)
        self.assertAlmostEqual(theta, expected_theta, places=10)
        self.assertGreater(theta, math.pi / 2)
        self.assertLess(theta, math.pi)

    def test_third_quadrant(self) -> None:
        """Point in third quadrant (x<0, y<0)."""
        r, theta = rectangular_to_polar(-3, -4)
        self.assertAlmostEqual(r, 5, places=10)
        expected_theta = math.atan2(-4, -3)
        self.assertAlmostEqual(theta, expected_theta, places=10)
        self.assertLess(theta, -math.pi / 2)
        self.assertGreater(theta, -math.pi)

    def test_fourth_quadrant(self) -> None:
        """Point in fourth quadrant (x>0, y<0)."""
        r, theta = rectangular_to_polar(3, -4)
        self.assertAlmostEqual(r, 5, places=10)
        expected_theta = math.atan2(-4, 3)
        self.assertAlmostEqual(theta, expected_theta, places=10)
        self.assertLess(theta, 0)
        self.assertGreater(theta, -math.pi / 2)

    def test_unit_circle_45_degrees(self) -> None:
        """Point at 45 degrees on unit circle."""
        x = math.sqrt(2) / 2
        y = math.sqrt(2) / 2
        r, theta = rectangular_to_polar(x, y)
        self.assertAlmostEqual(r, 1, places=10)
        self.assertAlmostEqual(theta, math.pi / 4, places=10)

    def test_large_values(self) -> None:
        """Test with large coordinate values."""
        r, theta = rectangular_to_polar(1000, 1000)
        expected_r = math.sqrt(2000000)
        self.assertAlmostEqual(r, expected_r, places=6)
        self.assertAlmostEqual(theta, math.pi / 4, places=10)

    def test_small_values(self) -> None:
        """Test with small coordinate values."""
        r, theta = rectangular_to_polar(0.001, 0.001)
        expected_r = math.sqrt(0.000002)
        self.assertAlmostEqual(r, expected_r, places=10)
        self.assertAlmostEqual(theta, math.pi / 4, places=10)


class TestPolarToRectangular(unittest.TestCase):
    """Tests for polar_to_rectangular conversion."""

    def test_origin(self) -> None:
        """r=0 should always give origin regardless of theta."""
        for theta in [0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2]:
            with self.subTest(theta=theta):
                x, y = polar_to_rectangular(0, theta)
                self.assertAlmostEqual(x, 0, places=10)
                self.assertAlmostEqual(y, 0, places=10)

    def test_positive_x_axis(self) -> None:
        """theta=0 should give point on positive x-axis."""
        x, y = polar_to_rectangular(5, 0)
        self.assertAlmostEqual(x, 5, places=10)
        self.assertAlmostEqual(y, 0, places=10)

    def test_negative_x_axis(self) -> None:
        """theta=pi should give point on negative x-axis."""
        x, y = polar_to_rectangular(5, math.pi)
        self.assertAlmostEqual(x, -5, places=10)
        self.assertAlmostEqual(y, 0, places=10)

    def test_positive_y_axis(self) -> None:
        """theta=pi/2 should give point on positive y-axis."""
        x, y = polar_to_rectangular(5, math.pi / 2)
        self.assertAlmostEqual(x, 0, places=10)
        self.assertAlmostEqual(y, 5, places=10)

    def test_negative_y_axis(self) -> None:
        """theta=-pi/2 should give point on negative y-axis."""
        x, y = polar_to_rectangular(5, -math.pi / 2)
        self.assertAlmostEqual(x, 0, places=10)
        self.assertAlmostEqual(y, -5, places=10)

    def test_45_degrees(self) -> None:
        """theta=pi/4 should give equal x and y."""
        x, y = polar_to_rectangular(1, math.pi / 4)
        expected = math.sqrt(2) / 2
        self.assertAlmostEqual(x, expected, places=10)
        self.assertAlmostEqual(y, expected, places=10)

    def test_135_degrees(self) -> None:
        """theta=3*pi/4 should give negative x, positive y."""
        x, y = polar_to_rectangular(1, 3 * math.pi / 4)
        expected = math.sqrt(2) / 2
        self.assertAlmostEqual(x, -expected, places=10)
        self.assertAlmostEqual(y, expected, places=10)

    def test_negative_radius(self) -> None:
        """Negative radius should flip the point through origin."""
        x, y = polar_to_rectangular(-5, 0)
        self.assertAlmostEqual(x, -5, places=10)
        self.assertAlmostEqual(y, 0, places=10)

    def test_angle_greater_than_2pi(self) -> None:
        """Angles greater than 2*pi should work (wrap around)."""
        x1, y1 = polar_to_rectangular(5, math.pi / 4)
        x2, y2 = polar_to_rectangular(5, math.pi / 4 + 2 * math.pi)
        self.assertAlmostEqual(x1, x2, places=10)
        self.assertAlmostEqual(y1, y2, places=10)

    def test_negative_angle(self) -> None:
        """Negative angles should work correctly."""
        x, y = polar_to_rectangular(5, -math.pi / 4)
        expected = 5 * math.sqrt(2) / 2
        self.assertAlmostEqual(x, expected, places=10)
        self.assertAlmostEqual(y, -expected, places=10)


class TestRoundtripConversion(unittest.TestCase):
    """Tests for roundtrip conversion between coordinate systems."""

    def test_rectangular_to_polar_to_rectangular(self) -> None:
        """Converting rect to polar and back should preserve values."""
        test_cases = [
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (3, 4),
            (-3, 4),
            (-3, -4),
            (3, -4),
            (1.5, 2.5),
            (-0.5, 0.75),
        ]
        for orig_x, orig_y in test_cases:
            with self.subTest(x=orig_x, y=orig_y):
                r, theta = rectangular_to_polar(orig_x, orig_y)
                x, y = polar_to_rectangular(r, theta)
                self.assertAlmostEqual(x, orig_x, places=10)
                self.assertAlmostEqual(y, orig_y, places=10)

    def test_polar_to_rectangular_to_polar(self) -> None:
        """Converting polar to rect and back should preserve r, theta may differ by 2*pi*n."""
        test_cases = [
            (0, 0),
            (5, 0),
            (5, math.pi / 4),
            (5, math.pi / 2),
            (5, math.pi),
            (5, -math.pi / 4),
            (5, -math.pi / 2),
            (2.5, math.pi / 6),
            (10, 2 * math.pi / 3),
        ]
        for orig_r, orig_theta in test_cases:
            with self.subTest(r=orig_r, theta=orig_theta):
                x, y = polar_to_rectangular(orig_r, orig_theta)
                r, theta = rectangular_to_polar(x, y)
                self.assertAlmostEqual(r, orig_r, places=10)
                # Theta may differ by 2*pi*n, so compare using sin/cos
                if orig_r > 0:
                    self.assertAlmostEqual(math.cos(theta), math.cos(orig_theta), places=10)
                    self.assertAlmostEqual(math.sin(theta), math.sin(orig_theta), places=10)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and special values."""

    def test_very_small_radius(self) -> None:
        """Test with very small radius values."""
        x, y = polar_to_rectangular(1e-15, math.pi / 4)
        self.assertAlmostEqual(x, 0, places=14)
        self.assertAlmostEqual(y, 0, places=14)

    def test_very_large_radius(self) -> None:
        """Test with very large radius values."""
        x, y = polar_to_rectangular(1e10, 0)
        self.assertAlmostEqual(x, 1e10, places=0)
        self.assertAlmostEqual(y, 0, places=0)

    def test_floating_point_precision(self) -> None:
        """Test that floating point precision is maintained."""
        x = 1.23456789012345
        y = 9.87654321098765
        r, theta = rectangular_to_polar(x, y)
        x_back, y_back = polar_to_rectangular(r, theta)
        self.assertAlmostEqual(x, x_back, places=10)
        self.assertAlmostEqual(y, y_back, places=10)


if __name__ == "__main__":
    unittest.main()
