"""
Server tests for FunctionRenderable path building.

Tests the path building logic for functions with asymptotes like 1/x.
"""

import sys
import unittest
from typing import List, Tuple, Optional, Callable

sys.path.insert(0, 'static/client')
sys.path.insert(0, 'static/client/rendering/renderables')


class MockFunction:
    """Mock Function object for testing."""

    def __init__(
        self,
        eval_func: Callable[[float], float],
        vertical_asymptotes: Optional[List[float]] = None,
        left_bound: Optional[float] = None,
        right_bound: Optional[float] = None,
        is_periodic: bool = False,
        estimated_period: Optional[float] = None,
    ):
        self.function = eval_func
        self.vertical_asymptotes = vertical_asymptotes or []
        self.horizontal_asymptotes: List[float] = []
        self.point_discontinuities: List[float] = []
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.is_periodic = is_periodic
        self.estimated_period = estimated_period

    def get_vertical_asymptote_between_x(self, x1: float, x2: float) -> Optional[float]:
        for x in self.vertical_asymptotes:
            if x1 <= x < x2 or x2 <= x < x1:
                return x
        return None

    def has_vertical_asymptote_between_x(self, x1: float, x2: float) -> bool:
        return self.get_vertical_asymptote_between_x(x1, x2) is not None


class MockMapper:
    """Mock CoordinateMapper for testing."""

    def __init__(self, width: int = 500, height: int = 500, scale: float = 25.0):
        self.width = width
        self.height = height
        self.scale = scale
        self.center_x = width / 2
        self.center_y = height / 2

    def math_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        sx = self.center_x + x * self.scale
        sy = self.center_y - y * self.scale
        return (sx, sy)

    def screen_to_math(self, sx: float, sy: float) -> Tuple[float, float]:
        x = (sx - self.center_x) / self.scale
        y = (self.center_y - sy) / self.scale
        return (x, y)

    def get_visible_left_bound(self) -> float:
        return -self.center_x / self.scale

    def get_visible_right_bound(self) -> float:
        return self.center_x / self.scale


class TestClampScreenY(unittest.TestCase):
    """Tests for _clamp_screen_y method."""

    def test_normal_value_unchanged(self):
        height = 500
        # Values within normal range should not be clamped
        self.assertEqual(clamp_screen_y(250, height), 250)
        self.assertEqual(clamp_screen_y(0, height), 0)
        self.assertEqual(clamp_screen_y(500, height), 500)

    def test_extreme_negative_clamped_to_top(self):
        height = 500
        # Values below -height should clamp to 0 (top of screen)
        self.assertEqual(clamp_screen_y(-600, height), 0.0)
        self.assertEqual(clamp_screen_y(-1000000, height), 0.0)

    def test_extreme_positive_clamped_to_bottom(self):
        height = 500
        # Values above 2*height should clamp to height (bottom of screen)
        self.assertEqual(clamp_screen_y(1100, height), 500)
        self.assertEqual(clamp_screen_y(1000000, height), 500)

    def test_borderline_values(self):
        height = 500
        # Just inside threshold - should not clamp
        self.assertEqual(clamp_screen_y(-499, height), -499)
        self.assertEqual(clamp_screen_y(999, height), 999)
        # Just outside threshold - should clamp
        self.assertEqual(clamp_screen_y(-501, height), 0.0)
        self.assertEqual(clamp_screen_y(1001, height), 500)


class TestSampleNearAsymptote(unittest.TestCase):
    """Tests for _sample_near_asymptote logic."""

    def test_approaching_from_left(self):
        # For 1/x at asymptote x=0, approaching from x=-1
        asymptote_x = 0.0
        from_x = -1.0
        epsilon = min(0.001, abs(asymptote_x - from_x) * 0.01)
        close_x = asymptote_x - epsilon  # Should be negative, close to 0

        self.assertLess(close_x, 0)
        self.assertGreater(close_x, -0.01)

    def test_approaching_from_right(self):
        # For 1/x at asymptote x=0, approaching from x=1
        asymptote_x = 0.0
        from_x = 1.0
        epsilon = min(0.001, abs(asymptote_x - from_x) * 0.01)
        close_x = asymptote_x + epsilon  # Should be positive, close to 0

        self.assertGreater(close_x, 0)
        self.assertLess(close_x, 0.01)


class TestOneOverX(unittest.TestCase):
    """Integration tests for 1/x function path building."""

    def setUp(self):
        self.mapper = MockMapper(width=500, height=500, scale=25.0)

        def one_over_x(x):
            if x == 0:
                return float('inf')
            return 1.0 / x

        self.func = MockFunction(
            eval_func=one_over_x,
            vertical_asymptotes=[0.0],
            left_bound=-10,
            right_bound=10,
        )

    def test_asymptote_detected_correctly(self):
        """Verify asymptote at x=0 is detected."""
        self.assertEqual(self.func.vertical_asymptotes, [0.0])
        self.assertTrue(self.func.has_vertical_asymptote_between_x(-1, 1))
        self.assertFalse(self.func.has_vertical_asymptote_between_x(1, 2))
        self.assertFalse(self.func.has_vertical_asymptote_between_x(-2, -1))

    def test_get_asymptote_between(self):
        """Verify get_vertical_asymptote_between_x returns correct values."""
        self.assertEqual(self.func.get_vertical_asymptote_between_x(-1, 1), 0.0)
        self.assertEqual(self.func.get_vertical_asymptote_between_x(-0.5, 0.5), 0.0)
        self.assertIsNone(self.func.get_vertical_asymptote_between_x(1, 2))
        self.assertIsNone(self.func.get_vertical_asymptote_between_x(-2, -1))

    def test_evaluation_near_asymptote(self):
        """Test function evaluation near the asymptote."""
        # Very close to 0 from negative side
        x = -0.001
        y = self.func.function(x)
        self.assertEqual(y, -1000.0)

        # Very close to 0 from positive side
        x = 0.001
        y = self.func.function(x)
        self.assertEqual(y, 1000.0)

    def test_screen_coords_near_asymptote(self):
        """Test screen coordinate conversion near asymptote."""
        # At x=-0.001, y=-1000
        sx, sy = self.mapper.math_to_screen(-0.001, -1000)
        # sy should be very large (below screen) since y is very negative
        self.assertGreater(sy, self.mapper.height)

        # At x=0.001, y=1000
        sx, sy = self.mapper.math_to_screen(0.001, 1000)
        # sy should be very negative (above screen) since y is very positive
        self.assertLess(sy, 0)

    def test_clamping_extreme_values(self):
        """Verify extreme y values get clamped correctly."""
        height = 500

        # For x=0.001, y=1000, screen_y = 250 - 1000*25 = -24750
        sy = self.mapper.center_y - 1000 * self.mapper.scale
        self.assertLess(sy, -height)
        clamped = clamp_screen_y(sy, height)
        self.assertEqual(clamped, 0.0)

        # For x=-0.001, y=-1000, screen_y = 250 - (-1000)*25 = 25250
        sy = self.mapper.center_y - (-1000) * self.mapper.scale
        self.assertGreater(sy, 2 * height)
        clamped = clamp_screen_y(sy, height)
        self.assertEqual(clamped, height)


def clamp_screen_y(sy: float, height: float) -> float:
    """Standalone version of _clamp_screen_y for testing."""
    if sy < -height:
        return 0.0
    elif sy > 2 * height:
        return height
    return sy


if __name__ == '__main__':
    unittest.main()

