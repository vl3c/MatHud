import unittest
from coordinate_mapper import CoordinateMapper
from rendering.functions_area_renderable import FunctionsBoundedAreaRenderable
from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea
from drawables.function import Function


class TestFunctionBoundedColoredAreaIntegration(unittest.TestCase):
    def setUp(self) -> None:
        # 500x500 mapper with origin at center by default
        self.mapper = CoordinateMapper(500, 500)

    def _build_area(self, f1_str, f2_str, left, right):
        f1 = Function(f1_str, name='f1', left_bound=left, right_bound=right)
        f2 = Function(f2_str, name='f2', left_bound=left, right_bound=right)
        area = FunctionsBoundedColoredArea(f1, f2, left_bound=left, right_bound=right, num_sample_points=50)
        renderable = FunctionsBoundedAreaRenderable(area, self.mapper)
        return area, renderable

    def test_area_respects_function_bounds_simple(self) -> None:
        left, right = -300, 300
        _, renderable = self._build_area('50 * sin(x / 50)', '100 * sin(x / 30)', left, right)
        closed = renderable.build_screen_area()
        self.assertIsNotNone(closed)
        self.assertGreater(len(closed.forward_points), 0)
        self.assertGreater(len(closed.reverse_points), 0)
        # Check that first and last x of forward equals first and last x (in reverse) of reverse
        fxs = [p[0] for p in closed.forward_points]
        rxs = [p[0] for p in closed.reverse_points]
        self.assertAlmostEqual(fxs[0], rxs[-1], places=6)
        self.assertAlmostEqual(fxs[-1], rxs[0], places=6)
        # Additionally validate y matches function values at the matched x
        # Left end
        left_screen_x = fxs[0]
        left_math_x, _ = self.mapper.screen_to_math(left_screen_x, 0)
        import math
        y1_left = 50 * math.sin(left_math_x / 50.0)
        y2_left = 100 * math.sin(left_math_x / 30.0)
        _, y1s = self.mapper.math_to_screen(left_math_x, y1_left)
        _, y2s = self.mapper.math_to_screen(left_math_x, y2_left)
        self.assertAlmostEqual(closed.forward_points[0][1], y1s, places=5)
        self.assertAlmostEqual(closed.reverse_points[-1][1], y2s, places=5)
        # Right end
        right_screen_x = fxs[-1]
        right_math_x, _ = self.mapper.screen_to_math(right_screen_x, 0)
        y1_right = 50 * math.sin(right_math_x / 50.0)
        y2_right = 100 * math.sin(right_math_x / 30.0)
        _, y1rs = self.mapper.math_to_screen(right_math_x, y1_right)
        _, y2rs = self.mapper.math_to_screen(right_math_x, y2_right)
        self.assertAlmostEqual(closed.forward_points[-1][1], y1rs, places=5)
        self.assertAlmostEqual(closed.reverse_points[0][1], y2rs, places=5)

    def test_area_respects_function_bounds_with_asymptotes(self) -> None:
        # f3 with tan introduces vertical asymptotes; ensure pairing still aligns ends
        left, right = -300, 300
        _, renderable = self._build_area('50 * sin(x / 50)', '100 * sin(x / 50) + 50 * tan(x / 100)', left, right)
        closed = renderable.build_screen_area()
        self.assertIsNotNone(closed)
        self.assertGreater(len(closed.forward_points), 0)
        self.assertGreater(len(closed.reverse_points), 0)
        fxs = [p[0] for p in closed.forward_points]
        rxs = [p[0] for p in closed.reverse_points]
        self.assertAlmostEqual(fxs[0], rxs[-1], places=6)
        self.assertAlmostEqual(fxs[-1], rxs[0], places=6)
        # Validate y against function evaluations at ends (avoid asymptote issues by using screen x ends)
        left_screen_x = fxs[0]
        left_math_x, _ = self.mapper.screen_to_math(left_screen_x, 0)
        import math
        y1_left = 50 * math.sin(left_math_x / 50.0)
        y2_left = 100 * math.sin(left_math_x / 50.0) + 50 * math.tan(left_math_x / 100.0)
        # If tangent is near asymptote, skip assertion for y2 at that end
        if abs(math.cos(left_math_x / 100.0)) > 1e-3:
            _, y2s = self.mapper.math_to_screen(left_math_x, y2_left)
            self.assertAlmostEqual(closed.reverse_points[-1][1], y2s, places=5)
        _, y1s = self.mapper.math_to_screen(left_math_x, y1_left)
        self.assertAlmostEqual(closed.forward_points[0][1], y1s, places=5)


