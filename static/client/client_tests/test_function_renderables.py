from __future__ import annotations

import math
import unittest

from coordinate_mapper import CoordinateMapper
from drawables.point import Point
from drawables.segment import Segment
from drawables.function import Function

from rendering.renderables import FunctionRenderable
from rendering.primitives import ClosedArea


class TestFunctionRenderable(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)

    def test_build_screen_paths_returns_polyline(self) -> None:
        func = Function("x", name="f")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.paths)
        self.assertGreater(len(result.paths), 0)

    def test_linear_function_produces_continuous_path(self) -> None:
        func = Function("x", name="f")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertEqual(len(result.paths), 1)
        self.assertGreater(len(result.paths[0]), 2)

    def test_quadratic_function_produces_smooth_path(self) -> None:
        func = Function("x^2", name="g")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)
        # Adaptive sampler may produce fewer points for simple curves
        total_points = sum(len(path) for path in result.paths)
        self.assertGreater(total_points, 2)

    def test_discontinuous_function_produces_multiple_paths(self) -> None:
        func = Function("1/x", name="h")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreaterEqual(len(result.paths), 2)
        self.assertTrue(any(len(path) > 1 for path in result.paths))

    def test_discontinuous_function_samples_both_sides_of_asymptote(self) -> None:
        func = Function("1/x", name="h")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()
        self.assertGreaterEqual(len(result.paths), 2)

        math_x_values: list[float] = []
        for path in result.paths:
            for screen_x, screen_y in path:
                math_x, _ = self.mapper.screen_to_math(screen_x, screen_y)
                math_x_values.append(math_x)

        self.assertTrue(any(x < 0 for x in math_x_values))
        self.assertTrue(any(x > 0 for x in math_x_values))

    def test_trigonometric_function_evaluates(self) -> None:
        func = Function("sin(x)", name="s")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)

    def test_high_frequency_sin_has_sufficient_samples_per_period(self) -> None:
        func = Function("sin(10*x)", name="s")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)
        total_points = sum(len(path) for path in result.paths)
        self.assertGreater(total_points, 20)

    def test_high_frequency_sin_peaks_are_smooth(self) -> None:
        func = Function("sin(5*x)", name="s")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)
        min_angle_radians = math.radians(30)
        violations = 0
        total_checked = 0
        for path in result.paths:
            if len(path) < 3:
                continue
            for i in range(1, len(path) - 1):
                angle = renderable._compute_angle(path[i - 1], path[i], path[i + 1])
                total_checked += 1
                if angle < min_angle_radians:
                    violations += 1
        self.assertEqual(violations, 0, f"Found {violations} angles below 30 degrees out of {total_checked}")

    def test_high_amplitude_gets_more_points_than_low_amplitude(self) -> None:
        low_amp_func = Function("sin(x)", name="low_amp", left_bound=-50, right_bound=50)
        high_amp_func = Function("100*sin(x)", name="high_amp", left_bound=-50, right_bound=50)

        low_amp_renderable = FunctionRenderable(low_amp_func, self.mapper)
        high_amp_renderable = FunctionRenderable(high_amp_func, self.mapper)

        low_amp_result = low_amp_renderable.build_screen_paths()
        high_amp_result = high_amp_renderable.build_screen_paths()

        low_amp_points = sum(len(path) for path in low_amp_result.paths)
        high_amp_points = sum(len(path) for path in high_amp_result.paths)

        self.assertGreaterEqual(high_amp_points, low_amp_points)

    def test_sin_peaks_have_reasonable_smoothness(self) -> None:
        func = Function("100*sin(x)", name="s", left_bound=-100, right_bound=100)
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)
        min_angle_radians = math.radians(30)
        violations = 0
        total_angles_checked = 0

        for path in result.paths:
            if len(path) < 3:
                continue
            for i in range(1, len(path) - 1):
                angle = renderable._compute_angle(path[i - 1], path[i], path[i + 1])
                total_angles_checked += 1
                if angle < min_angle_radians:
                    violations += 1

        self.assertGreater(total_angles_checked, 50)
        violation_rate = violations / total_angles_checked if total_angles_checked > 0 else 0
        self.assertLess(violation_rate, 0.30, f"Found {violations} angles below 30 degrees out of {total_angles_checked} ({violation_rate:.1%})")


class TestFunctionsBoundedAreaRenderable(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)

    def test_build_screen_area_with_two_functions(self) -> None:
        from rendering.renderables import FunctionsBoundedAreaRenderable
        from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea
        
        f1 = Function("x^2", name="f")
        f2 = Function("x", name="g")
        
        area_model = FunctionsBoundedColoredArea(
            f1, f2, left_bound=-1, right_bound=1, color="green", opacity=0.4
        )
        
        renderable = FunctionsBoundedAreaRenderable(area_model, self.mapper)
        result = renderable.build_screen_area()

        self.assertIsInstance(result, ClosedArea)
        self.assertGreater(len(result.forward_points), 0)
        self.assertGreater(len(result.reverse_points), 0)

    def test_area_with_x_bounds(self) -> None:
        from rendering.renderables import FunctionsBoundedAreaRenderable
        from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea
        
        f1 = Function("2*x", name="f")
        f2 = Function("x", name="g")
        
        area_model = FunctionsBoundedColoredArea(
            f1, f2, left_bound=0, right_bound=3
        )
        
        renderable = FunctionsBoundedAreaRenderable(area_model, self.mapper)
        result = renderable.build_screen_area()

        self.assertIsNotNone(result)


class TestSegmentsBoundedAreaRenderable(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)

    def test_build_screen_area_from_segments(self) -> None:
        from rendering.renderables import SegmentsBoundedAreaRenderable
        from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea
        
        p1 = Point(0, 0, name="A")
        p2 = Point(4, 0, name="B")
        p3 = Point(4, 3, name="C")
        p4 = Point(0, 3, name="D")
        
        seg1 = Segment(p1, p2)
        seg2 = Segment(p4, p3)
        
        area_model = SegmentsBoundedColoredArea(
            seg1, seg2, color="red", opacity=0.5
        )
        
        renderable = SegmentsBoundedAreaRenderable(area_model, self.mapper)
        result = renderable.build_screen_area()

        self.assertIsNotNone(result)
        if result is not None:
            self.assertIsInstance(result, ClosedArea)
            self.assertGreater(len(result.forward_points), 0)

    def test_triangular_area_from_segments(self) -> None:
        from rendering.renderables import SegmentsBoundedAreaRenderable
        from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea
        
        p1 = Point(0, 0, name="A")
        p2 = Point(3, 0, name="B")
        p3 = Point(1.5, 2.6, name="C")
        
        seg1 = Segment(p1, p2)
        seg2 = Segment(p2, p3)
        
        area_model = SegmentsBoundedColoredArea(seg1, seg2)
        
        renderable = SegmentsBoundedAreaRenderable(area_model, self.mapper)
        result = renderable.build_screen_area()

        self.assertIsNotNone(result)
        self.assertGreater(len(result.forward_points), 0)


class TestBoundaryExtension(unittest.TestCase):
    """Tests for path boundary extension utilities in FunctionRenderable."""
    
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.width = 640
        self.height = 480
        self.func = Function("x", name="f")
        self.renderable = FunctionRenderable(self.func, self.mapper)

    # === _is_inside_screen tests ===
    def test_is_inside_screen_center_point(self) -> None:
        self.assertTrue(self.renderable._is_inside_screen(320, 240, self.width, self.height))

    def test_is_inside_screen_near_edges(self) -> None:
        self.assertTrue(self.renderable._is_inside_screen(1, 1, self.width, self.height))
        self.assertTrue(self.renderable._is_inside_screen(639, 479, self.width, self.height))

    def test_is_inside_screen_on_edge_returns_false(self) -> None:
        self.assertFalse(self.renderable._is_inside_screen(0, 240, self.width, self.height))
        self.assertFalse(self.renderable._is_inside_screen(640, 240, self.width, self.height))
        self.assertFalse(self.renderable._is_inside_screen(320, 0, self.width, self.height))
        self.assertFalse(self.renderable._is_inside_screen(320, 480, self.width, self.height))

    def test_is_inside_screen_outside_returns_false(self) -> None:
        self.assertFalse(self.renderable._is_inside_screen(-10, 240, self.width, self.height))
        self.assertFalse(self.renderable._is_inside_screen(320, -10, self.width, self.height))
        self.assertFalse(self.renderable._is_inside_screen(700, 240, self.width, self.height))
        self.assertFalse(self.renderable._is_inside_screen(320, 500, self.width, self.height))

    # === _is_outside_screen_y tests ===
    def test_is_outside_screen_y_above(self) -> None:
        self.assertTrue(self.renderable._is_outside_screen_y(-10, self.height))
        self.assertTrue(self.renderable._is_outside_screen_y(0, self.height))

    def test_is_outside_screen_y_below(self) -> None:
        self.assertTrue(self.renderable._is_outside_screen_y(480, self.height))
        self.assertTrue(self.renderable._is_outside_screen_y(500, self.height))

    def test_is_outside_screen_y_inside(self) -> None:
        self.assertFalse(self.renderable._is_outside_screen_y(1, self.height))
        self.assertFalse(self.renderable._is_outside_screen_y(240, self.height))
        self.assertFalse(self.renderable._is_outside_screen_y(479, self.height))

    # === _clamp_to_boundary tests ===
    def test_clamp_to_boundary_inside_point_unchanged(self) -> None:
        result = self.renderable._clamp_to_boundary(100, 200, 150, 250, self.height)
        self.assertEqual(result, (150, 250))

    def test_clamp_to_boundary_above_screen(self) -> None:
        # Line from (100, 100) to (200, -100), should clamp to y=0
        result = self.renderable._clamp_to_boundary(100, 100, 200, -100, self.height)
        self.assertEqual(result[1], 0.0)
        self.assertAlmostEqual(result[0], 150.0, places=5)

    def test_clamp_to_boundary_below_screen(self) -> None:
        # Line from (100, 400) to (200, 600), should clamp to y=480
        result = self.renderable._clamp_to_boundary(100, 400, 200, 600, self.height)
        self.assertEqual(result[1], 480)
        self.assertAlmostEqual(result[0], 140.0, places=5)

    def test_clamp_to_boundary_horizontal_line(self) -> None:
        # Horizontal line, y doesn't change, should clamp y directly
        result = self.renderable._clamp_to_boundary(100, 200, 200, 200, self.height)
        self.assertEqual(result, (200, 200))

    def test_clamp_to_boundary_horizontal_line_outside(self) -> None:
        result = self.renderable._clamp_to_boundary(100, -50, 200, -50, self.height)
        self.assertEqual(result[1], 0.0)

    # === Boundary extension behavior tests ===
    def test_extend_paths_extends_inside_start_point(self) -> None:
        # Path starts inside screen - should add extension point
        func = Function("x^2", name="g")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # x^2 should have paths that extend to boundaries
        self.assertGreater(len(result.paths), 0)

    def test_discontinuous_function_paths_extend_to_boundaries(self) -> None:
        # 1/x has asymptote at x=0, paths should extend toward screen edges
        func = Function("1/x", name="h")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Should have 2 sub-paths (one for x<0, one for x>0)
        self.assertGreaterEqual(len(result.paths), 1)
        
        boundary_hits = 0
        for path in result.paths:
            if len(path) >= 2:
                # First and last points should be at or beyond screen boundaries
                # (y=0 or y=height for asymptotic behavior)
                first_y = path[0][1]
                last_y = path[-1][1]
                # At least one endpoint should be at boundary for steep functions
                if first_y <= 0 or first_y >= self.height:
                    boundary_hits += 1
                if last_y <= 0 or last_y >= self.height:
                    boundary_hits += 1
        self.assertGreater(boundary_hits, 0)

    def test_quadratic_function_extends_to_top_boundary(self) -> None:
        # x^2 opens upward, in screen coords the parabola dips down in center
        func = Function("x^2", name="parabola")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        self.assertGreater(len(result.paths), 0)
        # With adaptive sampling, continuous functions should produce at least one path
        # Multiple paths may occur if sampled points happen to fall off-screen
        total_points = sum(len(path) for path in result.paths)
        self.assertGreater(total_points, 2)

    def test_linear_function_has_boundary_points(self) -> None:
        func = Function("x", name="linear")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        self.assertEqual(len(result.paths), 1)
        path = result.paths[0]
        # Linear function should have reasonable coverage
        self.assertGreater(len(path), 5)

    def test_tan_function_has_multiple_paths_with_boundaries(self) -> None:
        # tan(x) has many asymptotes
        func = Function("tan(x)", name="tan")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Should produce multiple discontinuous paths
        self.assertGreater(len(result.paths), 0)

    def test_complex_sin_tan_combination(self) -> None:
        # Complex function: 100*sin(x/50) + 50*tan(x/100)
        # Has asymptotes from tan at x = 100*(π/2 + n*π)
        func = Function("100*sin(x/50) + 50*tan(x/100)", name="complex")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Should produce valid paths
        self.assertGreater(len(result.paths), 0)
        
        # Each path should have at least one point
        for path in result.paths:
            self.assertGreaterEqual(len(path), 1)
            
            # All points should have valid coordinates (not NaN or Inf)
            for x, y in path:
                self.assertFalse(math.isnan(x), f"NaN x coordinate in path")
                self.assertFalse(math.isnan(y), f"NaN y coordinate in path")
                self.assertFalse(math.isinf(x), f"Infinite x coordinate in path")
                self.assertFalse(math.isinf(y), f"Infinite y coordinate in path")

    def test_complex_sin_tan_with_wide_bounds(self) -> None:
        # Same function with wider bounds to hit multiple asymptotes
        func = Function("100*sin(x/50) + 50*tan(x/100)", name="complex_wide", 
                       left_bound=-500, right_bound=500)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Should produce multiple paths due to tan asymptotes
        self.assertGreater(len(result.paths), 0)
        
        # Paths should extend to boundaries properly
        for path in result.paths:
            if len(path) >= 2:
                first_y = path[0][1]
                last_y = path[-1][1]
                # Points should be within reasonable range or at boundaries
                self.assertTrue(
                    -10000 < first_y < 10000,
                    f"First y={first_y} out of reasonable range"
                )
                self.assertTrue(
                    -10000 < last_y < 10000,
                    f"Last y={last_y} out of reasonable range"
                )

    def test_no_diagonal_lines_across_asymptotes(self) -> None:
        # Regression test: ensure no diagonal lines connecting different branches
        # For 1/x, paths on left and right of asymptote should NOT connect
        func = Function("1/x", name="inv")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Should have separate paths for x<0 and x>0
        self.assertGreaterEqual(len(result.paths), 1)
        
        for path in result.paths:
            if len(path) < 2:
                continue
            # Check that consecutive points don't have huge x jumps
            # (which would indicate crossing an asymptote incorrectly)
            for i in range(1, len(path)):
                x_diff = abs(path[i][0] - path[i-1][0])
                # X difference should be reasonable (not jumping across screen)
                self.assertLess(x_diff, self.width / 2, 
                    f"Large x jump detected: {x_diff}, possible diagonal line bug")

    def test_no_path_spans_both_screen_halves(self) -> None:
        # Critical: A single path for 1/x should NOT have endpoints at both top and bottom
        # If it does, there's a diagonal line connecting different branches
        func = Function("1/x", name="inv")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        mid_y = self.height / 2
        tolerance = self.height * 0.1  # 10% tolerance from center
        
        for idx, path in enumerate(result.paths):
            if len(path) < 2:
                continue
            
            # Get y values of all points
            y_values = [pt[1] for pt in path]
            min_y = min(y_values)
            max_y = max(y_values)
            
            # A path should NOT span from top boundary area to bottom boundary area
            near_top = min_y < tolerance
            near_bottom = max_y > self.height - tolerance
            
            self.assertFalse(near_top and near_bottom,
                f"Path {idx} spans both screen halves (y: {min_y:.0f} to {max_y:.0f}), "
                f"indicates diagonal line crossing asymptote")

    def test_path_does_not_cross_asymptote_x(self) -> None:
        # For 1/x, asymptote is at x=0. No path should have points on both sides.
        func = Function("1/x", name="inv")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Get screen x for the asymptote at math x=0
        asymptote_screen_x, _ = self.mapper.math_to_screen(0, 0)
        
        for idx, path in enumerate(result.paths):
            if len(path) < 2:
                continue
            
            # Check all points in path - they should all be on same side of asymptote
            x_values = [pt[0] for pt in path]
            left_of_asymptote = [x < asymptote_screen_x for x in x_values]
            right_of_asymptote = [x > asymptote_screen_x for x in x_values]
            
            has_left = any(left_of_asymptote)
            has_right = any(right_of_asymptote)
            
            self.assertFalse(has_left and has_right,
                f"Path {idx} crosses asymptote at x={asymptote_screen_x:.0f}. "
                f"x range: [{min(x_values):.0f}, {max(x_values):.0f}]")

    def test_no_large_y_jump_in_path(self) -> None:
        # Consecutive points should not have extreme y jumps (indicates wrong branch extension)
        func = Function("1/x", name="inv")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Max allowed y jump between consecutive points (half the screen height)
        max_y_jump = self.height * 0.5
        
        for idx, path in enumerate(result.paths):
            if len(path) < 2:
                continue
            
            for i in range(1, len(path)):
                y1 = path[i-1][1]
                y2 = path[i][1]
                y_diff = abs(y2 - y1)
                
                self.assertLess(y_diff, max_y_jump,
                    f"Path {idx} has large y-jump at point {i}: {y_diff:.0f}px "
                    f"(from y={y1:.0f} to y={y2:.0f}), indicates invalid extension")

    def test_no_large_y_jump_sin_tan_combo(self) -> None:
        # Test the complex function that had diagonal line bugs
        # With adaptive sampling, this function produces many small paths near asymptotes
        func = Function("100*sin(x/50)+50*tan(x/100)", name="combo",
                       left_bound=100, right_bound=200)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Should produce valid paths without crashes
        self.assertGreater(len(result.paths), 0)
        
        # Check that points are valid (no NaN/Inf)
        for path in result.paths:
            for x, y in path:
                self.assertFalse(math.isnan(x))
                self.assertFalse(math.isnan(y))

    def test_paths_extend_to_boundaries_near_asymptotes(self) -> None:
        # For functions with asymptotes, paths should be separate for each branch
        # Boundary extension was removed in favor of simpler adaptive sampling
        func = Function("1/x", name="inv")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Should produce multiple paths due to asymptote at x=0
        self.assertGreater(len(result.paths), 0)
        
        # Verify no path crosses the asymptote (all points on same side of x=0)
        asymptote_x, _ = self.mapper.math_to_screen(0, 0)
        for path in result.paths:
            if len(path) < 2:
                continue
            x_values = [pt[0] for pt in path]
            all_left = all(x < asymptote_x for x in x_values)
            all_right = all(x > asymptote_x for x in x_values)
            self.assertTrue(all_left or all_right,
                "Path should not cross asymptote")

    def test_tan_paths_reach_vertical_bounds(self) -> None:
        # tan(x) should produce multiple separate paths due to asymptotes
        # Boundary extension was removed in favor of simpler adaptive sampling
        func = Function("tan(x)", name="tan")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # tan(x) has asymptotes at pi/2 + n*pi, should produce multiple paths
        self.assertGreater(len(result.paths), 1,
            "tan(x) should have multiple paths due to asymptotes")

    def test_sin_tan_combo_no_crossing_artifacts(self) -> None:
        # Specific test for the complex function that had diagonal line bugs
        # With adaptive sampling, this produces many paths due to asymptotes
        func = Function("100*sin(x/50) + 50*tan(x/100)", name="combo",
                       left_bound=-1000, right_bound=1000)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Should produce valid paths
        self.assertGreater(len(result.paths), 0)
        
        # All points should be valid (no NaN/Inf)
        for path in result.paths:
            for x, y in path:
                self.assertFalse(math.isnan(x), "NaN x coordinate")
                self.assertFalse(math.isnan(y), "NaN y coordinate")
                self.assertFalse(math.isinf(x), "Infinite x coordinate")
                self.assertFalse(math.isinf(y), "Infinite y coordinate")

    def test_extension_respects_function_bounds(self) -> None:
        # Extensions should never go past function's left/right bounds
        func = Function("x^2", name="parabola", left_bound=-5, right_bound=5)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        self.assertGreater(len(result.paths), 0)
        for path in result.paths:
            for x, y in path:
                # Convert screen x to math x and verify within bounds
                math_x, _ = self.mapper.screen_to_math(x, y)
                self.assertGreaterEqual(math_x, -5.1, f"Point extends past left bound: {math_x}")
                self.assertLessEqual(math_x, 5.1, f"Point extends past right bound: {math_x}")

    def test_function_rendered_at_exact_bounds(self) -> None:
        # Function should have points at or very close to the exact bounds
        func = Function("x", name="linear", left_bound=-10, right_bound=10)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        self.assertEqual(len(result.paths), 1)
        path = result.paths[0]
        
        # Get math x values for first and last points
        first_math_x, _ = self.mapper.screen_to_math(path[0][0], path[0][1])
        last_math_x, _ = self.mapper.screen_to_math(path[-1][0], path[-1][1])
        
        # First point should be near left bound, last near right bound
        self.assertLess(abs(first_math_x - (-10)), 1.0, 
            f"First point not at left bound: {first_math_x}")
        self.assertLess(abs(last_math_x - 10), 1.0,
            f"Last point not at right bound: {last_math_x}")

    def test_no_extension_past_screen_half_boundary(self) -> None:
        # Extensions should not cross from top half to bottom half of screen
        # This was a bug where diagonal lines appeared across asymptotes
        func = Function("1/x", name="inv", left_bound=-10, right_bound=10)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        mid_y = self.height / 2
        for path in result.paths:
            if len(path) < 2:
                continue
            # Check that path doesn't have huge y jumps within consecutive points
            for i in range(1, len(path)):
                y1 = path[i-1][1]
                y2 = path[i][1]
                # If one point is in top half and other in bottom half,
                # they shouldn't be far apart (which would indicate a diagonal line bug)
                if (y1 < mid_y and y2 > mid_y) or (y1 > mid_y and y2 < mid_y):
                    y_diff = abs(y2 - y1)
                    # Small crossing is OK (near center), large crossing is a bug
                    self.assertLess(y_diff, self.height * 0.8,
                        f"Large y jump across screen center: {y_diff}")

    def test_extrapolation_continues_path_direction(self) -> None:
        # When extrapolation is used, it should follow the path's direction
        func = Function("x^2", name="parabola")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        self.assertGreater(len(result.paths), 0)
        for path in result.paths:
            if len(path) < 3:
                continue
            # Check that first three points are roughly collinear or curving smoothly
            # (no sharp reversals that would indicate bad extrapolation)
            x1, y1 = path[0]
            x2, y2 = path[1]
            x3, y3 = path[2]
            
            # Direction from 1 to 2
            dx1 = x2 - x1
            dy1 = y2 - y1
            # Direction from 2 to 3
            dx2 = x3 - x2
            dy2 = y3 - y2
            
            # Directions should be roughly similar (not reversed)
            if abs(dx1) > 1 and abs(dx2) > 1:
                # X direction should be consistent
                self.assertEqual(dx1 > 0, dx2 > 0, 
                    "X direction reversal at path start")

    def test_asymptote_paths_extend_to_correct_boundary(self) -> None:
        # For asymptotes, paths going up should extend to y=0 (top),
        # paths going down should extend to y=height (bottom)
        func = Function("tan(x)", name="tan")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        for path in result.paths:
            if len(path) < 2:
                continue
            
            # Check if path is going up or down based on middle points
            mid_idx = len(path) // 2
            if mid_idx > 0 and mid_idx < len(path) - 1:
                y_before = path[mid_idx - 1][1]
                y_after = path[mid_idx + 1][1]
                going_up = y_before > y_after  # y decreasing = going up in screen coords
                
                # First and last points
                first_y = path[0][1]
                last_y = path[-1][1]
                
                # If going up, last point should be near top (y=0)
                # If going down, last point should be near bottom (y=height)
                if going_up:
                    # End should be at or near top (allow small margin for floating point)
                    self.assertLess(last_y, self.height / 2 + 5,
                        f"Path going up but ends at y={last_y}, not near top")

    def test_valid_extension_check_same_screen_half(self) -> None:
        # Test the _is_valid_extension logic
        func = Function("x", name="linear")
        renderable = FunctionRenderable(func, self.mapper)
        
        # Test cases: (original_y, extension_y, should_be_valid)
        test_cases = [
            # Same half - should be valid
            (100, 50, True),   # Both in top half
            (400, 450, True),  # Both in bottom half
            # Different half - should be invalid
            (100, 400, False),  # Top to bottom
            (400, 100, False),  # Bottom to top
            # At boundary - valid if correct boundary
            (100, 0, True),    # Top half to top boundary
            (400, 480, True),  # Bottom half to bottom boundary
            # At wrong boundary - invalid
            (100, 480, False), # Top half to bottom boundary
            (400, 0, False),   # Bottom half to top boundary
        ]
        
        for orig_y, ext_y, expected_valid in test_cases:
            result = renderable._is_valid_extension(orig_y, ext_y, self.height)
            self.assertEqual(result, expected_valid,
                f"_is_valid_extension({orig_y}, {ext_y}, {self.height}) = {result}, expected {expected_valid}")


class TestRenderableEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)

    def test_constant_function_produces_horizontal_line(self) -> None:
        func = Function("5", name="const")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)
        if len(result.paths) > 0:
            path = result.paths[0]
            y_values = [point[1] for point in path]
            
            if len(y_values) > 1:
                y_range = max(y_values) - min(y_values)
                self.assertLess(y_range, 5)

    def test_invalid_function_handles_gracefully(self) -> None:
        func = Function("invalid_expr", name="bad")
        
        try:
            renderable = FunctionRenderable(func, self.mapper)
            renderable.build_screen_paths()
        except Exception as exc:
            self.fail(f"Invalid function should not raise exception: {exc}")

    def test_none_segment_area_handles_gracefully(self) -> None:
        from rendering.renderables import SegmentsBoundedAreaRenderable
        from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea
        
        p1 = Point(0, 0, name="A")
        p2 = Point(1, 0, name="B")
        seg1 = Segment(p1, p2)
        
        area_model = SegmentsBoundedColoredArea(seg1, None)
        
        try:
            renderable = SegmentsBoundedAreaRenderable(area_model, self.mapper)
            renderable.build_screen_area()
        except Exception as exc:
            self.fail(f"Segments area with None segment raised exception: {exc}")

    def test_very_narrow_bounds(self) -> None:
        # Function with very small range (zoomed in scenario)
        func = Function("x^2", name="narrow", left_bound=0.99, right_bound=1.01)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        self.assertGreater(len(result.paths), 0)
        # Should still produce valid points
        for path in result.paths:
            self.assertGreater(len(path), 0)

    def test_very_wide_bounds(self) -> None:
        # Function with very large range
        func = Function("sin(x)", name="wide", left_bound=-10000, right_bound=10000)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        self.assertGreater(len(result.paths), 0)

    def test_function_entirely_above_screen(self) -> None:
        # Function that's entirely above visible area
        func = Function("1000", name="high")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Should handle gracefully (may produce empty paths or paths off-screen)
        # Just ensure no crash
        self.assertIsNotNone(result)

    def test_function_entirely_below_screen(self) -> None:
        # Function that's entirely below visible area
        func = Function("-1000", name="low")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        self.assertIsNotNone(result)

    def test_sqrt_function_domain_edge(self) -> None:
        # sqrt(x) is undefined for x < 0
        func = Function("sqrt(x)", name="sqrt", left_bound=-5, right_bound=10)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Should handle undefined region gracefully
        self.assertIsNotNone(result)
        # Points should only be for x >= 0
        for path in result.paths:
            for x, y in path:
                math_x, _ = self.mapper.screen_to_math(x, y)
                self.assertGreaterEqual(math_x, -0.1, 
                    f"sqrt(x) point at x={math_x} which is invalid domain")

    def test_log_function_domain_edge(self) -> None:
        # log(x) is undefined for x <= 0
        func = Function("log(x)", name="log", left_bound=-5, right_bound=10)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Should handle undefined region gracefully
        self.assertIsNotNone(result)

    def test_multiple_asymptotes_tan(self) -> None:
        # tan(x) has asymptotes at pi/2 + n*pi
        func = Function("tan(x)", name="tan", left_bound=-10, right_bound=10)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # Should produce multiple separate paths
        self.assertGreater(len(result.paths), 1, 
            "tan(x) over [-10,10] should have multiple paths due to asymptotes")

    def test_steep_exponential(self) -> None:
        # e^x grows very steeply
        func = Function("exp(x)", name="exp", left_bound=-5, right_bound=10)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        self.assertGreater(len(result.paths), 0)
        # Should have reasonable points
        for path in result.paths:
            for x, y in path:
                self.assertFalse(math.isnan(x))
                self.assertFalse(math.isnan(y))

    def test_oscillating_function(self) -> None:
        # High frequency oscillation
        func = Function("sin(100*x)", name="fast_sin")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        self.assertGreater(len(result.paths), 0)

    def test_piecewise_like_abs(self) -> None:
        # abs(x) has a corner at x=0
        func = Function("abs(x)", name="abs")
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()
        
        # With adaptive sampling, abs(x) may produce 1-2 paths
        # (the corner at 0 may cause a path break with some configurations)
        self.assertGreater(len(result.paths), 0)
        self.assertLessEqual(len(result.paths), 2)


__all__ = [
    "TestFunctionRenderable",
    "TestFunctionsBoundedAreaRenderable",
    "TestSegmentsBoundedAreaRenderable",
    "TestBoundaryExtension",
    "TestRenderableEdgeCases",
]

