from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import math
import unittest
from types import SimpleNamespace
from typing import Any, Callable, List, Tuple

from coordinate_mapper import CoordinateMapper
from rendering import canvas2d_renderer
from rendering.helpers import parametric_function_renderer
from rendering.renderables.parametric_function_renderable import ParametricFunctionRenderable


def _parametric(x: Callable[[float], float], y: Callable[[float], float], t_min: float, t_max: float) -> Any:
    return SimpleNamespace(
        name="p",
        color="#000",
        t_min=t_min,
        t_max=t_max,
        evaluate_x=x,
        evaluate_y=y,
        get_class_name=lambda: "ParametricFunction",
        get_state=lambda: {"name": "p", "args": {"t_min": t_min, "t_max": t_max}},
    )


def _make_mapper() -> CoordinateMapper:
    mapper = CoordinateMapper(640, 480)
    mapper.scale_factor = 32.0
    return mapper


class _PolylineRecorder:
    def __init__(self) -> None:
        self.polylines: List[List[Tuple[float, float]]] = []

    def stroke_polyline(self, points: List[Tuple[float, float]], _stroke: Any) -> None:
        self.polylines.append(list(points))

    def draw_text(self, *_args: Any, **_kwargs: Any) -> None:
        pass


class TestParametricPaths(unittest.TestCase):
    def test_path_breaks_on_large_horizontal_jump(self) -> None:
        # x(t) = 1/(t-1) jumps from +inf to -inf at t = 1 while y stays 0.
        func = _parametric(lambda t: 1.0 / (t - 1.0) if t != 1.0 else float("nan"), lambda t: 0.0, 0.0, 2.0)
        mapper = _make_mapper()
        paths = ParametricFunctionRenderable(func, mapper).build_screen_paths().paths

        # Long steps along one branch (both ends off screen on the same side) are kept;
        # no step may join the branch on the left to the branch on the right.
        width = mapper.canvas_width
        self.assertEqual(len(paths), 2)
        for path in paths:
            for (x1, _y1), (x2, _y2) in zip(path, path[1:]):
                self.assertFalse(min(x1, x2) < 0 and max(x1, x2) > width, (x1, x2))

    def test_steep_continuous_crossing_is_not_broken(self) -> None:
        # y(t) = 1e4 * t^3 crosses the whole screen between two samples near t = 0.
        func = _parametric(lambda t: t, lambda t: 1e4 * t**3, -1.0, 1.0)
        paths = ParametricFunctionRenderable(func, _make_mapper()).build_screen_paths().paths
        self.assertEqual(len(paths), 1)

    def test_helper_keeps_off_screen_parts_for_reprojection(self) -> None:
        # A circle of radius 8 centred at (8, 0) is half off-screen at scale 32 (visible x-range is +-10).
        func = _parametric(lambda t: 8 + 8 * math.cos(t), lambda t: 8 * math.sin(t), 0.0, 2 * math.pi)
        mapper = _make_mapper()
        recorder = _PolylineRecorder()

        parametric_function_renderer.render_parametric_function_helper(recorder, func, mapper, {})

        drawn = {point for polyline in recorder.polylines for point in polyline}
        sampled = ParametricFunctionRenderable(func, mapper).build_screen_paths().paths
        off_screen = [(x, y) for path in sampled for x, y in path if x > mapper.canvas_width + 100]
        self.assertTrue(off_screen)
        for point in off_screen:
            self.assertIn(point, drawn)


class TestParametricPlanSignature(unittest.TestCase):
    def _signature(self, mapper: CoordinateMapper) -> Any:
        renderer = canvas2d_renderer.Canvas2DRenderer.__new__(canvas2d_renderer.Canvas2DRenderer)
        func = _parametric(math.cos, math.sin, 0.0, 2 * math.pi)
        return renderer._compute_drawable_signature(func, mapper)

    def test_pan_does_not_invalidate_parametric_plan(self) -> None:
        mapper = _make_mapper()
        before = self._signature(mapper)
        mapper.apply_pan(120, -45)
        self.assertEqual(self._signature(mapper), before)

    def test_zoom_invalidates_parametric_plan(self) -> None:
        mapper = _make_mapper()
        before = self._signature(mapper)
        mapper.scale_factor = 64.0
        self.assertNotEqual(self._signature(mapper), before)


class TestDomainEdges(unittest.TestCase):
    """Paths end where the curve's domain ends between two samples, not at the last sample."""

    def _paths(self, func: Any, mapper: CoordinateMapper) -> List[List[Tuple[float, float]]]:
        return [list(path) for path in ParametricFunctionRenderable(func, mapper).build_screen_paths().paths]

    def test_lemniscate_lobes_reach_the_pole(self) -> None:
        from drawables.polar_function import PolarFunction

        mapper = _make_mapper()
        pole = mapper.math_to_screen(0.0, 0.0)
        # (r(θ), θ range, path ends at the pole: one per domain edge inside the range)
        cases = (
            ("sqrt(9cos(2θ))", (0.0, 2 * math.pi), 4),
            ("sqrt(cos(2θ))", (0.0, 6.283), 4),
            ("3sqrt(sin(2θ))", (0.1, 3.3), 2),
        )
        for r_text, (theta_min, theta_max), edges in cases:
            with self.subTest(r=r_text, theta_max=theta_max):
                curve = PolarFunction(r_text, theta_min=theta_min, theta_max=theta_max)
                ends = [end for path in self._paths(curve, mapper) for end in (path[0], path[-1])]
                at_pole = [end for end in ends if math.hypot(end[0] - pole[0], end[1] - pole[1]) < 1.0]
                self.assertEqual(len(at_pole), edges, ends)

    def test_a_pole_of_r_is_not_joined_to_infinity(self) -> None:
        from drawables.polar_function import PolarFunction

        # A parabola r = 2/(1 - cos θ): r is infinite at θ = 0 and 2π, the range's ends
        mapper = _make_mapper()
        paths = self._paths(PolarFunction("2/(1 - cos(θ))"), mapper)
        self.assertEqual(len(paths), 1)
        for x, y in paths[0]:
            self.assertTrue(math.isfinite(x) and math.isfinite(y))
            self.assertLess(abs(x), 1e7)
            self.assertLess(abs(y), 1e7)

    def test_hyperbola_branches_are_not_joined(self) -> None:
        from drawables.polar_function import PolarFunction

        # r = 2/(1 - 2cos θ): r changes sign through infinity at θ = ±π/3, two branches
        mapper = _make_mapper()
        paths = self._paths(PolarFunction("2/(1 - 2cos(θ))"), mapper)
        width, height = mapper.canvas_width, mapper.canvas_height
        self.assertGreaterEqual(len(paths), 2)
        for path in paths:
            for (x1, y1), (x2, y2) in zip(path, path[1:]):
                # A step may run along a branch off screen, never across the screen between branches
                self.assertFalse(min(x1, x2) < 0 and max(x1, x2) > width, ((x1, y1), (x2, y2)))
                self.assertFalse(min(y1, y2) < 0 and max(y1, y2) > height, ((x1, y1), (x2, y2)))

    def test_removable_gap_at_the_range_start_is_closed(self) -> None:
        # x(t) = sin(t)/t is undefined only at t = 0, the first sample
        func = _parametric(lambda t: math.sin(t) / t, lambda t: t, 0.0, 1.0)
        mapper = _make_mapper()
        paths = self._paths(func, mapper)
        self.assertEqual(len(paths), 1)
        start = paths[0][0]
        expected = mapper.math_to_screen(1.0, 0.0)
        self.assertLess(math.hypot(start[0] - expected[0], start[1] - expected[1]), 1e-3)


if __name__ == "__main__":
    unittest.main()
