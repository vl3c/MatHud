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

        width = mapper.canvas_width
        for path in paths:
            for (x1, _y1), (x2, _y2) in zip(path, path[1:]):
                self.assertLessEqual(abs(x2 - x1), 2 * width)

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


if __name__ == "__main__":
    unittest.main()
