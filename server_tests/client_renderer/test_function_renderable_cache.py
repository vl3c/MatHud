from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import math
import unittest
from typing import Any, Callable, Dict, List, Optional

from coordinate_mapper import CoordinateMapper
from drawables.function import Function
from rendering.cached_render_plan import _CachedCoordinateMapper
from rendering.helpers import function_renderer
from rendering.renderables.function_renderable import FunctionRenderable


class _FunctionModel:
    """Minimal function model double that counts evaluations."""

    def __init__(self, expression: str, func: Callable[[float], float], **extra: Any) -> None:
        self.name = "f"
        self.expression = expression
        self._func = func
        self.left_bound: Optional[float] = None
        self.right_bound: Optional[float] = None
        self.vertical_asymptotes: List[float] = []
        self.point_discontinuities: List[float] = []
        self.evaluated: List[float] = []
        for key, value in extra.items():
            setattr(self, key, value)

    def function(self, x: float) -> float:
        self.evaluated.append(x)
        return self._func(x)

    def get_state(self) -> Dict[str, Any]:
        return {"name": self.name, "args": {"function_string": self.expression}}

    def get_vertical_asymptote_between_x(self, x1: float, x2: float) -> Optional[float]:
        return Function.get_vertical_asymptote_between_x(self, x1, x2)  # type: ignore[arg-type]


def _make_mapper() -> CoordinateMapper:
    mapper = CoordinateMapper(640, 480)
    mapper.scale_factor = 32.0
    return mapper


class TestFunctionRenderableReuse(unittest.TestCase):
    def test_renderable_is_reused_across_plan_builds(self) -> None:
        func = _FunctionModel("x^2", lambda x: x * x)
        mapper = _make_mapper()

        first = function_renderer._get_or_create_renderable(func, _CachedCoordinateMapper(mapper))
        second = function_renderer._get_or_create_renderable(func, _CachedCoordinateMapper(mapper))

        self.assertIs(first, second)

    def test_unchanged_view_reuses_cached_paths(self) -> None:
        func = _FunctionModel("x^2", lambda x: x * x)
        mapper = _make_mapper()
        function_renderer._get_or_create_renderable(func, _CachedCoordinateMapper(mapper)).build_screen_paths()
        evaluations = len(func.evaluated)

        function_renderer._get_or_create_renderable(func, _CachedCoordinateMapper(mapper)).build_screen_paths()

        self.assertEqual(len(func.evaluated), evaluations)

    def test_vertical_pan_regenerates_paths(self) -> None:
        func = _FunctionModel("x^2", lambda x: x * x)
        mapper = _make_mapper()
        before = function_renderer._get_or_create_renderable(func, _CachedCoordinateMapper(mapper))
        paths_before = before.build_screen_paths().paths

        mapper.apply_pan(0, 40)
        after = function_renderer._get_or_create_renderable(func, _CachedCoordinateMapper(mapper))
        paths_after = after.build_screen_paths().paths

        self.assertNotEqual(paths_before, paths_after)

    def test_model_change_regenerates_paths(self) -> None:
        func = _FunctionModel("x^2", lambda x: x * x)
        mapper = _make_mapper()
        paths_before = function_renderer._get_or_create_renderable(func, mapper).build_screen_paths().paths

        func.expression = "x^2 + 1"
        func._func = lambda x: x * x + 1
        paths_after = function_renderer._get_or_create_renderable(func, mapper).build_screen_paths().paths

        self.assertNotEqual(paths_before, paths_after)


class TestFunctionRenderableEvaluation(unittest.TestCase):
    def test_each_sample_is_evaluated_once_per_build(self) -> None:
        func = _FunctionModel("sin(x)", math.sin)
        renderable = FunctionRenderable(func, _make_mapper())

        renderable.build_screen_paths()

        self.assertEqual(len(func.evaluated), len(set(func.evaluated)))

    def test_point_discontinuities_break_the_path(self) -> None:
        func = _FunctionModel("x", lambda x: x, point_discontinuities=[1.0])
        renderable = FunctionRenderable(func, _make_mapper())

        paths = renderable.build_screen_paths().paths

        self.assertGreaterEqual(len(paths), 2)

    def test_path_breaks_across_undefined_region(self) -> None:
        def func_value(x: float) -> float:
            value = x * x - 1
            return math.sqrt(value) if value >= 0 else float("nan")

        func = _FunctionModel("sqrt(x^2-1)", func_value)
        mapper = _make_mapper()
        paths = FunctionRenderable(func, mapper).build_screen_paths().paths

        ox, _ = mapper.math_to_screen(0, 0)
        half_unit = mapper.scale_factor / 2
        for path in paths:
            for (x1, _y1), (x2, _y2) in zip(path, path[1:]):
                self.assertFalse(
                    x1 < ox - half_unit and x2 > ox + half_unit,
                    "path must not bridge the undefined interval (-1, 1)",
                )


class _PolylineRecorder:
    def __init__(self) -> None:
        self.polylines: List[List[Any]] = []
        self.texts: List[Any] = []

    def stroke_polyline(self, points: List[Any], _stroke: Any) -> None:
        self.polylines.append(list(points))

    def draw_text(self, text: str, position: Any, *_args: Any, **_kwargs: Any) -> None:
        self.texts.append((text, position))


class TestFunctionViewMargin(unittest.TestCase):
    def test_margin_samples_beyond_each_viewport_edge(self) -> None:
        func = _FunctionModel("x", lambda x: 0.2 * x)
        mapper = _make_mapper()
        renderable = FunctionRenderable(func, mapper)
        renderable.view_margin = 0.5

        points = [point for path in renderable.build_screen_paths().paths for point in path]

        width = mapper.canvas_width
        self.assertLessEqual(min(x for x, _y in points), -0.5 * width + 1)
        self.assertGreaterEqual(max(x for x, _y in points), 1.5 * width - 1)

    def test_margin_keeps_curve_above_and_below_viewport(self) -> None:
        func = _FunctionModel("x^2", lambda x: x * x)
        mapper = _make_mapper()
        renderable = FunctionRenderable(func, mapper)
        renderable.view_margin = 0.5

        ys = [y for path in renderable.build_screen_paths().paths for _x, y in path]

        self.assertAlmostEqual(min(ys), -0.5 * mapper.canvas_height, places=6)

    def test_default_renderable_is_clipped_to_viewport(self) -> None:
        func = _FunctionModel("x^2", lambda x: x * x)
        mapper = _make_mapper()
        ys = [y for path in FunctionRenderable(func, mapper).build_screen_paths().paths for _x, y in path]
        self.assertGreaterEqual(min(ys), 0.0)

    def test_helper_uses_style_margin_and_labels_a_visible_point(self) -> None:
        func = _FunctionModel("x", lambda x: 0.2 * x)
        mapper = _make_mapper()
        recorder = _PolylineRecorder()

        function_renderer.render_function_helper(recorder, func, mapper, {"function_view_margin": 0.5})

        drawn_x = [x for polyline in recorder.polylines for x, _y in polyline]
        self.assertLess(min(drawn_x), -100)
        self.assertGreater(max(drawn_x), mapper.canvas_width + 100)
        ((_text, (label_x, label_y)),) = recorder.texts
        self.assertGreaterEqual(label_x, 0)
        self.assertLessEqual(label_y, mapper.canvas_height)


class TestVerticalAsymptoteLookup(unittest.TestCase):
    def _model(self, asymptotes: List[float]) -> Function:
        model = Function.__new__(Function)
        model.vertical_asymptotes = asymptotes
        return model

    def test_lookup_matches_half_open_interval(self) -> None:
        model = self._model([3.0, -1.0, 1.0])
        self.assertEqual(model.get_vertical_asymptote_between_x(-2.0, 0.0), -1.0)
        self.assertEqual(model.get_vertical_asymptote_between_x(-1.0, 0.0), -1.0)
        self.assertIsNone(model.get_vertical_asymptote_between_x(-0.5, 1.0))
        self.assertEqual(model.get_vertical_asymptote_between_x(0.0, 5.0), 1.0)
        self.assertIsNone(model.get_vertical_asymptote_between_x(4.0, 5.0))
        self.assertIsNone(model.get_vertical_asymptote_between_x(2.0, 0.0))
        self.assertTrue(model.has_vertical_asymptote_between_x(2.5, 3.5))
        self.assertFalse(model.has_vertical_asymptote_between_x(3.5, 4.5))

    def test_lookup_tracks_replaced_asymptote_list(self) -> None:
        model = self._model([1.0])
        self.assertEqual(model.get_vertical_asymptote_between_x(0.0, 2.0), 1.0)
        model.vertical_asymptotes = [5.0]
        self.assertIsNone(model.get_vertical_asymptote_between_x(0.0, 2.0))

    def test_lookup_does_not_rescan_the_asymptote_list(self) -> None:
        class _CountingList(list):
            iterations = 0

            def __iter__(self):
                _CountingList.iterations += 1
                return super().__iter__()

        asymptotes = _CountingList(math.pi / 2 + k * math.pi for k in range(-318, 318))
        model = self._model(asymptotes)
        found = [model.get_vertical_asymptote_between_x(-1000 + i * 0.5, -1000 + (i + 1) * 0.5) for i in range(4000)]

        self.assertEqual(sum(1 for value in found if value is not None), 636)
        self.assertLessEqual(_CountingList.iterations, 1)


if __name__ == "__main__":
    unittest.main()
