from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import math
import unittest
from typing import Any, Callable, Dict, List, Optional

from coordinate_mapper import CoordinateMapper
from drawables.function import Function
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
