"""Tests for implicit curves and inequality regions on a real canvas: the tools, undo, names,
the colored-area tools on regions, refusals from tools that take parametrized curves,
rendering (fill, solid and dashed strokes), the mid-zoom plot reuse, and the plotter
itself in Brython (whose dicts and floats differ from CPython in places).

Parsing and the plotter's geometry are covered by pytest in server_tests/test_implicit_plotter.py.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from constants import successful_call_message
from rendering.helpers.implicit_renderer import render_implicit_curve_helper, render_inequality_area_helper
from utils.implicit_plotter import plot
from utils.implicit_relations import parse_equation, parse_inequality

from .test_tool_batch_results import _ToolBatchTestCase

Point = Tuple[float, float]


class _Primitives:
    """Records what a render helper draws."""

    def __init__(self) -> None:
        self.polylines: List[List[Point]] = []
        self.fills: List[List[Point]] = []
        self.texts: List[str] = []

    def stroke_polyline(self, points: Any, stroke: Any, **_kwargs: Any) -> None:
        self.polylines.append(list(points))

    def fill_polygon(self, points: Any, fill: Any, stroke: Any = None, **_kwargs: Any) -> None:
        self.fills.append(list(points))

    def draw_text(self, text: str, *_args: Any, **_kwargs: Any) -> None:
        self.texts.append(text)


_STYLE: Dict[str, Any] = {"function_view_margin": 0.5, "function_stroke_width": 1}


class TestImplicitCurveTools(_ToolBatchTestCase):
    def draw(self, equation: str, **args: Any) -> Dict[str, Any]:
        call: Dict[str, Any] = {"name": None, "color": None}
        call.update(args)
        call["equation"] = equation
        _, traced = self.run_batch(("draw_implicit_curve", call))
        return traced[0]

    def shade(self, inequality: str, **args: Any) -> Dict[str, Any]:
        call: Dict[str, Any] = {"name": None, "color": None, "opacity": None}
        call.update(args)
        call["inequality"] = inequality
        _, traced = self.run_batch(("shade_inequality", call))
        return traced[0]

    def curves(self) -> Any:
        return self.canvas.drawable_manager.drawables.ImplicitCurves

    def regions(self) -> Any:
        return self.canvas.drawable_manager.drawables.InequalityColoredAreas

    def test_draw_names_and_result(self) -> None:
        traced = self.draw("x^2 + y^2 = 25")
        self.assertFalse(traced.get("is_error"), traced)
        result = traced["result"]
        self.assertEqual(result["name"], "c1")
        self.assertTrue(result["in_view"]["points_found"])
        self.draw("x = y^2")
        self.draw("x*y = 4", name="hyp", color="red")
        self.assertEqual([curve.name for curve in self.curves()], ["c1", "c2", "hyp"])
        self.assertEqual(self.curves()[2].color, "red")

    def test_one_undo_step_per_draw_and_none_for_a_refusal(self) -> None:
        depth = self.undo_depth()
        self.draw("x^2/9 + y^2/4 = 1")
        self.assertEqual(self.undo_depth(), depth + 1)
        for refused in ("x^2 + y^2 < 4", "x^2 + y^2", "xy = 1x", "X^2 + Y^2 = 4", "sinx = y"):
            traced = self.draw(refused)
            self.assertTrue(traced.get("is_error"), refused)
        self.assertEqual(self.undo_depth(), depth + 1)
        self.assertEqual(len(self.curves()), 1)

    def test_curve_out_of_view_and_fractional_power_notes(self) -> None:
        note = self.draw("x^2 + y^2 = -1")["result"]["in_view"]["note"]
        self.assertIn("No point", note)
        note = self.draw("x^(2/3) + y^(2/3) = 1")["result"]["in_view"]["note"]
        self.assertIn("cbrt(x)", note)

    def test_update_and_delete(self) -> None:
        self.draw("x = 3", name="v")
        self.assertEqual(self.run_single("update_implicit_curve", name="v", new_color="green"), successful_call_message)
        self.assertEqual(self.curves()[0].color, "green")
        depth = self.undo_depth()
        self.assertNotEqual(
            self.run_single("update_implicit_curve", name="v", new_color="green"), successful_call_message
        )
        self.assertEqual(self.undo_depth(), depth)
        _, traced = self.run_batch(("delete_implicit_curve", {"name": "v"}))
        self.assertFalse(traced[0].get("is_error"), traced)
        self.assertEqual(len(self.curves()), 0)
        self.canvas.undo_redo_manager.undo()
        self.assertEqual([curve.name for curve in self.curves()], ["v"])

    def test_shade_names_result_and_colored_area_tools(self) -> None:
        traced = self.shade("x + y <= 10, x >= 0, y >= 0", color="orange", opacity=0.4)
        self.assertFalse(traced.get("is_error"), traced)
        result = traced["result"]
        self.assertEqual(result["name"], "R1")
        self.assertGreater(result["in_view"]["shaded_share"], 0.0)
        region = self.regions()[0]
        self.assertEqual((region.color, region.opacity), ("orange", 0.4))
        _, traced = self.run_batch(
            (
                "update_colored_area",
                {"name": "R1", "new_color": "red", "new_opacity": 0.2, "new_left_bound": None, "new_right_bound": None},
            )
        )
        self.assertFalse(traced[0].get("is_error"), traced)
        self.assertEqual((region.color, region.opacity), ("red", 0.2))
        _, traced = self.run_batch(("delete_colored_area", {"name": "R1"}))
        self.assertFalse(traced[0].get("is_error"), traced)
        self.assertEqual(len(self.regions()), 0)

    def test_shade_refusals_change_nothing(self) -> None:
        depth = self.undo_depth()
        for refused in ("x^2 + y^2 = 4", "y != x", "x > 0 && y > 0", "1 < x > 0", "y < f(x)"):
            traced = self.shade(refused)
            self.assertTrue(traced.get("is_error"), refused)
        traced = self.shade("y > x", opacity=2)
        self.assertTrue(traced.get("is_error"), traced)
        self.assertEqual(self.undo_depth(), depth)
        self.assertEqual(len(self.regions()), 0)

    def test_tools_for_parametrized_curves_name_the_alternative(self) -> None:
        self.draw("x^2 + y^2 = 25", name="circ")
        self.run_single("draw_function", function_string="x", name="f", left_bound=None, right_bound=None)
        _, traced = self.run_batch(
            ("find_intersections", {"object_names": ["circ", "f"], "extend_lines": None, "place_points": None})
        )
        self.assertTrue(traced[0].get("is_error"), traced)
        self.assertIn("solve_system_of_equations", str(traced[0]["result"]))
        _, traced = self.run_batch(
            (
                "arc_length",
                {
                    "curve_name": "circ",
                    "function_string": None,
                    "x_expression": None,
                    "y_expression": None,
                    "r_expression": None,
                    "lower_bound": None,
                    "upper_bound": None,
                },
            )
        )
        self.assertIn("implicit curve", str(traced[0]["result"]))
        _, traced = self.run_batch(
            ("draw_tangent_line", {"curve_name": "circ", "parameter": 1, "name": None, "length": None, "color": None})
        )
        self.assertIn("implicit curve", str(traced[0]["result"]))

    def test_translate_is_refused(self) -> None:
        self.draw("x = y^2", name="p")
        _, traced = self.run_batch(("translate_object", {"name": "p", "x_offset": 1, "y_offset": 0}))
        self.assertTrue(traced[0].get("is_error"), traced)
        self.assertEqual(self.curves()[0].equation, "x = y^2")


class TestImplicitRendering(_ToolBatchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.mapper = self.canvas.coordinate_mapper
        self.mapper.scale_factor = 40.0
        self.mapper.zoom_in_progress = False

    def test_curve_is_stroked_and_labelled(self) -> None:
        curve = self.canvas.drawable_manager.implicit_curve_manager.create_implicit_curve("x^2 + y^2 = 9", "c")
        primitives = _Primitives()
        render_implicit_curve_helper(primitives, curve, self.mapper, _STYLE)
        self.assertEqual(len(primitives.polylines), 1)
        path = primitives.polylines[0]
        self.assertEqual(path[0], path[-1])
        # Radius 3 at 40 px per unit: 120 px from the origin's screen point
        cx, cy = self.mapper.origin.x + self.mapper.offset.x, self.mapper.origin.y + self.mapper.offset.y
        for x, y in path[:: max(1, len(path) // 12)]:
            self.assertAlmostEqual(math.hypot(x - cx, y - cy), 120.0, delta=0.5)
        self.assertEqual(primitives.texts, ["c"])

    def test_region_fill_once_with_solid_and_dashed_boundary(self) -> None:
        manager = self.canvas.drawable_manager.implicit_curve_manager
        area = manager.create_inequality_area("y > x^2 - 4, y <= 2", "R")
        primitives = _Primitives()
        render_inequality_area_helper(primitives, area, self.mapper, _STYLE)
        self.assertEqual(len(primitives.fills), 1)
        self.assertGreater(len(primitives.polylines), 2)  # one solid line, many dashes
        lengths = sorted(len(line) for line in primitives.polylines)
        self.assertLessEqual(lengths[0], 3)

    def test_mid_zoom_draw_reuses_the_last_plot(self) -> None:
        curve = self.canvas.drawable_manager.implicit_curve_manager.create_implicit_curve("x^2 + y^2 = 9", "c")
        render_implicit_curve_helper(_Primitives(), curve, self.mapper, _STYLE)
        first = curve._renderable._result
        self.mapper.zoom_in_progress = True
        self.mapper.scale_factor = 44.0
        primitives = _Primitives()
        render_implicit_curve_helper(primitives, curve, self.mapper, _STYLE)
        self.assertIs(curve._renderable._result, first)
        cx = self.mapper.origin.x + self.mapper.offset.x
        self.assertAlmostEqual(max(x for x, _ in primitives.polylines[0]) - cx, 132.0, delta=0.5)
        self.mapper.zoom_in_progress = False
        render_implicit_curve_helper(_Primitives(), curve, self.mapper, _STYLE)
        self.assertIsNot(curve._renderable._result, first)


class TestImplicitPlotterInBrython(_ToolBatchTestCase):
    """The plotter's results in Brython match what pytest checks in CPython."""

    def test_circle_closes_and_lies_on_the_curve(self) -> None:
        equation = parse_equation("x^2 + y^2 = 25")
        result = plot(equation.g, (-8.0, -6.0, 8.0, 6.0), 0.05, factors=equation.factors)
        self.assertTrue(result.complete)
        self.assertEqual(len(result.curves), 1)
        curve = result.curves[0]
        self.assertEqual(curve[0], curve[-1])
        for x, y in curve:
            self.assertAlmostEqual(math.hypot(x, y), 5.0, delta=1e-3)

    def test_poles_are_not_joined(self) -> None:
        equation = parse_equation("y = 1/x")
        result = plot(equation.g, (-4.0, -3.0, 4.0, 3.0), 0.05, factors=equation.factors)
        self.assertEqual(len(result.curves), 2)
        for curve in result.curves:
            xs = [x for x, _ in curve]
            self.assertTrue(min(xs) > 0 or max(xs) < 0)

    def test_region_rings_and_area(self) -> None:
        tree = parse_inequality("x + y <= 4, x >= 0, y >= 0")
        result = plot(tree.value, (-2.0, -2.0, 6.0, 6.0), 0.05, fill=True, atom_of=tree.atom_of)
        self.assertTrue(result.complete)
        self.assertEqual(len(result.rings), 1)
        ring = result.rings[0]
        area = 0.5 * sum(
            ring[k][0] * ring[(k + 1) % len(ring)][1] - ring[(k + 1) % len(ring)][0] * ring[k][1]
            for k in range(len(ring))
        )
        self.assertAlmostEqual(area, 8.0, delta=0.01)
