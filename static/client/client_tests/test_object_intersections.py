"""Brython tests for the find_intersections tool (Canvas -> DrawableManager -> IntersectionManager).

The geometry itself (utils/object_intersections.py) is covered by the pytest suite
server_tests/test_object_intersections_pure.py.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional

from canvas import Canvas
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from workspace_manager import WorkspaceManager


class TestObjectIntersectionsCanvas(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)

    def _find(
        self, names: List[str], extend_lines: Optional[bool] = None, place_points: Optional[bool] = None
    ) -> Dict[str, Any]:
        return self.canvas.find_intersections(names, extend_lines=extend_lines, place_points=place_points)

    def _point_count(self) -> int:
        return len(self.canvas.get_drawables_by_class_name("Point"))

    def _xy(self, result: Dict[str, Any]) -> List[tuple]:
        return [(p["x"], p["y"]) for p in result["intersections"]]

    def test_segment_and_circle(self) -> None:
        segment = self.canvas.create_segment(-5, 0, 5, 0, extra_graphics=False)
        circle = self.canvas.create_circle(0, 0, 3, extra_graphics=False)
        result = self._find([segment.name, circle.name])
        self.assertEqual(result["object_names"], [segment.name, circle.name])
        self.assertEqual(result["object_types"], ["segment", "circle"])
        self.assertEqual(self._xy(result), [(-3.0, 0.0), (3.0, 0.0)])
        self.assertEqual(result["count"], 2)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["overlaps"], [])
        self.assertEqual(result["intersections"][0]["params"][segment.name], {"t": 0.2})
        self.assertAlmostEqual(result["intersections"][0]["params"][circle.name]["angle"], math.pi, places=9)
        self.assertNotIn("note", result)
        self.assertNotIn("x_range", result)

    def test_extend_lines_reaches_past_the_segment(self) -> None:
        segment = self.canvas.create_segment(0, 0, 1, 0, extra_graphics=False)
        circle = self.canvas.create_circle(0, 0, 3, extra_graphics=False)
        short = self._find([segment.name, circle.name])
        self.assertEqual(short["intersections"], [])
        self.assertIn("extend_lines", short["note"])
        extended = self._find([segment.name, circle.name], extend_lines=True)
        self.assertEqual(self._xy(extended), [(-3.0, 0.0), (3.0, 0.0)])

    def test_vector_and_rotated_ellipse(self) -> None:
        vector = self.canvas.create_vector(-10, 2, 10, 2, extra_graphics=False)
        ellipse = self.canvas.create_ellipse(0, 0, 2, 4, rotation_angle=90, extra_graphics=False)
        result = self._find([vector.name, ellipse.name])
        self.assertEqual(result["object_types"], ["vector", "ellipse"])
        self.assertEqual(self._xy(result), [(0.0, 2.0)])
        self.assertTrue(result["intersections"][0]["tangent"])

    def test_minor_and_major_arcs(self) -> None:
        minor = self.canvas.create_circle_arc(
            2, 0, 0, 2, center_x=0, center_y=0, radius=2, arc_name="minor", extra_graphics=False
        )
        major = self.canvas.create_circle_arc(
            2, 0, 0, 2, center_x=0, center_y=0, radius=2, arc_name="major", use_major_arc=True, extra_graphics=False
        )
        horizontal = self.canvas.create_segment(-3, 1, 3, 1, extra_graphics=False)
        on_minor = self._find([minor.name, horizontal.name])
        self.assertEqual(on_minor["object_types"], ["circle_arc", "segment"])
        self.assertEqual(len(on_minor["intersections"]), 1)
        self.assertAlmostEqual(on_minor["intersections"][0]["x"], math.sqrt(3), places=9)
        on_major = self._find([major.name, horizontal.name])
        self.assertEqual(len(on_major["intersections"]), 1)
        self.assertAlmostEqual(on_major["intersections"][0]["x"], -math.sqrt(3), places=9)

    def test_function_and_circle_report_the_search_range(self) -> None:
        self.canvas.draw_function("x^2", name="f", left_bound=-5, right_bound=5)
        circle = self.canvas.create_circle(0, 0, 2, extra_graphics=False)
        result = self._find(["f", circle.name])
        y = (-1 + math.sqrt(17)) / 2
        self.assertEqual(len(result["intersections"]), 2)
        self.assertAlmostEqual(result["intersections"][1]["x"], math.sqrt(y), places=9)
        self.assertAlmostEqual(result["intersections"][1]["y"], y, places=9)
        self.assertEqual(result["x_range"], [-5.0, 5.0])

    def test_function_without_bounds_uses_the_visible_range(self) -> None:
        self.canvas.draw_function("x - 3", name="g")
        segment = self.canvas.create_segment(0, 0, 1, 0, extra_graphics=False)
        mapper = self.canvas.coordinate_mapper
        result = self._find(["g", segment.name], extend_lines=True)
        self.assertAlmostEqual(result["x_range"][0], mapper.get_visible_left_bound(), places=6)
        self.assertEqual(self._xy(result), [(3.0, 0.0)])

    def test_two_functions_and_a_piecewise_function(self) -> None:
        self.canvas.draw_function("x^2", name="f", left_bound=-5, right_bound=5)
        self.canvas.draw_function("x + 2", name="g", left_bound=-5, right_bound=5)
        self.assertEqual(self._xy(self._find(["f", "g"])), [(-1.0, 1.0), (2.0, 4.0)])
        pieces = [
            {"expression": "x^2 - 1", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x - 2", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        self.canvas.draw_piecewise_function(pieces, name="p")
        axis = self.canvas.create_segment(-5, 0, 5, 0, extra_graphics=False)
        result = self._find(["p", axis.name])
        self.assertEqual(result["object_types"], ["piecewise_function", "segment"])
        self.assertEqual([p["x"] for p in result["intersections"]], [-1.0, 2.0])

    def test_parametric_curve_and_segment(self) -> None:
        curve = self.canvas.draw_parametric_function("cos(t)", "sin(t)", name="unit")
        segment = self.canvas.create_segment(-2, 0.5, 2, 0.5, extra_graphics=False)
        result = self._find([curve.name, segment.name])
        self.assertEqual(result["object_types"], ["parametric_function", "segment"])
        self.assertEqual(len(result["intersections"]), 2)
        self.assertAlmostEqual(result["intersections"][1]["params"][curve.name]["t"], math.pi / 6, places=9)

    def test_collinear_segments_are_an_overlap(self) -> None:
        first = self.canvas.create_segment(0, 0, 2, 0, extra_graphics=False)
        second = self.canvas.create_segment(1, 0, 3, 0, extra_graphics=False)
        result = self._find([first.name, second.name], place_points=True)
        self.assertEqual(result["intersections"], [])
        self.assertEqual(result["overlaps"], [{"kind": "segment", "start": [1.0, 0.0], "end": [2.0, 0.0]}])
        self.assertIn("overlap", result["note"])
        self.assertNotIn("point_names", result)

    def test_no_intersection_note(self) -> None:
        first = self.canvas.create_circle(0, 0, 1, extra_graphics=False)
        second = self.canvas.create_circle(5, 0, 1, extra_graphics=False)
        result = self._find([first.name, second.name])
        self.assertEqual(result["count"], 0)
        self.assertIn("do not intersect", result["note"])

    def test_bad_names_raise(self) -> None:
        circle = self.canvas.create_circle(0, 0, 1, extra_graphics=False)
        point = self.canvas.create_point(5, 5, name="Q", extra_graphics=False)
        with self.assertRaises(ValueError):
            self._find([circle.name])
        with self.assertRaises(ValueError):
            self._find([circle.name, circle.name])
        with self.assertRaises(ValueError):
            self._find([circle.name, "missing"])
        with self.assertRaises(ValueError):
            self._find([circle.name, point.name])

    def test_place_points_creates_named_points(self) -> None:
        segment = self.canvas.create_segment(-5, 0, 5, 0, extra_graphics=False)
        circle = self.canvas.create_circle(0, 0, 3, extra_graphics=False)
        before = self._point_count()
        result = self._find([segment.name, circle.name], place_points=True)
        self.assertEqual(len(result["point_names"]), 2)
        self.assertEqual(result["created_point_names"], result["point_names"])
        self.assertEqual(result["reused_point_names"], [])
        self.assertEqual(self._point_count(), before + 2)
        for item in result["intersections"]:
            point = self.canvas.get_point_by_name(item["point_name"])
            self.assertEqual((point.x, point.y), (item["x"], item["y"]))

    def test_placed_points_are_one_undo_step(self) -> None:
        first = self.canvas.create_ellipse(0, 0, 4, 2, extra_graphics=False)
        second = self.canvas.create_ellipse(0, 0, 2, 4, extra_graphics=False)
        before = self._point_count()
        result = self._find([first.name, second.name], place_points=True)
        self.assertEqual(len(result["point_names"]), 4)
        self.assertEqual(self._point_count(), before + 4)

        self.assertTrue(self.canvas.undo())
        self.assertEqual(self._point_count(), before)
        self.assertIsNotNone(self.canvas.get_ellipse_by_name(second.name))

        self.assertTrue(self.canvas.redo())
        self.assertEqual(self._point_count(), before + 4)

    def test_existing_point_is_reused_and_reported(self) -> None:
        segment = self.canvas.create_segment(-5, 0, 5, 0, extra_graphics=False)
        circle = self.canvas.create_circle(0, 0, 3, extra_graphics=False)
        existing = self.canvas.create_point(3, 0, name="P", extra_graphics=False)
        result = self._find([segment.name, circle.name], place_points=True)
        self.assertIn(existing.name, result["point_names"])
        self.assertEqual(result["reused_point_names"], [existing.name])
        self.assertEqual(len(result["created_point_names"]), 1)
        self.assertNotIn(existing.name, result["created_point_names"])
        self.assertIn(existing.name, result["note"])

    def test_without_place_points_nothing_is_archived(self) -> None:
        segment = self.canvas.create_segment(-5, 0, 5, 0, extra_graphics=False)
        circle = self.canvas.create_circle(0, 0, 3, extra_graphics=False)
        undo_count = len(self.canvas.undo_redo_manager.undo_stack)
        self._find([segment.name, circle.name])
        self.assertEqual(len(self.canvas.undo_redo_manager.undo_stack), undo_count)

    def _run_tool_calls(self, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run tool calls through the same registry and result processor the AI uses."""
        available = FunctionRegistry.get_available_functions(self.canvas, WorkspaceManager(self.canvas))
        undoable = FunctionRegistry.get_undoable_functions()
        return dict(ProcessFunctionCalls.get_results(calls, available, undoable, self.canvas))

    def test_tool_call_returns_the_values_to_the_ai(self) -> None:
        self.assertNotIn("find_intersections", FunctionRegistry.get_undoable_functions())
        segment = self.canvas.create_segment(-5, 0, 5, 0, extra_graphics=False)
        circle = self.canvas.create_circle(0, 0, 3, extra_graphics=False)
        arguments = {"object_names": [segment.name, circle.name], "extend_lines": None, "place_points": None}
        results = self._run_tool_calls([{"function_name": "find_intersections", "arguments": arguments}])
        value = list(results.values())[0]
        self.assertIsInstance(value, dict)
        self.assertEqual([p["x"] for p in value["intersections"]], [-3.0, 3.0])
