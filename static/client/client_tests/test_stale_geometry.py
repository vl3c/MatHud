"""Regression tests for derived geometry that must follow the points (K8, K9, K16, K17, K19).

Each test drives the canvas with the same function registry and result processor
the AI uses, so it covers the tool path the model takes.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, List, Tuple

from canvas import Canvas
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from utils.math_utils import MathUtils
from workspace_manager import WorkspaceManager

Coord = Tuple[float, float]

EQUILATERAL_APEX_Y = 3.4641016151377544


class TestStaleGeometry(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.workspace_manager = WorkspaceManager(self.canvas)
        self.available_functions = FunctionRegistry.get_available_functions(self.canvas, self.workspace_manager)
        self.undoable_functions = FunctionRegistry.get_undoable_functions()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run(self, function_name: str, **arguments: Any) -> Any:
        """Run one tool call through the AI path and return its raw result value."""
        calls = [{"function_name": function_name, "arguments": arguments}]
        results = ProcessFunctionCalls.get_results(
            calls, self.available_functions, self.undoable_functions, self.canvas
        )
        self.assertEqual(len(results), 1, f"Expected one result for {function_name}: {results}")
        return list(results.values())[0]

    def _call(self, function_name: str, **arguments: Any) -> Any:
        """Run one tool call that must succeed."""
        value = self._run(function_name, **arguments)
        if isinstance(value, str):
            self.assertFalse(value.startswith("Error"), f"{function_name} failed: {value}")
        return value

    def _call_error(self, function_name: str, **arguments: Any) -> str:
        """Run one tool call that must fail, and return its error text."""
        value = self._run(function_name, **arguments)
        self.assertIsInstance(value, str)
        self.assertTrue(value.startswith("Error"), f"{function_name} should fail, got: {value}")
        return str(value)

    def _polygon(self, vertices: List[Coord], polygon_type: str, name: str, subtype: Any = None) -> Any:
        args: dict[str, Any] = {
            "vertices": [{"x": x, "y": y} for x, y in vertices],
            "polygon_type": polygon_type,
            "name": name,
        }
        if subtype is not None:
            args["subtype"] = subtype
        self._call("create_polygon", **args)
        return self._only(self._polygon_class(polygon_type))

    @staticmethod
    def _polygon_class(polygon_type: str) -> str:
        return {"triangle": "Triangle", "pentagon": "Pentagon"}.get(polygon_type, "Rectangle")

    def _only(self, class_name: str) -> Any:
        drawables = list(self.canvas.get_drawables_by_class_name(class_name))
        self.assertEqual(len(drawables), 1, f"Expected one {class_name}, found {len(drawables)}")
        return drawables[0]

    def _point(self, name: str) -> Any:
        point = self.canvas.get_point_by_name(name)
        self.assertIsNotNone(point, f"Point {name} not found")
        return point

    def _coords(self, point: Any) -> Coord:
        return (float(point.x), float(point.y))

    def _points(self) -> List[Coord]:
        return sorted(self._coords(p) for p in self.canvas.get_drawables_by_class_name("Point"))

    def _state_of(self, drawable: Any) -> Any:
        bucket = drawable.get_class_name() + "s"
        states = self.canvas.get_drawables_state().get(bucket, [])
        for state in states:
            if state.get("name") == drawable.name:
                return state
        self.fail(f"{drawable.name} missing from the canvas state")

    # ------------------------------------------------------------------
    # K8: circle and ellipse formulas follow their centre
    # ------------------------------------------------------------------
    def test_circle_formula_follows_translated_triangle_vertex(self) -> None:
        self._polygon([(0, 0), (4, 0), (0, 3)], "triangle", "ABC")
        self._call("create_circle", center_x=0, center_y=0, radius=5)
        circle = self._only("Circle")

        self._call("translate_object", name="ABC", x_offset=10, y_offset=0)

        expected = MathUtils.get_circle_formula(10.0, 0.0, 5)
        self.assertEqual(circle.circle_formula, expected)
        self.assertEqual(self._state_of(circle)["args"]["circle_formula"], expected)

    def test_circle_formula_follows_translated_centre_point(self) -> None:
        self._call("create_circle", center_x=1, center_y=2, radius=3)
        circle = self._only("Circle")

        self._call("translate_object", name=circle.center.name, x_offset=-4, y_offset=5)

        self.assertEqual(circle.circle_formula, MathUtils.get_circle_formula(-3.0, 7.0, 3))

    def test_ellipse_formula_follows_translated_triangle_vertex(self) -> None:
        self._polygon([(0, 0), (4, 0), (0, 3)], "triangle", "ABC")
        self._call("create_ellipse", center_x=0, center_y=0, radius_x=3, radius_y=2, rotation_angle=30)
        ellipse = self._only("Ellipse")

        self._call("translate_object", name="ABC", x_offset=2, y_offset=-1)

        expected = MathUtils.get_ellipse_formula(2.0, -1.0, 3, 2, 30)
        self.assertEqual(ellipse.ellipse_formula, expected)
        self.assertEqual(self._state_of(ellipse)["args"]["ellipse_formula"], expected)
