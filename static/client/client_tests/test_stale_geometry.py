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

    # ------------------------------------------------------------------
    # K16: polygon types follow the vertices
    # ------------------------------------------------------------------
    def test_square_scaled_non_uniformly_is_no_longer_a_square(self) -> None:
        square = self._polygon([(0, 0), (4, 0), (4, 4), (0, 4)], "square", "ABCD")
        self.assertIn("square", square.get_type_names())

        self._call("scale_object", name=square.name, sx=2, sy=1, cx=0, cy=0)

        types = self._state_of(square)["types"]
        self.assertNotIn("square", types)
        self.assertNotIn("rhombus", types)
        self.assertIn("rectangle", types)

    def test_sheared_rectangle_is_no_longer_a_rectangle(self) -> None:
        rectangle = self._polygon([(0, 0), (4, 0), (4, 2), (0, 2)], "rectangle", "ABCD")

        self._call("shear_object", name=rectangle.name, axis="horizontal", factor=1, cx=0, cy=0)

        types = self._state_of(rectangle)["types"]
        self.assertNotIn("rectangle", types)
        self.assertNotIn("square", types)
        self.assertIn("quadrilateral", types)

    def test_sheared_equilateral_triangle_is_no_longer_equilateral(self) -> None:
        triangle = self._polygon([(0, 0), (4, 0), (2, EQUILATERAL_APEX_Y)], "triangle", "ABC", subtype="equilateral")
        self.assertIn("equilateral", triangle.get_type_names())

        self._call("shear_object", name="ABC", axis="horizontal", factor=1, cx=0, cy=0)

        types = self._state_of(triangle)["types"]
        self.assertNotIn("equilateral", types)
        self.assertNotIn("isosceles", types)
        self.assertIn("scalene", types)

    def test_triangle_types_follow_a_moved_vertex(self) -> None:
        triangle = self._polygon([(0, 0), (4, 0), (0, 3)], "triangle", "ABC")
        self.assertIn("right", triangle.get_type_names())

        self._call("translate_object", name="A", x_offset=1, y_offset=1)

        self.assertNotIn("right", self._state_of(triangle)["types"])

    def test_square_types_follow_a_moved_vertex(self) -> None:
        square = self._polygon([(0, 0), (4, 0), (4, 4), (0, 4)], "square", "ABCD")

        self._call("translate_object", name="C", x_offset=1, y_offset=1)

        types = self._state_of(square)["types"]
        self.assertNotIn("square", types)
        self.assertNotIn("rectangle", types)
        self.assertIn("irregular", types)

    def test_pentagon_regularity_follows_scaling(self) -> None:
        vertices = [(math.cos(math.radians(90 + 72 * k)), math.sin(math.radians(90 + 72 * k))) for k in range(5)]
        pentagon = self._polygon(vertices, "pentagon", "ABCDE")
        self.assertIn("regular", pentagon.get_type_names())

        self._call("scale_object", name=pentagon.name, sx=3, sy=1, cx=0, cy=0)

        types = self._state_of(pentagon)["types"]
        self.assertNotIn("regular", types)
        self.assertIn("irregular", types)

    # ------------------------------------------------------------------
    # K17: angle values follow the points
    # ------------------------------------------------------------------
    def test_angle_degrees_follow_shear(self) -> None:
        self._polygon([(0, 0), (4, 0), (2, EQUILATERAL_APEX_Y)], "triangle", "ABC")
        self._call("create_angle", vx=4, vy=0, p1x=0, p1y=0, p2x=2, p2y=EQUILATERAL_APEX_Y)
        angle = self._only("Angle")
        self.assertAlmostEqual(angle.angle_degrees, 60.0, places=6)

        self._call("shear_object", name="ABC", axis="horizontal", factor=1, cx=0, cy=0)

        vertex, arm1, arm2 = angle.vertex_point, angle.arm1_point, angle.arm2_point
        expected_raw = MathUtils.calculate_angle_degrees(self._coords(vertex), self._coords(arm1), self._coords(arm2))
        expected = expected_raw if expected_raw <= 180 else 360 - expected_raw
        self.assertAlmostEqual(angle.angle_degrees, expected, places=9)
        self.assertAlmostEqual(angle.angle_degrees, 112.91133691901508, places=6)
        self.assertAlmostEqual(angle.raw_angle_degrees, expected_raw, places=9)

    def test_angle_degrees_follow_a_moved_arm_point(self) -> None:
        self._call("create_angle", vx=0, vy=0, p1x=4, p1y=0, p2x=0, p2y=4)
        angle = self._only("Angle")
        self.assertAlmostEqual(angle.angle_degrees, 90.0, places=6)

        self._call("translate_object", name=angle.arm2_point.name, x_offset=4, y_offset=0)

        self.assertAlmostEqual(angle.angle_degrees, 45.0, places=6)

    def test_reflex_angle_follows_moved_points(self) -> None:
        self._call("create_angle", vx=0, vy=0, p1x=4, p1y=0, p2x=0, p2y=4, is_reflex=True)
        angle = self._only("Angle")
        self.assertAlmostEqual(angle.angle_degrees, 270.0, places=6)

        self._call("translate_object", name=angle.arm2_point.name, x_offset=4, y_offset=0)

        self.assertAlmostEqual(angle.angle_degrees, 315.0, places=6)

    # ------------------------------------------------------------------
    # K9: arcs never move existing points
    # ------------------------------------------------------------------
    def test_arc_does_not_move_an_existing_point_given_by_coordinates(self) -> None:
        self._call("create_segment", x1=0, y1=0, x2=3, y2=0)
        segment = self._only("Segment")
        point_b = segment.point2 if segment.point2.x == 3 else segment.point1

        result = self._call(
            "create_circle_arc",
            point1_x=3,
            point1_y=0,
            point2_x=0,
            point2_y=5,
            center_x=0,
            center_y=0,
            radius=5,
            use_major_arc=False,
        )

        self.assertEqual(self._coords(point_b), (3.0, 0.0))
        self.assertAlmostEqual(math.hypot(segment.point2.x - segment.point1.x, segment.point2.y - segment.point1.y), 3)
        arc = self._only("CircleArc")
        self.assertIsNot(arc.point1, point_b)
        self.assertAlmostEqual(arc.point1.x, 5.0)
        self.assertAlmostEqual(arc.point1.y, 0.0)
        self.assertIsInstance(result, str)
        self.assertIn(point_b.name, result)
        self.assertIn(arc.point1.name, result)
        self.assertIn("not moved", result)

    def test_arc_does_not_move_an_existing_point_given_by_name(self) -> None:
        self._call("create_point", x=1, y=1, name="P")
        point_p = self._point("P")

        result = self._call(
            "create_circle_arc",
            point1_name="P",
            point2_x=0,
            point2_y=2,
            center_x=0,
            center_y=0,
            radius=2,
        )

        self.assertEqual(self._coords(point_p), (1.0, 1.0))
        arc = self._only("CircleArc")
        self.assertIsNot(arc.point1, point_p)
        self.assertAlmostEqual(math.hypot(arc.point1.x, arc.point1.y), 2.0)
        self.assertIn("P", str(result))

    def test_arc_reuses_an_existing_point_already_on_the_circle(self) -> None:
        self._call("create_point", x=5, y=0, name="Q")
        point_q = self._point("Q")

        result = self._call(
            "create_circle_arc",
            point1_x=5,
            point1_y=0,
            point2_x=0,
            point2_y=5,
            center_x=0,
            center_y=0,
            radius=5,
        )

        arc = self._only("CircleArc")
        self.assertIs(arc.point1, point_q)
        self.assertNotIn("not moved", str(result))

    def test_arc_still_snaps_new_reference_coordinates(self) -> None:
        self._call(
            "create_circle_arc",
            point1_x=3,
            point1_y=0,
            point2_x=0,
            point2_y=4,
            center_x=0,
            center_y=0,
            radius=5,
        )

        arc = self._only("CircleArc")
        self.assertAlmostEqual(arc.point1.x, 5.0)
        self.assertAlmostEqual(arc.point2.y, 5.0)
        self.assertEqual(self._points(), sorted([self._coords(arc.point1), self._coords(arc.point2)]))

    # ------------------------------------------------------------------
    # K19: polygon subtypes keep the given vertices
    # ------------------------------------------------------------------
    def test_square_subtype_keeps_exact_corners_and_order(self) -> None:
        square = self._polygon([(0, 0), (2, 0), (2, 2), (0, 2)], "quadrilateral", "ABCD", subtype="square")

        self.assertEqual(self._coords(self._point("A")), (0.0, 0.0))
        self.assertEqual(self._coords(self._point("B")), (2.0, 0.0))
        self.assertEqual(self._coords(self._point("C")), (2.0, 2.0))
        self.assertEqual(self._coords(self._point("D")), (0.0, 2.0))
        self.assertIn("square", square.get_type_names())

    def test_equilateral_subtype_keeps_given_coordinates(self) -> None:
        self._polygon([(0, 0), (4, 0), (2, EQUILATERAL_APEX_Y)], "triangle", "ABC", subtype="equilateral")

        self.assertEqual(self._coords(self._point("A")), (0.0, 0.0))
        self.assertEqual(self._coords(self._point("B")), (4.0, 0.0))
        self.assertEqual(self._coords(self._point("C")), (2.0, EQUILATERAL_APEX_Y))

    def test_subtype_mismatch_is_refused_without_side_effects(self) -> None:
        cases = [
            ("quadrilateral", "square", [(0, 0), (3, 0), (3, 2), (0, 2)]),
            ("quadrilateral", "rectangle", [(0, 0), (4, 0), (5, 2), (1, 2)]),
            ("quadrilateral", "rhombus", [(0, 0), (4, 0), (5, 2), (1, 2)]),
            ("quadrilateral", "parallelogram", [(0, 0), (4, 0), (4, 2), (1, 3)]),
            ("quadrilateral", "kite", [(0, 0), (4, 0), (4, 2), (1, 3)]),
            ("quadrilateral", "trapezoid", [(0, 0), (4, 0), (4, 2), (1, 3)]),
            ("quadrilateral", "isosceles_trapezoid", [(0, 0), (4, 0), (3, 2), (0, 2)]),
            ("quadrilateral", "right_trapezoid", [(0, 0), (4, 0), (3, 2), (1, 2)]),
            ("triangle", "equilateral", [(0, 0), (4, 0), (2, 3)]),
            ("triangle", "isosceles", [(0, 0), (4, 0), (1, 3)]),
            ("triangle", "scalene", [(0, 0), (4, 0), (2, 3)]),
            ("triangle", "right", [(0, 0), (4, 0), (1, 3)]),
            ("triangle", "right_isosceles", [(0, 0), (4, 0), (0, 3)]),
        ]
        for polygon_type, subtype, vertices in cases:
            with self.subTest(subtype=subtype):
                error = self._call_error(
                    "create_polygon",
                    vertices=[{"x": x, "y": y} for x, y in vertices],
                    polygon_type=polygon_type,
                    subtype=subtype,
                )
                self.assertIn(subtype.replace("_", " "), error)
                self.assertEqual(self._points(), [])
                self.assertEqual(self.canvas.undo_redo_manager.undo_stack, [])

    def test_subtype_match_is_accepted_for_each_quadrilateral_subtype(self) -> None:
        cases = [
            ("rectangle", [(0, 0), (4, 0), (4, 2), (0, 2)]),
            ("rhombus", [(0, 0), (3, 1), (4, 4), (1, 3)]),
            ("parallelogram", [(0, 0), (4, 0), (5, 3), (1, 3)]),
            ("kite", [(0, 0), (2, -1), (5, 0), (2, 1)]),
            ("trapezoid", [(0, 0), (6, 0), (4, 2), (1, 2)]),
            ("isosceles_trapezoid", [(0, 0), (6, 0), (5, 2), (1, 2)]),
            ("right_trapezoid", [(0, 0), (6, 0), (4, 2), (0, 2)]),
        ]
        for subtype, vertices in cases:
            with self.subTest(subtype=subtype):
                self._call("clear_canvas")
                self._call(
                    "create_polygon",
                    vertices=[{"x": x, "y": y} for x, y in vertices],
                    polygon_type="quadrilateral",
                    subtype=subtype,
                )
                self.assertEqual(self._points(), sorted((float(x), float(y)) for x, y in vertices))
