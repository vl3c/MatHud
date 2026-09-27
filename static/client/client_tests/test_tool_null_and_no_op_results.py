"""Regression tests for tool arguments sent as null and for truthful results (fix batch 7).

- ``update_circle_arc`` always failed: the canvas passed endpoint arguments the arc manager
  does not take.
- Strict-schema models send ``null`` for every optional argument; each create and update
  tool must treat it as "use the default" or "keep the current value" (the K10 pattern).
- An update, translation or label change that changes nothing says so (the K26 family).

Every tool call runs through ``ProcessFunctionCalls.get_results_traced``, the same path a
model tool batch takes, against a real canvas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from constants import successful_call_message
from expression_validator import ExpressionValidator

from .test_tool_batch_results import _ToolBatchTestCase

NullCall = Tuple[str, Dict[str, Any]]


class _NullToolTestCase(_ToolBatchTestCase):
    """Helpers: run a call, and read the newest drawable of a bucket."""

    def call(self, tool: str, **args: Any) -> Dict[str, Any]:
        _, traced = self.run_batch((tool, args))
        return traced[0]

    def newest(self, bucket: str) -> Any:
        drawables = getattr(self.canvas.drawable_manager.drawables, bucket)
        self.assertTrue(drawables, f"no {bucket}")
        return drawables[-1]

    def newest_name(self, bucket: str) -> str:
        return str(self.newest(bucket).name)


class TestCircleArcUpdate(_NullToolTestCase):
    """update_circle_arc passed endpoint arguments the arc manager does not accept, so it always failed."""

    def make_arc(self, use_major_arc: bool = False) -> Any:
        traced = self.call(
            "create_circle_arc",
            point1_x=5,
            point1_y=0,
            point2_x=0,
            point2_y=5,
            point1_name=None,
            point2_name=None,
            point3_x=None,
            point3_y=None,
            point3_name=None,
            center_point_choice=None,
            circle_name=None,
            center_x=0,
            center_y=0,
            radius=5,
            use_major_arc=use_major_arc,
            arc_name=None,
            color=None,
        )
        self.assertFalse(traced["is_error"], traced["result"])
        return self.newest("CircleArcs")

    def test_update_color(self) -> None:
        arc = self.make_arc()
        depth = self.undo_depth()

        traced = self.call("update_circle_arc", name=arc.name, new_color="red", use_major_arc=None)

        self.assertFalse(traced["is_error"], traced["result"])
        self.assertEqual(arc.color, "red")
        self.assertEqual(self.undo_depth(), depth + 1)

    def test_switch_to_the_major_arc_and_undo(self) -> None:
        arc = self.make_arc()

        traced = self.call("update_circle_arc", name=arc.name, new_color=None, use_major_arc=True)

        self.assertFalse(traced["is_error"], traced["result"])
        self.assertTrue(arc.use_major_arc)
        self.run_single("undo")
        self.assertFalse(self.newest("CircleArcs").use_major_arc)

    def test_the_sweep_already_selected_is_a_no_op(self) -> None:
        arc = self.make_arc()
        depth = self.undo_depth()

        traced = self.call("update_circle_arc", name=arc.name, new_color=None, use_major_arc=False)

        self.assertFalse(traced["is_error"], traced["result"])
        self.assertEqual(
            traced["result"], f"Circle arc '{arc.name}' already has the minor arc selected; nothing changed."
        )
        self.assertEqual(self.undo_depth(), depth)

    def test_update_with_nothing_to_change_is_an_error(self) -> None:
        arc = self.make_arc()

        traced = self.call("update_circle_arc", name=arc.name, new_color=None, use_major_arc=None)

        self.assertTrue(traced["is_error"], traced["result"])
        self.assertIn("at least one property", str(traced["result"]))

    def test_endpoint_arguments_are_not_accepted(self) -> None:
        """The schema offers no endpoint arguments; the canvas no longer takes them either."""
        arc = self.make_arc()

        with self.assertRaises(TypeError):
            self.canvas.update_circle_arc(arc.name, point1_x=1, point1_y=2)  # type: ignore[call-arg]


class TestNullArguments(_NullToolTestCase):
    """A null optional argument means the default (create) or "keep" (update), never a crash."""

    def assert_calls_succeed(self, calls: List[NullCall]) -> None:
        failures = []
        for tool, args in calls:
            traced = self.call(tool, **args)
            if traced["is_error"] or "NoneType" in str(traced["result"]):
                failures.append(f"{tool}: {traced['result']}")
        self.assertEqual(failures, [])

    def test_create_ellipse_with_null_rotation(self) -> None:
        traced = self.call(
            "create_ellipse",
            center_x=1,
            center_y=2,
            radius_x=3,
            radius_y=1.5,
            rotation_angle=None,
            color=None,
            name=None,
        )

        self.assertFalse(traced["is_error"], traced["result"])
        ellipse = self.newest("Ellipses")
        self.assertEqual(ellipse.rotation_angle, 0)
        self.assertEqual((ellipse.radius_x, ellipse.radius_y), (3, 1.5))

    def test_every_create_tool_accepts_null_optionals(self) -> None:
        self.assert_calls_succeed(
            [
                ("create_point", {"x": 1, "y": 1, "color": None, "name": None}),
                (
                    "create_segment",
                    {
                        "x1": 0,
                        "y1": 0,
                        "x2": 4,
                        "y2": 0,
                        "color": None,
                        "name": None,
                        "label_text": None,
                        "label_visible": None,
                    },
                ),
                ("create_vector", {"origin_x": 0, "origin_y": 1, "tip_x": 3, "tip_y": 1, "color": None, "name": None}),
                (
                    "create_polygon",
                    {
                        "vertices": [{"x": 10, "y": 10}, {"x": 14, "y": 10}, {"x": 10, "y": 13}],
                        "polygon_type": None,
                        "color": None,
                        "name": None,
                        "subtype": None,
                    },
                ),
                ("create_circle", {"center_x": 20, "center_y": 20, "radius": 2, "color": None, "name": None}),
                (
                    "create_ellipse",
                    {
                        "center_x": -10,
                        "center_y": -10,
                        "radius_x": 3,
                        "radius_y": 1.5,
                        "rotation_angle": None,
                        "color": None,
                        "name": None,
                    },
                ),
                (
                    "create_label",
                    {
                        "x": 5,
                        "y": 5,
                        "text": "hi",
                        "name": None,
                        "color": None,
                        "font_size": None,
                        "rotation_degrees": None,
                    },
                ),
                (
                    "draw_function",
                    {
                        "function_string": "x^2",
                        "name": None,
                        "left_bound": None,
                        "right_bound": None,
                        "color": None,
                        "undefined_at": None,
                    },
                ),
                (
                    "draw_parametric_function",
                    {
                        "x_expression": "cos(t)",
                        "y_expression": "sin(t)",
                        "name": None,
                        "t_min": None,
                        "t_max": None,
                        "color": None,
                    },
                ),
                (
                    "create_angle",
                    {
                        "vx": 30,
                        "vy": 0,
                        "p1x": 34,
                        "p1y": 0,
                        "p2x": 30,
                        "p2y": 4,
                        "color": None,
                        "angle_name": None,
                        "is_reflex": None,
                    },
                ),
                (
                    "plot_bars",
                    {
                        "name": None,
                        "values": [1, 2],
                        "labels_below": ["a", "b"],
                        "labels_above": None,
                        "bar_spacing": None,
                        "bar_width": None,
                        "stroke_color": None,
                        "fill_color": None,
                        "fill_opacity": None,
                        "x_start": None,
                        "y_base": None,
                    },
                ),
                (
                    "plot_distribution",
                    {
                        "name": None,
                        "representation": "continuous",
                        "distribution_type": "normal",
                        "distribution_params": None,
                        "plot_bounds": None,
                        "shade_bounds": None,
                        "curve_color": None,
                        "fill_color": None,
                        "fill_opacity": None,
                        "bar_count": None,
                    },
                ),
                (
                    "plot_distribution",
                    {
                        "name": None,
                        "representation": "discrete",
                        "distribution_type": "normal",
                        "distribution_params": None,
                        "plot_bounds": None,
                        "shade_bounds": None,
                        "curve_color": None,
                        "fill_color": None,
                        "fill_opacity": None,
                        "bar_count": None,
                    },
                ),
                (
                    "fit_regression",
                    {
                        "name": None,
                        "x_data": [1, 2, 3, 4],
                        "y_data": [2, 4, 6, 8.5],
                        "model_type": "linear",
                        "degree": None,
                        "plot_bounds": None,
                        "curve_color": None,
                        "show_points": None,
                        "point_color": None,
                    },
                ),
            ]
        )

    def test_constructions_accept_null_optionals(self) -> None:
        self.run_single(
            "create_polygon",
            vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}],
            polygon_type="triangle",
            name="ABC",
        )
        self.run_single("draw_function", function_string="x^2", name="f")
        self.assert_calls_succeed(
            [
                (
                    "construct_midpoint",
                    {"p1_name": None, "p2_name": None, "segment_name": "AB", "name": None, "color": None},
                ),
                (
                    "construct_perpendicular_bisector",
                    {"segment_name": "AB", "length": None, "name": None, "color": None},
                ),
                (
                    "construct_perpendicular_from_point",
                    {"point_name": "C", "segment_name": "AB", "name": None, "color": None},
                ),
                (
                    "construct_angle_bisector",
                    {
                        "vertex_name": "A",
                        "p1_name": "B",
                        "p2_name": "C",
                        "angle_name": None,
                        "length": None,
                        "name": None,
                        "color": None,
                    },
                ),
                (
                    "construct_parallel_line",
                    {"segment_name": "AB", "point_name": "C", "length": None, "name": None, "color": None},
                ),
                (
                    "construct_circumcircle",
                    {
                        "triangle_name": "ABC",
                        "p1_name": None,
                        "p2_name": None,
                        "p3_name": None,
                        "name": None,
                        "color": None,
                    },
                ),
                ("construct_incircle", {"triangle_name": "ABC", "name": None, "color": None}),
                ("draw_tangent_line", {"curve_name": "f", "parameter": 1, "name": None, "length": None, "color": None}),
                ("draw_normal_line", {"curve_name": "f", "parameter": 1, "name": None, "length": None, "color": None}),
                (
                    "create_colored_area",
                    {
                        "drawable1_name": "f",
                        "drawable2_name": None,
                        "left_bound": None,
                        "right_bound": None,
                        "color": None,
                        "opacity": None,
                    },
                ),
                ("rotate_object", {"name": "ABC", "angle": 10, "center_x": None, "center_y": None}),
                (
                    "reflect_object",
                    {
                        "name": "ABC",
                        "axis": "x_axis",
                        "line_a": None,
                        "line_b": None,
                        "line_c": None,
                        "segment_name": None,
                    },
                ),
            ]
        )

    def test_areas_arcs_and_piecewise_accept_null_optionals(self) -> None:
        self.run_single("create_circle", center_x=0, center_y=0, radius=5)
        circle = self.newest_name("Circles")
        self.run_single("draw_function", function_string="x^2", name="f")
        self.assert_calls_succeed(
            [
                (
                    "create_circle_arc",
                    {
                        "point1_x": 5,
                        "point1_y": 0,
                        "point2_x": 0,
                        "point2_y": 5,
                        "point1_name": None,
                        "point2_name": None,
                        "point3_x": None,
                        "point3_y": None,
                        "point3_name": None,
                        "center_point_choice": None,
                        "circle_name": circle,
                        "center_x": None,
                        "center_y": None,
                        "radius": None,
                        "use_major_arc": False,
                        "arc_name": None,
                        "color": None,
                    },
                ),
                (
                    "create_region_colored_area",
                    {
                        "expression": None,
                        "triangle_name": None,
                        "rectangle_name": None,
                        "polygon_segment_names": None,
                        "circle_name": circle,
                        "ellipse_name": None,
                        "chord_segment_name": None,
                        "arc_clockwise": None,
                        "resolution": None,
                        "color": None,
                        "opacity": None,
                    },
                ),
                (
                    "draw_piecewise_function",
                    {
                        "pieces": [
                            {
                                "expression": "x",
                                "left": None,
                                "right": 0,
                                "left_inclusive": False,
                                "right_inclusive": False,
                                "undefined_at": None,
                            },
                            {
                                "expression": "x^2",
                                "left": 0,
                                "right": None,
                                "left_inclusive": True,
                                "right_inclusive": False,
                                "undefined_at": None,
                            },
                        ],
                        "name": None,
                        "color": None,
                    },
                ),
            ]
        )
        # update_colored_area edits areas bounded by functions or segments, not region areas
        self.run_single("create_colored_area", drawable1_name="f", left_bound=0, right_bound=1)
        area = self.newest_name("FunctionsBoundedColoredAreas")
        piecewise = self.newest_name("PiecewiseFunctions")
        self.assert_calls_succeed(
            [
                (
                    "update_colored_area",
                    {
                        "name": area,
                        "new_color": "red",
                        "new_opacity": None,
                        "new_left_bound": None,
                        "new_right_bound": None,
                    },
                ),
                ("update_piecewise_function", {"name": piecewise, "new_color": "red"}),
            ]
        )

    def test_updates_keep_the_fields_sent_as_null(self) -> None:
        self.run_single("create_point", x=1, y=1, name="P")
        self.run_single("create_segment", x1=0, y1=0, x2=4, y2=0, name="AB", label_text="s", label_visible=True)
        self.run_single("create_ellipse", center_x=-10, center_y=-10, radius_x=3, radius_y=1.5, rotation_angle=20)
        self.run_single("create_label", x=5, y=5, text="hi", name="L1")
        self.run_single("draw_function", function_string="x^2", name="f", left_bound=-2, right_bound=2)
        self.run_single("draw_parametric_function", x_expression="cos(t)", y_expression="sin(t)", name="p")
        ellipse = self.newest_name("Ellipses")
        self.assert_calls_succeed(
            [
                (
                    "update_point",
                    {"point_name": "P", "new_name": None, "new_x": None, "new_y": None, "new_color": "red"},
                ),
                (
                    "update_segment",
                    {"name": "AB", "new_color": "red", "new_label_text": None, "new_label_visible": None},
                ),
                ("update_segment", {"name": "AB", "new_color": None, "new_label_text": "t", "new_label_visible": None}),
                (
                    "update_ellipse",
                    {
                        "name": ellipse,
                        "new_color": "red",
                        "new_radius_x": None,
                        "new_radius_y": None,
                        "new_rotation_angle": None,
                        "new_center_x": None,
                        "new_center_y": None,
                    },
                ),
                (
                    "update_ellipse",
                    {
                        "name": ellipse,
                        "new_color": None,
                        "new_radius_x": None,
                        "new_radius_y": None,
                        "new_rotation_angle": 45,
                        "new_center_x": None,
                        "new_center_y": None,
                    },
                ),
                (
                    "update_label",
                    {
                        "name": "L1",
                        "new_text": "ho",
                        "new_x": None,
                        "new_y": None,
                        "new_color": None,
                        "new_font_size": None,
                        "new_rotation_degrees": None,
                    },
                ),
                ("update_function", {"name": "f", "new_color": "red", "new_left_bound": None, "new_right_bound": None}),
                ("update_parametric_function", {"name": "p", "new_color": "red", "new_t_min": None, "new_t_max": None}),
            ]
        )
        self.assertEqual(self.newest("Ellipses").rotation_angle, 45)
        self.assertEqual(self.canvas.get_segment_by_name("AB").label.text, "t")
