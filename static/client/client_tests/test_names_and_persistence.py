"""Regression tests for naming, persistence and side-effect bugs K3, K5, K10, K15 and K27.

Tool calls run through ``ProcessFunctionCalls.get_results_traced`` (the model's tool path)
against a real canvas; workspace round trips go through ``get_canvas_state`` and the
workspace manager's restore, as a save followed by a load does.
"""

from __future__ import annotations

from typing import Any, Dict

from .test_tool_batch_results import _ToolBatchTestCase


class TestRequestedNamesAreReported(_ToolBatchTestCase):
    """K3: a requested name is used, or the result says which name was used instead."""

    def test_create_point_reports_the_name_it_used(self) -> None:
        result = self.run_single("create_point", x=1, y=1, name="A", color=None)

        self.assertEqual(result, "Created Point 'A'.")

    def test_taken_name_is_replaced_and_reported(self) -> None:
        self.run_single("create_point", x=0, y=0, name="A")

        result = self.run_single("create_point", x=7, y=7, name="A")

        created = self.canvas.get_point(7, 7)
        self.assertIsNotNone(created)
        self.assertNotEqual(created.name, "A")
        self.assertEqual(result, f"Created Point '{created.name}' instead of the requested name 'A'.")
        self.assertEqual(self.canvas.get_point_by_name("A").x, 0)

    def test_invalid_point_name_is_replaced_and_reported(self) -> None:
        result = self.run_single("create_point", x=1, y=2, name="Far")

        created = self.canvas.get_point(1, 2)
        self.assertEqual(result, f"Created Point '{created.name}' instead of the requested name 'Far'.")

    def test_name_is_available_again_after_undo(self) -> None:
        self.run_single("create_point", x=1, y=1, name="K")
        self.run_single("undo")

        result = self.run_single("create_point", x=3, y=3, name="K")

        self.assertEqual(self.canvas.get_point_by_name("K").x, 3)
        self.assertEqual(result, "Created Point 'K'.")

    def test_name_is_available_again_after_delete(self) -> None:
        self.run_single("create_point", x=1, y=1, name="K")
        self.run_single("delete_point", x=1, y=1)

        self.run_single("create_point", x=3, y=3, name="K")

        self.assertEqual(self.canvas.get_point_by_name("K").y, 3)

    def test_construction_with_a_taken_name_reports_the_real_name(self) -> None:
        self.build_triangle()

        result = self.run_single("construct_midpoint", segment_name="BC", name="A")

        midpoint = self.canvas.get_point(2, 1.5)
        self.assertIsNotNone(midpoint)
        self.assertEqual(result, f"Created Point '{midpoint.name}' instead of the requested name 'A'.")

    def test_segment_names_its_new_segment(self) -> None:
        result = self.run_single("create_segment", x1=0, y1=0, x2=4, y2=0, name="AB")

        self.assertEqual(result, "Created Segment 'AB'.")

    def test_existing_segment_is_reported_as_reused(self) -> None:
        self.run_single("create_segment", x1=0, y1=0, x2=4, y2=0, name="AB")

        result = self.run_single("create_segment", x1=0, y1=0, x2=4, y2=0)

        self.assertEqual(result, "Used the existing Segment 'AB'; no new segment was created.")

    def test_composite_result_names_each_part(self) -> None:
        self.run_single("create_segment", x1=0, y1=0, x2=4, y2=0, name="AB")
        self.run_single("create_point", x=2, y=3, name="C")

        result = self.run_single("construct_perpendicular_from_point", point_name="C", segment_name="AB")

        foot = self.canvas.get_point(2, 0)
        self.assertIsNotNone(foot)
        self.assertIn(f"foot: created Point '{foot.name}'", result)
        self.assertIn(f"segment: created Segment 'C{foot.name}'", result)

    def test_non_create_tools_keep_the_success_message(self) -> None:
        self.run_single("create_point", x=1, y=1, name="A")

        result = self.run_single("translate_object", name="A", x_offset=1, y_offset=0)

        self.assertEqual(result, "Call successful!")


class TestColoredAreaNullStyle(_ToolBatchTestCase):
    """K10: color and opacity null (strict-schema models send it) mean the defaults."""

    def test_create_colored_area_accepts_null_color_and_opacity(self) -> None:
        self.run_single("draw_function", function_string="x^2", name="f", left_bound=-5, right_bound=5)

        _, traced = self.run_batch(
            (
                "create_colored_area",
                {
                    "drawable1_name": "f",
                    "drawable2_name": None,
                    "left_bound": 1,
                    "right_bound": 2,
                    "color": None,
                    "opacity": None,
                },
            )
        )

        self.assertFalse(traced[0]["is_error"], traced[0]["result"])
        areas = self.canvas.get_drawables_by_class_name("FunctionsBoundedColoredArea")
        self.assertEqual(len(areas), 1)
        self.assertEqual(areas[0].color, "lightblue")
        self.assertEqual(areas[0].opacity, 0.3)

    def test_create_region_colored_area_accepts_null_style_and_resolution(self) -> None:
        self.run_single("create_circle", center_x=0, center_y=0, radius=5)
        self.build_triangle()
        circle_name = self.canvas.get_drawables_by_class_name("Circle")[0].name
        triangle_name = self.canvas.get_drawables_by_class_name("Triangle")[0].name
        args: Dict[str, Any] = {
            "expression": f"{circle_name} - {triangle_name}",
            "triangle_name": None,
            "rectangle_name": None,
            "polygon_segment_names": None,
            "circle_name": None,
            "ellipse_name": None,
            "chord_segment_name": None,
            "arc_clockwise": None,
            "resolution": None,
            "color": None,
            "opacity": None,
        }

        _, traced = self.run_batch(("create_region_colored_area", args))

        self.assertFalse(traced[0]["is_error"], traced[0]["result"])
        areas = self.canvas.get_drawables_by_class_name("ClosedShapeColoredArea")
        self.assertEqual(len(areas), 1)
        self.assertEqual(areas[0].color, "lightblue")


class TestSegmentAreaHasNoSideEffects(_ToolBatchTestCase):
    """K15: shading between two segments adds no points and splits no segment."""

    def test_area_between_segments_changes_nothing_else(self) -> None:
        self.run_single("create_segment", x1=0, y1=0, x2=4, y2=0)
        self.run_single("create_segment", x1=1, y1=2, x2=3, y2=2)
        before = self.snapshot()
        depth = self.undo_depth()

        _, traced = self.run_batch(
            ("create_colored_area", {"drawable1_name": "AB", "drawable2_name": "CD", "color": "pink", "opacity": 0.3})
        )

        self.assertFalse(traced[0]["is_error"], traced[0]["result"])
        after = self.snapshot()
        self.assertEqual(len(after.pop("SegmentsBoundedColoredArea", [])), 1)
        self.assertEqual(after, before)
        self.assertEqual(self.undo_depth(), depth + 1)


class TestRegressionReportsReusedPoints(_ToolBatchTestCase):
    """K27: fit_regression tells created data points from reused existing ones."""

    def fit(self) -> Dict[str, Any]:
        result = self.run_single(
            "fit_regression",
            name="fit1",
            x_data=[1, 2, 3, 4],
            y_data=[3, 5, 7, 9],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=True,
            point_color="red",
        )
        self.assertIsInstance(result, dict)
        return dict(result)

    def test_reused_point_is_listed_apart_from_created_points(self) -> None:
        self.run_single("create_point", x=1, y=3, name="A")

        result = self.fit()

        self.assertEqual(result["point_names"], ["A", "B", "C", "D"])
        self.assertEqual(result["created_point_names"], ["B", "C", "D"])
        self.assertEqual(result["reused_point_names"], ["A"])
        self.assertIn("already existed", result["note"])
        self.assertEqual(self.canvas.get_point_by_name("A").color, "black")
        self.assertEqual(self.canvas.get_point_by_name("B").color, "red")

    def test_without_existing_points_nothing_is_reported_as_reused(self) -> None:
        result = self.fit()

        self.assertEqual(result["created_point_names"], result["point_names"])
        self.assertNotIn("reused_point_names", result)
        self.assertNotIn("note", result)
