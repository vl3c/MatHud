"""Regression tests for naming, persistence and side-effect bugs K3, K5, K10, K15 and K27.

Tool calls run through ``ProcessFunctionCalls.get_results_traced`` (the model's tool path)
against a real canvas; workspace round trips go through ``get_canvas_state`` and the
workspace manager's restore, as a save followed by a load does.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List

from .test_tool_batch_results import TRIANGLE_VERTICES, _ToolBatchTestCase


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

    def test_existing_segment_is_reported_as_reused_and_adds_no_undo_entry(self) -> None:
        self.run_single("create_segment", x1=0, y1=0, x2=4, y2=0, name="AB")
        depth = self.undo_depth()

        _, traced = self.run_batch(("create_segment", {"x1": 0, "y1": 0, "x2": 4, "y2": 0, "color": "red"}))

        self.assertEqual(
            traced[0]["result"],
            "Used the existing Segment 'AB'; no new segment was created. The requested color was not applied.",
        )
        self.assertFalse(traced[0]["is_error"])
        self.assertEqual(self.undo_depth(), depth)

    def test_segment_name_is_available_again_after_undo(self) -> None:
        self.run_single("create_segment", x1=0, y1=0, x2=4, y2=0, name="PQ")
        self.run_single("undo")

        result = self.run_single("create_segment", x1=1, y1=1, x2=5, y2=1, name="PQ")

        self.assertEqual(result, "Created Segment 'PQ'.")
        self.assertEqual(self.canvas.get_point_by_name("P").x, 1)

    def test_triangle_name_is_available_again_after_undo(self) -> None:
        vertices = [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}]
        self.run_single("create_polygon", vertices=vertices, polygon_type="triangle", name="XYZ")
        self.run_single("undo")

        moved = [{"x": v["x"] + 10, "y": v["y"]} for v in vertices]
        result = self.run_single("create_polygon", vertices=moved, polygon_type="triangle", name="XYZ")

        self.assertEqual(result, "Created Triangle 'XYZ'.")
        self.assertEqual(sorted(self.snapshot()["Point"]), ["X(10.0, 0.0)", "Y(14.0, 0.0)", "Z(10.0, 3.0)"])

    def test_redefined_function_is_reported_as_updated_and_stays_undoable(self) -> None:
        self.run_single("draw_function", function_string="x^2", name="f", left_bound=-5, right_bound=5)
        depth = self.undo_depth()

        result = self.run_single("draw_function", function_string="x^3", name="f", left_bound=-5, right_bound=5)

        self.assertTrue(str(result).startswith("Updated the existing Function 'f' to "), result)
        self.assertEqual(self.undo_depth(), depth + 1)
        self.run_single("undo")
        self.assertIn("2", self.canvas.get_drawables_by_class_name("Function")[0].function_string)

    def test_circle_result_names_its_centre(self) -> None:
        result = self.run_single("create_circle", center_x=0, center_y=0, radius=2, name="O")

        self.assertEqual(result, "Created Circle 'O(2)' centred on 'O'.")

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

        self.assertEqual(result["point_names"], ["A", "B", "C", "D"])
        self.assertNotIn("created_point_names", result)
        self.assertNotIn("reused_point_names", result)
        self.assertNotIn("note", result)


class TestWorkspaceRoundTripKeepsStyles(_ToolBatchTestCase):
    """K5: save/load and undo keep colours, custom angle names and grid visibility."""

    def build_styled_scene(self) -> None:
        self.run_batch(
            ("create_point", {"x": -9, "y": 9, "name": "P", "color": "red"}),
            (
                "create_polygon",
                {"vertices": TRIANGLE_VERTICES, "polygon_type": "triangle", "name": "ABC", "color": "green"},
            ),
            ("create_circle", {"center_x": 10, "center_y": 10, "radius": 2, "color": "purple"}),
            (
                "create_ellipse",
                {
                    "center_x": -8,
                    "center_y": 5,
                    "radius_x": 3,
                    "radius_y": 1.5,
                    "rotation_angle": 30,
                    "color": "orange",
                },
            ),
            ("create_vector", {"origin_x": -5, "origin_y": -5, "tip_x": -2, "tip_y": -1, "color": "red"}),
            (
                "draw_function",
                {"function_string": "x^2", "name": "f", "left_bound": -2, "right_bound": 2, "color": "red"},
            ),
            ("create_segment", {"x1": 20, "y1": 0, "x2": 24, "y2": 0, "color": "teal"}),
            (
                "create_polygon",
                {
                    "vertices": [{"x": 30, "y": 0}, {"x": 34, "y": 0}, {"x": 34, "y": 2}, {"x": 30, "y": 2}],
                    "polygon_type": "rectangle",
                    "color": "maroon",
                },
            ),
            (
                "create_angle",
                {"vx": 0, "vy": 0, "p1x": 4, "p1y": 0, "p2x": 0, "p2y": 3, "angle_name": "alpha", "is_reflex": True},
            ),
        )

    def colors(self) -> Dict[str, str]:
        colors: Dict[str, str] = {}
        for bucket, drawables in self.canvas.drawable_manager.drawables._drawables.items():
            for drawable in drawables:
                colors[f"{bucket}:{drawable.name}"] = str(drawable.color)
        return colors

    def save_and_load(self) -> None:
        saved: Dict[str, Any] = json.loads(json.dumps(self.canvas.get_canvas_state()))
        self.canvas.clear()
        self.canvas.coordinate_system_manager.set_mode("cartesian", redraw=False)
        self.canvas.coordinate_system_manager.polar_grid.visible = True
        self.canvas.cartesian2axis.visible = True
        self.workspace_manager._restore_workspace_state(saved)

    def test_colors_survive_save_and_load(self) -> None:
        self.build_styled_scene()
        before = self.colors()
        self.assertEqual(before["Point:P"], "red")
        self.assertIn("purple", [color for key, color in before.items() if key.startswith("Circle:")])

        self.save_and_load()

        self.assertEqual(self.colors(), before)

    def test_piecewise_function_color_survives_save_and_load(self) -> None:
        pieces = [
            {"expression": "x", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x^2", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        self.canvas.draw_piecewise_function(pieces, name="p", color="green")

        self.save_and_load()

        self.assertEqual(self.canvas.get_drawables_by_class_name("PiecewiseFunction")[0].color, "green")

    def test_separately_recolored_polygon_edge_survives_save_and_load(self) -> None:
        self.run_single(
            "create_polygon", vertices=TRIANGLE_VERTICES, polygon_type="triangle", name="ABC", color="green"
        )
        self.run_single("update_segment", name="AB", new_color="red")
        before = self.colors()
        self.assertEqual(before["Segment:AB"], "red")

        self.save_and_load()

        self.assertEqual(self.colors(), before)

    def test_custom_angle_name_survives_save_and_load(self) -> None:
        self.build_styled_scene()

        self.save_and_load()

        angles = self.canvas.get_drawables_by_class_name("Angle")
        self.assertEqual([angle.name for angle in angles], ["alpha"])
        self.assertTrue(angles[0].is_reflex)

    def test_grid_visibility_survives_save_and_load(self) -> None:
        self.run_single("create_point", x=1, y=1)
        self.canvas.set_coordinate_system("polar")
        self.canvas.set_grid_visible(False)
        self.canvas.cartesian2axis.visible = False

        self.save_and_load()

        self.assertEqual(self.canvas.coordinate_system_manager.mode, "polar")
        self.assertFalse(self.canvas.coordinate_system_manager.polar_grid.visible)
        self.assertFalse(self.canvas.cartesian2axis.visible)

    def test_default_colors_are_not_written_to_the_state(self) -> None:
        self.run_single("create_segment", x1=0, y1=0, x2=4, y2=0)

        state = self.canvas.get_canvas_state()

        self.assertNotIn("color", state["Points"][0]["args"])
        self.assertNotIn("color", state["Segments"][0]["args"])
        self.assertEqual(state["coordinate_system"], {"mode": "cartesian"})

    def test_saves_without_colors_still_load_with_defaults(self) -> None:
        legacy: Dict[str, Any] = {
            "Points": [
                {"name": "A", "args": {"position": {"x": 0, "y": 0}}},
                {"name": "B", "args": {"position": {"x": 4, "y": 0}}},
            ],
            "Segments": [{"name": "AB", "args": {"p1": "A", "p2": "B"}}],
            "Circles": [{"name": "A(2)", "args": {"center": "A", "radius": 2}}],
            "coordinate_system": {"mode": "polar"},
        }

        self.workspace_manager._restore_workspace_state(legacy)

        colors = self.colors()
        self.assertEqual(colors["Point:A"], "black")
        self.assertEqual(colors["Segment:AB"], "black")
        self.assertEqual(colors["Circle:A(2)"], "black")
        self.assertTrue(self.canvas.coordinate_system_manager.polar_grid.visible)

    def test_undo_keeps_custom_angle_name_and_colors(self) -> None:
        self.build_styled_scene()
        before = self.colors()

        self.run_single("create_point", x=50, y=50)
        self.run_single("undo")

        self.assertEqual(self.colors(), before)
        self.assertEqual([a.name for a in self.canvas.get_drawables_by_class_name("Angle")], ["alpha"])

    def test_copies_for_undo_keep_custom_shape_names(self) -> None:
        self.build_styled_scene()
        self.run_single(
            "create_polygon",
            vertices=[{"x": 40, "y": 0}, {"x": 44, "y": 0}, {"x": 45, "y": 3}, {"x": 42, "y": 5}, {"x": 39, "y": 3}],
            polygon_type="pentagon",
        )
        renamed: List[Any] = []
        for class_name in ("Circle", "Ellipse", "Triangle", "Rectangle", "Pentagon"):
            drawable = self.canvas.get_drawables_by_class_name(class_name)[0]
            drawable.name = f"custom_{class_name}"
            renamed.append(drawable)

        for drawable in renamed:
            copied = deepcopy(drawable)
            self.assertEqual(copied.name, drawable.name)
            self.assertEqual(copied.color, drawable.color)
