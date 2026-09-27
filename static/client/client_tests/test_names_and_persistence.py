"""Regression tests for naming, persistence and side-effect bugs K3, K5, K10, K15 and K27.

Tool calls run through ``ProcessFunctionCalls.get_results_traced`` (the model's tool path)
against a real canvas; workspace round trips go through ``get_canvas_state`` and the
workspace manager's restore, as a save followed by a load does.
"""

from __future__ import annotations

from typing import Any, Dict

from .test_tool_batch_results import _ToolBatchTestCase


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
