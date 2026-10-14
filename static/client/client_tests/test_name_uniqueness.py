"""Drawable names are unique across types, and circles can carry a custom name."""

from __future__ import annotations

import copy
import json
import unittest
from typing import Any, List

from canvas import Canvas
from drawables.circle import Circle
from drawables.point import Point
from managers.drawables_container import DrawablesContainer
from name_generator.base import with_unique_suffix
from utils.area_expression_evaluator import AreaExpressionEvaluator
from workspace_manager import WorkspaceManager


class TestUniqueSuffix(unittest.TestCase):
    def test_first_free_suffix(self) -> None:
        self.assertEqual(with_unique_suffix("G", set()), "G")
        self.assertEqual(with_unique_suffix("G", {"G"}), "G_1")
        self.assertEqual(with_unique_suffix("G", {"G", "G_1"}), "G_2")

    def test_container_unique_name_ignores_the_excluded_drawable(self) -> None:
        container = DrawablesContainer()
        point = Point(0.0, 0.0, name="A")
        container.add(point)
        self.assertEqual(container.unique_name("A"), "A_1")
        self.assertEqual(container.unique_name("A", exclude=point), "A")


class TestCircleNameModel(unittest.TestCase):
    def _circle(self, radius: float = 3.0, name: str = "") -> Circle:
        return Circle(Point(0.0, 0.0, name="A"), radius, name=name)

    def test_default_and_custom_names(self) -> None:
        default = self._circle()
        self.assertEqual(default.name, "A(3)")
        self.assertFalse(default.has_custom_name)
        custom = self._circle(name="orbit")
        self.assertEqual(custom.name, "orbit")
        self.assertTrue(custom.has_custom_name)

    def test_is_default_name_accepts_numeric_suffix_only(self) -> None:
        circle = self._circle()
        self.assertTrue(circle.is_default_name("A(3)"))
        self.assertTrue(circle.is_default_name("A(3)_2"))
        self.assertFalse(circle.is_default_name("A(3)_x"))
        self.assertFalse(circle.is_default_name("A(4)"))

    def test_regenerate_name_avoids_taken_names_and_keeps_custom(self) -> None:
        circle = self._circle()
        circle.radius = 4.0
        circle.regenerate_name({"A(4)"})
        self.assertEqual(circle.name, "A(4)_1")
        custom = self._circle(name="orbit")
        custom.scale(2, 2, 0, 0)
        custom.translate(1, 1)
        custom.regenerate_name({"orbit"})
        self.assertEqual(custom.name, "orbit")

    def test_deepcopy_keeps_custom_flag(self) -> None:
        copied = copy.deepcopy(self._circle(name="orbit"))
        self.assertEqual(copied.name, "orbit")
        self.assertTrue(copied.has_custom_name)


class TestNamesOnCanvas(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.drawables = self.canvas.drawable_manager.drawables

    def _all_names(self) -> List[str]:
        return [drawable.name for drawable in self.drawables.get_all()]

    def _assert_unique(self) -> None:
        names = self._all_names()
        self.assertEqual(len(names), len(set(names)), f"duplicate names in {sorted(names)}")

    def _reload_workspace(self) -> None:
        manager = WorkspaceManager(self.canvas)
        saved = json.loads(json.dumps(manager._snapshot_persistable_canvas_state()))
        manager._restore_workspace_state(saved)

    def _graph(self, name: str, vertex_names: str) -> Any:
        return self.canvas.generate_graph(
            name=name,
            graph_type="graph",
            vertices=[{"name": v, "x": 10.0 * i, "y": 0.0} for i, v in enumerate(vertex_names)],
            edges=[{"source": 0, "target": 1}],
            adjacency_matrix=None,
            directed=None,
            root=None,
            layout=None,
            placement_box=None,
            metadata=None,
        )

    def _circle(self, name: str) -> Any:
        circle = self.canvas.get_circle_by_name(name)
        self.assertIsNotNone(circle, f"no circle named {name!r}")
        return circle

    # Names across types

    def test_function_and_label_avoid_an_ellipse_name(self) -> None:
        self.canvas.create_ellipse(0, 0, 3, 2, name="orbit", extra_graphics=False)
        function = self.canvas.draw_function("x^2", name="orbit")
        label = self.canvas.create_label(1, 1, "hi", name="orbit")
        self.assertNotEqual(function.name, "orbit")
        self.assertNotEqual(label.name, "orbit")
        self.assertIsNotNone(self.canvas.get_ellipse_by_name("orbit"))
        self._assert_unique()

    def test_graph_and_point_never_share_a_name(self) -> None:
        self.canvas.create_point(50, 50, name="G", extra_graphics=False)
        graph = self._graph("G", "AB")
        self.assertEqual(graph["name"], "G_1")
        self._graph("H", "CD")
        point = self.canvas.create_point(60, 60, extra_graphics=False)
        self.assertNotIn(point.name, {"G", "H"})
        self._assert_unique()

    def test_graph_vertex_name_in_use_is_not_taken(self) -> None:
        self._graph("K", "AB")
        self._graph("M", "KL")
        self._assert_unique()

    def test_point_rename_to_a_name_in_use_is_refused(self) -> None:
        self.canvas.draw_function("x", name="f")
        self.canvas.create_point(1, 2, name="A", extra_graphics=False)
        self._graph("Q", "CD")
        with self.assertRaises(ValueError):
            self.canvas.update_point("A", new_name="Q")
        self.assertIsNotNone(self.canvas.get_point_by_name("A"))

    def test_parametric_function_and_angle_names_get_suffixes(self) -> None:
        self.canvas.draw_function("x", name="curve")
        parametric = self.canvas.draw_parametric_function("cos(t)", "sin(t)", name="curve")
        self.assertEqual(parametric.name, "curve_1")
        angle = self.canvas.create_angle(0, 0, 5, 0, 0, 5, angle_name="curve", extra_graphics=False)
        self.assertEqual(angle.name, "curve_2")
        self._assert_unique()

    def test_reload_keeps_a_custom_ellipse_name_a_function_asked_for(self) -> None:
        self.canvas.create_ellipse(0, 0, 3, 2, name="E1", extra_graphics=False)
        self.canvas.draw_function("x^2", name="E1")
        self.canvas.create_region_colored_area(ellipse_name="E1")
        self._reload_workspace()
        self.assertIsNotNone(self.canvas.get_ellipse_by_name("E1"))
        areas = self.drawables.ClosedShapeColoredAreas
        self.assertEqual([area.get_state()["args"].get("ellipse") for area in areas], ["E1"])
        self._assert_unique()

    # Circle names

    def test_point_name_names_the_centre(self) -> None:
        circle = self.canvas.create_circle(1, 1, 3, name="B", extra_graphics=False)
        self.assertEqual(circle.center.name, "B")
        self.assertEqual(circle.name, "B(3)")
        self.assertFalse(circle.has_custom_name)

    def test_requested_name_names_the_circle(self) -> None:
        circle = self.canvas.create_circle(1, 1, 3, name="c1", extra_graphics=False)
        self.assertEqual(circle.name, "c1")
        self.assertEqual(circle.center.name, "C")
        self.assertTrue(circle.has_custom_name)
        self.assertIs(self.canvas.get_circle_by_name("c1"), circle)

    def test_default_form_name_is_not_custom(self) -> None:
        circle = self.canvas.create_circle(1, 1, 3, name="D(3)", extra_graphics=False)
        self.assertEqual(circle.name, "D(3)")
        self.assertFalse(circle.has_custom_name)

    def test_custom_name_in_use_gets_suffix(self) -> None:
        self.canvas.draw_function("x^2", name="f1")
        circle = self.canvas.create_circle(1, 1, 3, name="f1", extra_graphics=False)
        self.assertEqual(circle.name, "f1_1")

    def test_unreadable_custom_name_is_refused(self) -> None:
        for bad_name in ("my circle", "c-1", "2nd"):
            with self.assertRaises(ValueError):
                self.canvas.create_circle(1, 1, 3, name=bad_name, extra_graphics=False)
        self.assertEqual(list(self.drawables.Circles), [])

    def test_custom_name_survives_transforms_and_undo(self) -> None:
        circle = self.canvas.create_circle(0, 0, 3, name="orbit", extra_graphics=False)
        self.canvas.scale_object("orbit", 2, 2, 0, 0)
        self.canvas.translate_object("orbit", 1, 1)
        self.assertEqual(circle.name, "orbit")
        self.assertEqual(circle.radius, 6)
        self.canvas.undo()
        restored = self._circle("orbit")
        self.assertTrue(restored.has_custom_name)
        self.canvas.redo()
        self.assertEqual(self._circle("orbit").radius, 6)

    def test_scaling_into_a_name_in_use_gets_suffix(self) -> None:
        self.canvas.create_circle(0, 0, 6, name="A", extra_graphics=False)
        small = self.canvas.create_circle(0, 0, 3, extra_graphics=False)
        self.canvas.scale_object(small.name, 2, 2, 0, 0)
        self.assertEqual(small.name, "A(6)_1")
        self._assert_unique()

    def test_workspace_reload_keeps_custom_suffixed_and_twin_circles(self) -> None:
        self.canvas.create_circle(0, 0, 6, name="A", extra_graphics=False)
        small = self.canvas.create_circle(0, 0, 3, extra_graphics=False)
        self.canvas.scale_object(small.name, 2, 2, 0, 0)
        self.canvas.create_circle(20, 20, 2, name="orbit", extra_graphics=False)
        self.canvas.create_region_colored_area(circle_name="orbit")
        self._reload_workspace()
        names = sorted(circle.name for circle in self.drawables.Circles)
        self.assertEqual(names, ["A(6)", "A(6)_1", "orbit"])
        self.assertTrue(self._circle("orbit").has_custom_name)
        self.assertFalse(self._circle("A(6)_1").has_custom_name)
        areas = self.drawables.ClosedShapeColoredAreas
        self.assertEqual([area.get_state()["args"].get("circle") for area in areas], ["orbit"])

    def test_area_expression_reads_custom_circle_name(self) -> None:
        self.canvas.create_circle(0, 0, 2, name="disk", extra_graphics=False)
        result = AreaExpressionEvaluator.evaluate("disk", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 3.14159265 * 4, places=2)


__all__ = [
    "TestUniqueSuffix",
    "TestCircleNameModel",
    "TestNamesOnCanvas",
]
