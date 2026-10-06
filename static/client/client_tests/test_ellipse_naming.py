"""Ellipse naming: requested names, unique names, and names kept through edits, undo and reload."""

from __future__ import annotations

import copy
import json
import unittest
from typing import Any, List

from canvas import Canvas
from drawables.ellipse import Ellipse
from drawables.point import Point
from utils.area_expression_evaluator import AreaExpressionEvaluator
from workspace_manager import WorkspaceManager


class TestEllipseNameModel(unittest.TestCase):
    def _ellipse(self, radius_x: float = 3.0, radius_y: float = 1.5, name: str = "") -> Ellipse:
        return Ellipse(Point(0.0, 0.0, name="A"), radius_x, radius_y, name=name)

    def test_default_name_comes_from_centre_and_radii(self) -> None:
        ellipse = self._ellipse()
        self.assertEqual(ellipse.name, "A(3, 1.5)")
        self.assertFalse(ellipse.has_custom_name)

    def test_custom_name_is_kept(self) -> None:
        ellipse = self._ellipse(name="orbit")
        self.assertEqual(ellipse.name, "orbit")
        self.assertTrue(ellipse.has_custom_name)

    def test_is_default_name_accepts_numeric_suffix_only(self) -> None:
        ellipse = self._ellipse()
        self.assertTrue(ellipse.is_default_name("A(3, 1.5)"))
        self.assertTrue(ellipse.is_default_name("A(3, 1.5)_2"))
        self.assertFalse(ellipse.is_default_name("A(3, 1.5)_"))
        self.assertFalse(ellipse.is_default_name("A(3, 1.5)_x"))
        self.assertFalse(ellipse.is_default_name("A(4, 1.5)"))
        self.assertFalse(ellipse.is_default_name("orbit"))

    def test_unique_name_takes_first_free_suffix(self) -> None:
        self.assertEqual(Ellipse.unique_name("A(3, 1.5)", set()), "A(3, 1.5)")
        self.assertEqual(Ellipse.unique_name("A(3, 1.5)", {"A(3, 1.5)"}), "A(3, 1.5)_1")
        self.assertEqual(Ellipse.unique_name("A(3, 1.5)", {"A(3, 1.5)", "A(3, 1.5)_1"}), "A(3, 1.5)_2")

    def test_regenerate_name_avoids_taken_names(self) -> None:
        ellipse = self._ellipse()
        ellipse.radius_x = 4.0
        ellipse.regenerate_name({"A(4, 1.5)"})
        self.assertEqual(ellipse.name, "A(4, 1.5)_1")

    def test_regenerate_name_keeps_a_free_suffix(self) -> None:
        ellipse = self._ellipse()
        ellipse.name = "A(3, 1.5)_1"
        ellipse.regenerate_name(set())
        self.assertEqual(ellipse.name, "A(3, 1.5)_1")

    def test_regenerate_name_follows_new_radii(self) -> None:
        ellipse = self._ellipse()
        ellipse.name = "A(3, 1.5)_1"
        ellipse.scale(2, 2, 0, 0)
        self.assertEqual(ellipse.name, "A(6, 3)")

    def test_transforms_keep_custom_name(self) -> None:
        ellipse = self._ellipse(name="orbit")
        ellipse.scale(2, 2, 0, 0)
        ellipse.translate(1, 1)
        ellipse.rotate_around(30, 0, 0)
        ellipse.update_radius_y(5)
        ellipse.regenerate_name({"orbit_1"})
        self.assertEqual(ellipse.name, "orbit")

    def test_deepcopy_keeps_name_and_custom_flag(self) -> None:
        ellipse = self._ellipse(name="orbit")
        copied = copy.deepcopy(ellipse)
        self.assertEqual(copied.name, "orbit")
        self.assertTrue(copied.has_custom_name)
        suffixed = self._ellipse()
        suffixed.name = "A(3, 1.5)_1"
        copied_suffixed = copy.deepcopy(suffixed)
        self.assertEqual(copied_suffixed.name, "A(3, 1.5)_1")
        self.assertFalse(copied_suffixed.has_custom_name)

    def test_area_expression_reads_suffixed_ellipse_name(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("A(3, 1.5)_1 - A(3, 1.5)")
        self.assertEqual(tokens, ["A(3, 1.5)_1", "-", "A(3, 1.5)"])


class TestEllipseNamingOnCanvas(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)

    def _ellipse_names(self) -> List[str]:
        return sorted(ellipse.name for ellipse in self.canvas.drawable_manager.drawables.Ellipses)

    def _ellipse(self, name: str) -> Any:
        ellipse = self.canvas.get_ellipse_by_name(name)
        self.assertIsNotNone(ellipse, f"no ellipse named {name!r}")
        return ellipse

    def _two_ellipses_with_one_default_name(self) -> Any:
        """Two ellipses on centre A that end up with the same radii; returns the second."""
        self.canvas.create_ellipse(0, 0, 3, 2, name="A", extra_graphics=False)
        second = self.canvas.create_ellipse(0, 0, 4, 2, rotation_angle=30, extra_graphics=False)
        self.canvas.update_ellipse(second.name, new_radius_x=3)
        return second

    def test_point_name_names_the_centre(self) -> None:
        ellipse = self.canvas.create_ellipse(1, 1, 3, 2, name="B", extra_graphics=False)
        self.assertEqual(ellipse.center.name, "B")
        self.assertEqual(ellipse.name, "B(3, 2)")
        self.assertFalse(ellipse.has_custom_name)

    def test_requested_name_names_the_ellipse(self) -> None:
        ellipse = self.canvas.create_ellipse(1, 1, 3, 2, name="E1", extra_graphics=False)
        self.assertEqual(ellipse.name, "E1")
        self.assertEqual(ellipse.center.name, "E")
        self.assertTrue(ellipse.has_custom_name)
        self.assertIs(self.canvas.get_ellipse_by_name("E1"), ellipse)

    def test_default_form_name_is_not_custom(self) -> None:
        ellipse = self.canvas.create_ellipse(1, 1, 3, 2, name="C(3, 2)", extra_graphics=False)
        self.assertEqual(ellipse.name, "C(3, 2)")
        self.assertFalse(ellipse.has_custom_name)
        mismatched = self.canvas.create_ellipse(5, 5, 3, 2, name="D(9, 9)", extra_graphics=False)
        self.assertEqual(mismatched.name, "D(3, 2)")
        self.assertFalse(mismatched.has_custom_name)

    def test_requested_name_in_use_gets_suffix(self) -> None:
        self.canvas.create_segment(10, 10, 12, 12, name="PQ", extra_graphics=False)
        self.canvas.create_ellipse(1, 1, 3, 2, name="orbit", extra_graphics=False)
        second = self.canvas.create_ellipse(5, 5, 3, 2, name="orbit", extra_graphics=False)
        third = self.canvas.create_ellipse(-5, -5, 3, 2, name="PQ", extra_graphics=False)
        self.assertEqual(second.name, "orbit_1")
        self.assertEqual(third.name, "PQ_1")

    def test_colliding_default_name_gets_suffix(self) -> None:
        second = self._two_ellipses_with_one_default_name()
        self.assertEqual(second.name, "A(3, 2)_1")
        self.assertEqual(self._ellipse_names(), ["A(3, 2)", "A(3, 2)_1"])

    def test_name_tools_reach_the_second_ellipse(self) -> None:
        second = self._two_ellipses_with_one_default_name()
        self.canvas.rotate_object("A(3, 2)_1", 15)
        self.assertAlmostEqual(second.rotation_angle, 45)
        self.assertAlmostEqual(self._ellipse("A(3, 2)").rotation_angle, 0)
        self.canvas.delete_ellipse("A(3, 2)_1")
        self.assertEqual(self._ellipse_names(), ["A(3, 2)"])

    def test_scaling_into_a_name_in_use_gets_suffix(self) -> None:
        self.canvas.create_ellipse(0, 0, 6, 4, name="A", extra_graphics=False)
        small = self.canvas.create_ellipse(0, 0, 3, 2, extra_graphics=False)
        self.canvas.scale_object(small.name, 2, 2, 0, 0)
        self.assertEqual(small.name, "A(6, 4)_1")

    def test_suffix_is_stable_through_transforms(self) -> None:
        second = self._two_ellipses_with_one_default_name()
        self.canvas.delete_ellipse("A(3, 2)")
        self.canvas.rotate_object("A(3, 2)_1", 15)
        self.canvas.translate_object("A(3, 2)_1", 1, 1)
        self.assertEqual(second.name, "A(3, 2)_1")

    def test_custom_name_survives_edits_and_transforms(self) -> None:
        ellipse = self.canvas.create_ellipse(0, 0, 3, 2, name="orbit", extra_graphics=False)
        self.canvas.update_ellipse("orbit", new_radius_x=5)
        self.canvas.scale_object("orbit", 2, 2, 0, 0)
        self.canvas.rotate_object("orbit", 20)
        self.assertEqual(ellipse.name, "orbit")
        self.assertEqual(ellipse.radius_x, 10)

    def test_renamed_centre_keeps_ellipse_names_unique(self) -> None:
        second = self._two_ellipses_with_one_default_name()
        first = self._ellipse("A(3, 2)")
        point_manager = self.canvas.drawable_manager.point_manager
        first.center.update_name("P")
        point_manager._rename_dependent_rotational_drawables(first.center)
        self.assertEqual(sorted([first.name, second.name]), ["P(3, 2)", "P(3, 2)_1"])

    def test_undo_and_redo_keep_custom_and_suffixed_names(self) -> None:
        self._two_ellipses_with_one_default_name()
        self.canvas.create_ellipse(5, 5, 3, 2, name="orbit", extra_graphics=False)
        self.canvas.rotate_object("orbit", 30)
        self.canvas.undo()
        self.assertEqual(self._ellipse_names(), ["A(3, 2)", "A(3, 2)_1", "orbit"])
        self.canvas.redo()
        self.assertEqual(self._ellipse_names(), ["A(3, 2)", "A(3, 2)_1", "orbit"])
        restored = self._ellipse("orbit")
        self.assertTrue(restored.has_custom_name)
        self.canvas.scale_object("orbit", 2, 2, 5, 5)
        self.assertEqual(restored.name, "orbit")

    def _reload_workspace(self) -> None:
        manager = WorkspaceManager(self.canvas)
        saved = json.loads(json.dumps(manager._snapshot_persistable_canvas_state()))
        manager._restore_workspace_state(saved)

    def test_workspace_reload_keeps_custom_and_suffixed_names(self) -> None:
        self._two_ellipses_with_one_default_name()
        self.canvas.delete_ellipse("A(3, 2)")
        self.canvas.create_ellipse(5, 5, 3, 2, name="orbit", extra_graphics=False)
        self._reload_workspace()
        self.assertEqual(self._ellipse_names(), ["A(3, 2)_1", "orbit"])
        self.assertTrue(self._ellipse("orbit").has_custom_name)
        suffixed = self._ellipse("A(3, 2)_1")
        self.assertFalse(suffixed.has_custom_name)
        self.canvas.scale_object("A(3, 2)_1", 2, 2, 0, 0)
        self.assertEqual(suffixed.name, "A(6, 4)")

    def test_workspace_reload_keeps_area_on_suffixed_ellipse(self) -> None:
        self._two_ellipses_with_one_default_name()
        self.canvas.delete_ellipse("A(3, 2)")
        self.canvas.create_region_colored_area(ellipse_name="A(3, 2)_1")
        self._reload_workspace()
        areas = self.canvas.drawable_manager.drawables.ClosedShapeColoredAreas
        self.assertEqual(len(areas), 1)
        self.assertEqual(areas[0].get_state()["args"].get("ellipse"), "A(3, 2)_1")
