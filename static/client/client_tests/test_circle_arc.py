from __future__ import annotations

import unittest

from drawables.circle import Circle
from drawables.circle_arc import CircleArc
from drawables.point import Point


class TestCircleArc(unittest.TestCase):
    def setUp(self) -> None:
        self.center = Point(0, 0, name="O")
        self.circle = Circle(self.center, radius=5)
        self.point_a = Point(5, 0, name="A")
        self.point_b = Point(0, 5, name="B")
        self.point_c = Point(-5, 0, name="C")

    def test_initialization_with_valid_points(self) -> None:
        arc = CircleArc(
            self.point_a,
            self.point_b,
            center_x=0,
            center_y=0,
            radius=5,
            circle=self.circle,
        )
        self.assertEqual(arc.point1, self.point_a)
        self.assertEqual(arc.point2, self.point_b)
        self.assertFalse(arc.use_major_arc)
        self.assertEqual(arc.circle_name, self.circle.name)

    def test_initialization_fails_for_duplicate_points(self) -> None:
        with self.assertRaises(ValueError):
            CircleArc(
                self.point_a,
                self.point_a,
                center_x=0,
                center_y=0,
                radius=5,
            )

    def test_initialization_requires_points_on_circle(self) -> None:
        off_circle_point = Point(3, 0, name="P")
        with self.assertRaises(ValueError):
            CircleArc(
                off_circle_point,
                self.point_b,
                center_x=0,
                center_y=0,
                radius=5,
            )

    def test_get_state_contains_expected_fields(self) -> None:
        arc = CircleArc(
            self.point_a,
            self.point_c,
            center_x=0,
            center_y=0,
            radius=5,
            use_major_arc=True,
            color="orange",
            name="arc_test",
        )
        state = arc.get_state()
        self.assertEqual(state["name"], "arc_test")
        self.assertEqual(state["args"]["point1_name"], "A")
        self.assertEqual(state["args"]["point2_name"], "C")
        self.assertTrue(state["args"]["use_major_arc"])
        self.assertEqual(state["args"]["color"], "orange")

    def test_sync_with_circle_updates_geometry(self) -> None:
        arc = CircleArc(
            self.point_a,
            self.point_b,
            center_x=0,
            center_y=0,
            radius=5,
            circle=self.circle,
        )
        self.circle.update_center_position(1, 2)
        self.circle.radius = 6
        arc.sync_with_circle()
        self.assertEqual(arc.center_x, 1)
        self.assertEqual(arc.center_y, 2)
        self.assertEqual(arc.radius, 6)

    def test_set_use_major_arc_switches_flag(self) -> None:
        arc = CircleArc(
            self.point_a,
            self.point_b,
            center_x=0,
            center_y=0,
            radius=5,
        )
        arc.set_use_major_arc(True)
        self.assertTrue(arc.use_major_arc)
        arc.set_use_major_arc(False)
        self.assertFalse(arc.use_major_arc)

