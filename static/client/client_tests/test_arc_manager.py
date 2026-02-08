from __future__ import annotations

import math
import unittest
from typing import Any, Dict, Optional

from drawables.circle import Circle
from drawables.point import Point
from managers.arc_manager import ArcManager
from managers.drawables_container import DrawablesContainer
from .simple_mock import SimpleMock


class TestArcManager(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = SimpleMock(
            name="CanvasMock",
            undo_redo_manager=SimpleMock(archive=lambda: None),
            draw_enabled=True,
            draw=lambda: None,
        )
        self.drawables = DrawablesContainer()
        self.dependency_manager = SimpleMock(
            register_dependency=SimpleMock(),
            unregister_dependency=SimpleMock(),
            remove_drawable=SimpleMock(),
        )
        self.points_by_name: Dict[str, Point] = {}

        def create_point(x: float, y: float, name: Optional[str] = None, extra_graphics: bool = True, **kwargs: Any) -> Point:
            assigned_name = name or f"P_{len(self.points_by_name)}"
            point = Point(x, y, assigned_name)
            self.points_by_name[assigned_name] = point
            return point

        def get_point_by_name(name: str) -> Optional[Point]:
            return self.points_by_name.get(name)

        def get_point(x: float, y: float) -> Optional[Point]:
            for point in self.points_by_name.values():
                if abs(point.x - x) < 1e-9 and abs(point.y - y) < 1e-9:
                    return point
            return None

        self.point_manager = SimpleMock(
            create_point=create_point,
            get_point_by_name=get_point_by_name,
            get_point=get_point,
        )

        self.circles: Dict[str, Circle] = {}

        def get_circle_by_name(name: str) -> Optional[Circle]:
            return self.circles.get(name)

        self.drawable_manager_proxy = SimpleMock(
            create_drawables_from_new_connections=lambda: None,
            get_circle_by_name=get_circle_by_name,
        )

        self.name_generator = SimpleMock(
            extract_point_names_from_arc_name=lambda arc_name: (None, None),
            generate_arc_name=lambda proposed, p1, p2, major, existing: proposed if proposed else f"{'ArcMaj' if major else 'ArcMin'}_{p1}{p2}",
        )

        self.arc_manager = ArcManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            point_manager=self.point_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

        center_point = Point(0, 0, name="O")
        self.circle = Circle(center_point, radius=5)
        self.circles[self.circle.name] = self.circle

    def test_create_circle_arc_with_existing_circle(self) -> None:
        arc = self.arc_manager.create_circle_arc(
            point1_x=5,
            point1_y=0,
            point2_x=0,
            point2_y=5,
            point1_name="A",
            point2_name="B",
            circle_name=self.circle.name,
        )
        self.assertIsNotNone(arc)
        self.assertEqual(len(self.drawables.CircleArcs), 1)
        self.assertEqual(self.drawables.CircleArcs[0].circle_name, self.circle.name)

    def test_create_circle_arc_requires_circle_or_center(self) -> None:
        with self.assertRaises(ValueError):
            self.arc_manager.create_circle_arc(
                point1_x=5,
                point1_y=0,
                point2_x=0,
                point2_y=5,
            )

    def test_create_circle_arc_snaps_new_points_to_circle(self) -> None:
        arc = self.arc_manager.create_circle_arc(
            point1_x=1,
            point1_y=0,
            point2_x=0,
            point2_y=1,
            point1_name="A",
            point2_name="B",
            center_x=0,
            center_y=0,
            radius=5,
            arc_name="arc_AB",
        )

        self.assertIsNotNone(arc)
        self.assertTrue(
            math.isclose(math.hypot(arc.point1.x, arc.point1.y), 5.0, rel_tol=1e-9, abs_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(math.hypot(arc.point2.x, arc.point2.y), 5.0, rel_tol=1e-9, abs_tol=1e-9)
        )

    def test_create_circle_arc_projects_existing_points(self) -> None:
        existing_point_a = self.point_manager.create_point(1, 0, name="A")
        existing_point_b = self.point_manager.create_point(0, 2, name="B")

        arc = self.arc_manager.create_circle_arc(
            point1_name="A",
            point2_name="B",
            circle_name=self.circle.name,
        )

        self.assertIsNotNone(arc)
        self.assertTrue(
            math.isclose(math.hypot(existing_point_a.x, existing_point_a.y), 5.0, rel_tol=1e-9, abs_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(math.hypot(existing_point_b.x, existing_point_b.y), 5.0, rel_tol=1e-9, abs_tol=1e-9)
        )

    def test_create_circle_arc_raises_when_points_coincide(self) -> None:
        self.point_manager.create_point(1, 0, name="A")
        with self.assertRaises(ValueError):
            self.arc_manager.create_circle_arc(
                point1_name="A",
                point2_name="A",
                circle_name=self.circle.name,
            )

    def test_create_circle_arc_derived_from_three_points(self) -> None:
        arc = self.arc_manager.create_circle_arc(
            point1_x=6,
            point1_y=0,
            point2_x=0,
            point2_y=8,
            point3_x=0,
            point3_y=0,
            point1_name="P1",
            point2_name="P2",
            point3_name="O",
            center_point_choice="point3",
            arc_name="arc_three",
        )

        self.assertIsNotNone(arc)
        self.assertEqual(arc.center_x, 0)
        self.assertEqual(arc.center_y, 0)
        self.assertTrue(math.isclose(arc.radius, 6.0, rel_tol=1e-9, abs_tol=1e-9))
        self.assertTrue(math.isclose(math.hypot(arc.point1.x, arc.point1.y), 6.0, rel_tol=1e-9, abs_tol=1e-9))
        self.assertTrue(math.isclose(math.hypot(arc.point2.x, arc.point2.y), 6.0, rel_tol=1e-9, abs_tol=1e-9))

    def test_create_circle_arc_three_point_requires_all_points(self) -> None:
        with self.assertRaises(ValueError):
            self.arc_manager.create_circle_arc(
                point1_x=5,
                point1_y=0,
                point2_x=0,
                point2_y=5,
                point1_name="A",
                point2_name="B",
                center_point_choice="point3",
            )

    def test_create_circle_arc_ignores_center_choice_with_circle(self) -> None:
        arc = self.arc_manager.create_circle_arc(
            point1_x=5,
            point1_y=0,
            point2_x=0,
            point2_y=5,
            point1_name="A",
            point2_name="B",
            circle_name=self.circle.name,
            center_point_choice="point1",
        )
        self.assertIsNotNone(arc)

    def test_delete_circle_arc_removes_from_container(self) -> None:
        arc = self.arc_manager.create_circle_arc(
            point1_x=5,
            point1_y=0,
            point2_x=0,
            point2_y=5,
            point1_name="A",
            point2_name="B",
            center_x=0,
            center_y=0,
            radius=5,
            arc_name="arc_AB",
        )
        self.assertIsNotNone(arc)
        deleted = self.arc_manager.delete_circle_arc("arc_AB")
        self.assertTrue(deleted)
        self.assertEqual(len(self.drawables.CircleArcs), 0)

    def test_update_circle_arc_changes_properties(self) -> None:
        arc = self.arc_manager.create_circle_arc(
            point1_x=5,
            point1_y=0,
            point2_x=0,
            point2_y=5,
            point1_name="A",
            point2_name="B",
            center_x=0,
            center_y=0,
            radius=5,
            arc_name="arc_AB",
        )
        self.assertIsNotNone(arc)
        self.arc_manager.update_circle_arc(
            "arc_AB",
            new_color="red",
            use_major_arc=True,
        )
        updated_arc = self.arc_manager.get_circle_arc_by_name("arc_AB")
        self.assertEqual(updated_arc.color, "red")
        self.assertTrue(updated_arc.use_major_arc)

    def test_update_circle_arc_requires_editable_property(self) -> None:
        self.arc_manager.create_circle_arc(
            point1_x=5,
            point1_y=0,
            point2_x=0,
            point2_y=5,
            point1_name="A",
            point2_name="B",
            center_x=0,
            center_y=0,
            radius=5,
            arc_name="arc_AB",
        )

        with self.assertRaises(ValueError):
            self.arc_manager.update_circle_arc("arc_AB")


    def test_handle_circle_removed_deletes_arcs(self) -> None:
        arc = self.arc_manager.create_circle_arc(
            point1_x=5,
            point1_y=0,
            point2_x=0,
            point2_y=5,
            point1_name="A",
            point2_name="B",
            circle_name=self.circle.name,
            arc_name="arc_AB",
        )
        self.assertIsNotNone(arc)
        self.arc_manager.handle_circle_removed(self.circle.name)
        self.assertEqual(len(self.drawables.CircleArcs), 0)

    def test_load_circle_arcs_restores_from_state(self) -> None:
        self.point_manager.create_point(5, 0, name="A")
        self.point_manager.create_point(0, 5, name="B")
        state = [
            {
                "name": "arc_AB",
                "args": {
                    "point1_name": "A",
                    "point2_name": "B",
                    "center_x": 0,
                    "center_y": 0,
                    "radius": 5,
                    "circle_name": self.circle.name,
                    "color": "green",
                    "use_major_arc": False,
                },
            }
        ]
        self.arc_manager.load_circle_arcs(state)
        self.assertEqual(len(self.drawables.CircleArcs), 1)
        self.assertEqual(self.drawables.CircleArcs[0].name, "arc_AB")

