from __future__ import annotations

import unittest

from drawables.point import Point
from drawables.vector import Vector
from managers.drawables_container import DrawablesContainer
from managers.vector_manager import VectorManager
from .simple_mock import SimpleMock


class TestVectorManager(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = SimpleMock(
            name="CanvasMock",
            draw_enabled=True,
            draw=SimpleMock(),
            undo_redo_manager=SimpleMock(
                name="UndoRedoMock",
                archive=SimpleMock(),
            ),
        )

        self.drawables = DrawablesContainer()
        self.name_generator = SimpleMock(
            name="NameGeneratorMock",
            split_point_names=lambda expr, count: ["", ""][:count],
        )
        self.dependency_manager = SimpleMock(
            name="DependencyManagerMock",
            analyze_drawable_for_dependencies=SimpleMock(),
            get_children=lambda vector: set(),
            get_all_children=lambda vector: set(),
            get_all_parents=lambda vector: set(),
            remove_drawable=SimpleMock(),
        )
        self.point_manager = SimpleMock(
            name="PointManagerMock",
            create_point=lambda x, y, name="", extra_graphics=True: Point(x, y, name=name),
        )
        self.drawable_manager_proxy = SimpleMock(
            name="DrawableManagerProxyMock",
            create_drawables_from_new_connections=SimpleMock(),
            delete_segment=SimpleMock(),
        )

        self.vector_manager = VectorManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            point_manager=self.point_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

    def _add_vector(self, name: str = "AB", color: str = "#111111") -> Vector:
        origin = Point(0.0, 0.0, name="A")
        tip = Point(1.0, 0.0, name="B")
        vector = Vector(origin, tip, color=color)
        vector.name = name
        self.drawables.add(vector)
        return vector

    def test_update_vector_changes_color_and_draws(self) -> None:
        vector = self._add_vector()

        result = self.vector_manager.update_vector("AB", new_color="#ff00ff")

        self.assertTrue(result)
        self.assertEqual(vector.color, "#ff00ff")
        self.assertEqual(vector.segment.color, "#ff00ff")
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()

    def test_update_vector_requires_existing_vector(self) -> None:
        with self.assertRaises(ValueError):
            self.vector_manager.update_vector("missing", new_color="#00ff00")

    def test_update_vector_requires_color_value(self) -> None:
        self._add_vector()
        with self.assertRaises(ValueError):
            self.vector_manager.update_vector("AB", new_color="  ")

    def test_update_vector_requires_properties(self) -> None:
        self._add_vector()
        with self.assertRaises(ValueError):
            self.vector_manager.update_vector("AB")


if __name__ == "__main__":
    unittest.main()

