from __future__ import annotations

import unittest

from drawables.function import Function
from managers.drawables_container import DrawablesContainer
from managers.function_manager import FunctionManager
from .simple_mock import SimpleMock


class TestFunctionManager(unittest.TestCase):
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
        self.canvas.drawable_manager = SimpleMock(delete_colored_areas_for_function=SimpleMock())
        self.drawables = DrawablesContainer()
        self.name_generator = SimpleMock(name="NameGeneratorMock")
        self.dependency_manager = SimpleMock(name="DependencyManagerMock")
        self.dependency_manager.remove_drawable = SimpleMock()
        self.drawable_manager_proxy = SimpleMock(name="DrawableManagerProxyMock")

        self.manager = FunctionManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

    def _add_function(self, name: str = "f1", color: str = "#111111") -> Function:
        func = Function("x^2", name=name, left_bound=-2.0, right_bound=4.0, color=color)
        self.drawables.add(func)
        return func

    def test_update_function_changes_color_and_bounds(self) -> None:
        func = self._add_function()

        result = self.manager.update_function(
            "f1",
            new_color="#ff00ff",
            new_left_bound=-1.0,
            new_right_bound=5.0,
        )

        self.assertTrue(result)
        self.assertEqual(func.color, "#ff00ff")
        self.assertEqual(func.left_bound, -1.0)
        self.assertEqual(func.right_bound, 5.0)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()

    def test_update_function_requires_existing_function(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.update_function("missing", new_color="#ff00ff")

    def test_update_function_validates_color(self) -> None:
        self._add_function()
        with self.assertRaises(ValueError):
            self.manager.update_function("f1", new_color="  ")

    def test_update_function_requires_bounds_relation(self) -> None:
        self._add_function()
        with self.assertRaises(ValueError):
            self.manager.update_function("f1", new_left_bound=5.0, new_right_bound=0.0)

    def test_update_function_left_only_keeps_right(self) -> None:
        func = self._add_function()

        result = self.manager.update_function("f1", new_left_bound=-3.0)

        self.assertTrue(result)
        self.assertEqual(func.left_bound, -3.0)
        self.assertEqual(func.right_bound, 4.0)

    def test_delete_function_removes_dependency_entry(self) -> None:
        func = self._add_function(name="f1")

        removed = self.manager.delete_function("f1")

        self.assertTrue(removed)
        self.dependency_manager.remove_drawable.assert_called_once_with(func)
        self.canvas.drawable_manager.delete_colored_areas_for_function.assert_called_once()


if __name__ == "__main__":
    unittest.main()
