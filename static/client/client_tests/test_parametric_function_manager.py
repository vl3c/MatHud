"""Tests for ParametricFunctionManager."""

from __future__ import annotations

import math
import unittest

from drawables.parametric_function import ParametricFunction
from managers.drawables_container import DrawablesContainer
from managers.parametric_function_manager import ParametricFunctionManager
from .simple_mock import SimpleMock


class TestParametricFunctionManager(unittest.TestCase):
    """Test suite for ParametricFunctionManager class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
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
            generate_parametric_function_name=SimpleMock(return_value="p1"),
        )
        self.dependency_manager = SimpleMock(name="DependencyManagerMock")
        self.dependency_manager.remove_drawable = SimpleMock()
        self.proxy = SimpleMock(name="ProxyMock")

        self.manager = ParametricFunctionManager(
            canvas=self.canvas,
            drawables=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            proxy=self.proxy,
        )

    def _add_parametric_function(
        self,
        name: str = "Circle",
        x_expr: str = "cos(t)",
        y_expr: str = "sin(t)",
        color: str = "#111111",
    ) -> ParametricFunction:
        """Helper to add a parametric function directly to drawables."""
        func = ParametricFunction(
            x_expression=x_expr,
            y_expression=y_expr,
            name=name,
            t_min=0,
            t_max=2 * math.pi,
            color=color,
        )
        self.drawables.add(func)
        return func

    def test_draw_parametric_function_creates_function(self) -> None:
        """Test that draw_parametric_function creates a function."""
        func = self.manager.draw_parametric_function(
            x_expression="cos(t)",
            y_expression="sin(t)",
            name="Circle",
        )

        self.assertIsInstance(func, ParametricFunction)
        self.assertEqual(func.name, "Circle")
        self.assertEqual(len(self.drawables.ParametricFunctions), 1)

    def test_draw_parametric_function_archives_for_undo(self) -> None:
        """Test that draw_parametric_function archives state for undo."""
        self.manager.draw_parametric_function(
            x_expression="t",
            y_expression="t",
            name="Line",
        )

        self.canvas.undo_redo_manager.archive.assert_called_once()

    def test_draw_parametric_function_triggers_draw(self) -> None:
        """Test that draw_parametric_function triggers canvas draw."""
        self.manager.draw_parametric_function(
            x_expression="t",
            y_expression="t",
            name="Line",
        )

        self.canvas.draw.assert_called_once()

    def test_draw_parametric_function_with_custom_params(self) -> None:
        """Test creation with custom t_min, t_max, and color."""
        func = self.manager.draw_parametric_function(
            x_expression="t*cos(t)",
            y_expression="t*sin(t)",
            name="Spiral",
            t_min=0,
            t_max=6 * math.pi,
            color="#ff0000",
        )

        self.assertEqual(func.t_min, 0)
        self.assertAlmostEqual(func.t_max, 6 * math.pi, places=5)
        self.assertEqual(func.color, "#ff0000")

    def test_get_parametric_function_returns_function(self) -> None:
        """Test get_parametric_function retrieves existing function."""
        self._add_parametric_function(name="Circle")

        func = self.manager.get_parametric_function("Circle")

        self.assertIsNotNone(func)
        self.assertEqual(func.name, "Circle")

    def test_get_parametric_function_returns_none_for_missing(self) -> None:
        """Test get_parametric_function returns None for missing function."""
        func = self.manager.get_parametric_function("NonExistent")

        self.assertIsNone(func)

    def test_delete_parametric_function_removes_function(self) -> None:
        """Test delete_parametric_function removes the function."""
        self._add_parametric_function(name="Circle")
        self.assertEqual(len(self.drawables.ParametricFunctions), 1)

        result = self.manager.delete_parametric_function("Circle")

        self.assertTrue(result)
        self.assertEqual(len(self.drawables.ParametricFunctions), 0)

    def test_delete_parametric_function_archives_for_undo(self) -> None:
        """Test delete archives state for undo."""
        self._add_parametric_function(name="Circle")

        self.manager.delete_parametric_function("Circle")

        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.dependency_manager.remove_drawable.assert_called_once()

    def test_delete_parametric_function_returns_false_for_missing(self) -> None:
        """Test delete returns False for missing function."""
        result = self.manager.delete_parametric_function("NonExistent")

        self.assertFalse(result)

    def test_update_parametric_function_changes_color(self) -> None:
        """Test update_parametric_function changes color."""
        func = self._add_parametric_function(name="Circle", color="#111111")

        result = self.manager.update_parametric_function("Circle", new_color="#ff00ff")

        self.assertTrue(result)
        self.assertEqual(func.color, "#ff00ff")

    def test_update_parametric_function_changes_t_min(self) -> None:
        """Test update_parametric_function changes t_min."""
        func = self._add_parametric_function(name="Circle")

        result = self.manager.update_parametric_function("Circle", new_t_min=1.0)

        self.assertTrue(result)
        self.assertEqual(func.t_min, 1.0)

    def test_update_parametric_function_changes_t_max(self) -> None:
        """Test update_parametric_function changes t_max."""
        func = self._add_parametric_function(name="Circle")

        result = self.manager.update_parametric_function("Circle", new_t_max=10.0)

        self.assertTrue(result)
        self.assertEqual(func.t_max, 10.0)

    def test_update_parametric_function_archives_for_undo(self) -> None:
        """Test update archives state for undo."""
        self._add_parametric_function(name="Circle")

        self.manager.update_parametric_function("Circle", new_color="#ff00ff")

        self.canvas.undo_redo_manager.archive.assert_called_once()

    def test_update_parametric_function_returns_false_for_missing(self) -> None:
        """Test update returns False for missing function."""
        result = self.manager.update_parametric_function("NonExistent", new_color="#ff00ff")

        self.assertFalse(result)

    def test_update_parametric_function_no_changes_no_archive(self) -> None:
        """Test update with no changes doesn't archive."""
        func = self._add_parametric_function(name="Circle", color="#111111")

        # Pass same values - no changes
        result = self.manager.update_parametric_function(
            "Circle",
            new_color="#111111",
            new_t_min=func.t_min,
            new_t_max=func.t_max,
        )

        self.assertTrue(result)
        # Archive should NOT be called since no changes were made
        self.canvas.undo_redo_manager.archive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
