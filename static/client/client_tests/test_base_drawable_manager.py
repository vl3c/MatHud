"""Tests for BaseDrawableManager base class."""

import unittest

from managers.base_drawable_manager import BaseDrawableManager
from managers.drawables_container import DrawablesContainer
from .simple_mock import SimpleMock


class ConcreteManager(BaseDrawableManager):
    """Minimal concrete subclass for testing the base class."""

    drawable_type = "Point"


class TestBaseDrawableManager(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = SimpleMock(name="CanvasMock")
        self.drawables = DrawablesContainer()
        self.name_generator = SimpleMock(name="NameGeneratorMock")
        self.dependency_manager = SimpleMock(name="DependencyManagerMock")
        self.drawable_manager_proxy = SimpleMock(name="DrawableManagerProxyMock")

        self.manager = ConcreteManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

    def _make_drawable(self, class_name: str, name: str) -> SimpleMock:
        """Create a mock drawable with the given class name and name."""
        return SimpleMock(
            name=name,
            get_class_name=lambda _cn=class_name: _cn,
            is_renderable=True,
        )

    # ------------------------------------------------------------------
    # Constructor stores all dependencies
    # ------------------------------------------------------------------

    def test_constructor_stores_canvas(self) -> None:
        self.assertIs(self.manager.canvas, self.canvas)

    def test_constructor_stores_drawables(self) -> None:
        self.assertIs(self.manager.drawables, self.drawables)

    def test_constructor_stores_name_generator(self) -> None:
        self.assertIs(self.manager.name_generator, self.name_generator)

    def test_constructor_stores_dependency_manager(self) -> None:
        self.assertIs(self.manager.dependency_manager, self.dependency_manager)

    def test_constructor_stores_drawable_manager_proxy(self) -> None:
        self.assertIs(self.manager.drawable_manager, self.drawable_manager_proxy)

    # ------------------------------------------------------------------
    # Edit policy resolved for known type
    # ------------------------------------------------------------------

    def test_edit_policy_resolved_for_known_type(self) -> None:
        self.assertIsNotNone(self.manager.edit_policy)
        self.assertEqual(self.manager.edit_policy.drawable_type, "Point")

    def test_edit_policy_is_none_for_empty_drawable_type(self) -> None:
        class EmptyTypeManager(BaseDrawableManager):
            drawable_type = ""

        manager = EmptyTypeManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )
        self.assertIsNone(manager.edit_policy)

    # ------------------------------------------------------------------
    # _get_by_name returns matching drawable
    # ------------------------------------------------------------------

    def test_get_by_name_returns_matching_drawable(self) -> None:
        point_a = self._make_drawable("Point", "A")
        self.drawables.add(point_a)

        result = self.manager._get_by_name("A")

        self.assertIs(result, point_a)

    # ------------------------------------------------------------------
    # _get_by_name returns None for no match
    # ------------------------------------------------------------------

    def test_get_by_name_returns_none_for_no_match(self) -> None:
        point_a = self._make_drawable("Point", "A")
        self.drawables.add(point_a)

        result = self.manager._get_by_name("Z")

        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # _get_by_name returns None for empty string
    # ------------------------------------------------------------------

    def test_get_by_name_returns_none_for_empty_string(self) -> None:
        point_a = self._make_drawable("Point", "A")
        self.drawables.add(point_a)

        result = self.manager._get_by_name("")

        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # _get_by_name filters by drawable_type
    # ------------------------------------------------------------------

    def test_get_by_name_filters_by_drawable_type(self) -> None:
        segment = self._make_drawable("Segment", "A")
        point_a = self._make_drawable("Point", "A")
        self.drawables.add(segment)
        self.drawables.add(point_a)

        result = self.manager._get_by_name("A")

        self.assertIs(result, point_a)

    def test_get_by_name_ignores_other_types_with_same_name(self) -> None:
        segment = self._make_drawable("Segment", "B")
        self.drawables.add(segment)

        result = self.manager._get_by_name("B")

        self.assertIsNone(result)
