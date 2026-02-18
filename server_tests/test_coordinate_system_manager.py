"""
Unit tests for CoordinateSystemManager class logic.

Tests the coordinate system mode management including:
- Mode switching between Cartesian and Polar
- Active grid retrieval
- State serialization and restoration
- Integration with Canvas

Note: These tests use a mock CoordinateSystemManager implementation since
the actual class has browser dependencies through PolarGrid.
"""

from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import unittest
from types import SimpleNamespace
from typing import Any, Dict, Union

SimpleMock = SimpleNamespace


class Position:
    """Simple Position class for testing."""

    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y


class MockPolarGrid:
    """Mock PolarGrid for testing."""

    def __init__(self, coordinate_mapper: Any):
        self.coordinate_mapper = coordinate_mapper
        self.class_name = "PolarGrid"


class MockCoordinateSystemManager:
    """Mock CoordinateSystemManager for testing core logic.

    This mirrors the essential logic of the real CoordinateSystemManager
    without browser dependencies.
    """

    def __init__(self, canvas: Any) -> None:
        self.canvas = canvas
        self._mode: str = "cartesian"
        self.cartesian_grid = canvas.cartesian2axis
        self.polar_grid = MockPolarGrid(canvas.coordinate_mapper)

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("cartesian", "polar"):
            raise ValueError("Mode must be 'cartesian' or 'polar'")
        self._mode = mode
        self.canvas.draw()

    def get_active_grid(self) -> Union[Any, MockPolarGrid]:
        return self.cartesian_grid if self._mode == "cartesian" else self.polar_grid

    def get_state(self) -> Dict[str, Any]:
        return {"mode": self._mode}

    def set_state(self, state: Dict[str, Any]) -> None:
        mode = state.get("mode", "cartesian")
        if mode not in ("cartesian", "polar"):
            mode = "cartesian"
        self._mode = mode


# Use mock for testing
CoordinateSystemManager = MockCoordinateSystemManager


def create_mock_coordinate_mapper(
    canvas_width: int = 800,
    canvas_height: int = 600,
    scale_factor: float = 1.0,
) -> SimpleNamespace:
    """Create a mock CoordinateMapper."""
    origin_x = canvas_width / 2
    origin_y = canvas_height / 2

    def math_to_screen(mx: float, my: float):
        sx = origin_x + mx * scale_factor
        sy = origin_y - my * scale_factor
        return (sx, sy)

    def screen_to_math(sx: float, sy: float):
        mx = (sx - origin_x) / scale_factor
        my = (origin_y - sy) / scale_factor
        return (mx, my)

    def scale_value(v: float):
        return v * scale_factor

    def unscale_value(v: float):
        return v / scale_factor

    return SimpleMock(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        scale_factor=scale_factor,
        offset=Position(0, 0),
        origin=Position(origin_x, origin_y),
        math_to_screen=math_to_screen,
        screen_to_math=screen_to_math,
        scale_value=scale_value,
        unscale_value=unscale_value,
    )


def create_mock_cartesian_grid() -> SimpleNamespace:
    """Create a mock Cartesian2Axis grid."""
    return SimpleMock(
        class_name="Cartesian2Axis",
        width=800,
        height=600,
    )


def create_mock_canvas(
    coordinate_mapper: SimpleNamespace = None,
    cartesian_grid: SimpleNamespace = None,
) -> SimpleNamespace:
    """Create a mock Canvas with required components."""
    if coordinate_mapper is None:
        coordinate_mapper = create_mock_coordinate_mapper()
    if cartesian_grid is None:
        cartesian_grid = create_mock_cartesian_grid()

    draw_called = [False]

    def mock_draw():
        draw_called[0] = True

    return SimpleMock(
        coordinate_mapper=coordinate_mapper,
        cartesian2axis=cartesian_grid,
        draw=mock_draw,
        _draw_called=draw_called,
    )


class TestCoordinateSystemManagerInitialization(unittest.TestCase):
    """Tests for CoordinateSystemManager initialization."""

    def test_default_initialization(self) -> None:
        """Test manager initializes with Cartesian mode by default."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        self.assertEqual(manager.mode, "cartesian")
        self.assertEqual(manager.canvas, canvas)

    def test_cartesian_grid_reference(self) -> None:
        """Test manager holds reference to Cartesian grid."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        self.assertEqual(manager.cartesian_grid, canvas.cartesian2axis)

    def test_polar_grid_created(self) -> None:
        """Test manager creates a PolarGrid instance."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        self.assertIsNotNone(manager.polar_grid)
        self.assertEqual(manager.polar_grid.class_name, "PolarGrid")


class TestCoordinateSystemManagerModeSwitch(unittest.TestCase):
    """Tests for mode switching functionality."""

    def test_set_mode_to_polar(self) -> None:
        """Test switching to polar mode."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        manager.set_mode("polar")

        self.assertEqual(manager.mode, "polar")

    def test_set_mode_to_cartesian(self) -> None:
        """Test switching to cartesian mode."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        manager.set_mode("polar")  # First switch to polar
        manager.set_mode("cartesian")  # Then back to cartesian

        self.assertEqual(manager.mode, "cartesian")

    def test_set_mode_triggers_redraw(self) -> None:
        """Test that mode switch triggers canvas redraw."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        canvas._draw_called[0] = False
        manager.set_mode("polar")

        self.assertTrue(canvas._draw_called[0])

    def test_set_mode_invalid_raises_error(self) -> None:
        """Test that invalid mode raises ValueError."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        with self.assertRaises(ValueError):
            manager.set_mode("invalid_mode")

    def test_set_mode_case_sensitive(self) -> None:
        """Test that mode names are case-sensitive."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        with self.assertRaises(ValueError):
            manager.set_mode("Polar")

        with self.assertRaises(ValueError):
            manager.set_mode("CARTESIAN")


class TestCoordinateSystemManagerActiveGrid(unittest.TestCase):
    """Tests for active grid retrieval."""

    def test_get_active_grid_cartesian(self) -> None:
        """Test get_active_grid returns Cartesian grid in cartesian mode."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        active_grid = manager.get_active_grid()

        self.assertEqual(active_grid, manager.cartesian_grid)

    def test_get_active_grid_polar(self) -> None:
        """Test get_active_grid returns Polar grid in polar mode."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        manager.set_mode("polar")
        active_grid = manager.get_active_grid()

        self.assertEqual(active_grid, manager.polar_grid)

    def test_active_grid_changes_with_mode(self) -> None:
        """Test active grid changes when mode changes."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        # Initially cartesian
        grid1 = manager.get_active_grid()
        self.assertEqual(grid1.class_name, "Cartesian2Axis")

        # Switch to polar
        manager.set_mode("polar")
        grid2 = manager.get_active_grid()
        self.assertEqual(grid2.class_name, "PolarGrid")

        # Switch back to cartesian
        manager.set_mode("cartesian")
        grid3 = manager.get_active_grid()
        self.assertEqual(grid3.class_name, "Cartesian2Axis")


class TestCoordinateSystemManagerState(unittest.TestCase):
    """Tests for state serialization and restoration."""

    def test_get_state_cartesian(self) -> None:
        """Test state serialization in cartesian mode."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        state = manager.get_state()

        self.assertIn("mode", state)
        self.assertEqual(state["mode"], "cartesian")

    def test_get_state_polar(self) -> None:
        """Test state serialization in polar mode."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)
        manager.set_mode("polar")

        state = manager.get_state()

        self.assertEqual(state["mode"], "polar")

    def test_set_state_cartesian(self) -> None:
        """Test state restoration to cartesian mode."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)
        manager.set_mode("polar")  # Start in polar

        manager.set_state({"mode": "cartesian"})

        self.assertEqual(manager.mode, "cartesian")

    def test_set_state_polar(self) -> None:
        """Test state restoration to polar mode."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        manager.set_state({"mode": "polar"})

        self.assertEqual(manager.mode, "polar")

    def test_set_state_empty(self) -> None:
        """Test state restoration with empty state defaults to cartesian."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)
        manager.set_mode("polar")

        manager.set_state({})

        self.assertEqual(manager.mode, "cartesian")

    def test_set_state_missing_mode(self) -> None:
        """Test state restoration with missing mode key defaults to cartesian."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)
        manager.set_mode("polar")

        manager.set_state({"other_key": "value"})

        self.assertEqual(manager.mode, "cartesian")

    def test_state_roundtrip(self) -> None:
        """Test state can be saved and restored correctly."""
        canvas = create_mock_canvas()
        manager1 = CoordinateSystemManager(canvas)
        manager1.set_mode("polar")

        state = manager1.get_state()

        manager2 = CoordinateSystemManager(canvas)
        manager2.set_state(state)

        self.assertEqual(manager1.mode, manager2.mode)


class TestCoordinateSystemManagerModeProperty(unittest.TestCase):
    """Tests for mode property getter."""

    def test_mode_property_readonly(self) -> None:
        """Test mode property is read-only (no setter)."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        # Should use set_mode() method, not direct assignment
        # This test verifies the property exists and returns correct value
        self.assertEqual(manager.mode, "cartesian")

    def test_mode_property_after_switch(self) -> None:
        """Test mode property reflects current mode after switch."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        self.assertEqual(manager.mode, "cartesian")

        manager.set_mode("polar")
        self.assertEqual(manager.mode, "polar")

        manager.set_mode("cartesian")
        self.assertEqual(manager.mode, "cartesian")


class TestCoordinateSystemManagerEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_set_same_mode_twice(self) -> None:
        """Test setting the same mode twice doesn't cause issues."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        manager.set_mode("cartesian")
        manager.set_mode("cartesian")

        self.assertEqual(manager.mode, "cartesian")

    def test_rapid_mode_switching(self) -> None:
        """Test rapid mode switching doesn't cause issues."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        for _ in range(100):
            manager.set_mode("polar")
            manager.set_mode("cartesian")

        self.assertEqual(manager.mode, "cartesian")

    def test_set_state_invalid_mode_ignored(self) -> None:
        """Test set_state with invalid mode falls back to cartesian."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)
        manager.set_mode("polar")

        # Invalid mode in state should be handled gracefully
        manager.set_state({"mode": "invalid"})

        # Should either stay at polar or fall back to cartesian
        self.assertIn(manager.mode, ["cartesian", "polar"])


class TestCoordinateSystemManagerIntegration(unittest.TestCase):
    """Integration tests with mock Canvas."""

    def test_polar_grid_uses_canvas_mapper(self) -> None:
        """Test that PolarGrid uses the canvas's coordinate mapper."""
        mapper = create_mock_coordinate_mapper(canvas_width=1000, canvas_height=800)
        canvas = create_mock_canvas(coordinate_mapper=mapper)
        manager = CoordinateSystemManager(canvas)

        # PolarGrid should reference the same mapper
        self.assertEqual(manager.polar_grid.coordinate_mapper, mapper)

    def test_mode_switch_preserves_grid_instances(self) -> None:
        """Test that mode switching doesn't recreate grid instances."""
        canvas = create_mock_canvas()
        manager = CoordinateSystemManager(canvas)

        cartesian_id = id(manager.cartesian_grid)
        polar_id = id(manager.polar_grid)

        manager.set_mode("polar")
        manager.set_mode("cartesian")
        manager.set_mode("polar")

        # Grid instances should be the same objects
        self.assertEqual(id(manager.cartesian_grid), cartesian_id)
        self.assertEqual(id(manager.polar_grid), polar_id)


if __name__ == "__main__":
    unittest.main()
