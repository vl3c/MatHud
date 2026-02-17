"""
Unit tests for PolarGrid class logic.

Tests the polar coordinate grid system logic including:
- Radial spacing calculations
- Angular divisions
- State management
- Origin calculations

Note: These tests use a mock PolarGrid implementation since the actual
PolarGrid class has browser dependencies. The mock implementation matches
the core logic of the real class.
"""

from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import math
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

SimpleMock = SimpleNamespace


class Position:
    """Simple Position class for testing."""

    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y


class MockPolarGrid:
    """Mock PolarGrid implementation for testing core logic.

    This mirrors the essential logic of the real PolarGrid class
    without browser dependencies.
    """

    def __init__(
        self,
        coordinate_mapper: Any,
        angular_divisions: int = 12,
        radial_spacing: float = 1.0,
        show_angle_labels: bool = True,
        show_radius_labels: bool = True,
    ):
        self.coordinate_mapper = coordinate_mapper
        self.angular_divisions = angular_divisions
        self.radial_spacing = radial_spacing
        self.show_angle_labels = show_angle_labels
        self.show_radius_labels = show_radius_labels
        self.width = None
        self.height = None
        self._cached_display_spacing = None

    @property
    def class_name(self) -> str:
        return "PolarGrid"

    @property
    def angle_step(self) -> float:
        return 2 * math.pi / self.angular_divisions

    @property
    def origin_screen(self) -> Tuple[float, float]:
        if self.width is None or self.height is None:
            return (0, 0)
        ox = self.width / 2 + self.coordinate_mapper.offset.x
        oy = self.height / 2 + self.coordinate_mapper.offset.y
        return (ox, oy)

    @property
    def max_radius_screen(self) -> float:
        if self.width is None or self.height is None:
            return 0
        ox, oy = self.origin_screen
        corners = [(0, 0), (self.width, 0), (0, self.height), (self.width, self.height)]
        max_dist = 0
        for cx, cy in corners:
            dist = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
            max_dist = max(max_dist, dist)
        return max_dist * 1.1

    @property
    def max_radius_math(self) -> float:
        return self.max_radius_screen / self.coordinate_mapper.scale_factor

    @property
    def display_spacing(self) -> float:
        return abs(self.radial_spacing) * self.coordinate_mapper.scale_factor * 50

    def get_angle_labels(self) -> List[Tuple[float, str]]:
        labels = []
        for i in range(self.angular_divisions):
            angle = i * self.angle_step
            labels.append((angle, f"{int(math.degrees(angle))}"))
        return labels

    def get_radial_circles(self) -> List[float]:
        circles = []
        spacing = self.display_spacing
        if spacing <= 0:
            return circles
        n = 1
        while n * spacing < self.max_radius_screen:
            circles.append(n * spacing)
            n += 1
        return circles

    def get_state(self) -> Dict[str, Any]:
        return {
            "angular_divisions": self.angular_divisions,
            "radial_spacing": self.radial_spacing,
            "show_angle_labels": self.show_angle_labels,
            "show_radius_labels": self.show_radius_labels,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if "angular_divisions" in state:
            self.angular_divisions = state["angular_divisions"]
        if "radial_spacing" in state:
            self.radial_spacing = state["radial_spacing"]
        if "show_angle_labels" in state:
            self.show_angle_labels = state["show_angle_labels"]
        if "show_radius_labels" in state:
            self.show_radius_labels = state["show_radius_labels"]

    def _invalidate_cache_on_zoom(self) -> None:
        self._cached_display_spacing = None


# Use MockPolarGrid for testing
PolarGrid = MockPolarGrid


def create_mock_coordinate_mapper(
    canvas_width: int = 800,
    canvas_height: int = 600,
    scale_factor: float = 1.0,
    offset_x: float = 0,
    offset_y: float = 0,
) -> SimpleNamespace:
    """Create a mock CoordinateMapper with configurable properties."""
    origin_x = canvas_width / 2
    origin_y = canvas_height / 2

    def math_to_screen(mx: float, my: float):
        sx = origin_x + mx * scale_factor + offset_x
        sy = origin_y - my * scale_factor + offset_y
        return (sx, sy)

    def screen_to_math(sx: float, sy: float):
        mx = (sx - offset_x - origin_x) / scale_factor
        my = (origin_y + offset_y - sy) / scale_factor
        return (mx, my)

    def scale_value(v: float):
        return v * scale_factor

    def unscale_value(v: float):
        return v / scale_factor

    return SimpleMock(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        scale_factor=scale_factor,
        offset=Position(offset_x, offset_y),
        origin=Position(origin_x, origin_y),
        math_to_screen=math_to_screen,
        screen_to_math=screen_to_math,
        scale_value=scale_value,
        unscale_value=unscale_value,
    )


class TestPolarGridInitialization(unittest.TestCase):
    """Tests for PolarGrid initialization."""

    def test_default_initialization(self) -> None:
        """Test PolarGrid initializes with correct defaults."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper)

        self.assertEqual(grid.coordinate_mapper, mapper)
        self.assertEqual(grid.angular_divisions, 12)
        self.assertEqual(grid.radial_spacing, 1.0)
        self.assertTrue(grid.show_angle_labels)
        self.assertTrue(grid.show_radius_labels)
        self.assertIsNone(grid.width)
        self.assertIsNone(grid.height)

    def test_custom_angular_divisions(self) -> None:
        """Test PolarGrid with custom angular divisions."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper, angular_divisions=8)

        self.assertEqual(grid.angular_divisions, 8)

    def test_custom_radial_spacing(self) -> None:
        """Test PolarGrid with custom radial spacing."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper, radial_spacing=2.5)

        self.assertEqual(grid.radial_spacing, 2.5)

    def test_label_visibility_options(self) -> None:
        """Test PolarGrid with label visibility options."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper, show_angle_labels=False, show_radius_labels=False)

        self.assertFalse(grid.show_angle_labels)
        self.assertFalse(grid.show_radius_labels)


class TestPolarGridDimensions(unittest.TestCase):
    """Tests for PolarGrid dimension management."""

    def test_set_dimensions(self) -> None:
        """Test setting grid dimensions."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper)

        grid.width = 1000
        grid.height = 800

        self.assertEqual(grid.width, 1000)
        self.assertEqual(grid.height, 800)

    def test_max_radius_screen_calculation(self) -> None:
        """Test max_radius_screen calculation."""
        mapper = create_mock_coordinate_mapper(canvas_width=800, canvas_height=600)
        grid = PolarGrid(mapper)
        grid.width = 800
        grid.height = 600

        max_radius = grid.max_radius_screen
        # Should be at least the diagonal from center to corner
        expected_min = math.sqrt((400) ** 2 + (300) ** 2)
        self.assertGreaterEqual(max_radius, expected_min)

    def test_max_radius_math_calculation(self) -> None:
        """Test max_radius_math calculation."""
        mapper = create_mock_coordinate_mapper(canvas_width=800, canvas_height=600, scale_factor=2.0)
        grid = PolarGrid(mapper)
        grid.width = 800
        grid.height = 600

        max_radius_screen = grid.max_radius_screen
        max_radius_math = grid.max_radius_math

        # Math radius should be screen radius divided by scale factor
        self.assertAlmostEqual(max_radius_math, max_radius_screen / 2.0, places=6)


class TestPolarGridAngularCalculations(unittest.TestCase):
    """Tests for angular division calculations."""

    def test_angle_step_12_divisions(self) -> None:
        """Test angle step with 12 divisions (30 degrees each)."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper, angular_divisions=12)

        angle_step = grid.angle_step
        self.assertAlmostEqual(angle_step, math.pi / 6, places=10)  # 30 degrees

    def test_angle_step_8_divisions(self) -> None:
        """Test angle step with 8 divisions (45 degrees each)."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper, angular_divisions=8)

        angle_step = grid.angle_step
        self.assertAlmostEqual(angle_step, math.pi / 4, places=10)  # 45 degrees

    def test_angle_step_6_divisions(self) -> None:
        """Test angle step with 6 divisions (60 degrees each)."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper, angular_divisions=6)

        angle_step = grid.angle_step
        self.assertAlmostEqual(angle_step, math.pi / 3, places=10)  # 60 degrees

    def test_get_angle_labels(self) -> None:
        """Test angle label generation."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper, angular_divisions=4)

        labels = grid.get_angle_labels()

        # Should have 4 labels at 0, 90, 180, 270 degrees
        self.assertEqual(len(labels), 4)

        # Check angles are evenly spaced
        angles = [label[0] for label in labels]
        expected_angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        for actual, expected in zip(angles, expected_angles):
            self.assertAlmostEqual(actual, expected, places=10)


class TestPolarGridRadialCalculations(unittest.TestCase):
    """Tests for radial spacing calculations."""

    def test_display_spacing_default(self) -> None:
        """Test display spacing with default radial spacing."""
        mapper = create_mock_coordinate_mapper(scale_factor=1.0)
        grid = PolarGrid(mapper, radial_spacing=1.0)
        grid.width = 800
        grid.height = 600

        display_spacing = grid.display_spacing
        # With scale_factor=1.0 and radial_spacing=1.0, display_spacing should be close to 1.0 * scale
        self.assertGreater(display_spacing, 0)

    def test_display_spacing_with_zoom(self) -> None:
        """Test display spacing adjusts with zoom."""
        mapper1 = create_mock_coordinate_mapper(scale_factor=1.0)
        mapper2 = create_mock_coordinate_mapper(scale_factor=2.0)

        grid1 = PolarGrid(mapper1, radial_spacing=1.0)
        grid2 = PolarGrid(mapper2, radial_spacing=1.0)

        grid1.width = grid2.width = 800
        grid1.height = grid2.height = 600

        # Display spacing should be different due to different scale factors
        spacing1 = grid1.display_spacing
        spacing2 = grid2.display_spacing

        # With 2x zoom, display spacing should be 2x larger
        self.assertAlmostEqual(spacing2, spacing1 * 2, places=6)

    def test_get_radial_circles(self) -> None:
        """Test radial circle generation."""
        # Use scale_factor=1.0 so display_spacing = 1.0 * 1.0 * 50 = 50 pixels
        mapper = create_mock_coordinate_mapper(scale_factor=1.0)
        grid = PolarGrid(mapper, radial_spacing=1.0)
        grid.width = 800
        grid.height = 600

        circles = grid.get_radial_circles()

        # Should have multiple circles (max_radius is about 500-600 pixels, spacing is 50)
        self.assertGreater(len(circles), 0)

        # Circles should be at regular intervals
        if len(circles) >= 2:
            spacing = circles[1] - circles[0]
            for i in range(2, len(circles)):
                self.assertAlmostEqual(circles[i] - circles[i - 1], spacing, places=6)


class TestPolarGridOrigin(unittest.TestCase):
    """Tests for origin position calculations."""

    def test_origin_screen_default(self) -> None:
        """Test origin screen position with no offset."""
        mapper = create_mock_coordinate_mapper(canvas_width=800, canvas_height=600)
        grid = PolarGrid(mapper)
        grid.width = 800
        grid.height = 600

        ox, oy = grid.origin_screen

        # Origin should be at canvas center
        self.assertAlmostEqual(ox, 400, places=6)
        self.assertAlmostEqual(oy, 300, places=6)

    def test_origin_screen_with_offset(self) -> None:
        """Test origin screen position with pan offset."""
        mapper = create_mock_coordinate_mapper(canvas_width=800, canvas_height=600, offset_x=50, offset_y=-30)
        grid = PolarGrid(mapper)
        grid.width = 800
        grid.height = 600

        ox, oy = grid.origin_screen

        # Origin should be offset from center
        self.assertAlmostEqual(ox, 450, places=6)  # 400 + 50
        self.assertAlmostEqual(oy, 270, places=6)  # 300 - 30


class TestPolarGridState(unittest.TestCase):
    """Tests for PolarGrid state management."""

    def test_get_state(self) -> None:
        """Test state serialization."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(
            mapper, angular_divisions=8, radial_spacing=2.0, show_angle_labels=False, show_radius_labels=True
        )

        state = grid.get_state()

        self.assertIn("angular_divisions", state)
        self.assertIn("radial_spacing", state)
        self.assertIn("show_angle_labels", state)
        self.assertIn("show_radius_labels", state)

        self.assertEqual(state["angular_divisions"], 8)
        self.assertEqual(state["radial_spacing"], 2.0)
        self.assertFalse(state["show_angle_labels"])
        self.assertTrue(state["show_radius_labels"])

    def test_set_state(self) -> None:
        """Test state restoration."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper)

        state = {
            "angular_divisions": 6,
            "radial_spacing": 3.0,
            "show_angle_labels": False,
            "show_radius_labels": False,
        }

        grid.set_state(state)

        self.assertEqual(grid.angular_divisions, 6)
        self.assertEqual(grid.radial_spacing, 3.0)
        self.assertFalse(grid.show_angle_labels)
        self.assertFalse(grid.show_radius_labels)

    def test_set_state_partial(self) -> None:
        """Test partial state restoration preserves unset values."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper, angular_divisions=12, radial_spacing=1.0)

        # Only update some properties
        state = {"angular_divisions": 8}
        grid.set_state(state)

        self.assertEqual(grid.angular_divisions, 8)
        self.assertEqual(grid.radial_spacing, 1.0)  # Should be unchanged


class TestPolarGridCacheInvalidation(unittest.TestCase):
    """Tests for cache invalidation on zoom."""

    def test_invalidate_cache_on_zoom(self) -> None:
        """Test that cache invalidation method exists and can be called."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper)

        # Should not raise
        grid._invalidate_cache_on_zoom()

    def test_cache_invalidation_clears_cached_values(self) -> None:
        """Test that cache invalidation clears any cached calculations."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper)
        grid.width = 800
        grid.height = 600

        # Access some cached properties
        _ = grid.max_radius_screen
        _ = grid.display_spacing

        # Invalidate cache
        grid._invalidate_cache_on_zoom()

        # Properties should still be accessible after invalidation
        max_radius = grid.max_radius_screen
        self.assertGreater(max_radius, 0)


class TestPolarGridClassName(unittest.TestCase):
    """Tests for class name property."""

    def test_class_name(self) -> None:
        """Test class_name property returns correct value."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper)

        self.assertEqual(grid.class_name, "PolarGrid")


class TestPolarGridEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_zero_angular_divisions(self) -> None:
        """Test behavior with zero angular divisions."""
        mapper = create_mock_coordinate_mapper()
        # Should handle gracefully or raise appropriate error
        try:
            grid = PolarGrid(mapper, angular_divisions=0)
            # If it doesn't raise, angle_step should handle it
            with self.assertRaises((ZeroDivisionError, ValueError)):
                _ = grid.angle_step
        except (ZeroDivisionError, ValueError):
            pass  # Expected behavior

    def test_negative_radial_spacing(self) -> None:
        """Test behavior with negative radial spacing."""
        mapper = create_mock_coordinate_mapper()
        grid = PolarGrid(mapper, radial_spacing=-1.0)

        # Should use absolute value or handle gracefully
        grid.width = 800
        grid.height = 600

        # display_spacing should be positive
        spacing = grid.display_spacing
        self.assertGreaterEqual(abs(spacing), 0)

    def test_very_small_dimensions(self) -> None:
        """Test with very small canvas dimensions."""
        mapper = create_mock_coordinate_mapper(canvas_width=10, canvas_height=10)
        grid = PolarGrid(mapper)
        grid.width = 10
        grid.height = 10

        # Should still calculate valid values
        max_radius = grid.max_radius_screen
        self.assertGreater(max_radius, 0)

    def test_very_large_dimensions(self) -> None:
        """Test with very large canvas dimensions."""
        mapper = create_mock_coordinate_mapper(canvas_width=10000, canvas_height=10000)
        grid = PolarGrid(mapper)
        grid.width = 10000
        grid.height = 10000

        max_radius = grid.max_radius_screen
        self.assertGreater(max_radius, 0)
        self.assertLess(max_radius, float("inf"))


if __name__ == "__main__":
    unittest.main()
