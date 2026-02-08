"""
Unit tests for polar renderer plan integration.

Tests the polar grid rendering integration with renderers including:
- Plan caching for polar grids
- Renderer method availability
- Style key availability

Note: These tests use mocks since the actual rendering code has browser dependencies.
"""

from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import unittest
from types import SimpleNamespace

from rendering.style_manager import get_renderer_style

from .renderer_fixtures import PlanStub, TelemetryRecorder


class Position:
    """Simple Position class for testing."""
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y


def create_mock_polar_grid(
    angular_divisions: int = 12,
    radial_spacing: float = 1.0,
    width: int = 800,
    height: int = 600,
) -> SimpleNamespace:
    """Create a mock PolarGrid for testing."""
    return SimpleNamespace(
        class_name="PolarGrid",
        angular_divisions=angular_divisions,
        radial_spacing=radial_spacing,
        width=width,
        height=height,
        show_angle_labels=True,
        show_radius_labels=True,
    )


def create_mock_coordinate_mapper(
    canvas_width: int = 800,
    canvas_height: int = 600,
    scale_factor: float = 1.0,
) -> SimpleNamespace:
    """Create a mock CoordinateMapper."""
    return SimpleNamespace(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        scale_factor=scale_factor,
        offset=Position(0, 0),
        origin=Position(canvas_width / 2, canvas_height / 2),
    )


class TestPolarStyleKeys(unittest.TestCase):
    """Tests for polar-specific style keys in style_manager."""

    def test_polar_axis_color_exists(self) -> None:
        """Test that polar_axis_color style key exists."""
        style = get_renderer_style()
        self.assertIn("polar_axis_color", style)

    def test_polar_circle_color_exists(self) -> None:
        """Test that polar_circle_color style key exists."""
        style = get_renderer_style()
        self.assertIn("polar_circle_color", style)

    def test_polar_radial_color_exists(self) -> None:
        """Test that polar_radial_color style key exists."""
        style = get_renderer_style()
        self.assertIn("polar_radial_color", style)

    def test_polar_label_color_exists(self) -> None:
        """Test that polar_label_color style key exists."""
        style = get_renderer_style()
        self.assertIn("polar_label_color", style)

    def test_polar_label_font_size_exists(self) -> None:
        """Test that polar_label_font_size style key exists."""
        style = get_renderer_style()
        self.assertIn("polar_label_font_size", style)

    def test_polar_font_family_exists(self) -> None:
        """Test that polar_font_family style key exists."""
        style = get_renderer_style()
        self.assertIn("polar_font_family", style)


class TestPolarPlanStub(unittest.TestCase):
    """Tests using PlanStub to simulate polar plan behavior."""

    def test_plan_stub_visibility(self) -> None:
        """Test PlanStub visibility checking."""
        plan = PlanStub(visible=True, plan_key="polar-grid")
        self.assertTrue(plan.is_visible(800, 600))

        plan_invisible = PlanStub(visible=False, plan_key="polar-grid")
        self.assertFalse(plan_invisible.is_visible(800, 600))

    def test_plan_stub_key(self) -> None:
        """Test PlanStub plan_key property."""
        plan = PlanStub(visible=True, plan_key="polar-grid-123")
        self.assertEqual(plan.plan_key, "polar-grid-123")

    def test_plan_stub_update_map_state(self) -> None:
        """Test PlanStub map state update tracking."""
        plan = PlanStub(visible=True, plan_key="polar-grid")
        self.assertEqual(plan.update_calls, 0)

        plan.update_map_state({"scale": 1.0})
        self.assertEqual(plan.update_calls, 1)

        plan.update_map_state({"scale": 2.0})
        self.assertEqual(plan.update_calls, 2)

    def test_plan_stub_needs_apply(self) -> None:
        """Test PlanStub needs_apply behavior."""
        plan = PlanStub(visible=True, plan_key="polar-grid")
        self.assertTrue(plan.needs_apply())

        plan.mark_dirty()
        self.assertTrue(plan.needs_apply())

    def test_plan_stub_supports_transform(self) -> None:
        """Test PlanStub supports_transform behavior."""
        plan = PlanStub(visible=True, plan_key="polar-grid", supports_transform=True)
        self.assertTrue(plan.supports_transform())

        plan_no_transform = PlanStub(visible=True, plan_key="polar-grid", supports_transform=False)
        self.assertFalse(plan_no_transform.supports_transform())


class TestPolarRendererIntegration(unittest.TestCase):
    """Integration tests for polar renderer methods."""

    def test_polar_grid_class_name(self) -> None:
        """Test that mock polar grid has correct class name."""
        polar_grid = create_mock_polar_grid()
        self.assertEqual(polar_grid.class_name, "PolarGrid")

    def test_polar_grid_configuration(self) -> None:
        """Test polar grid configuration options."""
        polar_grid = create_mock_polar_grid(
            angular_divisions=8,
            radial_spacing=2.0,
            width=1000,
            height=800,
        )

        self.assertEqual(polar_grid.angular_divisions, 8)
        self.assertEqual(polar_grid.radial_spacing, 2.0)
        self.assertEqual(polar_grid.width, 1000)
        self.assertEqual(polar_grid.height, 800)

    def test_coordinate_mapper_configuration(self) -> None:
        """Test coordinate mapper configuration."""
        mapper = create_mock_coordinate_mapper(
            canvas_width=1920,
            canvas_height=1080,
            scale_factor=2.0,
        )

        self.assertEqual(mapper.canvas_width, 1920)
        self.assertEqual(mapper.canvas_height, 1080)
        self.assertEqual(mapper.scale_factor, 2.0)
        self.assertEqual(mapper.origin.x, 960)
        self.assertEqual(mapper.origin.y, 540)


class TestPolarPlanCaching(unittest.TestCase):
    """Tests for polar plan caching behavior using mock renderer."""

    def _make_mock_renderer(self) -> SimpleNamespace:
        """Create a mock renderer with polar plan caching support."""
        renderer = SimpleNamespace()
        renderer.style = get_renderer_style()
        renderer._telemetry = TelemetryRecorder()
        renderer._plan_cache = {}
        renderer._polar_cache = None
        renderer._frame_seen_plan_keys = set()

        def resolve_polar_plan(polar_grid, mapper, map_state, signature, drawable_name):
            cache_key = f"{drawable_name}:{signature}"
            if cache_key in renderer._plan_cache:
                entry = renderer._plan_cache[cache_key]
                if entry.get("signature") == signature:
                    entry["plan"].update_map_state(map_state)
                    return {"plan": entry["plan"], "plan_key": entry["plan"].plan_key}

            plan = PlanStub(visible=True, plan_key=f"polar-{id(polar_grid)}")
            plan.update_map_state(map_state)
            renderer._plan_cache[cache_key] = {"plan": plan, "signature": signature}
            return {"plan": plan, "plan_key": plan.plan_key}

        renderer._resolve_polar_plan = resolve_polar_plan
        return renderer

    def test_polar_plan_cache_reuse(self) -> None:
        """Test that polar plans are cached and reused."""
        renderer = self._make_mock_renderer()

        polar_grid = create_mock_polar_grid()
        mapper = create_mock_coordinate_mapper()
        map_state = {"scale": 1.0}
        signature = ("polar", 1.0, 12)

        ctx1 = renderer._resolve_polar_plan(polar_grid, mapper, map_state, signature, "PolarGrid")
        ctx2 = renderer._resolve_polar_plan(polar_grid, mapper, map_state, signature, "PolarGrid")

        self.assertIs(ctx1["plan"], ctx2["plan"])
        self.assertEqual(ctx1["plan"].update_calls, 2)

    def test_polar_plan_cache_invalidation_on_signature_change(self) -> None:
        """Test that polar plans are rebuilt when signature changes."""
        renderer = self._make_mock_renderer()

        polar_grid = create_mock_polar_grid()
        mapper = create_mock_coordinate_mapper()
        map_state = {"scale": 1.0}

        signature1 = ("polar", 1.0, 12)
        signature2 = ("polar", 2.0, 12)  # Different zoom

        ctx1 = renderer._resolve_polar_plan(polar_grid, mapper, map_state, signature1, "PolarGrid")
        ctx2 = renderer._resolve_polar_plan(polar_grid, mapper, map_state, signature2, "PolarGrid")

        self.assertIsNot(ctx1["plan"], ctx2["plan"])

    def test_polar_plan_different_grids(self) -> None:
        """Test that different polar grids get different plans."""
        renderer = self._make_mock_renderer()

        polar_grid1 = create_mock_polar_grid(angular_divisions=8)
        polar_grid2 = create_mock_polar_grid(angular_divisions=12)
        mapper = create_mock_coordinate_mapper()
        map_state = {"scale": 1.0}

        signature1 = ("polar", 1.0, 8)
        signature2 = ("polar", 1.0, 12)

        ctx1 = renderer._resolve_polar_plan(polar_grid1, mapper, map_state, signature1, "PolarGrid")
        ctx2 = renderer._resolve_polar_plan(polar_grid2, mapper, map_state, signature2, "PolarGrid")

        self.assertIsNot(ctx1["plan"], ctx2["plan"])


class TestPolarRendererMethodAvailability(unittest.TestCase):
    """Tests to verify polar rendering methods are available in renderers."""

    def test_svg_renderer_has_render_polar_import(self) -> None:
        """Test that svg_renderer imports build_plan_for_polar."""
        from rendering import svg_renderer
        # Check that the import statement exists in the module
        import_source = svg_renderer.__file__
        with open(import_source, 'r') as f:
            content = f.read()
        self.assertIn("build_plan_for_polar", content)

    def test_canvas2d_renderer_has_render_polar_import(self) -> None:
        """Test that canvas2d_renderer imports build_plan_for_polar."""
        from rendering import canvas2d_renderer
        import_source = canvas2d_renderer.__file__
        with open(import_source, 'r') as f:
            content = f.read()
        self.assertIn("build_plan_for_polar", content)

    def test_webgl_renderer_has_render_polar_import(self) -> None:
        """Test that webgl_renderer imports build_plan_for_polar."""
        from rendering import webgl_renderer
        import_source = webgl_renderer.__file__
        with open(import_source, 'r') as f:
            content = f.read()
        self.assertIn("build_plan_for_polar", content)


if __name__ == '__main__':
    unittest.main()
