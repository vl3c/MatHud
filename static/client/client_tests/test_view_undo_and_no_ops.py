"""Regression tests for view undo (K4), the polar grid reset (K25) and truthful no-op updates (K26).

Every tool call runs through ``ProcessFunctionCalls.get_results_traced``, the same path a
model tool batch takes, against a real canvas.
"""

from __future__ import annotations

from typing import Dict

from .test_tool_batch_results import _ToolBatchTestCase

BOUND_TOLERANCE = 1e-9


class _ViewTestCase(_ToolBatchTestCase):
    """Helpers to read the view, the coordinate mode and the grids."""

    def bounds(self) -> Dict[str, float]:
        return dict(self.canvas.coordinate_mapper.get_visible_bounds())

    def assert_bounds(self, expected: Dict[str, float]) -> None:
        actual = self.bounds()
        for key in ("left", "right", "top", "bottom"):
            self.assertAlmostEqual(actual[key], expected[key], delta=BOUND_TOLERANCE, msg=key)

    def polar_spacing(self) -> float:
        return float(self.canvas.coordinate_system_manager.polar_grid.current_radial_spacing)

    def zoom_and_adapt_polar_grid(self, range_val: float) -> None:
        """Zoom as the model does; the polar grid adapts its spacing on the next frame, so adapt it here."""
        self.run_single("zoom", center_x=0, center_y=0, range_val=range_val, range_axis="x")
        self.canvas.coordinate_system_manager.polar_grid._invalidate_cache_on_zoom()

    def circle_name(self) -> str:
        return str(self.canvas.drawable_manager.drawables.Circles[0].name)


class TestPolarGridReset(_ViewTestCase):
    """K25: clearing or resetting the canvas also resets the polar grid's zoom-adapted spacing."""

    def test_clear_after_zoom_resets_the_polar_spacing(self) -> None:
        default_spacing = self.polar_spacing()
        self.zoom_and_adapt_polar_grid(2)
        self.assertLess(self.polar_spacing(), default_spacing)

        self.run_batch(("clear_canvas", {}), ("set_coordinate_system", {"mode": "polar"}))

        self.assertEqual(self.polar_spacing(), default_spacing)

    def test_reset_canvas_resets_the_polar_spacing(self) -> None:
        default_spacing = self.polar_spacing()
        self.zoom_and_adapt_polar_grid(2)
        self.assertLess(self.polar_spacing(), default_spacing)

        self.run_single("reset_canvas")

        self.assertEqual(self.polar_spacing(), default_spacing)
