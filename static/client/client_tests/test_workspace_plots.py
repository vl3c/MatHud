from __future__ import annotations

import unittest

from canvas import Canvas
from workspace_manager import WorkspaceManager


class TestWorkspacePlotsRestore(unittest.TestCase):
    def _names_for_class(self, canvas: Canvas, class_name: str) -> list[str]:
        return [d.name for d in canvas.get_drawables_by_class_name(class_name)]

    def test_restore_workspace_rebuilds_plot_bars_and_supports_delete_plot(self) -> None:
        canvas1 = Canvas(500, 500, draw_enabled=False)
        canvas1.plot_distribution(
            name="MyPlot",
            representation="continuous",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            left_bound=None,
            right_bound=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=None,
        )
        canvas1.plot_distribution(
            name="MyBars",
            representation="discrete",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            left_bound=-2.0,
            right_bound=2.0,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=5,
        )
        canvas1.plot_bars(
            name="Poll",
            values=[10.0, 20.0, 5.0],
            labels_below=["A", "B", "C"],
            labels_above=["10", "20", "5"],
            bar_spacing=0.5,
            bar_width=2.0,
            stroke_color="#111",
            fill_color="#222",
            fill_opacity=0.5,
            x_start=1.0,
            y_base=0.0,
        )

        state = canvas1.get_canvas_state()
        # Match WorkspaceManager.save_workspace behavior: bars are derived and not persisted.
        if isinstance(state, dict) and "Bars" in state:
            del state["Bars"]

        canvas2 = Canvas(500, 500, draw_enabled=False)
        manager = WorkspaceManager(canvas2)
        manager._restore_workspace_state(state)

        self.assertIn("MyPlot", self._names_for_class(canvas2, "ContinuousPlot"))
        self.assertIn("MyBars", self._names_for_class(canvas2, "DiscretePlot"))
        self.assertIn("Poll", self._names_for_class(canvas2, "BarsPlot"))

        bar_names = set(self._names_for_class(canvas2, "Bar"))
        for i in range(5):
            self.assertIn(f"MyBars_bar_{i}", bar_names)
        for i in range(3):
            self.assertIn(f"Poll_bar_{i}", bar_names)

        # delete_plot should work after restore for all plot types.
        self.assertTrue(canvas2.delete_plot("MyBars"))
        self.assertTrue(canvas2.delete_plot("Poll"))
        self.assertTrue(canvas2.delete_plot("MyPlot"))

        self.assertNotIn("MyBars", self._names_for_class(canvas2, "DiscretePlot"))
        self.assertNotIn("Poll", self._names_for_class(canvas2, "BarsPlot"))
        self.assertNotIn("MyPlot", self._names_for_class(canvas2, "ContinuousPlot"))


