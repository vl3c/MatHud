from __future__ import annotations

import unittest
from typing import Any, Dict

from static.canvas_state_summarizer import compare_canvas_states, summarize_canvas_state


class TestCanvasStateSummarizer(unittest.TestCase):
    def _sample_state(self) -> Dict[str, Any]:
        return {
            "Points": [
                {"name": "B", "args": {"position": {"x": 1, "y": 2}}},
                {"name": "A", "args": {"position": {"x": 0, "y": 0}}},
            ],
            "Segments": [
                {
                    "name": "s1",
                    "args": {"p1": "A", "p2": "B", "label": {"text": "", "visible": False}},
                    "_p1_coords": [0, 0],
                    "_p2_coords": [1, 2],
                }
            ],
            "Circles": [
                {
                    "name": "c1",
                    "args": {"center": "A", "radius": 5},
                    "circle_formula": {"a": 1, "b": 1, "c": 0},
                }
            ],
            "Functions": [
                {
                    "name": "f",
                    "args": {
                        "function_string": "x^2",
                        "left_bound": None,
                        "right_bound": None,
                        "vertical_asymptotes": [],
                        "horizontal_asymptotes": [],
                        "point_discontinuities": [],
                        "undefined_at": [],
                    },
                }
            ],
            "coordinate_system": {"mode": "cartesian", "grid_visible": True},
        }

    def test_summarize_prunes_noise_and_keeps_identity(self) -> None:
        summary = summarize_canvas_state(self._sample_state())

        self.assertIn("Segments", summary)
        seg = summary["Segments"][0]
        self.assertEqual(seg["name"], "s1")
        self.assertIn("args", seg)
        self.assertNotIn("_p1_coords", seg)
        self.assertNotIn("_p2_coords", seg)
        self.assertNotIn("label", seg["args"])

        circle = summary["Circles"][0]
        self.assertEqual(circle["name"], "c1")
        self.assertNotIn("circle_formula", circle)
        self.assertEqual(circle["args"]["radius"], 5)

        fn = summary["Functions"][0]
        self.assertIn("function_string", fn["args"])
        self.assertNotIn("left_bound", fn["args"])
        self.assertNotIn("undefined_at", fn["args"])

    def test_compare_reports_metrics_and_reduction(self) -> None:
        comparison = compare_canvas_states(self._sample_state())
        metrics = comparison["metrics"]

        self.assertGreater(metrics["full_bytes"], 0)
        self.assertGreater(metrics["summary_bytes"], 0)
        self.assertGreaterEqual(metrics["reduction_pct"], 0.0)
        self.assertLess(metrics["summary_bytes"], metrics["full_bytes"])

    def test_summary_is_idempotent(self) -> None:
        once = summarize_canvas_state(self._sample_state())
        twice = summarize_canvas_state(once)
        self.assertEqual(once, twice)

    def test_comparison_output_is_deterministic(self) -> None:
        comparison = compare_canvas_states(self._sample_state())
        points = comparison["summary"]["Points"]
        self.assertEqual(points[0]["name"], "A")
        self.assertEqual(points[1]["name"], "B")

    def test_large_scene_reduces_size(self) -> None:
        state: Dict[str, Any] = {"Segments": [], "Points": []}
        for i in range(200):
            state["Points"].append({"name": f"P{i}", "args": {"position": {"x": i, "y": i}}})
            state["Segments"].append(
                {
                    "name": f"s{i}",
                    "args": {"p1": f"P{i}", "p2": f"P{i + 1}", "label": {"text": "", "visible": False}},
                    "_p1_coords": [i, i],
                    "_p2_coords": [i + 1, i + 1],
                }
            )

        comparison = compare_canvas_states(state)
        metrics = comparison["metrics"]

        self.assertGreater(metrics["full_bytes"], metrics["summary_bytes"])
        self.assertGreater(metrics["reduction_pct"], 20.0)
        self.assertEqual(len(comparison["summary"]["Points"]), 200)
        self.assertEqual(len(comparison["summary"]["Segments"]), 200)


if __name__ == "__main__":
    unittest.main()
