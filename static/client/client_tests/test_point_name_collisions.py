"""
Brython tests for point name uniqueness when preferred names collide.

Covers the case where many points ask for preferred names that filter down to
the same letter: once the letter and its apostrophe variants are exhausted, the
generator must fall back to an unused name instead of reusing a taken one.
"""

from __future__ import annotations

import unittest
from typing import Any, List

from canvas import Canvas
from name_generator.point import PointNameGenerator
from .simple_mock import SimpleMock


class TestPointNameCollisions(unittest.TestCase):
    """Tests that generated point names never repeat an existing name."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)

    def _point_names(self) -> List[str]:
        return [p.name for p in self.canvas.get_drawables_by_class_name("Point")]

    def _assert_unique(self, names: List[str]) -> None:
        duplicates = sorted({n for n in names if names.count(n) > 1})
        self.assertEqual(duplicates, [], f"Duplicate point names: {duplicates}")

    def test_generator_skips_exhausted_letter(self) -> None:
        taken: List[Any] = [SimpleMock(name="F" + "'" * i) for i in range(6)]
        mock_canvas: Any = SimpleMock()
        setattr(mock_canvas, "get_drawables_by_class_name", SimpleMock(return_value=taken))
        generator = PointNameGenerator(mock_canvas)

        result = generator.generate_point_name("F")

        self.assertNotIn(result, [p.name for p in taken])

    def test_preferred_names_sharing_a_letter_stay_unique(self) -> None:
        for i in range(10):
            self.canvas.create_point(float(i), float(i), name=f"F_{i}", extra_graphics=False)

        names = self._point_names()
        self.assertEqual(len(names), 10)
        self._assert_unique(names)

    def test_fit_regression_with_many_points_has_unique_names(self) -> None:
        x_data = [float(i) for i in range(30)]
        y_data = [2.0 * x + 1.0 for x in x_data]

        result = self.canvas.fit_regression(
            name="fit1",
            x_data=x_data,
            y_data=y_data,
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=True,
            point_color=None,
        )

        point_names = result["point_names"]
        self.assertEqual(len(point_names), 30)
        self._assert_unique(point_names)

        state_names = [p["name"] for p in self.canvas.get_canvas_state().get("Points", [])]
        self.assertEqual(len(state_names), 30)
        self._assert_unique(state_names)
        self.assertEqual(sorted(state_names), sorted(point_names))
