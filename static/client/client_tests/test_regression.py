"""
Brython tests for regression analysis via Canvas.

These tests verify the fit_regression functionality through the Canvas
interface in the browser environment.
"""

from __future__ import annotations

import unittest
from typing import List

from canvas import Canvas


class TestRegressionCanvas(unittest.TestCase):
    """Tests for regression fitting via Canvas."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)

    def _names_for_class(self, class_name: str) -> List[str]:
        return [d.name for d in self.canvas.get_drawables_by_class_name(class_name)]

    # -------------------------------------------------------------------------
    # Linear regression tests
    # -------------------------------------------------------------------------

    def test_fit_regression_linear_creates_function(self) -> None:
        result = self.canvas.fit_regression(
            name="LinReg",
            x_data=[1.0, 2.0, 3.0, 4.0],
            y_data=[2.0, 4.0, 6.0, 8.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=True,
            point_color=None,
        )

        self.assertEqual(result["model_type"], "linear")
        self.assertIn("LinReg", result["function_name"])
        self.assertIn("expression", result)
        self.assertIn("coefficients", result)
        self.assertIn("r_squared", result)
        self.assertIn("function_name", result)

        # Check function was created
        function_names = self._names_for_class("Function")
        self.assertIn(result["function_name"], function_names)

    def test_fit_regression_linear_creates_points(self) -> None:
        result = self.canvas.fit_regression(
            name="WithPoints",
            x_data=[1.0, 2.0, 3.0],
            y_data=[2.0, 4.0, 6.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=True,
            point_color=None,
        )

        self.assertIn("point_names", result)
        point_names = result["point_names"]
        self.assertEqual(len(point_names), 3)

        all_points = self._names_for_class("Point")
        for pt_name in point_names:
            self.assertIn(pt_name, all_points)

    def test_fit_regression_show_points_false_omits_points(self) -> None:
        result = self.canvas.fit_regression(
            name="NoPoints",
            x_data=[1.0, 2.0, 3.0],
            y_data=[2.0, 4.0, 6.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        # Should not have point_names or have empty list
        point_names = result.get("point_names", [])
        self.assertEqual(len(point_names), 0)

    def test_fit_regression_linear_r_squared_perfect(self) -> None:
        result = self.canvas.fit_regression(
            name="PerfectFit",
            x_data=[1.0, 2.0, 3.0, 4.0],
            y_data=[2.0, 4.0, 6.0, 8.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        self.assertAlmostEqual(result["r_squared"], 1.0, places=8)
        self.assertAlmostEqual(result["coefficients"]["m"], 2.0, places=8)
        self.assertAlmostEqual(result["coefficients"]["b"], 0.0, places=8)

    # -------------------------------------------------------------------------
    # Polynomial regression tests
    # -------------------------------------------------------------------------

    def test_fit_regression_polynomial_requires_degree(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.fit_regression(
                name="NoDegree",
                x_data=[1.0, 2.0, 3.0, 4.0],
                y_data=[1.0, 4.0, 9.0, 16.0],
                model_type="polynomial",
                degree=None,  # Missing degree
                plot_bounds=None,
                curve_color=None,
                show_points=False,
                point_color=None,
            )

    def test_fit_regression_polynomial_degree_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.fit_regression(
                name="BadDegree",
                x_data=[1.0, 2.0, 3.0],
                y_data=[1.0, 4.0, 9.0],
                model_type="polynomial",
                degree=0,
                plot_bounds=None,
                curve_color=None,
                show_points=False,
                point_color=None,
            )

        with self.assertRaises(ValueError):
            self.canvas.fit_regression(
                name="BadDegree2",
                x_data=[1.0, 2.0, 3.0],
                y_data=[1.0, 4.0, 9.0],
                model_type="polynomial",
                degree=-1,
                plot_bounds=None,
                curve_color=None,
                show_points=False,
                point_color=None,
            )

    def test_fit_regression_polynomial_quadratic(self) -> None:
        result = self.canvas.fit_regression(
            name="Quadratic",
            x_data=[0.0, 1.0, 2.0, 3.0, 4.0],
            y_data=[0.0, 1.0, 4.0, 9.0, 16.0],
            model_type="polynomial",
            degree=2,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        self.assertEqual(result["model_type"], "polynomial")
        self.assertAlmostEqual(result["r_squared"], 1.0, places=6)
        self.assertIn("a0", result["coefficients"])
        self.assertIn("a1", result["coefficients"])
        self.assertIn("a2", result["coefficients"])

    # -------------------------------------------------------------------------
    # Input validation tests
    # -------------------------------------------------------------------------

    def test_fit_regression_rejects_mismatched_data_lengths(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.fit_regression(
                name="Mismatched",
                x_data=[1.0, 2.0, 3.0],
                y_data=[1.0, 2.0],  # Different length
                model_type="linear",
                degree=None,
                plot_bounds=None,
                curve_color=None,
                show_points=False,
                point_color=None,
            )

    def test_fit_regression_rejects_invalid_model_type(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.fit_regression(
                name="BadModel",
                x_data=[1.0, 2.0, 3.0],
                y_data=[1.0, 2.0, 3.0],
                model_type="invalid_model",
                degree=None,
                plot_bounds=None,
                curve_color=None,
                show_points=False,
                point_color=None,
            )

    def test_fit_regression_rejects_insufficient_points(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.fit_regression(
                name="TooFew",
                x_data=[1.0],
                y_data=[1.0],
                model_type="linear",
                degree=None,
                plot_bounds=None,
                curve_color=None,
                show_points=False,
                point_color=None,
            )

    # -------------------------------------------------------------------------
    # Domain validation tests
    # -------------------------------------------------------------------------

    def test_fit_regression_exponential_rejects_non_positive_y(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.fit_regression(
                name="BadExpY",
                x_data=[1.0, 2.0, 3.0, 4.0],
                y_data=[1.0, -2.0, 3.0, 4.0],
                model_type="exponential",
                degree=None,
                plot_bounds=None,
                curve_color=None,
                show_points=False,
                point_color=None,
            )

    def test_fit_regression_logarithmic_rejects_non_positive_x(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.fit_regression(
                name="BadLogX",
                x_data=[1.0, -2.0, 3.0, 4.0],
                y_data=[1.0, 2.0, 3.0, 4.0],
                model_type="logarithmic",
                degree=None,
                plot_bounds=None,
                curve_color=None,
                show_points=False,
                point_color=None,
            )

    def test_fit_regression_power_rejects_non_positive_x(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.fit_regression(
                name="BadPowerX",
                x_data=[-1.0, 2.0, 3.0, 4.0],
                y_data=[1.0, 2.0, 3.0, 4.0],
                model_type="power",
                degree=None,
                plot_bounds=None,
                curve_color=None,
                show_points=False,
                point_color=None,
            )

    def test_fit_regression_power_rejects_non_positive_y(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.fit_regression(
                name="BadPowerY",
                x_data=[1.0, 2.0, 3.0, 4.0],
                y_data=[1.0, 2.0, -3.0, 4.0],
                model_type="power",
                degree=None,
                plot_bounds=None,
                curve_color=None,
                show_points=False,
                point_color=None,
            )

    # -------------------------------------------------------------------------
    # Other model types
    # -------------------------------------------------------------------------

    def test_fit_regression_exponential(self) -> None:
        result = self.canvas.fit_regression(
            name="Exp",
            x_data=[0.0, 1.0, 2.0, 3.0],
            y_data=[1.0, 2.7, 7.4, 20.0],
            model_type="exponential",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        self.assertEqual(result["model_type"], "exponential")
        self.assertIn("a", result["coefficients"])
        self.assertIn("b", result["coefficients"])
        self.assertGreater(result["r_squared"], 0.9)

    def test_fit_regression_logarithmic(self) -> None:
        result = self.canvas.fit_regression(
            name="Log",
            x_data=[1.0, 2.0, 3.0, 4.0, 5.0],
            y_data=[0.0, 0.69, 1.1, 1.39, 1.61],
            model_type="logarithmic",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        self.assertEqual(result["model_type"], "logarithmic")
        self.assertIn("a", result["coefficients"])
        self.assertIn("b", result["coefficients"])

    def test_fit_regression_power(self) -> None:
        result = self.canvas.fit_regression(
            name="Power",
            x_data=[1.0, 2.0, 3.0, 4.0],
            y_data=[1.0, 4.0, 9.0, 16.0],
            model_type="power",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        self.assertEqual(result["model_type"], "power")
        self.assertIn("a", result["coefficients"])
        self.assertIn("b", result["coefficients"])

    def test_fit_regression_logistic(self) -> None:
        result = self.canvas.fit_regression(
            name="Logistic",
            x_data=[1.0, 2.0, 3.0, 4.0, 5.0],
            y_data=[0.1, 0.3, 0.5, 0.7, 0.9],
            model_type="logistic",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        self.assertEqual(result["model_type"], "logistic")
        self.assertIn("L", result["coefficients"])
        self.assertIn("k", result["coefficients"])
        self.assertIn("x0", result["coefficients"])

    def test_fit_regression_sinusoidal(self) -> None:
        result = self.canvas.fit_regression(
            name="Sin",
            x_data=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            y_data=[0.0, 0.84, 0.91, 0.14, -0.76, -0.96],
            model_type="sinusoidal",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        self.assertEqual(result["model_type"], "sinusoidal")
        self.assertIn("a", result["coefficients"])
        self.assertIn("b", result["coefficients"])
        self.assertIn("c", result["coefficients"])
        self.assertIn("d", result["coefficients"])

    # -------------------------------------------------------------------------
    # Plot bounds tests
    # -------------------------------------------------------------------------

    def test_fit_regression_custom_plot_bounds(self) -> None:
        result = self.canvas.fit_regression(
            name="CustomBounds",
            x_data=[1.0, 2.0, 3.0, 4.0],
            y_data=[2.0, 4.0, 6.0, 8.0],
            model_type="linear",
            degree=None,
            plot_bounds={"left_bound": -5.0, "right_bound": 10.0},
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        self.assertEqual(result["bounds"]["left"], -5.0)
        self.assertEqual(result["bounds"]["right"], 10.0)

    def test_fit_regression_invalid_plot_bounds_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.fit_regression(
                name="BadBounds",
                x_data=[1.0, 2.0, 3.0, 4.0],
                y_data=[2.0, 4.0, 6.0, 8.0],
                model_type="linear",
                degree=None,
                plot_bounds={"left_bound": 10.0, "right_bound": 5.0},  # Reversed
                curve_color=None,
                show_points=False,
                point_color=None,
            )

    # -------------------------------------------------------------------------
    # Unique name generation tests
    # -------------------------------------------------------------------------

    def test_fit_regression_generates_unique_names(self) -> None:
        r1 = self.canvas.fit_regression(
            name="Reg",
            x_data=[1.0, 2.0, 3.0],
            y_data=[2.0, 4.0, 6.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        r2 = self.canvas.fit_regression(
            name="Reg",  # Same name
            x_data=[1.0, 2.0, 3.0],
            y_data=[3.0, 6.0, 9.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        # Function names should be unique
        self.assertNotEqual(r1["function_name"], r2["function_name"])
        self.assertIn("Reg", r1["function_name"])
        self.assertIn("Reg", r2["function_name"])

    # -------------------------------------------------------------------------
    # Deletion tests
    # -------------------------------------------------------------------------

    def test_delete_function_removes_regression_curve(self) -> None:
        result = self.canvas.fit_regression(
            name="ToDelete",
            x_data=[1.0, 2.0, 3.0, 4.0],
            y_data=[2.0, 4.0, 6.0, 8.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=True,
            point_color=None,
        )

        function_name = result["function_name"]
        point_names = result.get("point_names", [])

        # Verify function exists
        self.assertIn(function_name, self._names_for_class("Function"))

        # Delete the function directly
        self.assertTrue(self.canvas.delete_function(function_name))

        # Function should be removed, points remain (independent)
        self.assertNotIn(function_name, self._names_for_class("Function"))
        for pt in point_names:
            self.assertIn(pt, self._names_for_class("Point"))

    # -------------------------------------------------------------------------
    # Case insensitivity test
    # -------------------------------------------------------------------------

    def test_fit_regression_model_type_case_insensitive(self) -> None:
        result = self.canvas.fit_regression(
            name="CaseTest",
            x_data=[1.0, 2.0, 3.0, 4.0],
            y_data=[2.0, 4.0, 6.0, 8.0],
            model_type="LINEAR",  # Uppercase
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        self.assertEqual(result["model_type"], "linear")

    # -------------------------------------------------------------------------
    # Edge cases and additional coverage
    # -------------------------------------------------------------------------

    def test_fit_regression_null_name_generates_name(self) -> None:
        result = self.canvas.fit_regression(
            name=None,
            x_data=[1.0, 2.0, 3.0, 4.0],
            y_data=[2.0, 4.0, 6.0, 8.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        # Should auto-generate a name based on model type
        self.assertIn("linear", result["function_name"].lower())

    def test_fit_regression_empty_name_generates_name(self) -> None:
        result = self.canvas.fit_regression(
            name="",
            x_data=[1.0, 2.0, 3.0, 4.0],
            y_data=[2.0, 4.0, 6.0, 8.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        # Should auto-generate a name
        self.assertTrue(len(result["function_name"]) > 0)

    def test_fit_regression_default_show_points_is_true(self) -> None:
        result = self.canvas.fit_regression(
            name="DefaultPoints",
            x_data=[1.0, 2.0, 3.0],
            y_data=[2.0, 4.0, 6.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=None,  # Default should be True
            point_color=None,
        )

        # Points should be created by default
        self.assertIn("point_names", result)
        self.assertEqual(len(result["point_names"]), 3)

    def test_fit_regression_with_curve_color(self) -> None:
        result = self.canvas.fit_regression(
            name="ColoredCurve",
            x_data=[1.0, 2.0, 3.0, 4.0],
            y_data=[2.0, 4.0, 6.0, 8.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color="red",
            show_points=False,
            point_color=None,
        )

        # Verify function was created (color is applied internally)
        self.assertIn("function_name", result)
        function_names = self._names_for_class("Function")
        self.assertIn(result["function_name"], function_names)

    def test_fit_regression_partial_plot_bounds(self) -> None:
        # Only left bound specified
        result = self.canvas.fit_regression(
            name="PartialBounds",
            x_data=[1.0, 2.0, 3.0, 4.0],
            y_data=[2.0, 4.0, 6.0, 8.0],
            model_type="linear",
            degree=None,
            plot_bounds={"left_bound": -10.0, "right_bound": None},
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        self.assertEqual(result["bounds"]["left"], -10.0)
        # Right bound should default to data range + padding
        self.assertGreater(result["bounds"]["right"], 4.0)

    def test_fit_regression_negative_values(self) -> None:
        result = self.canvas.fit_regression(
            name="NegativeData",
            x_data=[-4.0, -2.0, 0.0, 2.0, 4.0],
            y_data=[-8.0, -4.0, 0.0, 4.0, 8.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=True,
            point_color=None,
        )

        self.assertAlmostEqual(result["coefficients"]["m"], 2.0, places=6)
        self.assertAlmostEqual(result["coefficients"]["b"], 0.0, places=6)
        self.assertEqual(len(result.get("point_names", [])), 5)

    def test_fit_regression_multiple_independent_regressions(self) -> None:
        r1 = self.canvas.fit_regression(
            name="Reg1",
            x_data=[1.0, 2.0, 3.0],
            y_data=[2.0, 4.0, 6.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=True,
            point_color=None,
        )

        r2 = self.canvas.fit_regression(
            name="Reg2",
            x_data=[1.0, 2.0, 3.0],
            y_data=[1.0, 4.0, 9.0],
            model_type="polynomial",
            degree=2,
            plot_bounds=None,
            curve_color=None,
            show_points=True,
            point_color=None,
        )

        # Both should exist independently with different function names
        self.assertNotEqual(r1["function_name"], r2["function_name"])

        # Delete first function shouldn't affect second
        self.assertTrue(self.canvas.delete_function(r1["function_name"]))
        self.assertIn(r2["function_name"], self._names_for_class("Function"))

    def test_fit_regression_empty_data_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.fit_regression(
                name="Empty",
                x_data=[],
                y_data=[],
                model_type="linear",
                degree=None,
                plot_bounds=None,
                curve_color=None,
                show_points=False,
                point_color=None,
            )

    def test_fit_regression_high_degree_polynomial(self) -> None:
        # Degree 4 polynomial with 5 points - exactly solvable
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 1.0, 16.0, 81.0, 256.0]  # y = x^4
        result = self.canvas.fit_regression(
            name="HighDegree",
            x_data=x,
            y_data=y,
            model_type="polynomial",
            degree=4,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        self.assertEqual(result["model_type"], "polynomial")
        self.assertGreater(result["r_squared"], 0.99)

    def test_fit_regression_bounds_in_result(self) -> None:
        result = self.canvas.fit_regression(
            name="BoundsCheck",
            x_data=[1.0, 2.0, 3.0, 4.0],
            y_data=[2.0, 4.0, 6.0, 8.0],
            model_type="linear",
            degree=None,
            plot_bounds=None,
            curve_color=None,
            show_points=False,
            point_color=None,
        )

        # Result should include bounds
        self.assertIn("bounds", result)
        self.assertIn("left", result["bounds"])
        self.assertIn("right", result["bounds"])
        # Default padding is 10% of range
        x_range = 4.0 - 1.0
        expected_left = 1.0 - x_range * 0.1
        expected_right = 4.0 + x_range * 0.1
        self.assertAlmostEqual(result["bounds"]["left"], expected_left, places=4)
        self.assertAlmostEqual(result["bounds"]["right"], expected_right, places=4)
