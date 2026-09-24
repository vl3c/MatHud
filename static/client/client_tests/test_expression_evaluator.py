from __future__ import annotations

import unittest

from expression_evaluator import ExpressionEvaluator
from .simple_mock import SimpleMock


class TestExpressionEvaluatorZeroResults(unittest.TestCase):
    """A legitimate result of 0 must not be reported as an unsupported expression."""

    def test_numeric_zero_results(self) -> None:
        self.assertEqual(ExpressionEvaluator.evaluate_expression("2-2"), 0.0)
        self.assertEqual(ExpressionEvaluator.evaluate_expression("sin(0)"), 0.0)
        self.assertEqual(ExpressionEvaluator.evaluate_expression("x-3", {"x": 3}), 0.0)

    def test_large_whole_number_result(self) -> None:
        # Whole-number results above 2**53 must survive the JavaScript -> Python conversion
        result = ExpressionEvaluator.evaluate_expression("x^2", {"x": 1e9})
        self.assertAlmostEqual(result / 1e18, 1.0)

    def test_function_zero_result(self) -> None:
        quadratic = SimpleMock(name="Quadratic", function=lambda x: x * x)
        canvas = SimpleMock(get_drawables_by_class_name=SimpleMock(return_value=[quadratic]))
        self.assertEqual(ExpressionEvaluator.evaluate_expression("Quadratic(0)", canvas=canvas), 0.0)
        self.assertEqual(ExpressionEvaluator.evaluate_expression("Quadratic(3)", canvas=canvas), 9.0)

    def test_numeric_error_not_replaced_by_function_lookup_error(self) -> None:
        quadratic = SimpleMock(name="Quadratic", function=lambda x: x * x)
        canvas = SimpleMock(get_drawables_by_class_name=SimpleMock(return_value=[quadratic]))
        result = ExpressionEvaluator.evaluate_expression("exp(1000)", canvas=canvas)
        self.assertIsInstance(result, str)
        self.assertNotIn("No function found", result)
        self.assertIn("Result is infinite", result)

    def test_numeric_error_reported_for_non_call_with_canvas(self) -> None:
        canvas = SimpleMock(get_drawables_by_class_name=SimpleMock(return_value=[]))
        result = ExpressionEvaluator.evaluate_expression("1/0", canvas=canvas)
        self.assertIsInstance(result, str)
        self.assertIn("Result is infinite", result)

    def test_canvas_function_call_still_evaluated_after_numeric_error(self) -> None:
        cube = SimpleMock(name="g", function=lambda x: x**3)
        canvas = SimpleMock(get_drawables_by_class_name=SimpleMock(return_value=[cube]))
        self.assertEqual(ExpressionEvaluator.evaluate_expression("g(2)", canvas=canvas), 8.0)

    def test_error_result_still_reported(self) -> None:
        result = ExpressionEvaluator.evaluate_expression("1/0")
        self.assertIsInstance(result, str)
        self.assertIn("not a supported mathematical expression", result)
