"""Tests for ResultProcessor.get_results_traced — traced execution path."""

from __future__ import annotations

import unittest
from typing import Any, Dict

from canvas import Canvas
from process_function_calls import ProcessFunctionCalls
from drawables_aggregator import Position
from .simple_mock import SimpleMock


class TestGetResultsTraced(unittest.TestCase):
    """Verify get_results_traced returns correct results and trace metadata."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.mock_cartesian2axis = SimpleMock(
            draw=SimpleMock(return_value=None),
            reset=SimpleMock(return_value=None),
            get_state=SimpleMock(return_value={"Cartesian_System_Visibility": "cartesian_state"}),
            origin=Position(0, 0),
        )
        self.canvas.cartesian2axis = self.mock_cartesian2axis

    def test_returns_results_and_trace(self) -> None:
        available_functions: Dict[str, Any] = {
            "evaluate_expression": ProcessFunctionCalls.evaluate_expression,
        }
        calls = [{"function_name": "evaluate_expression", "arguments": {"expression": "3 + 7", "canvas": self.canvas}}]
        results, traced = ProcessFunctionCalls.get_results_traced(
            calls, available_functions, (), self.canvas,
        )
        # Results should contain the evaluation
        self.assertTrue(len(results) > 0)
        # Traced calls should have exactly 1 entry
        self.assertEqual(len(traced), 1)
        self.assertEqual(traced[0]["function_name"], "evaluate_expression")
        self.assertEqual(traced[0]["seq"], 0)
        self.assertFalse(traced[0]["is_error"])

    def test_records_timing(self) -> None:
        available_functions: Dict[str, Any] = {
            "evaluate_expression": ProcessFunctionCalls.evaluate_expression,
        }
        calls = [{"function_name": "evaluate_expression", "arguments": {"expression": "2 * 3", "canvas": self.canvas}}]
        _, traced = ProcessFunctionCalls.get_results_traced(
            calls, available_functions, (), self.canvas,
        )
        self.assertGreaterEqual(traced[0]["duration_ms"], 0)

    def test_records_errors(self) -> None:
        def always_fail(**kwargs: Any) -> str:
            raise ValueError("intentional error")

        available_functions: Dict[str, Any] = {"fail_func": always_fail}
        calls = [{"function_name": "fail_func", "arguments": {}}]
        results, traced = ProcessFunctionCalls.get_results_traced(
            calls, available_functions, (), self.canvas,
        )
        self.assertTrue(traced[0]["is_error"])
        self.assertIn("Error", str(results.get("fail_func", "")))

    def test_same_results_as_get_results(self) -> None:
        """get_results_traced should produce identical results dict as get_results."""
        available_functions: Dict[str, Any] = {
            "evaluate_expression": ProcessFunctionCalls.evaluate_expression,
        }
        calls = [
            {"function_name": "evaluate_expression", "arguments": {"expression": "5 + 5", "canvas": self.canvas}},
            {"function_name": "evaluate_expression", "arguments": {"expression": "10 * 2", "canvas": self.canvas}},
        ]
        results_normal = ProcessFunctionCalls.get_results(
            calls, available_functions, (), self.canvas,
        )
        results_traced, _ = ProcessFunctionCalls.get_results_traced(
            calls, available_functions, (), self.canvas,
        )
        self.assertEqual(results_normal, results_traced)

    def test_sanitized_arguments(self) -> None:
        """Traced call arguments should exclude the canvas reference."""
        available_functions: Dict[str, Any] = {
            "evaluate_expression": ProcessFunctionCalls.evaluate_expression,
        }
        calls = [{"function_name": "evaluate_expression", "arguments": {"expression": "1+1", "canvas": self.canvas}}]
        _, traced = ProcessFunctionCalls.get_results_traced(
            calls, available_functions, (), self.canvas,
        )
        self.assertNotIn("canvas", traced[0]["arguments"])
        self.assertIn("expression", traced[0]["arguments"])

    def test_missing_function(self) -> None:
        """Calling a non-existent function should record as error in trace."""
        available_functions: Dict[str, Any] = {}
        calls = [{"function_name": "nonexistent", "arguments": {}}]
        results, traced = ProcessFunctionCalls.get_results_traced(
            calls, available_functions, (), self.canvas,
        )
        self.assertTrue(traced[0]["is_error"])
        self.assertIn("nonexistent", results)

    def test_result_value_captured(self) -> None:
        """The traced call should capture the actual result value."""
        available_functions: Dict[str, Any] = {
            "evaluate_expression": ProcessFunctionCalls.evaluate_expression,
        }
        calls = [{"function_name": "evaluate_expression", "arguments": {"expression": "3 + 7", "canvas": self.canvas}}]
        _, traced = ProcessFunctionCalls.get_results_traced(
            calls, available_functions, (), self.canvas,
        )
        self.assertEqual(traced[0]["result"], 10)

    def test_result_value_for_expression_with_variables(self) -> None:
        """Expression evaluation with variables should capture correct result."""
        available_functions: Dict[str, Any] = {
            "evaluate_expression": ProcessFunctionCalls.evaluate_expression,
        }
        calls = [{"function_name": "evaluate_expression", "arguments": {
            "expression": "x + y", "variables": {"x": 5, "y": 3}, "canvas": self.canvas,
        }}]
        _, traced = ProcessFunctionCalls.get_results_traced(
            calls, available_functions, (), self.canvas,
        )
        self.assertEqual(traced[0]["result"], 8)

    def test_multiple_calls_sequential_order(self) -> None:
        """Multiple calls should have sequential seq numbers."""
        available_functions: Dict[str, Any] = {
            "evaluate_expression": ProcessFunctionCalls.evaluate_expression,
        }
        calls = [
            {"function_name": "evaluate_expression", "arguments": {"expression": "1+1", "canvas": self.canvas}},
            {"function_name": "evaluate_expression", "arguments": {"expression": "2+2", "canvas": self.canvas}},
            {"function_name": "evaluate_expression", "arguments": {"expression": "3+3", "canvas": self.canvas}},
        ]
        _, traced = ProcessFunctionCalls.get_results_traced(
            calls, available_functions, (), self.canvas,
        )
        self.assertEqual(len(traced), 3)
        self.assertEqual([t["seq"] for t in traced], [0, 1, 2])
