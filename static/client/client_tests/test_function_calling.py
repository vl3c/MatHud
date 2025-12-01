from __future__ import annotations

import unittest
from typing import Any, Dict, List

from drawables_aggregator import Position
from canvas import Canvas
from process_function_calls import ProcessFunctionCalls
from .simple_mock import SimpleMock
from utils.linear_algebra_utils import LinearAlgebraUtils
from utils.linear_algebra_utils import LinearAlgebraResult


class TestProcessFunctionCalls(unittest.TestCase):
    def setUp(self) -> None:
        # Setup the mock canvas and its functions as described
        self.canvas = Canvas(500, 500, draw_enabled=False)  # Assuming a basic mock or actual Canvas class
        self.mock_cartesian2axis = SimpleMock(draw=SimpleMock(return_value=None), 
                                              reset=SimpleMock(return_value=None),
                                              get_state=SimpleMock(return_value={'Cartesian_System_Visibility': 'cartesian_state'}),
                                              origin=Position(0, 0))  # Assuming Position is defined elsewhere
        self.canvas.cartesian2axis = self.mock_cartesian2axis
        
        # Mocking a function in canvas.drawables['Function']
        self.function_string, self.name = "x^2", "Quadratic"
        # Assuming draw_function method is available to add mock functions
        self.f = self.canvas.draw_function(self.function_string, self.name)
        self._original_linear_algebra_evaluate = LinearAlgebraUtils.evaluate_expression

    def tearDown(self) -> None:
        LinearAlgebraUtils.evaluate_expression = staticmethod(self._original_linear_algebra_evaluate)

    def test_evaluate_numeric_expression(self) -> None:
        expression = "3 + 7"
        result = ProcessFunctionCalls.evaluate_expression(expression, variables=None, canvas=self.canvas)
        self.assertEqual(result, 10)

    def test_evaluate_expression_with_variables(self) -> None:
        expression = "x - 4 + y * 5"
        variables = {'x': 7, 'y': 65}
        result = ProcessFunctionCalls.evaluate_expression(expression, variables=variables, canvas=self.canvas)
        self.assertEqual(result, 328)  # Expected result for "x - 4 + y * 5" with x = 7 and y = 65

    def test_evaluate_expression_combinatorics(self) -> None:
        result = ProcessFunctionCalls.evaluate_expression("arrangements(6, 3)", canvas=self.canvas)
        self.assertEqual(result, 120)

        result = ProcessFunctionCalls.evaluate_expression("permutations(5, 2)", canvas=self.canvas)
        self.assertEqual(result, 20)

        result = ProcessFunctionCalls.evaluate_expression("permutations(5)", canvas=self.canvas)
        self.assertEqual(result, 120)

        result = ProcessFunctionCalls.evaluate_expression("combinations(6, 3)", canvas=self.canvas)
        self.assertEqual(result, 20)

    def test_evaluate_expression_complex_combinatorics(self) -> None:
        expression = "2*combinations(8, 3) + permutations(4, 2) - arrangements(5, 2)"
        result = ProcessFunctionCalls.evaluate_expression(expression, canvas=self.canvas)
        self.assertEqual(result, 104)

        expression = "(combinations(10, 4) / combinations(6, 2)) * permutations(3)"
        result = ProcessFunctionCalls.evaluate_expression(expression, canvas=self.canvas)
        self.assertEqual(result, 84)

    def test_evaluate_function_expression(self) -> None:
        expression = "Quadratic(5)"
        result = ProcessFunctionCalls.evaluate_expression(expression, variables=None, canvas=self.canvas)
        self.assertEqual(result, 25)  # Expected result for "Quadratic(5)"

    def test_get_results1(self) -> None:
        available_functions = {'evaluate_expression': ProcessFunctionCalls.evaluate_expression}
        calls = [{'function_name': 'evaluate_expression', 'arguments': {'expression': 'Quadratic(5)', 'canvas': self.canvas}}]
        undoable_functions = ()  # Example, assuming no undoable functions for simplicity
        results: Dict[str, Any] = ProcessFunctionCalls.get_results(
            calls,
            available_functions,
            undoable_functions,
            self.canvas,
        )
        self.assertTrue(len(results) > 0)
        self.assertIn('Quadratic(5)', results) # Check if the result for "Quadratic(5)" is available
        self.assertEqual(results['Quadratic(5)'], 25)  # Expected result for "Quadratic(5)"

    def test_get_results2(self) -> None:
        available_functions = {'evaluate_expression': ProcessFunctionCalls.evaluate_expression}
        calls = [{'function_name': 'evaluate_expression', 'arguments': {'expression': 'x + y', 'variables': {'x': 5, 'y': 1}, 'canvas': self.canvas}}]
        undoable_functions = ()  # Example, assuming no undoable functions for simplicity
        results = ProcessFunctionCalls.get_results(calls, available_functions, undoable_functions, self.canvas)
        self.assertTrue(len(results) > 0)
        self.assertIn('x+y for x:5, y:1', results)
        self.assertEqual(results['x+y for x:5, y:1'], 6)

    def test_evaluate_expression_invalid_function(self) -> None:
        # Testing with an invalid function expression
        expression = "NonExistentFunction(10)"
        result = ProcessFunctionCalls.evaluate_expression(expression, variables=None, canvas=self.canvas)
        self.assertTrue("Sorry" in result)

    def test_evaluate_linear_algebra_expression_delegates_to_utils(self) -> None:
        captured_calls: List[Any] = []

        def fake_evaluate(objects: List[Dict[str, Any]], expression: str) -> LinearAlgebraResult:
            captured_calls.append((objects, expression))
            return {"type": "matrix", "value": [[1.0]]}

        LinearAlgebraUtils.evaluate_expression = staticmethod(fake_evaluate)

        objects = [{"name": "A", "value": [[1, 2], [3, 4]]}]
        expression = "A"

        result = ProcessFunctionCalls.evaluate_linear_algebra_expression(objects, expression)

        self.assertEqual(result, {"type": "matrix", "value": [[1.0]]})
        self.assertEqual(captured_calls, [(objects, expression)])

    def test_evaluate_linear_algebra_expression_propagates_errors(self) -> None:
        def fake_evaluate(*_: Any) -> Dict[str, Any]:
            raise ValueError("invalid data")

        LinearAlgebraUtils.evaluate_expression = staticmethod(fake_evaluate)

        with self.assertRaises(ValueError):
            ProcessFunctionCalls.evaluate_linear_algebra_expression([
                {"name": "A", "value": [[1.0]]}
            ], "A")

    def test_evaluate_linear_algebra_expression_supports_grouped_operations(self) -> None:
        captured_calls: List[Any] = []

        def fake_evaluate(objects: List[Dict[str, Any]], expression: str) -> LinearAlgebraResult:
            captured_calls.append((objects, expression))
            return {"type": "matrix", "value": [[42.0]]}

        LinearAlgebraUtils.evaluate_expression = staticmethod(fake_evaluate)

        objects = [
            {"name": "A", "value": [[1, 0], [0, 1]]},
            {"name": "Binv", "value": [[1, 0], [0, 1]]},
        ]

        result = ProcessFunctionCalls.evaluate_linear_algebra_expression(
            objects,
            "transpose(A) * Binv",
        )

        self.assertEqual(result["value"], [[42.0]])
        self.assertEqual(captured_calls[0][1], "transpose(A) * Binv")

    def test_get_results_records_matrix_sum(self) -> None:
        expected_result: LinearAlgebraResult = {
            "type": "matrix",
            "value": [
                [57, -105, 134, 15],
                [-121, 98, -118, 115],
                [100, -92, 110, 77],
                [8, 18, -44, 177],
            ],
        }

        def fake_evaluate(objects: List[Dict[str, Any]], expression: str) -> LinearAlgebraResult:
            self.assertEqual(expression, "A + B")
            self.assertEqual(len(objects), 2)
            return expected_result

        LinearAlgebraUtils.evaluate_expression = staticmethod(fake_evaluate)

        objects = [
            {"name": "A", "value": [[42, -17, 63, -5], [-28, 91, -74, 60], [39, -56, 81, -13], [22, -48, 9, 100]]},
            {"name": "B", "value": [[15, -88, 71, 20], [-93, 7, -44, 55], [61, -36, 29, 90], [-14, 66, -53, 77]]},
        ]

        calls = [
            {
                "function_name": "evaluate_linear_algebra_expression",
                "arguments": {"objects": objects, "expression": "A + B"},
            }
        ]

        available_functions = {
            "evaluate_linear_algebra_expression": ProcessFunctionCalls.evaluate_linear_algebra_expression,
        }

        results = ProcessFunctionCalls.get_results(calls, available_functions, (), self.canvas)

        self.assertEqual(len(results), 1)
        key, value = next(iter(results.items()))
        self.assertIn("expression:A + B", key)
        self.assertIn("objects:[{'name': 'A'", key)
        self.assertEqual(value, expected_result)

    def test_get_results_records_matrix_inverse(self) -> None:
        expected_result: LinearAlgebraResult = {
            "type": "matrix",
            "value": [
                [-0.01852789402109, -0.00012003596207, 0.01726042942533, -0.01527634792136],
                [0.03015604564334, -0.02073178307553, -0.02950658907737, 0.0414638983539],
                [0.05327575815057, -0.02568902469341, -0.04039437128784, 0.05172564429912],
                [0.007453579912, 0.00006621910084, 0.00062570406241, 0.01027237643632],
            ],
        }

        def fake_evaluate(objects: List[Dict[str, Any]], expression: str) -> LinearAlgebraResult:
            self.assertEqual(expression, "inv(B)")
            self.assertEqual(len(objects), 1)
            return expected_result

        LinearAlgebraUtils.evaluate_expression = staticmethod(fake_evaluate)

        objects = [
            {"name": "B", "value": [[15, -88, 71, 20], [-93, 7, -44, 55], [61, -36, 29, 90], [-14, 66, -53, 77]]},
        ]

        calls = [
            {
                "function_name": "evaluate_linear_algebra_expression",
                "arguments": {"objects": objects, "expression": "inv(B)"},
            }
        ]

        available_functions = {
            "evaluate_linear_algebra_expression": ProcessFunctionCalls.evaluate_linear_algebra_expression,
        }

        results = ProcessFunctionCalls.get_results(calls, available_functions, (), self.canvas)

        self.assertEqual(len(results), 1)
        _, value = next(iter(results.items()))
        self.assertEqual(value, expected_result)

    def test_get_results_records_error_payload(self) -> None:
        expected_result: LinearAlgebraResult = {
            "type": "error",
            "value": "Error: singular matrix",
        }

        def fake_evaluate(*_: Any) -> LinearAlgebraResult:
            return expected_result

        LinearAlgebraUtils.evaluate_expression = staticmethod(fake_evaluate)

        objects = [
            {"name": "A", "value": [[1, 0], [0, 0]]},
        ]

        calls = [
            {
                "function_name": "evaluate_linear_algebra_expression",
                "arguments": {"objects": objects, "expression": "inv(A)"},
            }
        ]

        available_functions = {
            "evaluate_linear_algebra_expression": ProcessFunctionCalls.evaluate_linear_algebra_expression,
        }

        results = ProcessFunctionCalls.get_results(calls, available_functions, (), self.canvas)

        self.assertEqual(list(results.values())[0], expected_result)

    def test_get_results_handles_combined_operations_expression(self) -> None:
        expected_result: LinearAlgebraResult = {
            "type": "matrix",
            "value": [[42.0, 0.0], [0.0, 42.0]],
        }

        def fake_evaluate(objects: List[Dict[str, Any]], expression: str) -> LinearAlgebraResult:
            self.assertEqual(expression, "inv(transpose(A)) * diag(1, 2, 3)")
            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0]["name"], "A")
            return expected_result

        LinearAlgebraUtils.evaluate_expression = staticmethod(fake_evaluate)

        objects = [
            {"name": "A", "value": [[42, -17], [-28, 91]]},
        ]

        calls = [
            {
                "function_name": "evaluate_linear_algebra_expression",
                "arguments": {
                    "objects": objects,
                    "expression": "inv(transpose(A)) * diag(1, 2, 3)",
                },
            }
        ]

        available_functions = {
            "evaluate_linear_algebra_expression": ProcessFunctionCalls.evaluate_linear_algebra_expression,
        }

        results = ProcessFunctionCalls.get_results(calls, available_functions, (), self.canvas)

        self.assertEqual(list(results.values())[0], expected_result)

    def test_get_results_handles_four_by_four_inverse(self) -> None:
        expected_result: LinearAlgebraResult = {
            "type": "matrix",
            "value": [
                [0.08978748025755, -0.0431349377372, -0.1110128961747, 0.01593866015249],
                [0.00113046897795, 0.01621907946636, 0.01479939945209, -0.00775100230215],
                [-0.04488430734323, 0.03425951076629, 0.07993324152089, -0.01240860042922],
                [-0.01517103288635, 0.01419148847707, 0.02433255715856, 0.00388978770005],
            ],
        }

        def fake_evaluate(objects: List[Dict[str, Any]], expression: str) -> LinearAlgebraResult:
            self.assertEqual(expression, "inv(A)")
            self.assertEqual(len(objects), 1)
            return expected_result

        LinearAlgebraUtils.evaluate_expression = staticmethod(fake_evaluate)

        objects = [
            {
                "name": "A",
                "value": [
                    [42, -17, 63, -5],
                    [-28, 91, -74, 60],
                    [39, -56, 81, -13],
                    [22, -48, 9, 100],
                ],
            }
        ]

        calls = [
            {
                "function_name": "evaluate_linear_algebra_expression",
                "arguments": {"objects": objects, "expression": "inv(A)"},
            }
        ]

        available_functions = {
            "evaluate_linear_algebra_expression": ProcessFunctionCalls.evaluate_linear_algebra_expression,
        }

        results = ProcessFunctionCalls.get_results(calls, available_functions, (), self.canvas)

        self.assertEqual(list(results.values())[0], expected_result)

    def test_get_results_validates_ba_inverse_identity(self) -> None:
        sum_objects = [
            {"name": "A", "value": [[42, -17, 63, -5], [-28, 91, -74, 60], [39, -56, 81, -13], [22, -48, 9, 100]]},
            {"name": "B", "value": [[15, -88, 71, 20], [-93, 7, -44, 55], [61, -36, 29, 90], [-14, 66, -53, 77]]},
        ]

        expected_product: LinearAlgebraResult = {
            "type": "matrix",
            "value": [
                [-0.00875984820092, 0.00373635492043, 0.00731680437918, -0.0087386531244],
                [0.00119883460219, -0.00071708153649, -0.00106171960295, 0.00134112408346],
                [0.00603075860861, -0.00275910171608, -0.00502222087331, 0.00611333377118],
                [0.0020343747376, -0.00091721587455, -0.00166106545361, 0.00211876697239],
            ],
        }

        def fake_evaluate(objects: List[Dict[str, Any]], expression: str) -> LinearAlgebraResult:
            if expression == "Ainv * Binv":
                self.assertEqual(len(objects), 2)
                return expected_product
            if expression == "inv(B * A)":
                self.assertEqual(len(objects), 2)
                return expected_product
            self.fail(f"Unexpected expression {expression}")

        LinearAlgebraUtils.evaluate_expression = staticmethod(fake_evaluate)

        calls = [
            {
                "function_name": "evaluate_linear_algebra_expression",
                "arguments": {
                    "objects": [
                        {"name": "Ainv", "value": [[0.08978748025755, -0.0431349377372, -0.1110128961747, 0.01593866015249], [0.00113046897795, 0.01621907946636, 0.01479939945209, -0.00775100230215], [-0.04488430734323, 0.03425951076629, 0.07993324152089, -0.01240860042922], [-0.01517103288635, 0.01419148847707, 0.02433255715856, 0.00388978770005]]},
                        {"name": "Binv", "value": [[-0.01852789402109, -0.00012003596207, 0.01726042942533, -0.01527634792136], [0.03015604564334, -0.02073178307553, -0.02950658907737, 0.0414638983539], [0.05327575815057, -0.02568902469341, -0.04039437128784, 0.05172564429912], [0.007453579912, 0.00006621910084, 0.00062570406241, 0.01027237643632]]},
                    ],
                    "expression": "Ainv * Binv",
                },
            },
            {
                "function_name": "evaluate_linear_algebra_expression",
                "arguments": {
                    "objects": sum_objects,
                    "expression": "inv(B * A)",
                },
            },
        ]

        available_functions = {
            "evaluate_linear_algebra_expression": ProcessFunctionCalls.evaluate_linear_algebra_expression,
        }

        results = ProcessFunctionCalls.get_results(calls, available_functions, (), self.canvas)

        self.assertEqual(len(results), 2)
        values = list(results.values())
        self.assertEqual(values[0], expected_product)
        self.assertEqual(values[1], expected_product)

    def test_validate_results_with_valid_input(self) -> None:
        # Testing result validation with valid types as dictionary values
        results = {"result1": 1, "result2": "a", "result3": True, "result4": 3.14}
        self.assertTrue(ProcessFunctionCalls.validate_results(results))

    def test_validate_results_with_invalid_input1(self) -> None:
        # Testing result validation with an invalid type (list) as one of the dictionary values
        results = {"result1": 1, "result2": "a", "result3": [1, 2, 3], "result4": True}
        self.assertFalse(ProcessFunctionCalls.validate_results(results))

    def test_validate_results_with_invalid_input2(self) -> None:
        # Testing result validation with an invalid type (None) as one of the dictionary values
        results = {"result1": 1, "result2": "a", "result3": None, "result4": True}
        self.assertFalse(ProcessFunctionCalls.validate_results(results))

    def test_validate_results_accepts_linear_algebra_payload(self) -> None:
        payload: Dict[str, Any] = {
            "expression:A+B": {
                "type": "matrix",
                "value": [[10.0, 10.0], [10.0, 10.0]],
            }
        }
        self.assertTrue(ProcessFunctionCalls.validate_results(payload))

    def test_is_successful_result_handles_error_payload(self) -> None:
        error_payload = {"type": "error", "value": "Error: singular"}
        self.assertFalse(ProcessFunctionCalls.is_successful_result(error_payload))

    def test_is_successful_result_accepts_scalar_payload(self) -> None:
        scalar_payload = {"type": "scalar", "value": -10.0}
        self.assertTrue(ProcessFunctionCalls.is_successful_result(scalar_payload))

    def test_validate_results_with_empty_string_key(self) -> None:
        # Testing result validation with an empty string as a key
        results = {"": 1, "result2": "a", "result3": True, "result4": 3.14}
        self.assertFalse(ProcessFunctionCalls.validate_results(results))

    def test_validate_results_with_none_key(self) -> None:
        # Testing result validation with None as a key
        results = {None: 1, "result2": "a", "result3": True, "result4": 3.14}
        self.assertFalse(ProcessFunctionCalls.validate_results(results))

    def test_validate_results_with_all_valid_keys_and_values(self) -> None:
        # Testing result validation with all valid keys and values
        results = {"result1": 1, "result2": "a", "result3": True, "result4": 3.14}
        self.assertTrue(ProcessFunctionCalls.validate_results(results))

    def test_validate_results_with_empty_dict(self) -> None:
        # Testing result validation with an empty dictionary
        results: Dict[str, Any] = {}
        self.assertTrue(ProcessFunctionCalls.validate_results(results))
