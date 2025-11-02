from __future__ import annotations

import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple

from .simple_mock import SimpleMock
from utils.linear_algebra_utils import LinearAlgebraObject, LinearAlgebraUtils
import utils.linear_algebra_utils as linear_algebra_utils_module

A_MATRIX: List[List[int]] = [
    [42, -17, 63, -5],
    [-28, 91, -74, 60],
    [39, -56, 81, -13],
    [22, -48, 9, 100],
]

B_MATRIX: List[List[int]] = [
    [15, -88, 71, 20],
    [-93, 7, -44, 55],
    [61, -36, 29, 90],
    [-14, 66, -53, 77],
]

A_PLUS_B_MATRIX: List[List[int]] = [
    [57, -105, 134, 15],
    [-121, 98, -118, 115],
    [100, -92, 110, 77],
    [8, 18, -44, 177],
]

B_INV_MATRIX: List[List[float]] = [
    [-0.01852789402109, -0.00012003596207, 0.01726042942533, -0.01527634792136],
    [0.03015604564334, -0.02073178307553, -0.02950658907737, 0.0414638983539],
    [0.05327575815057, -0.02568902469341, -0.04039437128784, 0.05172564429912],
    [0.007453579912, 0.00006621910084, 0.00062570406241, 0.01027237643632],
]

A_INV_MATRIX: List[List[float]] = [
    [0.08978748025755, -0.0431349377372, -0.1110128961747, 0.01593866015249],
    [0.00113046897795, 0.01621907946636, 0.01479939945209, -0.00775100230215],
    [-0.04488430734323, 0.03425951076629, 0.07993324152089, -0.01240860042922],
    [-0.01517103288635, 0.01419148847707, 0.02433255715856, 0.00388978770005],
]

A_INV_TIMES_B_INV_MATRIX: List[List[float]] = [
    [-0.00875984820092, 0.00373635492043, 0.00731680437918, -0.0087386531244],
    [0.00119883460219, -0.00071708153649, -0.00106171960295, 0.00134112408346],
    [0.00603075860861, -0.00275910171608, -0.00502222087331, 0.00611333377118],
    [0.0020343747376, -0.00091721587455, -0.00166106545361, 0.00211876697239],
]

BA_INV_MATRIX: List[List[float]] = [
    [-0.00875984820092, 0.00373635492043, 0.00731680437918, -0.0087386531244],
    [0.00119883460219, -0.00071708153649, -0.00106171960295, 0.00134112408346],
    [0.00603075860861, -0.00275910171608, -0.00502222087331, 0.00611333377118],
    [0.0020343747376, -0.00091721587455, -0.00166106545361, 0.00211876697239],
]


class FakeMatrix:
    def __init__(self, array: List[Any]):
        self._array: List[Any] = array
        self.isMatrix: bool = True

    def toArray(self) -> List[Any]:
        return self._array


class FakeBigNumber:
    def __init__(self, value: float):
        self._value: float = value

    def toNumber(self) -> float:
        return self._value


class FakeComplex:
    def __init__(self, value: complex):
        self.value: complex = value


class FakeMath:
    def __init__(
        self,
        return_value: Any = None,
        raise_exception: bool = False,
        *,
        type_of: Optional[Callable[[Any], str]] = None,
        format_func: Optional[Callable[[Any], str]] = None,
        number_func: Optional[Callable[[Any], float]] = None,
        has_matrix: bool = True,
    ):
        self.return_value: Any = return_value
        self.raise_exception: bool = raise_exception
        self.evaluate_calls: List[Tuple[str, Dict[str, Any]]] = []
        if has_matrix:
            self.matrix = lambda value: value
        self.transpose: Callable[[Any], Any] = lambda value: value
        self.inv: Callable[[Any], Any] = lambda value: value
        self.det: Callable[[Any], Any] = lambda value: value
        self.dot: Callable[[Any, Any], Tuple[Any, Any]] = lambda a, b: (a, b)
        self.cross: Callable[[Any, Any], Tuple[Any, Any]] = lambda a, b: (a, b)
        self.norm: Callable[[Any], Any] = lambda value: value
        self.trace: Callable[[Any], Any] = lambda value: value
        self.diag: Callable[..., List[Any]] = lambda *args: list(args)
        self.identity: Callable[[Any], Any] = lambda n: n
        self.zeros: Callable[..., Tuple[Any, ...]] = lambda *shape: shape
        self.ones: Callable[..., Tuple[Any, ...]] = lambda *shape: shape
        self.reshape: Callable[[Any, Any], Any] = lambda value, shape: (value, shape)
        self.size: Callable[[Any], Any] = lambda value: len(value)
        self.e: float = 2.718281828
        self.pi: float = 3.141592654
        self._type_of: Optional[Callable[[Any], str]] = type_of
        self._format_func: Optional[Callable[[Any], str]] = format_func
        self._number_func: Optional[Callable[[Any], float]] = number_func

    def evaluate(self, expression: str, scope: Dict[str, Any]) -> Any:
        self.evaluate_calls.append((expression, scope))
        if self.raise_exception:
            raise ValueError("evaluation failed")
        return self.return_value

    def typeOf(self, value: Any) -> str:
        if self._type_of is None:
            raise AttributeError("typeOf is not available")
        return self._type_of(value)

    def format(self, value: Any) -> str:
        if self._format_func is None:
            raise AttributeError("format is not available")
        return self._format_func(value)

    def number(self, value: Any) -> float:
        if self._number_func is None:
            raise AttributeError("number is not available")
        return self._number_func(value)


class TestLinearAlgebraUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.original_window: Any = linear_algebra_utils_module.window

    def tearDown(self) -> None:
        linear_algebra_utils_module.window = self.original_window

    def _set_math(self, math: FakeMath) -> None:
        linear_algebra_utils_module.window = SimpleMock(math=math)

    def test_evaluate_matrix_expression_returns_matrix(self) -> None:
        fake_math = FakeMath(FakeMatrix([[3.0, 5.0], [7.0, 9.0]]))
        self._set_math(fake_math)

        objects = [
            {"name": "A", "value": [[1, 2], [3, 4]]},
            {"name": "B", "value": [[2, 3], [4, 5]]},
        ]

        result = LinearAlgebraUtils.evaluate_expression(objects, "A + B")

        self.assertEqual(result["type"], "matrix")
        self.assertEqual(result["value"], [[3.0, 5.0], [7.0, 9.0]])
        self.assertEqual(fake_math.evaluate_calls[0][0], "A + B")
        self.assertIn("A", fake_math.evaluate_calls[0][1])
        self.assertIn("B", fake_math.evaluate_calls[0][1])

    def test_unknown_symbol_raises_value_error(self) -> None:
        fake_math = FakeMath(FakeMatrix([[0.0]]))
        self._set_math(fake_math)

        objects = [{"name": "A", "value": [[1, 2], [3, 4]]}]

        with self.assertRaises(ValueError):
            LinearAlgebraUtils.evaluate_expression(objects, "A + C")

        self.assertEqual(len(fake_math.evaluate_calls), 0)

    def test_invalid_matrix_definition_raises_value_error(self) -> None:
        fake_math = FakeMath(FakeMatrix([[0.0]]))
        self._set_math(fake_math)

        objects = [{"name": "A", "value": [[1, 2], [3]]}]

        with self.assertRaises(ValueError):
            LinearAlgebraUtils.evaluate_expression(objects, "A")

    def test_scalar_result_converts_to_scalar_type(self) -> None:
        fake_math = FakeMath(FakeBigNumber(5))
        self._set_math(fake_math)

        objects = [{"name": "s", "value": 2}]

        result = LinearAlgebraUtils.evaluate_expression(objects, "s * 2.5")
        self.assertEqual(result["type"], "scalar")
        self.assertEqual(result["value"], 5.0)

    def test_vector_result_detects_vector_type(self) -> None:
        fake_math = FakeMath(FakeMatrix([1, 2, 3]))
        self._set_math(fake_math)

        objects = [{"name": "v", "value": [1, 2, 3]}]

        result = LinearAlgebraUtils.evaluate_expression(objects, "v")
        self.assertEqual(result["type"], "vector")
        self.assertEqual(result["value"], [1, 2, 3])

    def test_allowed_function_added_to_scope(self) -> None:
        fake_math = FakeMath(FakeMatrix([[1.0], [2.0]]))
        self._set_math(fake_math)

        objects = [{"name": "A", "value": [[1], [2]]}]

        LinearAlgebraUtils.evaluate_expression(objects, "transpose(A)")

        expression, scope = fake_math.evaluate_calls[0]
        self.assertEqual(expression, "transpose(A)")
        self.assertIn("transpose", scope)
        self.assertIs(scope["transpose"], fake_math.transpose)

    def test_math_error_returns_error_payload(self) -> None:
        fake_math = FakeMath(raise_exception=True)
        self._set_math(fake_math)

        objects = [{"name": "A", "value": [[1.0]]}]

        result = LinearAlgebraUtils.evaluate_expression(objects, "A")
        self.assertEqual(result["type"], "error")
        self.assertTrue(isinstance(result["value"], str))
        self.assertIn("Error:", result["value"])

    def test_duplicate_names_raise_value_error(self) -> None:
        fake_math = FakeMath(FakeMatrix([[0.0]]))
        self._set_math(fake_math)

        objects = [
            {"name": "A", "value": [[1, 2], [3, 4]]},
            {"name": "A", "value": [[5, 6], [7, 8]]},
        ]

        with self.assertRaises(ValueError):
            LinearAlgebraUtils.evaluate_expression(objects, "A + B")

        self.assertEqual(len(fake_math.evaluate_calls), 0)

    def test_allowlist_accepts_common_math_functions(self) -> None:
        fake_math = FakeMath(FakeBigNumber(0))
        self._set_math(fake_math)

        objects = [{"name": "A", "value": 0}]

        result = LinearAlgebraUtils.evaluate_expression(objects, "sin(pi)")
        self.assertEqual(result["type"], "scalar")
        self.assertEqual(result["value"], 0.0)

    def test_scope_includes_supported_functions_and_constants(self) -> None:
        fake_math = FakeMath(FakeMatrix([[1.0]]))
        self._set_math(fake_math)

        objects = [{"name": "A", "value": [[1.0]]}]

        LinearAlgebraUtils.evaluate_expression(objects, "A")

        _, scope = fake_math.evaluate_calls[0]
        for name in LinearAlgebraUtils.ALLOWED_FUNCTION_NAMES:
            self.assertIn(name, scope)
        self.assertIn("pi", scope)
        self.assertIn("e", scope)

    def test_empty_expression_raises_value_error(self) -> None:
        fake_math = FakeMath()
        self._set_math(fake_math)

        objects = [{"name": "A", "value": [[1.0]]}]

        with self.assertRaises(ValueError):
            LinearAlgebraUtils.evaluate_expression(objects, " ")

        self.assertEqual(len(fake_math.evaluate_calls), 0)

    def test_mixed_dimension_array_raises_value_error(self) -> None:
        fake_math = FakeMath(FakeMatrix([[0.0]]))
        self._set_math(fake_math)

        objects = [{"name": "A", "value": [[1, 2], 3]}]

        with self.assertRaises(ValueError):
            LinearAlgebraUtils.evaluate_expression(objects, "A")

    def test_large_matrix_addition_matches_expected_values(self) -> None:
        fake_math = FakeMath(FakeMatrix(A_PLUS_B_MATRIX))
        self._set_math(fake_math)

        objects = [
            {"name": "A", "value": A_MATRIX},
            {"name": "B", "value": B_MATRIX},
        ]

        result = LinearAlgebraUtils.evaluate_expression(objects, "A + B")

        self.assertEqual(result["type"], "matrix")
        self.assertEqual(result["value"], A_PLUS_B_MATRIX)
        self.assertEqual(fake_math.evaluate_calls[0][0], "A + B")

    def test_matrix_inverse_matches_expected_values(self) -> None:
        fake_math = FakeMath(FakeMatrix(B_INV_MATRIX))
        self._set_math(fake_math)

        objects = [{"name": "B", "value": B_MATRIX}]

        result = LinearAlgebraUtils.evaluate_expression(objects, "inv(B)")

        self.assertEqual(result["type"], "matrix")
        self.assertEqual(result["value"], B_INV_MATRIX)
        self.assertEqual(fake_math.evaluate_calls[0][0], "inv(B)")
        scope = fake_math.evaluate_calls[0][1]
        self.assertIn("inv", scope)

    def test_matrix_product_matches_expected_values(self) -> None:
        fake_math = FakeMath(FakeMatrix(A_INV_TIMES_B_INV_MATRIX))
        self._set_math(fake_math)

        objects = [
            {"name": "Ainv", "value": A_INV_MATRIX},
            {"name": "Binv", "value": B_INV_MATRIX},
        ]

        result = LinearAlgebraUtils.evaluate_expression(objects, "Ainv * Binv")

        self.assertEqual(result["type"], "matrix")
        self.assertEqual(result["value"], A_INV_TIMES_B_INV_MATRIX)
        expression, scope = fake_math.evaluate_calls[0]
        self.assertEqual(expression, "Ainv * Binv")
        self.assertIn("Ainv", scope)
        self.assertIn("Binv", scope)

    def test_ba_inverse_matches_product_of_inverses(self) -> None:
        inverse_math = FakeMath(FakeMatrix(BA_INV_MATRIX))
        self._set_math(inverse_math)

        objects_ba = [
            {"name": "A", "value": A_MATRIX},
            {"name": "B", "value": B_MATRIX},
        ]

        inverse_result = LinearAlgebraUtils.evaluate_expression(objects_ba, "inv(B * A)")

        self.assertEqual(inverse_result["type"], "matrix")
        self.assertEqual(inverse_result["value"], BA_INV_MATRIX)
        self.assertEqual(inverse_math.evaluate_calls[0][0], "inv(B * A)")

        product_math = FakeMath(FakeMatrix(A_INV_TIMES_B_INV_MATRIX))
        self._set_math(product_math)

        objects_product = [
            {"name": "Ainv", "value": A_INV_MATRIX},
            {"name": "Binv", "value": B_INV_MATRIX},
        ]

        product_result = LinearAlgebraUtils.evaluate_expression(objects_product, "Ainv * Binv")

        self.assertEqual(product_result["type"], "matrix")
        self.assertEqual(product_result["value"], A_INV_TIMES_B_INV_MATRIX)
        self.assertEqual(product_math.evaluate_calls[0][0], "Ainv * Binv")

        self.assertEqual(product_result["value"], inverse_result["value"])

    def test_missing_math_matrix_returns_matrix_result(self) -> None:
        def type_of(value: Any) -> str:
            if isinstance(value, list):
                return "Array"
            return "Number"

        fake_math = FakeMath(
            return_value=A_PLUS_B_MATRIX,
            type_of=type_of,
            has_matrix=False,
        )
        self._set_math(fake_math)

        objects = [
            {"name": "A", "value": A_MATRIX},
            {"name": "B", "value": B_MATRIX},
        ]

        result = LinearAlgebraUtils.evaluate_expression(objects, "A + B")

        self.assertFalse(hasattr(fake_math, "matrix"))
        self.assertEqual(result["type"], "matrix")
        self.assertEqual(result["value"], A_PLUS_B_MATRIX)

    def test_complex_result_uses_format_conversion(self) -> None:
        complex_value = FakeComplex(1 + 2j)

        fake_math = FakeMath(
            return_value=complex_value,
            type_of=lambda value: "Complex",
            format_func=lambda value: f"{value.value.real}+{value.value.imag}i",
        )
        self._set_math(fake_math)

        objects = [{"name": "z", "value": 0}]

        result = LinearAlgebraUtils.evaluate_expression(objects, "z")

        self.assertEqual(result, {"type": "complex", "value": "1.0+2.0i"})

    def test_diag_expression_uses_allowlist_function(self) -> None:
        fake_math = FakeMath(
            return_value=[1, 2, 3],
            type_of=lambda value: "Array" if isinstance(value, list) else "Number",
        )
        self._set_math(fake_math)

        objects: List[LinearAlgebraObject] = []

        result = LinearAlgebraUtils.evaluate_expression(objects, "diag(1, 2, 3)")

        self.assertEqual(result["type"], "vector")
        self.assertEqual(result["value"], [1.0, 2.0, 3.0])
        _, scope = fake_math.evaluate_calls[0]
        self.assertIn("diag", scope)


if __name__ == "__main__":
    unittest.main()

