"""Tests for the pure rref/rank helpers and for how LinearAlgebraUtils exposes them.

matrix_rref has no browser imports, so it runs under plain pytest. The expression-level
checks against real math.js live in the Brython suite (test_linear_algebra_decompositions).
"""

from __future__ import annotations

import unittest
from typing import Any, List

from server_tests import client_renderer  # noqa: F401  (installs the browser stub)
import utils.linear_algebra_utils as linear_algebra_utils_module
from utils.linear_algebra_utils import LinearAlgebraUtils
from utils.matrix_rref import matrix_rank, rref


def _assert_rows_close(case: unittest.TestCase, actual: List[List[float]], expected: List[List[float]]) -> None:
    case.assertEqual(len(actual), len(expected))
    for actual_row, expected_row in zip(actual, expected):
        case.assertEqual(len(actual_row), len(expected_row))
        for a, e in zip(actual_row, expected_row):
            case.assertAlmostEqual(a, e, places=12)


class TestRref(unittest.TestCase):
    def test_full_rank_square_reduces_to_identity(self) -> None:
        reduced, rank = rref([[1, 2], [3, 4]])
        self.assertEqual(reduced, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(rank, 2)

    def test_full_rank_3x3_needs_row_swaps(self) -> None:
        reduced, rank = rref([[0, 2, 1], [1, 1, 1], [2, 1, 0]])
        self.assertEqual(reduced, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(rank, 3)

    def test_rank_deficient_matrix(self) -> None:
        reduced, rank = rref([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
        _assert_rows_close(self, reduced, [[1, 0, -1], [0, 1, 2], [0, 0, 0]])
        self.assertEqual(rank, 2)

    def test_rectangular_wide_matrix(self) -> None:
        reduced, rank = rref([[1, 2, 3], [4, 5, 6]])
        _assert_rows_close(self, reduced, [[1, 0, -1], [0, 1, 2]])
        self.assertEqual(rank, 2)

    def test_rectangular_tall_matrix(self) -> None:
        reduced, rank = rref([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(reduced, [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        self.assertEqual(rank, 2)

    def test_zero_matrix_has_rank_zero(self) -> None:
        reduced, rank = rref([[0, 0, 0], [0, 0, 0]])
        self.assertEqual(reduced, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertEqual(rank, 0)

    def test_pivot_free_column_in_the_middle(self) -> None:
        reduced, rank = rref([[1, 2, 0], [2, 4, 1]])
        self.assertEqual(reduced, [[1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(rank, 2)

    def test_single_row_and_single_column(self) -> None:
        self.assertEqual(rref([[2, 4, 6]]), ([[1.0, 2.0, 3.0]], 1))
        self.assertEqual(rref([[0], [3], [6]]), ([[1.0], [0.0], [0.0]], 1))

    def test_near_singular_matrix_is_rank_deficient_within_tolerance(self) -> None:
        # The second row is twice the first up to 1e-13 noise, far below 1e-10 of the largest entry
        reduced, rank = rref([[1, 2], [2, 4 + 1e-13]])
        self.assertEqual(rank, 1)
        self.assertEqual(reduced, [[1.0, 2.0], [0.0, 0.0]])

    def test_near_singular_matrix_keeps_rank_with_tight_tolerance(self) -> None:
        _, rank = rref([[1, 2], [2, 4 + 1e-13]], tol=1e-16)
        self.assertEqual(rank, 2)

    def test_small_but_genuine_entries_are_not_dropped(self) -> None:
        # Relative tolerance: a matrix scaled by 1e-8 keeps its rank
        reduced, rank = rref([[1e-8, 0], [0, 2e-8]])
        self.assertEqual(rank, 2)
        self.assertEqual(reduced, [[1.0, 0.0], [0.0, 1.0]])

    def test_small_result_entries_of_a_large_matrix_are_kept(self) -> None:
        # The pivot tolerance scales with the input; result entries are pivot-normalised,
        # so a small one must not be zeroed against the input's scale.
        reduced, rank = rref([[1e6, 1], [0, 1e6]])
        self.assertEqual(rank, 2)
        self.assertEqual(reduced, [[1.0, 0.0], [0.0, 1.0]])
        reduced, _ = rref([[1e6, 2, 0], [0, 0, 1e6]])
        self.assertEqual(reduced, [[1.0, 2e-6, 0.0], [0.0, 0.0, 1.0]])

    def test_explicit_tolerance_is_absolute(self) -> None:
        _, loose = rref([[1, 0], [0, 0.01]], tol=0.1)
        _, tight = rref([[1, 0], [0, 0.01]], tol=0.001)
        self.assertEqual((loose, tight), (1, 2))
        # Entries of a column without a pivot are zero within the tolerance
        self.assertEqual(rref([[1, 0], [0, 0.01]], tol=0.1)[0], [[1.0, 0.0], [0.0, 0.0]])

    def test_float_noise_is_snapped_to_clean_values(self) -> None:
        reduced, _ = rref([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]])
        for row in reduced:
            for value in row:
                self.assertEqual(value, round(value, 12))
        self.assertEqual(reduced[0][0], 1.0)
        self.assertEqual(reduced[1][0], 0.0)
        self.assertFalse(str(reduced[1][0]).startswith("-"))

    def test_result_is_idempotent(self) -> None:
        once, rank = rref([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]])
        twice, rank_again = rref(once)
        _assert_rows_close(self, twice, once)
        self.assertEqual(rank, rank_again)
        _assert_rows_close(self, once, [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, -1]])

    def test_input_is_not_modified(self) -> None:
        source = [[1, 2], [3, 4]]
        rref(source)
        self.assertEqual(source, [[1, 2], [3, 4]])

    def test_matrix_rank(self) -> None:
        self.assertEqual(matrix_rank([[1, 2], [2, 4]]), 1)
        self.assertEqual(matrix_rank([[1, 0], [0, 1]]), 2)
        self.assertEqual(matrix_rank([[0, 0]]), 0)

    def test_invalid_input_raises_value_error(self) -> None:
        bad_inputs: List[Any] = [[], [[]], [1, 2, 3], [[1, 2], [3]], [[1, float("nan")]], [[float("inf"), 1]], "A"]
        for bad in bad_inputs:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    rref(bad)

    def test_invalid_tolerance_raises_value_error(self) -> None:
        for bad_tol in (-1.0, True, "x"):
            with self.subTest(tol=bad_tol):
                with self.assertRaises(ValueError):
                    rref([[1, 2], [3, 4]], tol=bad_tol)  # type: ignore[arg-type]


class _Matrix:
    """Stand-in for a math.js matrix: only toArray() is used."""

    def __init__(self, rows: Any) -> None:
        self._rows = rows

    def toArray(self) -> Any:
        return self._rows


class _Json:
    @staticmethod
    def stringify(value: Any) -> str:
        import json

        return json.dumps(value)


class _FakeWindow:
    def __init__(self, evaluated: Any = None) -> None:
        self.math = _FakeMath(evaluated)
        self.JSON = _Json()


class _FakeMath:
    def __init__(self, evaluated: Any) -> None:
        self._evaluated = evaluated
        self.pi = 3.141592653589793
        self.e = 2.718281828459045
        self.lup = self.eigs = lambda value: value

    def matrix(self, value: Any) -> _Matrix:
        return _Matrix(value)

    def evaluate(self, expression: str, scope: Any) -> Any:
        return self._evaluated

    def typeOf(self, value: Any) -> str:
        return "Object" if isinstance(value, dict) else "Number"


class TestRrefBindings(unittest.TestCase):
    def setUp(self) -> None:
        self.original_window: Any = linear_algebra_utils_module.window

    def tearDown(self) -> None:
        linear_algebra_utils_module.window = self.original_window

    def _use(self, evaluated: Any = None) -> None:
        linear_algebra_utils_module.window = _FakeWindow(evaluated)

    def test_rref_and_rank_are_bound_into_the_scope(self) -> None:
        self._use(1.0)
        scope = LinearAlgebraUtils._build_scope([{"name": "A", "value": [[1, 2], [2, 4]]}])
        self.assertIn("rref", scope)
        self.assertIn("rank", scope)

    def test_object_named_rank_is_not_shadowed(self) -> None:
        self._use(1.0)
        scope = LinearAlgebraUtils._build_scope([{"name": "rank", "value": 3}])
        self.assertEqual(scope["rank"], 3.0)

    def test_rref_binding_returns_a_matrix_of_the_reduced_rows(self) -> None:
        self._use()
        result = LinearAlgebraUtils._rref_binding(_Matrix([[1.0, 2.0], [2.0, 4.0]]))
        self.assertEqual(result.toArray(), [[1.0, 2.0], [0.0, 0.0]])

    def test_rank_binding_accepts_tolerance(self) -> None:
        self._use()
        matrix = _Matrix([[1.0, 0.0], [0.0, 0.01]])
        self.assertEqual(LinearAlgebraUtils._rank_binding(matrix), 2)
        self.assertEqual(LinearAlgebraUtils._rank_binding(matrix, 0.1), 1)

    def test_vector_and_scalar_arguments_are_rejected(self) -> None:
        self._use()
        for bad in (_Matrix([1.0, 2.0]), 5.0):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    LinearAlgebraUtils._rref_binding(bad)

    def test_non_numeric_tolerance_is_rejected(self) -> None:
        self._use()
        with self.assertRaises(ValueError):
            LinearAlgebraUtils._rank_binding(_Matrix([[1.0]]), "tiny")

    def test_object_results_are_converted_to_json_ready_values(self) -> None:
        # Regression: eigs/lup/qr return plain objects, which used to come back as "unknown"
        tagged = {
            "L": {"mathjs": "DenseMatrix", "data": [[1, 0], [0.5, 1]], "size": [2, 2]},
            "U": {"mathjs": "DenseMatrix", "data": [[2, 4], [0, 0]], "size": [2, 2]},
            "p": [1, 0],
        }
        self._use(tagged)
        result = LinearAlgebraUtils.evaluate_expression([{"name": "A", "value": [[1, 2], [2, 4]]}], "lup(A)")
        self.assertEqual(result["type"], "object")
        self.assertEqual(result["value"], {"L": [[1, 0], [0.5, 1]], "U": [[2, 4], [0, 0]], "p": [1, 0]})

    def test_complex_entries_are_formatted_as_text(self) -> None:
        tagged = {
            "values": {
                "mathjs": "DenseMatrix",
                "data": [{"mathjs": "Complex", "re": 0, "im": 1}, {"mathjs": "Complex", "re": 0.5, "im": -2}],
                "size": [2],
            }
        }
        self._use(tagged)
        result = LinearAlgebraUtils.evaluate_expression([{"name": "A", "value": [[0, -1], [1, 0]]}], "eigs(A)")
        self.assertEqual(result["value"], {"values": ["0 + 1i", "0.5 - 2i"]})

    def test_object_results_may_not_be_json_serializable(self) -> None:
        self._use({"bad": 1})
        linear_algebra_utils_module.window.JSON = _FailingJson()
        with self.assertRaises(ValueError):
            LinearAlgebraUtils.evaluate_expression([{"name": "A", "value": [[1]]}], "A")


class _FailingJson:
    @staticmethod
    def stringify(value: Any) -> str:
        raise TypeError("cannot stringify")


if __name__ == "__main__":
    unittest.main()
