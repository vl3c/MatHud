"""Brython tests for eigs, lup, qr, rref and rank evaluated through the real math.js.

test_linear_algebra_utils.py drives LinearAlgebraUtils with a fake math object; these tests
use the math.js instance on the page so the decompositions and the rref/rank bindings are
checked end to end (result shape, reconstruction identities, error paths).
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from utils.linear_algebra_utils import LinearAlgebraResult, LinearAlgebraUtils

Rows = List[List[Any]]

TOLERANCE = 1e-9


def _evaluate(matrices: Dict[str, Any], expression: str) -> LinearAlgebraResult:
    objects: List[Any] = [{"name": name, "value": value} for name, value in matrices.items()]
    return LinearAlgebraUtils.evaluate_expression(objects, expression)


def _multiply(a: Rows, b: Rows) -> Rows:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def _transpose(a: Rows) -> Rows:
    return [list(column) for column in zip(*a)]


def _identity(n: int) -> Rows:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


class _MatrixAssertions(unittest.TestCase):
    def assert_matrix_close(self, actual: Rows, expected: Rows, tol: float = TOLERANCE) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_row, expected_row in zip(actual, expected):
            self.assertEqual(len(actual_row), len(expected_row))
            for a, e in zip(actual_row, expected_row):
                self.assertTrue(abs(a - e) <= tol, f"{actual} != {expected}")

    def object_value(self, result: LinearAlgebraResult, *keys: str) -> Any:
        self.assertEqual(result["type"], "object", result)
        value = result["value"]
        for key in keys:
            self.assertIn(key, value)
        return value


class TestEigs(_MatrixAssertions):
    def _assert_eigenpairs(self, matrix: Rows, pairs: List[Dict[str, Any]]) -> None:
        for pair in pairs:
            value, vector = pair["value"], pair["vector"]
            product = [sum(row[k] * vector[k] for k in range(len(vector))) for row in matrix]
            scaled = [value * component for component in vector]
            for got, want in zip(product, scaled):
                self.assertTrue(abs(got - want) <= TOLERANCE, f"A*v={product} but lambda*v={scaled}")

    def test_symmetric_matrix_has_real_orthogonal_eigenvectors(self) -> None:
        matrix = [[2, 1], [1, 2]]
        result = _evaluate({"A": matrix}, "eigs(A)")
        value = self.object_value(result, "values", "eigenvectors")
        self.assertEqual(sorted(round(v, 9) for v in value["values"]), [1.0, 3.0])
        self._assert_eigenpairs(matrix, value["eigenvectors"])
        first, second = (pair["vector"] for pair in value["eigenvectors"])
        self.assertTrue(abs(sum(a * b for a, b in zip(first, second))) <= TOLERANCE)

    def test_symmetric_3x3_matrix(self) -> None:
        matrix = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
        value = self.object_value(_evaluate({"A": matrix}, "eigs(A)"), "values", "eigenvectors")
        self.assertEqual(len(value["values"]), 3)
        self.assertTrue(abs(sum(value["values"]) - 9.0) <= TOLERANCE)  # trace
        self._assert_eigenpairs(matrix, value["eigenvectors"])

    def test_non_symmetric_matrix_with_real_eigenvalues(self) -> None:
        matrix = [[2, 1], [0, 3]]
        value = self.object_value(_evaluate({"A": matrix}, "eigs(A)"), "values", "eigenvectors")
        self.assertEqual(sorted(round(v, 9) for v in value["values"]), [2.0, 3.0])
        self._assert_eigenpairs(matrix, value["eigenvectors"])

    def test_complex_eigenvalues_are_reported_without_crashing(self) -> None:
        # A rotation by 90 degrees has eigenvalues +i and -i
        result = _evaluate({"R": [[0, -1], [1, 0]]}, "eigs(R)")
        value = self.object_value(result, "values", "eigenvectors")
        self.assertEqual(sorted(value["values"]), ["0 + 1i", "0 - 1i"])
        for pair in value["eigenvectors"]:
            self.assertIsInstance(pair["value"], str)
            self.assertEqual(len(pair["vector"]), 2)
            self.assertTrue(all(isinstance(entry, str) for entry in pair["vector"]))

    def test_eigenvalues_of_diagonal_matrix(self) -> None:
        value = self.object_value(_evaluate({"D": [[5, 0], [0, -2]]}, "eigs(D)"), "values")
        self.assertEqual(sorted(round(v, 9) for v in value["values"]), [-2.0, 5.0])

    def test_non_square_matrix_returns_error(self) -> None:
        result = _evaluate({"A": [[1, 2, 3], [4, 5, 6]]}, "eigs(A)")
        self.assertEqual(result["type"], "error")
        self.assertIn("square", result["value"])


class TestLup(_MatrixAssertions):
    def _reconstruct(self, matrix: Rows) -> None:
        value = self.object_value(_evaluate({"A": matrix}, "lup(A)"), "L", "U", "p")
        lower, upper, permutation = value["L"], value["U"], value["p"]
        # math.js reports p so that P[p[j]][j] = 1 and P * A = L * U, i.e. (L*U)[p[i]] = A[i]
        rows = len(matrix)
        permutation_matrix = [[1.0 if permutation[j] == i else 0.0 for j in range(rows)] for i in range(rows)]
        self.assert_matrix_close(_multiply(lower, upper), _multiply(permutation_matrix, matrix))
        for i, row in enumerate(lower):  # L is unit lower triangular
            if i < len(row):  # a tall L has more rows than columns
                self.assertEqual(row[i], 1.0)
            self.assertTrue(all(entry == 0 for entry in row[i + 1 :]))
        for i, row in enumerate(upper):  # U is upper triangular
            self.assertTrue(all(abs(entry) <= TOLERANCE for entry in row[:i]))

    def test_matrix_needing_a_row_swap(self) -> None:
        self._reconstruct([[1, 2], [3, 4]])

    def test_3x3_matrix_without_swaps(self) -> None:
        self._reconstruct([[4, 3, 2], [1, 2, 3], [0, 1, 5]])

    def test_4x4_matrix_with_mixed_signs(self) -> None:
        self._reconstruct([[42, -17, 63, -5], [-28, 91, -74, 60], [39, -56, 81, -13], [22, -48, 9, 100]])

    def test_zero_leading_entry_is_pivoted(self) -> None:
        self._reconstruct([[0, 1, 2], [1, 0, 3], [4, -3, 8]])

    def test_singular_matrix_has_zero_on_the_u_diagonal(self) -> None:
        matrix = [[1, 2], [2, 4]]
        self._reconstruct(matrix)
        upper = self.object_value(_evaluate({"A": matrix}, "lup(A)"), "U")["U"]
        self.assertTrue(abs(upper[1][1]) <= TOLERANCE)

    def test_rectangular_matrix(self) -> None:
        self._reconstruct([[1, 2], [3, 4], [5, 6]])

    def test_permutation_vector_is_a_permutation(self) -> None:
        value = self.object_value(_evaluate({"A": [[0, 1, 0], [0, 0, 1], [1, 0, 0]]}, "lup(A)"), "p")
        self.assertEqual(sorted(value["p"]), [0, 1, 2])


class TestQr(_MatrixAssertions):
    def _assert_qr(self, matrix: Rows) -> None:
        value = self.object_value(_evaluate({"A": matrix}, "qr(A)"), "Q", "R")
        q, r = value["Q"], value["R"]
        rows = len(matrix)
        self.assert_matrix_close(_multiply(q, r), matrix)
        self.assert_matrix_close(_multiply(_transpose(q), q), _identity(rows))
        for i, row in enumerate(r):
            self.assertTrue(all(abs(entry) <= TOLERANCE for entry in row[:i]), f"R is not upper triangular: {r}")

    def test_square_matrix(self) -> None:
        self._assert_qr([[1, 2], [3, 4]])

    def test_3x3_matrix(self) -> None:
        self._assert_qr([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])

    def test_tall_matrix(self) -> None:
        self._assert_qr([[1, 2], [3, 4], [5, 6]])

    def test_wide_matrix(self) -> None:
        self._assert_qr([[1, 2, 3], [4, 5, 6]])

    def test_rank_deficient_matrix(self) -> None:
        self._assert_qr([[1, 2], [2, 4]])

    def test_identity_matrix(self) -> None:
        self._assert_qr(_identity(3))


class TestRrefExpression(_MatrixAssertions):
    def _rref(self, matrix: Any, expression: str = "rref(A)") -> Rows:
        result = _evaluate({"A": matrix}, expression)
        self.assertEqual(result["type"], "matrix", result)
        rows: Rows = result["value"]
        return rows

    def _rank(self, matrix: Any, expression: str = "rank(A)") -> Any:
        result = _evaluate({"A": matrix}, expression)
        self.assertEqual(result["type"], "scalar", result)
        return result["value"]

    def test_full_rank_matrix_reduces_to_identity(self) -> None:
        self.assertEqual(self._rref([[1, 2], [3, 4]]), [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(self._rank([[1, 2], [3, 4]]), 2)

    def test_rank_deficient_matrix(self) -> None:
        matrix = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]
        self.assert_matrix_close(self._rref(matrix), [[1, 0, -1], [0, 1, 2], [0, 0, 0]])
        self.assertEqual(self._rank(matrix), 2)

    def test_rectangular_matrices(self) -> None:
        self.assert_matrix_close(self._rref([[1, 2, 3], [4, 5, 6]]), [[1, 0, -1], [0, 1, 2]])
        self.assertEqual(self._rref([[1, 2], [3, 4], [5, 6]]), [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        self.assertEqual(self._rank([[1, 2], [3, 4], [5, 6]]), 2)

    def test_zero_matrix(self) -> None:
        self.assertEqual(self._rref([[0, 0], [0, 0]]), [[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(self._rank([[0, 0], [0, 0]]), 0)

    def test_near_singular_matrix_uses_tolerance(self) -> None:
        matrix = [[1, 2], [2, 4 + 1e-13]]
        self.assertEqual(self._rref(matrix), [[1.0, 2.0], [0.0, 0.0]])
        self.assertEqual(self._rank(matrix), 1)
        self.assertEqual(self._rank(matrix, "rank(A, 1e-16)"), 2)

    def test_explicit_tolerance_argument(self) -> None:
        matrix = [[1, 0], [0, 0.01]]
        self.assertEqual(self._rank(matrix, "rank(A, 0.1)"), 1)
        self.assertEqual(self._rref(matrix, "rref(A, 0.1)"), [[1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(self._rref(matrix), [[1.0, 0.0], [0.0, 1.0]])

    def test_rref_composes_with_other_math_js_functions(self) -> None:
        matrix = [[2, 4], [1, 3]]
        self.assertEqual(self._rref(matrix, "rref(A) * 2"), [[2.0, 0.0], [0.0, 2.0]])
        self.assertEqual(self._rref(matrix, "rref(transpose(A))"), [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(self._rank(matrix, "rank(A * A)"), 2)

    def test_rref_solves_an_augmented_system(self) -> None:
        # x + 2y + z = 4, 2x + y - z = 2, x - y + 2z = 2 has the solution x = y = z = 1
        augmented = [[1, 2, 1, 4], [2, 1, -1, 2], [1, -1, 2, 2]]
        self.assert_matrix_close(self._rref(augmented), [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])

    def test_vector_argument_returns_an_error(self) -> None:
        raised = False
        try:
            result = _evaluate({"v": [1, 2, 3]}, "rref(v)")
        except Exception:
            raised = True
        else:
            self.assertEqual(result["type"], "error", result)
        self.assertTrue(raised or result["type"] == "error")

    def test_object_named_rank_still_works_as_a_variable(self) -> None:
        result = _evaluate({"rank": 3, "A": [[1, 0], [0, 1]]}, "rank * 2")
        self.assertEqual(result["type"], "scalar")
        self.assertEqual(result["value"], 6.0)
