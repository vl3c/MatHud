"""Tests for utils.numeric_equations: solve's real roots when SymPy has none, from the sign chart.

The equations go through the browser's numeric path in CPython (real_sets_harness), so these
are the answers solve gives without SymPy, or when SymPy finds no closed form.
"""

from __future__ import annotations

import unittest

from server_tests.real_sets_harness import solve_equation


def _roots(answer: str) -> str:
    """The root list of an answer: "[1,2]" from "[1,2] (the real roots found numerically ...)"."""
    return answer.split(" (")[0]


class TestNumericEquations(unittest.TestCase):
    def test_transcendental_roots(self) -> None:
        self.assertEqual(_roots(solve_equation("cos(x)=x")), "[0.739085133215161]")
        self.assertEqual(_roots(solve_equation("2^x=x^2")), "[-0.766664695962123,2,4]")
        self.assertEqual(_roots(solve_equation("x*e^x=1")), "[0.567143290409784]")

    def test_the_note_says_how_and_why(self) -> None:
        answer = solve_equation("e^x=3*x", reason="the server's SymPy engine is not available")
        self.assertTrue(answer.startswith("[0.619061286735945,1.51213455165784] (the real roots found numerically"))
        self.assertIn("for |x| up to 10^6", answer)
        self.assertIn("because the server's SymPy engine is not available", answer)

    def test_a_root_beside_a_grid_point(self) -> None:
        # a sample lands within a few ulps of the root 1
        self.assertEqual(_roots(solve_equation("sin(pi*x/6)=1/2", "[-2.8, 1.2]")), "[1]")

    def test_underflow_is_not_a_root(self) -> None:
        answer = solve_equation("(1-x)*e^(-x)=0")
        self.assertEqual(_roots(answer), "[1]")
        self.assertIn("-10^6 <= x <= 743.464", answer)

    def test_touching_roots(self) -> None:
        self.assertEqual(_roots(solve_equation("x^2=0")), "[0]")
        self.assertEqual(_roots(solve_equation("abs(x-1)=0")), "[1]")
        self.assertEqual(_roots(solve_equation("1-cos(x)=0", "[-1, 1]")), "[0]")

    def test_tiny_coefficients(self) -> None:
        self.assertEqual(_roots(solve_equation("1e-8*x=2e-8")), "[2]")
        self.assertEqual(_roots(solve_equation("1.6e-19*x=3.2e-19")), "[2]")

    def test_trig_equations_use_the_principal_window(self) -> None:
        answer = solve_equation("tan(x)=1")
        self.assertEqual(_roots(answer), "[-7*pi/4,-3*pi/4,pi/4,5*pi/4]")
        self.assertIn("in [-2*pi, 2*pi]", answer)
        self.assertEqual(_roots(solve_equation("sin(x)=0", "[0, 10]")), "[0,pi,2*pi,3*pi]")

    def test_identities_and_contradictions(self) -> None:
        self.assertTrue(solve_equation("(x+1)^2=x^2+2*x+1").startswith("Every real x satisfies"))
        self.assertTrue(solve_equation("sin(x)^2+cos(x)^2=1").startswith("Every x in [-2*pi, 2*pi] satisfies"))
        self.assertEqual(_roots(solve_equation("x=x+1")), "[]")
        self.assertIn("no real root was found numerically", solve_equation("x=x+1"))
        self.assertEqual(_roots(solve_equation("sqrt(x)=-1")), "[]")

    def test_poles_are_not_roots(self) -> None:
        self.assertEqual(_roots(solve_equation("1/x=0")), "[]")
        self.assertEqual(_roots(solve_equation("x/sin(x)=1", "[-5, 7]")), "[]")

    def test_a_range_of_solutions(self) -> None:
        self.assertTrue(solve_equation("abs(x)=x").startswith("x >= 0; interval notation: [0, Infinity)"))

    def test_other_names_need_sympy(self) -> None:
        answer = solve_equation("x+a=1")
        self.assertEqual(
            answer,
            "Error: SymPy found no closed form, and the numeric root finder needs an equation in x alone; "
            "besides x it contains a.",
        )


if __name__ == "__main__":
    unittest.main()
