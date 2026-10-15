"""Server tests for the SymPy CAS engine: the safe parser, the operations, the worker and /api/cas."""

from __future__ import annotations

import math
import unittest

import pytest

sympy = pytest.importorskip("sympy")

from static.sympy_cas import (  # noqa: E402
    ExpressionError,
    compute_derivative,
    compute_integral,
    compute_limit,
    compute_simplify,
    compute_solve,
    handle,
    parse_expression,
    to_text,
)
from static.sympy_engine import SympyEngine  # noqa: E402


class TestParser(unittest.TestCase):
    def test_client_notation(self) -> None:
        x = sympy.Symbol("x")
        self.assertEqual(parse_expression("2x^2 + 3(x+1)"), 2 * x**2 + 3 * (x + 1))
        self.assertEqual(parse_expression("e^x"), sympy.exp(x))
        self.assertEqual(parse_expression("i^2"), -1)
        self.assertEqual(parse_expression("Infinity"), sympy.oo)
        self.assertEqual(parse_expression("1e3"), 1000)
        self.assertEqual(parse_expression("0.5*x"), x / 2)
        self.assertEqual(parse_expression("log(x, 2)"), sympy.log(x, 2))
        self.assertEqual(parse_expression("(x+1)(x-1)"), (x + 1) * (x - 1))

    def test_refuses_everything_but_math(self) -> None:
        for text in (
            "__import__('os').system('ls')",
            "x.__class__",
            "(lambda: 1)()",
            "[1][0]",
            "{'a': 1}",
            "'text'",
            "sin(x, y=2)",
            "sin(*x)",
            "open('f')",
            "eval('1')",
            "x if x else 1",
            "x == 1",
            "__x",
            "sin",
            "sin(x, x)",
            "1" * 100,
            "x" * 2000,
        ):
            with self.subTest(text=text):
                with self.assertRaises(ExpressionError):
                    parse_expression(text)

    def test_huge_powers_stay_unevaluated(self) -> None:
        self.assertIsInstance(parse_expression("10^(10^10)"), sympy.Pow)

    def test_printing(self) -> None:
        x = sympy.Symbol("x")
        self.assertEqual(to_text(x**2 / 3 + sympy.sqrt(x) - 1 / x), "sqrt(x)+x^2/3-1/x")
        self.assertEqual(to_text(sympy.exp(2) + sympy.I + sympy.E), "e+exp(2)+i")
        self.assertEqual(to_text(sympy.Abs(x) + sympy.ceiling(x)), "abs(x)+ceil(x)")
        self.assertEqual(to_text(-sympy.oo), "-Infinity")
        self.assertEqual(to_text(x ** sympy.Rational(3, 2)), "x^(3/2)")
        self.assertEqual(to_text((x + 1) ** -2), "(x+1)^(-2)")


class TestOperations(unittest.TestCase):
    def test_limits(self) -> None:
        self.assertEqual(compute_limit("sqrt(x^2+x)-x", "x", "Infinity"), "1/2")
        self.assertEqual(compute_limit("sin(x)/x", "x", "0"), "1")
        self.assertEqual(compute_limit("(1+1/x)^x", "x", "Infinity"), "e")
        self.assertEqual(compute_limit("1/x^2", "x", "0"), "Infinity")
        self.assertEqual(compute_limit("ln(x)", "x", "0"), "-Infinity")
        # The sides differ or oscillate: no single value (the client explains why)
        self.assertIsNone(compute_limit("1/x", "x", "0"))
        self.assertIsNone(compute_limit("abs(x)/x", "x", "0"))
        self.assertIsNone(compute_limit("sin(1/x)", "x", "0"))
        self.assertIsNone(compute_limit("sin(x)", "x", "Infinity"))

    def test_integrals(self) -> None:
        self.assertEqual(compute_integral("1/(x^2+2*x+5)", "x"), "atan(x/2+1/2)/2")
        self.assertEqual(compute_integral("cos(x)*e^(sin(x))", "x"), "exp(sin(x))")
        self.assertEqual(compute_integral("sqrt(1-x^2)", "x", "-1", "1"), "pi/2")
        self.assertEqual(compute_integral("e^(-x^2)", "x", "-Infinity", "Infinity"), "sqrt(pi)")
        self.assertEqual(compute_integral("1/(x^2-1)", "x", "2", "3"), "-log(2)/2+log(3)/2")

    def test_simplify(self) -> None:
        self.assertEqual(compute_simplify("sin(x)^2+cos(x)^2"), "1")
        self.assertEqual(compute_simplify("(x^3-8)/(x-2)"), "x^2+2*x+4")
        self.assertEqual(compute_simplify("(1/x+1/y)/(1/(x*y))"), "x+y")
        self.assertEqual(compute_simplify("e^(ln(x))"), "x")

    def test_derivatives(self) -> None:
        self.assertEqual(compute_derivative("atan2(y,x)", "x"), "-y/(x^2+y^2)")
        self.assertEqual(compute_derivative("x^3", "x"), "3*x^2")

    def test_solve(self) -> None:
        self.assertEqual(compute_solve("x^2=4", "x"), {"roots": ["-2", "2"]})
        self.assertEqual(compute_solve("x^2+1=0", "x"), {"roots": ["-i", "i"]})
        self.assertEqual(compute_solve("x/(x-1)=2", "x"), {"roots": ["2"]})
        self.assertEqual(compute_solve("ln(x)=2", "x"), {"roots": ["exp(2)"]})
        self.assertEqual(compute_solve("x=x", "x"), {"identity": True, "roots": []})
        self.assertEqual(compute_solve("x=x+1", "x"), {"roots": []})
        self.assertEqual(compute_solve("1/x=0", "x"), {"roots": []})
        self.assertIsNone(compute_solve("cos(x)=x", "x"))

    def test_cubic_roots_without_cardano_forms(self) -> None:
        solution = compute_solve("x^3+x+1=0", "x")
        assert solution is not None
        self.assertEqual(solution["roots"][0], "-0.682327803828019")
        self.assertEqual(len(solution["roots"]), 3)

    def test_periodic_solutions(self) -> None:
        solution = compute_solve("sin(x)=0", "x")
        assert solution is not None
        self.assertEqual(solution["roots"], ["-2*pi", "-pi", "0", "pi", "2*pi"])
        self.assertEqual(solution["general"], ["x = 2*pi*n", "x = 2*pi*n+pi"])
        tangent = compute_solve("tan(x)=1", "x")
        assert tangent is not None
        self.assertEqual(len(tangent["roots"]), 4)
        for root in tangent["roots"]:
            self.assertLessEqual(abs(float(sympy.sympify(root.replace("^", "**")))), 2 * math.pi)

    def test_handle(self) -> None:
        self.assertEqual(handle({"operation": "simplify", "args": {"expression": "2x+3x"}}), {"result": "5*x"})
        self.assertIn("error", handle({"operation": "simplify", "args": {"expression": "x.y"}}))
        self.assertIn("error", handle({"operation": "eval", "args": {}}))
        self.assertIn("error", handle({"operation": "limit", "args": {"expression": 1}}))
        self.assertIsNone(handle({"operation": "limit", "args": {"expression": "x"}})["result"])


class TestEngine(unittest.TestCase):
    def test_runs_requests_and_stops_long_ones(self) -> None:
        engine = SympyEngine(time_limit_s=2.0)
        try:
            self.assertEqual(
                engine.compute("limit", {"expression": "sin(x)/x", "variable": "x", "target": "0"}),
                {"result": "1"},
            )
            self.assertEqual(engine.compute("simplify", {"expression": "10^10^10"}), {"timeout": True})
            # A new worker serves the next request
            self.assertEqual(engine.compute("simplify", {"expression": "x+x"}), {"result": "2*x"})
        finally:
            engine.close()


class TestRoute(unittest.TestCase):
    def setUp(self) -> None:
        from static.app_manager import AppManager

        self.app = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_cas_route(self) -> None:
        response = self.client.post(
            "/api/cas", json={"operation": "integrate", "args": {"expression": "2*x", "variable": "x"}}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["data"], {"result": "x^2"})

    def test_cas_route_refuses_bad_bodies(self) -> None:
        self.assertEqual(self.client.post("/api/cas", json=[1]).status_code, 400)
        response = self.client.post("/api/cas", json={"operation": "simplify", "args": {"expression": 3}})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
