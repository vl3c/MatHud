"""Server tests for the SymPy CAS engine: the safe parser, the operations, the worker and /api/cas."""

from __future__ import annotations

import math
import unittest
from typing import Any, List, Tuple

import pytest

sympy = pytest.importorskip("sympy")

from static.sympy_cas import (  # noqa: E402
    ExpressionError,
    compute_derivative,
    compute_integral,
    compute_limit,
    compute_simplify,
    compute_solve,
    compute_solve_inequality,
    handle,
    parse_expression,
    parse_relation,
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
        self.assertIsInstance(parse_expression("pow(10, 10^10)"), sympy.Pow)
        with self.assertRaises(ExpressionError):
            parse_expression("1e300")

    def test_printing(self) -> None:
        x = sympy.Symbol("x")
        self.assertEqual(to_text(x**2 / 3 + sympy.sqrt(x) - 1 / x), "sqrt(x)+x^2/3-1/x")
        self.assertEqual(to_text(sympy.exp(2) + sympy.I + sympy.E), "e+exp(2)+i")
        self.assertEqual(to_text(sympy.Abs(x) + sympy.ceiling(x)), "abs(x)+ceil(x)")
        self.assertEqual(to_text(-sympy.oo), "-Infinity")
        self.assertEqual(to_text(x ** sympy.Rational(3, 2)), "x^(3/2)")
        self.assertEqual(to_text((x + 1) ** -2), "(x+1)^(-2)")
        self.assertEqual(to_text(sympy.Min(x, 1)), "min(1,x)")
        # math.js is case-sensitive and has no Mod or conjugate
        self.assertEqual(to_text(sympy.Mod(x, 3)), "mod(x,3)")
        self.assertEqual(to_text(sympy.conjugate(sympy.Symbol("z"))), "conj(z)")

    def test_remainder_and_complex_helpers(self) -> None:
        x = sympy.Symbol("x")
        self.assertEqual(parse_expression("x % 3"), sympy.Mod(x, 3))
        self.assertEqual(parse_expression("mod(x, 3)"), sympy.Mod(x, 3))
        self.assertEqual(parse_expression("2^7 % 5"), 3)  # % binds like * and /, below ^
        self.assertEqual(parse_expression("-7 % 3"), 2)  # floored, like Python and math.js
        z = sympy.Symbol("z")
        self.assertEqual(parse_expression("re(z) + im(z)"), sympy.re(z) + sympy.im(z))
        self.assertEqual(parse_expression("arg(z)"), sympy.arg(z))
        self.assertEqual(parse_expression("conj(z)"), sympy.conjugate(z))


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
        # Not real: (-8)^(1/3) is SymPy's principal (complex) root
        self.assertIsNone(compute_limit("(-8)^(1/3)+x", "x", "0"))

    def test_integrals(self) -> None:
        self.assertEqual(compute_integral("1/(x^2+2*x+5)", "x"), "atan(x/2+1/2)/2")
        self.assertEqual(compute_integral("cos(x)*e^(sin(x))", "x"), "exp(sin(x))")
        self.assertEqual(compute_integral("sqrt(1-x^2)", "x", "-1", "1"), "pi/2")
        self.assertEqual(compute_integral("e^(-x^2)", "x", "-Infinity", "Infinity"), "sqrt(pi)")
        self.assertEqual(compute_integral("1/(x^2-1)", "x", "2", "3"), "-log(2)/2+log(3)/2")
        self.assertIsNone(compute_integral("1/x", "x", "-Infinity", "-1"))  # SymPy: -Infinity + i*pi

    def test_simplify(self) -> None:
        self.assertEqual(compute_simplify("sin(x)^2+cos(x)^2"), "1")
        self.assertEqual(compute_simplify("(x^3-8)/(x-2)"), "x^2+2*x+4")
        self.assertEqual(compute_simplify("(1/x+1/y)/(1/(x*y))"), "x+y")
        self.assertEqual(compute_simplify("e^(ln(x))"), "x")

    def test_simplify_keeps_numbers_exact(self) -> None:
        self.assertEqual(compute_simplify("1/3+1/4"), "7/12")
        self.assertEqual(compute_simplify("sqrt(12)"), "2*sqrt(3)")
        self.assertEqual(compute_simplify("0.1+0.2"), "3/10")

    def test_simplify_calculator_functions(self) -> None:
        cases = {
            "mod(-7,3)": "2",
            "17 % 5": "2",
            "2*7 % 3": "2",
            "re(3+4i)": "3",
            "im(3+4i)": "4",
            "arg(1+i)": "pi/4",
            "conj(3+4i)": "3-4*i",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(compute_simplify(expression), expected)
        # Variables are real in simplify, so the complex helpers only mean something for numbers
        self.assertEqual(compute_simplify("im(z)"), "0")
        self.assertEqual(compute_simplify("re(z)"), "z")

    def test_a_zero_divisor_is_a_note_not_a_crash(self) -> None:
        for expression in ("mod(7, 0)", "7 % 0"):
            with self.subTest(expression=expression):
                response = handle({"operation": "simplify", "args": {"expression": expression}})
                self.assertIsNone(response.get("result"))
                self.assertIn("note", response)

    def test_derivatives(self) -> None:
        self.assertEqual(compute_derivative("atan2(y,x)", "x"), "-y/(x^2+y^2)")
        self.assertEqual(compute_derivative("x^3", "x"), "3*x^2")
        self.assertIsNone(compute_derivative("floor(x)", "x"))  # SymPy leaves it unevaluated

    def test_solve(self) -> None:
        self.assertEqual(compute_solve("x^2=4", "x"), {"roots": ["-2", "2"]})
        self.assertEqual(compute_solve("x^2+1=0", "x"), {"roots": ["-i", "i"]})
        self.assertEqual(compute_solve("x/(x-1)=2", "x"), {"roots": ["2"]})
        self.assertEqual(compute_solve("ln(x)=2", "x"), {"roots": ["exp(2)"]})
        self.assertEqual(compute_solve("x=x", "x"), {"identity": True, "roots": []})
        self.assertEqual(compute_solve("x=x+1", "x"), {"roots": []})
        self.assertEqual(compute_solve("1/x=0", "x"), {"roots": []})
        self.assertIsNone(compute_solve("cos(x)=x", "x"))
        # A rational equation is solved over the complex numbers, like a polynomial
        self.assertEqual(compute_solve("1/x=-x", "x"), {"roots": ["-i", "i"]})

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
        self.assertEqual(solution["window"], "2*pi")
        tangent = compute_solve("tan(x)=1", "x")
        assert tangent is not None
        self.assertEqual(len(tangent["roots"]), 4)
        for root in tangent["roots"]:
            self.assertLessEqual(abs(float(sympy.sympify(root.replace("^", "**")))), 2 * math.pi)

    def test_dense_periodic_solutions_shrink_the_window(self) -> None:
        solution = compute_solve("tan(60*x)=0", "x")
        assert solution is not None
        self.assertEqual(len(solution["roots"]), 59)
        self.assertEqual(solution["window"], "29*pi/60")
        self.assertEqual(solution["general"], ["x = pi*n/60"])

    def test_periodic_solutions_without_affine_forms(self) -> None:
        solution = compute_solve("sin(x^2)=0", "x")
        assert solution is not None
        self.assertNotIn("general", solution)  # sqrt(n) is not real for n < 0
        self.assertEqual(solution["window"], "2*pi")
        # Roots that pile up at 0 cannot all be listed
        self.assertIsNone(compute_solve("sin(1/x)=0", "x"))

    def test_solve_on_an_interval(self) -> None:
        cosine = compute_solve("cos(x)=3/10", "x", "0", "20", "[]")
        assert cosine is not None
        self.assertEqual(cosine["roots"][:3], ["acos(3/10)", "-acos(3/10)+2*pi", "acos(3/10)+2*pi"])
        self.assertEqual(len(cosine["roots"]), 6)
        tangent = compute_solve("tan(x)=1", "x", "0", "20", "[]")
        assert tangent is not None
        self.assertEqual(tangent["roots"], ["pi/4", "5*pi/4", "9*pi/4", "13*pi/4", "17*pi/4", "21*pi/4", "25*pi/4"])
        self.assertEqual(compute_solve("x^2=2", "x", "0", "5", "[]"), {"roots": ["sqrt(2)"]})
        self.assertEqual(compute_solve("x^2=4", "x", "-Infinity", "0", "(]"), {"roots": ["-2"]})
        # The brackets are honoured
        self.assertEqual(compute_solve("sin(x)=0", "x", "0", "2*pi", "[)"), {"roots": ["0", "pi"]})
        self.assertEqual(compute_solve("sin(x)=0", "x", "0", "2*pi", "[]"), {"roots": ["0", "pi", "2*pi"]})
        self.assertEqual(compute_solve("sin(x)=0", "x", "0", "pi", "()"), {"roots": []})
        self.assertEqual(compute_solve("x-x=0", "x", "0", "5", "[]"), {"identity": True, "roots": []})
        self.assertEqual(compute_solve("x^2+1=0", "x", "0", "5", "[]"), {"roots": []})
        self.assertIsNone(compute_solve("cos(x)=x", "x", "0", "5", "[]"))

    def test_solve_on_an_interval_lists_at_most_sixty_roots(self) -> None:
        solution = compute_solve("sin(x)=0", "x", "0", "1000", "[]")
        assert solution is not None
        self.assertEqual(len(solution["roots"]), 60)
        self.assertEqual(solution["roots"][-1], "59*pi")
        self.assertEqual(solution["total"], 319)

    def test_handle(self) -> None:
        self.assertEqual(handle({"operation": "simplify", "args": {"expression": "2x+3x"}}), {"result": "5*x"})
        self.assertIn("error", handle({"operation": "simplify", "args": {"expression": "x.y"}}))
        self.assertIn("error", handle({"operation": "eval", "args": {}}))
        self.assertIn("error", handle({"operation": "limit", "args": {"expression": 1}}))
        self.assertIsNone(handle({"operation": "limit", "args": {"expression": "x"}})["result"])


def _ends(solution: Any) -> List[Tuple[str, str, bool, bool]]:
    """The intervals of a solve_inequality result as (lower, upper, lower_closed, upper_closed)."""
    assert solution is not None
    return [
        (item["lower"], item["upper"], item["lower_closed"], item["upper_closed"]) for item in solution["intervals"]
    ]


class TestInequalities(unittest.TestCase):
    def test_polynomial_rational_and_absolute_value(self) -> None:
        self.assertEqual(_ends(compute_solve_inequality("x^2 - 5x + 6 < 0", "x")), [("2", "3", False, False)])
        self.assertEqual(
            _ends(compute_solve_inequality("abs(x - 1) >= 2", "x")),
            [("-Infinity", "-1", False, True), ("3", "Infinity", True, False)],
        )
        self.assertEqual(
            _ends(compute_solve_inequality("(x-1)/(x+2) > 0", "x")),
            [("-Infinity", "-2", False, False), ("1", "Infinity", False, False)],
        )
        self.assertEqual(_ends(compute_solve_inequality("log(x) < 2", "x")), [("0", "exp(2)", False, False)])
        self.assertEqual(_ends(compute_solve_inequality("sqrt(x-1) < 3", "x")), [("1", "10", True, False)])

    def test_points_empty_and_every_real(self) -> None:
        self.assertEqual(_ends(compute_solve_inequality("x^2 <= 0", "x")), [("0", "0", True, True)])
        self.assertEqual(_ends(compute_solve_inequality("x^2 + 1 < 0", "x")), [])
        self.assertEqual(_ends(compute_solve_inequality("x^2 + 1 > 0", "x")), [("-Infinity", "Infinity", False, False)])
        self.assertEqual(
            _ends(compute_solve_inequality("(x-1)^2 > 0", "x")),
            [("-Infinity", "1", False, False), ("1", "Infinity", False, False)],
        )

    def test_chain_and_cubics(self) -> None:
        self.assertEqual(_ends(compute_solve_inequality("1 < 2x+1 <= 5", "x")), [("0", "2", False, True)])
        self.assertEqual(
            _ends(compute_solve_inequality("x^3 - 2x + 1 > 0", "x")),
            [("-sqrt(5)/2-1/2", "-1/2+sqrt(5)/2", False, False), ("1", "Infinity", False, False)],
        )
        # Cardano's form of the root is long: it is given in decimals
        self.assertEqual(
            _ends(compute_solve_inequality("x^3 + x + 1 > 0", "x")),
            [("-0.682327803828019", "Infinity", False, False)],
        )

    def test_periodic_inequalities_cover_the_window(self) -> None:
        # SymPy alone answers (pi/6, 5*pi/6) for one period, even on [-2*pi, 2*pi]
        solution = compute_solve_inequality("sin(x) > 1/2", "x")
        assert solution is not None
        self.assertEqual(_ends(solution), [("-11*pi/6", "-7*pi/6", False, False), ("pi/6", "5*pi/6", False, False)])
        self.assertEqual((solution["window"], solution["period"]), ("[-2*pi, 2*pi]", "2*pi"))
        on_interval = compute_solve_inequality("sin(x) > 1/2", "x", "0", "2*pi")
        assert on_interval is not None
        self.assertEqual(_ends(on_interval), [("pi/6", "5*pi/6", False, False)])
        self.assertEqual(on_interval["window"], "[0, 2*pi]")
        # Copies merge across period edges
        self.assertIn(("-pi/2", "pi/2", False, False), _ends(compute_solve_inequality("cos(x) > 0", "x")))

    def test_periodic_regressions(self) -> None:
        # SymPy gives only (pi/4, pi/2) for tan(x) > 1 on [0, 2*pi]
        self.assertEqual(
            _ends(compute_solve_inequality("tan(x) > 1", "x", "0", "2*pi")),
            [("pi/4", "pi/2", False, False), ("5*pi/4", "3*pi/2", False, False)],
        )
        # ... and loses the point 2*pi of sin(x) >= 0 on [0, 2*pi]
        self.assertEqual(
            _ends(compute_solve_inequality("sin(x) >= 0", "x", "0", "2*pi")),
            [("0", "pi", True, True), ("2*pi", "2*pi", True, True)],
        )

    def test_half_open_intervals(self) -> None:
        solution = compute_solve_inequality("sin(x) >= 0", "x", "0", "2*pi", "[)")
        assert solution is not None
        self.assertEqual(_ends(solution), [("0", "pi", True, True)])
        self.assertEqual(solution["window"], "[0, 2*pi)")
        self.assertEqual(_ends(compute_solve_inequality("x > 2", "x", "0", "1")), [])

    def test_many_periods_give_one_period(self) -> None:
        solution = compute_solve_inequality("sin(100*x) > 0", "x")
        assert solution is not None
        self.assertEqual(_ends(solution), [("0", "pi/100", False, False)])
        self.assertEqual((solution["window"], solution["period"]), ("[0, pi/50)", "pi/50"))
        on_interval = compute_solve_inequality("sin(x) > 1/2", "x", "0", "1000")
        assert on_interval is not None
        self.assertEqual(on_interval["window"], "[0, 2*pi)")

    def test_the_last_period_copy_is_kept(self) -> None:
        # pi / (pi/3) is just below 3 in floats, which lost the copy that holds pi
        self.assertEqual(
            _ends(compute_solve_inequality("cos(6*x) >= 1", "x", "0", "pi")),
            [
                ("0", "0", True, True),
                ("pi/3", "pi/3", True, True),
                ("2*pi/3", "2*pi/3", True, True),
                ("pi", "pi", True, True),
            ],
        )

    def test_decimal_close_to_a_fraction_stays_itself(self) -> None:
        # nsimplify reads 1.0000000001 as 1, which would make (x - 1)^2 + 1e-10 touch 0
        self.assertEqual(
            _ends(compute_solve_inequality("x^2 - 2x + 1.0000000001 > 0", "x")),
            [("-Infinity", "Infinity", False, False)],
        )
        self.assertEqual(
            _ends(compute_solve_inequality("x > 0.333333333333333", "x")), [("1/3", "Infinity", False, False)]
        )

    def test_decimal_input_gives_decimal_ends(self) -> None:
        self.assertEqual(
            _ends(compute_solve_inequality("x >= 1.23456789", "x")), [("1.23456789", "Infinity", True, False)]
        )
        self.assertEqual(
            _ends(compute_solve_inequality("(x-1)*(x-1.0000001) < 0", "x")), [("1", "1.0000001", False, False)]
        )
        solution = compute_solve_inequality("x > 0", "x", "1e-300", "1")
        assert solution is not None
        self.assertEqual(_ends(solution), [("1e-300", "1", True, True)])
        self.assertEqual(solution["window"], "[1e-300, 1]")
        self.assertEqual(
            _ends(compute_solve_inequality("x < 0.000001", "x")), [("-Infinity", "0.000001", False, False)]
        )
        # Without decimals in the input, fractions stay fractions
        self.assertEqual(_ends(compute_solve_inequality("2*x < 1", "x")), [("-Infinity", "1/2", False, False)])

    def test_sympy_is_wrong_on_some_periodic_inequalities(self) -> None:
        # The answer is [0, pi/2) U (pi/2, 7*pi/6) U (11*pi/6, 2*pi) in each period; SymPy gives
        # [0, pi/2). The browser's numeric check refutes it and answers with the sign chart.
        solution = compute_solve_inequality("2*sin(x)^2 - sin(x) - 1 < 0", "x", "0", "2*pi", "[)")
        self.assertEqual(_ends(solution), [("0", "pi/2", True, False)])

    def test_unsolvable_inequalities_give_none(self) -> None:
        self.assertIsNone(compute_solve_inequality("cos(x) > x", "x"))
        self.assertIsNone(compute_solve_inequality("x*sin(x) > 0", "x"))
        # Both of SymPy's routes refuse it
        self.assertIsNone(compute_solve_inequality("abs(sin(x)) < 1/2", "x"))

    def test_refusals(self) -> None:
        for text, variable in (
            ("x > y", "x"),
            ("x = 2", "x"),
            ("x != 2", "x"),
            ("1 < x > 0", "x"),
            ("1 < x < 2 < 3", "x"),
            ("2 < 3", "x"),
            ("x < 2", "pi"),
        ):
            with self.subTest(text=text):
                with self.assertRaises(ExpressionError):
                    compute_solve_inequality(text, variable)
        for lower, upper, bounds in (
            ("2", "1", None),
            ("0", "y", None),
            ("0", None, None),
            ("0", "1", "[["),
            ("i", "2", None),
        ):
            with self.subTest(lower=lower, upper=upper, bounds=bounds):
                with self.assertRaises(ExpressionError):
                    compute_solve_inequality("x > 0", "x", lower, upper, bounds)

    def test_parse_relation(self) -> None:
        x = sympy.Symbol("x", real=True)
        self.assertEqual(parse_relation("2x < 4"), [sympy.Lt(2 * x, 4)])
        self.assertEqual(parse_relation("1 <= x^2 < 4"), [sympy.Le(1, x**2), sympy.Lt(x**2, 4)])


class TestResponsesAreBrythonSafe(unittest.TestCase):
    """Every number in an /api/cas response is text, a bool or a small int: Brython's json.loads
    cannot read a float's exponent (1e+16)."""

    def _assert_safe(self, value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                self._assert_safe(item)
        elif isinstance(value, list):
            for item in value:
                self._assert_safe(item)
        elif isinstance(value, bool) or value is None or isinstance(value, str):
            return
        elif isinstance(value, int):
            self.assertLess(abs(value), 2**31, value)
        else:
            self.fail(f"{value!r} ({type(value).__name__}) in a response")

    def test_responses_hold_no_floats(self) -> None:
        requests = [
            ("solve_inequality", {"inequality": "x > 1e17", "variable": "x"}),
            ("solve_inequality", {"inequality": "x^2 < 1e20", "variable": "x"}),
            ("solve_inequality", {"inequality": "x^5 - x - 1 > 0", "variable": "x"}),
            ("solve_inequality", {"inequality": "sin(x) > 1/2", "variable": "x", "lower": "0", "upper": "1e5"}),
            ("solve", {"equation": "x^3+x+1=0", "variable": "x"}),
            ("solve", {"equation": "x^2=1e40", "variable": "x"}),
            ("solve", {"equation": "sin(x)=0", "variable": "x", "lower": "0", "upper": "1000", "bounds": "[)"}),
            ("simplify", {"expression": "2^60"}),
            ("integrate", {"expression": "x", "variable": "x", "lower": "0", "upper": "1e10"}),
        ]
        for operation, args in requests:
            with self.subTest(operation=operation, args=args):
                response = handle({"operation": operation, "args": args})
                self.assertIn("result", response)
                self._assert_safe(response)


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
            # and one after a crash
            assert engine._process is not None
            engine._process.kill()
            engine._process.wait()
            self.assertEqual(engine.compute("simplify", {"expression": "x+x"}), {"result": "2*x"})
        finally:
            engine.close()


class TestRoute(unittest.TestCase):
    def setUp(self) -> None:
        from static.app_manager import AppManager

        self.app = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        from static.sympy_engine import get_engine

        get_engine().close()

    def test_cas_route(self) -> None:
        response = self.client.post(
            "/api/cas", json={"operation": "integrate", "args": {"expression": "2*x", "variable": "x"}}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["data"], {"result": "x^2"})
        response = self.client.post(
            "/api/cas",
            json={"operation": "solve_inequality", "args": {"inequality": "x^2 < 4", "variable": "x"}},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["data"]["result"]["intervals"],
            [{"lower": "-2", "upper": "2", "lower_closed": False, "upper_closed": False}],
        )

    def test_cas_route_refuses_bad_bodies(self) -> None:
        self.assertEqual(self.client.post("/api/cas", json=[1]).status_code, 400)
        response = self.client.post("/api/cas", json={"operation": "simplify", "args": {"expression": 3}})
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/cas", json={"operation": "simplify", "args": {"expression": "x" * 5000}})
        self.assertEqual(response.status_code, 400)

    def test_cas_route_unavailable_answers(self) -> None:
        from unittest import mock

        body = {"operation": "simplify", "args": {"expression": "x"}}
        with mock.patch(
            "static.sympy_engine.SympyEngine.compute", return_value={"unavailable": True, "installed": False}
        ):
            self.assertEqual(self.client.post("/api/cas", json=body).status_code, 503)
        with mock.patch("static.sympy_engine.SympyEngine.compute", return_value={"unavailable": True}):
            self.assertEqual(self.client.post("/api/cas", json=body).status_code, 502)


if __name__ == "__main__":
    unittest.main()
