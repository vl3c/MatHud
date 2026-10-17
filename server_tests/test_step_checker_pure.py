"""Tests for utils.step_checker (check_steps) with Python stand-ins for the browser parts.

The steps go through the real input pipeline (InequalitySolver.prepared, which runs under
the browser stub). In the browser math.js then reads them and the SymPy engine is asked
over /api/cas; here compile_expression's Python evaluators read them, and
compute_equivalent (or a scripted answer) stands in for the server. Brython's own run is in
static/client/client_tests/test_step_checker.py.
"""

from __future__ import annotations

import math
import time
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

from server_tests import client_renderer  # noqa: F401  (browser stub, so compiled_expression imports)
from compiled_expression import compile_expression
from utils import step_checker
from utils.equivalence import Evaluator, Point
from utils.step_checker import StepChecker

try:
    from static.sympy_cas import ExpressionError, compute_equivalent

    HAVE_SYMPY = True
except ImportError:  # pragma: no cover
    HAVE_SYMPY = False


def _variables(text: str) -> List[str]:
    return list(compile_expression(text, ()).free_names)


def _evaluator(text: str) -> Optional[Evaluator]:
    try:
        names = tuple(_variables(text))
        bound = compile_expression(text, names).bind()
    except Exception:
        return None

    def evaluate(point: Point) -> Optional[float]:
        try:
            value = bound(*[point[name] for name in names])
        except OverflowError:
            return math.inf
        except (ArithmeticError, ValueError, TypeError, KeyError, NameError):
            return None
        return None if isinstance(value, complex) else float(value)

    return evaluate


def _sympy(first: str, second: str) -> Optional[Dict[str, Any]]:
    if not HAVE_SYMPY:
        return None
    try:
        return {"result": compute_equivalent(first, second)}
    except ExpressionError as error:
        return {"error": str(error)}


class _Engines:
    """Patches the browser parts; records the SymPy requests."""

    def __init__(self, sympy: Any = _sympy) -> None:
        self.requests: List[tuple] = []
        self._sympy = sympy

    def ask(self, first: str, second: str) -> Optional[Dict[str, Any]]:
        self.requests.append((first, second))
        return self._sympy(first, second)  # type: ignore[no-any-return]

    def check(self, steps: Any) -> Any:
        with mock.patch.multiple(
            StepChecker,
            _variables=staticmethod(_variables),
            _evaluator=staticmethod(_evaluator),
            _ask_sympy=staticmethod(self.ask),
        ):
            return StepChecker.check_steps(steps)


def _check(steps: List[str]) -> Dict[str, Any]:
    result = _Engines().check(steps)
    assert isinstance(result, dict), result
    return result


class TestDerivations(unittest.TestCase):
    def test_correct_derivation(self) -> None:
        engines = _Engines()
        result = engines.check(["(x+1)^2 - 1", "x^2 + 2x + 1 - 1", "x^2 + 2x", "x(x+2)"])
        self.assertEqual(result["verdict"], "equivalent")
        self.assertIsNone(result["first_wrong_step"])
        self.assertEqual([record["step"] for record in result["steps"]], [2, 3, 4])
        self.assertEqual({record["method"] for record in result["steps"]}, {"numeric"})
        self.assertEqual(
            result["summary"],
            "Step 2 follows from step 1. Step 3 follows from step 2. Step 4 follows from step 3.",
        )
        self.assertEqual(engines.requests, [])  # the numeric check settles every step
        self.assertNotIn("error", result)

    @unittest.skipUnless(HAVE_SYMPY, "needs SymPy")
    def test_first_wrong_step(self) -> None:
        result = _check(["(x+1)^2 - 1", "x^2 + 1 - 1", "x^2"])
        self.assertEqual(result["verdict"], "not_equivalent")
        self.assertEqual(result["first_wrong_step"], 2)
        wrong, right = result["steps"]
        self.assertEqual(wrong["verdict"], "not_equivalent")
        self.assertEqual(wrong["method"], "symbolic+numeric")
        self.assertEqual(wrong["reason"], "step 2 - step 1 = -2*x (not 0); at x = 1, step 1 = 3 and step 2 = 1")
        self.assertEqual(wrong["counterexample"], {"at": {"x": 1}, "previous": 3, "this": 1})
        self.assertEqual(right, {"step": 3, "verdict": "equivalent", "method": "numeric"})
        self.assertEqual(
            result["summary"],
            "Step 2 is not equivalent to step 1: step 2 - step 1 = -2*x (not 0); at x = 1, step 1 = 3 and "
            "step 2 = 1. Step 3 follows from step 2.",
        )

    def test_wrong_without_sympy(self) -> None:
        result = _Engines(sympy=lambda first, second: None).check(["(x+1)^2 - 1", "x^2 + 1 - 1", "x^2"])
        wrong = result["steps"][0]
        self.assertEqual((wrong["verdict"], wrong["method"]), ("not_equivalent", "numeric"))
        self.assertEqual(wrong["reason"], "at x = 1, step 1 = 3 and step 2 = 1")
        self.assertEqual(
            result["notes"], ["SymPy was not available: from step 2 on, the steps were checked numerically only."]
        )

    def test_trig_identities_and_rational_cancellation(self) -> None:
        for steps in (
            ["sin(2x)/(2cos(x))", "2sin(x)cos(x)/(2cos(x))", "sin(x)"],
            ["(1 - cos(x)^2)/sin(x)", "sin(x)^2/sin(x)", "sin(x)"],
            ["sec(x)^2 - tan(x)^2", "1"],
            ["cos(x)^4 - sin(x)^4", "(cos(x)^2 - sin(x)^2)(cos(x)^2 + sin(x)^2)", "cos(2x)"],
            ["(x^2 - 1)/(x - 1)", "(x-1)(x+1)/(x-1)", "x + 1"],
            ["(x^3 - 8)/(x^2 - 4)", "(x^2 + 2x + 4)/(x + 2)"],
            ["1/(x-1) - 1/(x+1)", "((x+1) - (x-1))/((x-1)(x+1))", "2/(x^2 - 1)"],
        ):
            with self.subTest(steps=steps):
                result = _check(steps)
                self.assertEqual(result["verdict"], "equivalent", result["summary"])

    @unittest.skipUnless(HAVE_SYMPY, "needs SymPy")
    def test_wrong_trig_and_rational_steps(self) -> None:
        for steps, wrong in (
            (["sin(x)^2 + cos(x)^2", "2"], 2),
            (["sin(2x)", "2sin(x)"], 2),
            (["cos(2x)", "cos(x)^2 + sin(x)^2"], 2),
            (["(x^2 - 4)/(x - 2)", "x - 2"], 2),
            (["(x^2 + 1)/(x + 1)", "x + 1"], 2),
            (["1/x + 1/y", "2/(x + y)"], 2),
            (["(a + b)^2", "a^2 + 2a*b + b^2", "a^2 + b^2"], 3),
        ):
            with self.subTest(steps=steps):
                result = _check(steps)
                self.assertEqual(result["verdict"], "not_equivalent", result["summary"])
                self.assertEqual(result["first_wrong_step"], wrong)

    def test_several_variables(self) -> None:
        result = _check(["(a + b)(a - b)", "a^2 - a*b + a*b - b^2", "a^2 - b^2"])
        self.assertEqual(result["verdict"], "equivalent")
        result = _check(["log(x*y)", "log(x) + log(y)"])
        self.assertEqual(result["verdict"], "equivalent")
        self.assertIn("note", result["steps"][0])


class TestDomains(unittest.TestCase):
    def test_log_of_a_square(self) -> None:
        result = _check(["log(x^2)", "2log(x)"])
        self.assertEqual(result["verdict"], "equivalent")
        record = result["steps"][0]
        self.assertEqual(record["note"], "equivalent where both are defined; step 2 is undefined at x = -1")
        self.assertEqual(
            result["summary"],
            "Step 2 follows from step 1 (equivalent where both are defined; step 2 is undefined at x = -1).",
        )

    def test_bars_after_a_function(self) -> None:
        # ln|x| is log(abs(x)) for both engines, so the steps are equivalent wherever x != 0
        result = _check(["ln|x|^2", "(ln|x|)^2", "ln(|x|)*ln|x|"])
        self.assertEqual(result["verdict"], "equivalent", result["summary"])
        result = _check(["ln(x^2)", "2ln|x|"])
        self.assertEqual(result["verdict"], "equivalent", result["summary"])
        self.assertNotIn("note", result["steps"][0])

    def test_sqrt_of_a_square(self) -> None:
        result = _check(["sqrt(x^2)", "x"])
        self.assertEqual(result["verdict"], "not_equivalent")
        record = result["steps"][0]
        self.assertEqual(record["counterexample"], {"at": {"x": -1}, "previous": 1, "this": -1})
        if HAVE_SYMPY:
            self.assertEqual(
                record["reason"], "step 2 - step 1 = x-abs(x) (not 0); at x = -1, step 1 = 1 and step 2 = -1"
            )

    def test_absolute_value_beyond_small_values(self) -> None:
        result = _check(["sqrt((x-5)^2)", "|x - 5|", "5 - x"])
        self.assertEqual(result["first_wrong_step"], 3)
        self.assertEqual(result["steps"][1]["counterexample"]["at"], {"x": 6})


class TestGuards(unittest.TestCase):
    def test_joined_letters(self) -> None:
        engines = _Engines()
        result = engines.check(["x^2 + 2xy + y^2", "(x + y)^2"])
        record = result["steps"][0]
        self.assertEqual((record["verdict"], record["method"]), ("undecided", "none"))
        self.assertEqual(record["reason"], "xy is read as one variable, but the other step has x and y; write x*y")
        self.assertEqual(result["verdict"], "undecided")
        self.assertEqual(engines.requests, [])
        # xy in both steps is one name, consistently
        self.assertEqual(_check(["xy + xy", "2xy"])["verdict"], "equivalent")

    def test_unreadable_step(self) -> None:
        result = _Engines(sympy=lambda first, second: {"error": "Cannot read the expression."}).check(
            ["x^2 + 1", "x^2 +* 1", "x^2 + 1"]
        )
        self.assertEqual(result["verdict"], "undecided")
        first, second = result["steps"]
        self.assertEqual(first["verdict"], "undecided")
        self.assertTrue(first["reason"].startswith("step 2 cannot be read ("), first["reason"])
        self.assertTrue(first["reason"].endswith("; SymPy: Cannot read the expression)"), first["reason"])
        self.assertTrue(second["reason"].startswith("step 2 cannot be read ("), second["reason"])

    def test_math_js_cannot_read_but_sympy_can(self) -> None:
        answers = {"verdict": "equivalent", "difference": None}
        with mock.patch.object(step_checker, "compare") as compare:
            result = _Engines(sympy=lambda first, second: {"result": answers}).check(["x^2", "x*x +* 0"])
            compare.assert_not_called()
        self.assertEqual(result["steps"][0], {"step": 2, "verdict": "equivalent", "method": "symbolic"})

    def test_checks_that_disagree(self) -> None:
        sympy = {"result": {"verdict": "equivalent", "difference": None}}
        result = _Engines(sympy=lambda first, second: sympy).check(["x^2", "x^2 + 1"])
        record = result["steps"][0]
        self.assertEqual(record["verdict"], "undecided")
        self.assertEqual(
            record["reason"],
            "the checks disagree: SymPy finds the steps equivalent, but at x = 1, step 1 = 1 and step 2 = 2",
        )

    def test_sympy_settles_an_undecided_pair(self) -> None:
        # sqrt(x-300) is defined at too few sample points for the numeric check
        engines = _Engines()
        result = engines.check(["sqrt(x-300)^2", "x - 300"])
        if HAVE_SYMPY:
            self.assertEqual(result["steps"][0]["method"], "symbolic")
            self.assertEqual(result["verdict"], "equivalent")
        self.assertEqual(len(engines.requests), 1)

    def test_undecided_reasons_name_the_steps(self) -> None:
        result = _Engines(sympy=lambda first, second: None).check(["sqrt(-x^2-1)", "sqrt(-x^2-2)"])
        self.assertEqual(
            result["steps"][0]["reason"],
            "the numeric check could not decide (neither step has a real value at any sample point)",
        )

    def test_timeout_and_budget(self) -> None:
        result = _Engines(sympy=lambda first, second: {"timeout": True}).check(["sqrt(x-300)", "sqrt(x-300)*1"])
        self.assertEqual(result["steps"][0]["verdict"], "undecided")
        self.assertIn("SymPy ran out of time", result["steps"][0]["reason"])
        with mock.patch.object(step_checker, "SYMPY_BUDGET_S", 0.0):
            engines = _Engines()
            result = engines.check(["sqrt(x-300)", "sqrt(x-300)*1", "sqrt(x-300)"])
        self.assertEqual(engines.requests, [])
        self.assertEqual(
            result["notes"],
            ["SymPy's time budget (0 s) ran out: from step 2 on, the steps were checked numerically only."],
        )


# Steps written as on a calculator: functions without parentheses, and e or pi glued to a letter
CALCULATOR_EQUIVALENT = [
    ["(x+1)e^x", "xe^x + e^x"],
    ["2xe^(x^2)", "2x*exp(x^2)"],
    ["xpi", "pi*x"],
    ["2 sin x cos x", "sin 2x"],
    ["ln x + ln y", "ln(x*y)"],
    ["sin^2 x + cos^2 x", "1"],
    ["sec x", "1/cos(x)"],
    ["x ln x", "x*ln(x)"],
    ["sinx", "sin(x)"],
    ["sin x", "sin(x)"],
    ["cos x", "cos(x)"],
    ["log x^2", "log(x^2)"],
    ["log x^2", "2log(x)"],
    ["cos^2 x - sin^2 x", "cos 2x"],
    ["xlnx - x", "x(lnx - 1)"],
    ["2xsinx", "2x sin x"],
    ["e^xsinx", "e^x sin x"],
    ["xsinx + cosx", "x sin x + cos x"],
    ["tcost", "t cos t"],
    ["nlogn", "n log n"],
    ["sin 2pi x", "sin(2pi*x)"],
    ["sin 2.5 x", "sin(2.5x)"],
    ["sin^-1 x", "asin(x)"],
    # descriptive names stay variables
    ["revenue - costs", "-(costs - revenue)"],
    ["max_height/2", "0.5*max_height"],
    ["signal + noise", "noise + signal"],
    ["minutes/60", "minutes*(1/60)"],
    ["2*exponent", "exponent + exponent"],
    ["cos_theta^2", "cos_theta*cos_theta"],
    ["seconds*1000", "1000seconds"],
    ["revenue - cost", "-(cost - revenue)"],
    ["sinhx", "sinh(x)"],
    ["coshx", "cosh x"],
    ["asinhx", "asinh(x)"],
    ["acoshx", "acosh(x)"],
    ["atanhx", "atanh x"],
    ["xasinhx", "x*asinh(x)"],
    # a glued call before / or *
    ["sinx/x", "sin(x)/x"],
    ["lnx/x", "ln(x)/x"],
    ["lnx / x", "ln(x)/x"],
    ["sqrtx/x", "1/sqrtx"],
    ["sinx*2", "2sin(x)"],
    ["2cosx*x", "2x*cos(x)"],
    ["sinx/2", "sin(x)/2"],
    ["sinhx/x", "sinh(x)/x"],
]
CALCULATOR_WRONG = [
    ["revenue - cost", "revenue - cos(t)"],
    ["(x+1)e^x", "xe^x + 1"],
    ["2xe^(x^2)", "2e^(x^2)"],
    ["xpi", "pi + x"],
    ["xe^x", "x + e^x"],
    ["2 sin x cos x", "sin x"],
    ["sin 2x", "2 sin x"],
    ["ln x + ln y", "ln(x + y)"],
    ["sin^2 x + cos^2 x", "2"],
    ["sec x", "1/sin x"],
    ["x ln x", "ln(x)"],
    ["sinx", "cos x"],
    ["sin x^2", "sin(x)^2"],
    ["ln x^2", "(ln x)^2"],
    ["xlnx - x", "x(lnx + 1)"],
    ["2xsinx", "2x cos x"],
    ["e^xsinx", "e^x + sin x"],
    ["tcost", "cos t"],
    ["nlogn", "log n"],
    ["sin pi x", "sin(pi)*x"],
]


class TestCalculatorNotation(unittest.TestCase):
    def _verdicts(self, cases: List[List[str]], sympy: bool) -> List[str]:
        engines = _Engines() if sympy else _Engines(sympy=lambda first, second: None)
        return [engines.check(steps)["verdict"] for steps in cases]

    def test_correct_steps_are_equivalent_without_sympy(self) -> None:
        self.assertEqual(self._verdicts(CALCULATOR_EQUIVALENT, False), ["equivalent"] * len(CALCULATOR_EQUIVALENT))

    def test_wrong_steps_stay_wrong_without_sympy(self) -> None:
        self.assertEqual(self._verdicts(CALCULATOR_WRONG, False), ["not_equivalent"] * len(CALCULATOR_WRONG))

    @unittest.skipUnless(HAVE_SYMPY, "needs SymPy")
    def test_with_sympy(self) -> None:
        self.assertEqual(self._verdicts(CALCULATOR_EQUIVALENT, True), ["equivalent"] * len(CALCULATOR_EQUIVALENT))
        self.assertEqual(self._verdicts(CALCULATOR_WRONG, True), ["not_equivalent"] * len(CALCULATOR_WRONG))
        reason = _check(["(x+1)e^x", "xe^x + 1"])["steps"][0]["reason"]
        self.assertEqual(reason, "step 2 - step 1 = 1-exp(x) (not 0); at x = 1, step 1 = 5.43656 and step 2 = 3.71828")

    def test_misread_names_are_undecided_with_a_hint(self) -> None:
        # function names math.js reads as variables (sin alone, gamma) are in the Brython test
        engines = _Engines()
        for steps, reason in (
            (["sinxy", "sin(x*y)"], "sinxy is read as one variable; write sin(x*y)"),
            (["xye^x", "x*y*e^x"], "xye is read as one variable; write x*y*e"),
            (["sinxcosx", "sin(x)cos(x)"], "sinxcosx is read as one variable; write sin(x)*cos(x)"),
            (["sin x2", "sin(2x)"], "x2 is read as one variable; write x^2 or 2*x"),
            (["xy", "yx"], "xy and yx are each read as one variable; write x*y"),
            (["log2x", "log2(x)"], "log2x is read as one variable; write log2(x) or log(2x)"),
            (["log10x", "log(10x)"], "log10x is read as one variable; write log10(x) or log(10x)"),
            (["sinx^-1", "1/sin(x)"], "sinx is read as one variable here; write sin(x)"),
            (["xlnx^-1", "x/ln(x)"], "xlnx is read as one variable here; write x*ln(x)"),
        ):
            with self.subTest(steps=steps):
                record = engines.check(steps)["steps"][0]
                self.assertEqual((record["verdict"], record["method"], record["reason"]), ("undecided", "none", reason))
        self.assertEqual(engines.requests, [])

    @unittest.skipUnless(HAVE_SYMPY, "needs SymPy")
    def test_reciprocal_hyperbolic_functions(self) -> None:
        # sech is sech(x), not sec(h)*x (the stand-in cannot evaluate sech, SymPy decides)
        self.assertEqual(_check(["sech x", "1/cosh(x)"])["verdict"], "equivalent")
        self.assertEqual(_check(["coth x", "cosh(x)/sinh(x)"])["verdict"], "equivalent")
        # without math.js's sech here, SymPy alone cannot show this difference is not 0
        self.assertNotEqual(_check(["sech x", "1/cos(x)"])["verdict"], "equivalent")

    def test_long_names_are_checked_quickly(self) -> None:
        started = time.time()
        _Engines(sympy=lambda first, second: None).check(["tanx" * 14 + "Q", "t+a+n+x+Q"])
        _Engines(sympy=lambda first, second: None).check(["tanx" * 9 + "Q", "t+a+n+x+Q"])
        self.assertLess(time.time() - started, 2.0)

    def test_function_names_are_never_bound_as_variables(self) -> None:
        with mock.patch.object(step_checker, "compare", wraps=step_checker.compare) as spy:
            _Engines(sympy=lambda first, second: None).check(["x + sin(x)", "sin(x) + x"])
        self.assertEqual(spy.call_args[0][2], ["x"])


class TestInput(unittest.TestCase):
    def test_refusals(self) -> None:
        engines = _Engines()
        for steps, message in (
            ("x + 1", "steps must be a list"),
            (["x"], "at least 2 steps"),
            ([f"x + {k}" for k in range(13)], "at most 12 steps"),
            (["x", ""], "step 2 must be a non-empty expression"),
            (["x", 3], "step 2 must be a non-empty expression"),
            (["x", "x" * 1001], "longer than 1000 characters"),
            (["2x + 1 = 5", "2x = 4"], "step 1 (2x + 1 = 5) is an equation or inequality"),
            (["x", "x ≤ 2"], "step 2 (x ≤ 2) is an equation or inequality"),
        ):
            with self.subTest(steps=steps):
                result = engines.check(steps)
                self.assertIsInstance(result, str)
                self.assertTrue(result.startswith("Error: "), result)
                self.assertIn(message, result)
        self.assertIn("For a chain a = b = c, pass a, b and c.", engines.check(["a", "b = c"]))


if __name__ == "__main__":
    unittest.main()
