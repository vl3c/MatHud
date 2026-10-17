"""Tests for the check_steps tool (utils.step_checker, utils.equivalence in Brython).

TestStepCheckerSympy pins the reasons with SymPy's difference and skips without the server's
SymPy engine; TestStepCheckerWithoutSympy pins the numeric verdicts, which answer when SymPy
is unavailable. TestStepCheckerHelpers pins what Brython reads (math.js variables and
values) and prints (CPython's pytest covers the same helpers).
"""

from __future__ import annotations

import json
import math
import unittest
from typing import Any, Dict, List

from utils.equivalence import distinct_value_texts, json_number, point_text, value_text
from utils.step_checker import StepChecker
from utils.symbolic_checks import SymbolicChecks

from .cas_engines import WithoutSympy, sympy_available


def _check(steps: List[str]) -> Dict[str, Any]:
    result = StepChecker.check_steps(steps)
    if not isinstance(result, dict):
        raise AssertionError(f"not a result: {result}")
    return result


WRONG_SECOND_STEP = ["(x+1)^2 - 1", "x^2 + 1 - 1", "x^2"]
CORRECT = ["(x+1)^2 - 1", "x^2 + 2x + 1 - 1", "x^2 + 2x", "x(x+2)"]


class TestStepCheckerHelpers(unittest.TestCase):
    def test_math_js_variables(self) -> None:
        self.assertEqual(SymbolicChecks.expression_variables("xy + 2x"), ["x", "xy"])
        self.assertEqual(SymbolicChecks.expression_variables("sin(x)^2 + pi*e + i"), ["x"])
        self.assertEqual(SymbolicChecks.expression_variables("log(a*b)"), ["a", "b"])
        self.assertEqual(SymbolicChecks.expression_variables("2^3"), [])
        with self.assertRaises(ValueError):
            SymbolicChecks.expression_variables("x +* 1")

    def test_math_js_values(self) -> None:
        evaluate = SymbolicChecks.scope_function("x^2 + y")
        assert evaluate is not None
        self.assertEqual(evaluate({"x": 3.0, "y": 0.5}), 9.5)
        log = SymbolicChecks.scope_function("log(x)")
        assert log is not None
        self.assertIsNone(log({"x": -1.0}))  # complex, not real
        self.assertEqual(log({"x": 0.0}), -math.inf)
        self.assertIsNone(SymbolicChecks.scope_function("x +* 1"))

    def test_brython_prints_numbers_like_cpython(self) -> None:
        self.assertEqual(distinct_value_texts(2 * math.sqrt(2), 4 * math.sqrt(2)), ("2.82843", "5.65685"))
        self.assertEqual(distinct_value_texts(1.0000001, 1.0000002), ("1.0000001", "1.0000002"))
        self.assertEqual(value_text(-3.0), "-3")
        self.assertEqual(point_text({"x": 1.0, "y": -0.5}), "x = 1, y = -0.5")
        self.assertEqual(json_number(1 / 3), 0.333333333333)
        self.assertEqual(json.dumps({"n": json_number(3.0), "q": json_number(0.4)}), '{"n": 3, "q": 0.4}')

    def test_refusals(self) -> None:
        for steps, message in (
            ("x + 1", "steps must be a list"),
            (["x"], "at least 2 steps"),
            ([f"x + {k}" for k in range(13)], "at most 12 steps"),
            (["2x + 1 = 5", "2x = 4"], "is an equation or inequality"),
            (["x", "x ≥ 2"], "is an equation or inequality"),
        ):
            result = StepChecker.check_steps(steps)
            self.assertTrue(isinstance(result, str) and result.startswith("Error: "), f"{steps}: {result}")
            self.assertIn(message, result)


class TestStepCheckerSympy(unittest.TestCase):
    def setUp(self) -> None:
        if not sympy_available():
            self.skipTest("the server's SymPy engine is not available")

    def test_first_wrong_step(self) -> None:
        result = _check(WRONG_SECOND_STEP)
        self.assertEqual(result["verdict"], "not_equivalent")
        self.assertEqual(result["first_wrong_step"], 2)
        self.assertEqual(
            result["steps"][0],
            {
                "step": 2,
                "verdict": "not_equivalent",
                "method": "symbolic+numeric",
                "reason": "step 2 - step 1 = -2*x (not 0); at x = 1, step 1 = 3 and step 2 = 1",
                "counterexample": {"at": {"x": 1}, "previous": 3, "this": 1},
            },
        )
        self.assertEqual(result["steps"][1], {"step": 3, "verdict": "equivalent", "method": "numeric"})
        self.assertEqual(
            result["summary"],
            "Step 2 is not equivalent to step 1: step 2 - step 1 = -2*x (not 0); at x = 1, step 1 = 3 "
            "and step 2 = 1. Step 3 follows from step 2.",
        )

    def test_domains(self) -> None:
        record = _check(["sqrt(x^2)", "x"])["steps"][0]
        self.assertEqual(record["reason"], "step 2 - step 1 = x-abs(x) (not 0); at x = -1, step 1 = 1 and step 2 = -1")
        record = _check(["log(x^2)", "2log(x)"])["steps"][0]
        self.assertEqual(record["verdict"], "equivalent")
        self.assertEqual(record["note"], "equivalent where both are defined; step 2 is undefined at x = -1")

    def test_wrong_steps(self) -> None:
        for steps, wrong in (
            (["sin(x)^2 + cos(x)^2", "2"], 2),
            (["sin(2x)", "2sin(x)"], 2),
            (["(x^2 - 4)/(x - 2)", "x - 2"], 2),
            (["1/x + 1/y", "2/(x + y)"], 2),
            (["(a + b)^2", "a^2 + 2a*b + b^2", "a^2 + b^2"], 3),
            (["sqrt((x-5)^2)", "|x - 5|", "5 - x"], 3),
            (["1/2 + 1/3", "2/5"], 2),
        ):
            result = _check(steps)
            self.assertEqual(result["first_wrong_step"], wrong, f"{steps}: {result['summary']}")

    def test_sympy_settles_what_the_samples_cannot(self) -> None:
        # sqrt(x - 300) is real at too few sample points for the numeric check
        record = _check(["sqrt(x-300)^2", "x - 300"])["steps"][0]
        self.assertEqual((record["verdict"], record["method"]), ("equivalent", "symbolic"))


class TestStepCheckerWithoutSympy(unittest.TestCase):
    def test_numeric_verdicts(self) -> None:
        with WithoutSympy():
            correct = _check(CORRECT)
            wrong = _check(WRONG_SECOND_STEP)
            identities = [
                _check(steps)["verdict"]
                for steps in (
                    ["sin(x)^2/(1-cos(x))", "(1-cos(x))(1+cos(x))/(1-cos(x))", "1+cos(x)"],
                    ["tan(x) + cot(x)", "(sin(x)^2+cos(x)^2)/(sin(x)cos(x))", "2/sin(2x)"],
                    ["(x^2+5x+6)/(x+2)", "(x+2)(x+3)/(x+2)", "x+3"],
                    ["(a + b)(a - b)", "a^2 - b^2"],
                    ["sin²(x) + cos²(x)", "1"],
                    ["ln|x|^2", "(ln|x|)^2"],
                )
            ]
        self.assertEqual(correct["verdict"], "equivalent")
        self.assertNotIn("notes", correct)  # no pair needed SymPy
        self.assertEqual(wrong["first_wrong_step"], 2)
        self.assertEqual(wrong["steps"][0]["reason"], "at x = 1, step 1 = 3 and step 2 = 1")
        self.assertEqual(wrong["steps"][0]["method"], "numeric")
        self.assertEqual(
            wrong["notes"], ["SymPy was not available: from step 2 on, the steps were checked numerically only."]
        )
        self.assertEqual(identities, ["equivalent"] * len(identities))

    def test_guards(self) -> None:
        with WithoutSympy():
            joined = _check(["x^2 + 2xy + y^2", "(x + y)^2"])
            unreadable = _check(["x^2 + 1", "x^2 +* 1"])
            domain = _check(["log(x^2)", "2log(x)"])
        self.assertEqual(
            joined["steps"][0]["reason"], "xy is read as one variable, but the other step has x and y; write x*y"
        )
        self.assertEqual(joined["verdict"], "undecided")
        self.assertEqual(unreadable["verdict"], "undecided")
        self.assertTrue(unreadable["steps"][0]["reason"].startswith("step 2 cannot be read (math.js: "))
        self.assertEqual(domain["steps"][0]["note"], "equivalent where both are defined; step 2 is undefined at x = -1")
