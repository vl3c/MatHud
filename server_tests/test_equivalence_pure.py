"""Tests for utils.equivalence: the numeric check of whether two expressions are equivalent.

The browser evaluates the expressions with math.js; here compile_expression's Python
evaluators stand in for it (an overflow is +inf, a complex or failed value undefined).
"""

from __future__ import annotations

import math
import unittest
from typing import Optional, Sequence, Tuple

from server_tests import client_renderer  # noqa: F401  (browser stub, so compiled_expression imports)
from compiled_expression import compile_expression
from utils.cas_text import split_top_level_terms
from utils.equivalence import (
    EQUIVALENT,
    MIN_AGREEING,
    NOT_EQUIVALENT,
    POINTS_PER_TIER,
    TIERS,
    UNDECIDED,
    EquivalenceReport,
    Evaluator,
    Point,
    compare,
    distinct_value_texts,
    json_number,
    point_text,
    rounding_noise,
    sample_points,
)


def _evaluator(text: str, names: Sequence[str]) -> Evaluator:
    bound = compile_expression(text, tuple(names)).bind()

    def evaluate(point: Point) -> Optional[float]:
        try:
            value = bound(*[point[name] for name in names])
        except OverflowError:
            return math.inf
        except (ArithmeticError, ValueError, TypeError):
            return None
        return None if isinstance(value, complex) else float(value)

    return evaluate


def _scale(text: str, names: Sequence[str]) -> Optional[Evaluator]:
    """The sum of |term| over the terms of text, as StepChecker gives it."""
    terms = [_evaluator(term, names) for _, term in split_top_level_terms(text)]
    if len(terms) < 2:
        return None

    def scale(point: Point) -> Optional[float]:
        values = [term(point) for term in terms]
        if any(value is None for value in values):
            return None
        return float(sum(abs(value) for value in values if value is not None))

    return scale


def _compare(first: str, second: str, names: Tuple[str, ...] = ("x",)) -> EquivalenceReport:
    return compare(
        _evaluator(first, names),
        _evaluator(second, names),
        names,
        _scale(first, names),
        _scale(second, names),
    )


EXPANDED_POWER = (
    "x^15 - 45*x^14 + 945*x^13 - 12285*x^12 + 110565*x^11 - 729729*x^10 + 3648645*x^9"
    " - 14073345*x^8 + 42220035*x^7 - 98513415*x^6 + 177324147*x^5 - 241805655*x^4"
    " + 241805655*x^3 - 167403915*x^2 + 71744535*x - 14348907"
)


class TestEquivalent(unittest.TestCase):
    def test_rewrites_are_equivalent(self) -> None:
        for first, second in (
            ("(x+1)^2 - 1", "x^2 + 2*x + 1 - 1"),
            ("x^2 + 2*x", "x*(x+2)"),
            ("sin(x)^2", "1 - cos(x)^2"),
            ("sin(2*x)", "2*sin(x)*cos(x)"),
            ("cos(x)^2 - sin(x)^2", "cos(2*x)"),
            ("tan(x)", "sin(x)/cos(x)"),
            ("sec(x)^2 - 1", "tan(x)^2"),
            ("(x^2 - 1)/(x - 1)", "x + 1"),
            ("(x^3 - 8)/(x - 2)", "x^2 + 2*x + 4"),
            ("1/(x-1) - 1/(x+1)", "2/(x^2-1)"),
            ("exp(2*x)/exp(x)", "exp(x)"),
            ("2^(x+1)", "2*2^x"),
            ("(x-3)^15", EXPANDED_POWER),
            ("sin(x^3)", "sin(x*x*x)"),
            ("x^8 + 1 - x^8", "1"),
        ):
            with self.subTest(first=first, second=second):
                report = _compare(first, second)
                self.assertEqual(report.verdict, EQUIVALENT, report.reason)
                self.assertEqual(report.mismatches, 0)
                self.assertGreaterEqual(report.agreeing, MIN_AGREEING)
                self.assertFalse(report.domain_note)

    def test_every_variable_varies(self) -> None:
        names = ("a", "b")
        self.assertEqual(_compare("(a+b)^2", "a^2 + 2*a*b + b^2", names).verdict, EQUIVALENT)
        self.assertEqual(_compare("sin(a+b)", "sin(a)*cos(b) + cos(a)*sin(b)", names).verdict, EQUIVALENT)
        report = _compare("(a+b)^2", "a^2 + b^2", names)
        self.assertEqual(report.verdict, NOT_EQUIVALENT)
        self.assertEqual(report.counterexample, {"a": 1.0, "b": 1.0})
        self.assertEqual((report.first_value, report.second_value), (4.0, 2.0))
        # b is fixed in a one-variable check; here it varies, so b*0 + a is not a*b
        self.assertEqual(_compare("a*b", "a", names).verdict, NOT_EQUIVALENT)

    def test_constants(self) -> None:
        self.assertEqual(compare(_evaluator("2+3", ()), _evaluator("5", ()), ()).verdict, EQUIVALENT)
        report = compare(_evaluator("2+3", ()), _evaluator("6", ()), ())
        self.assertEqual(report.verdict, NOT_EQUIVALENT)
        self.assertEqual(report.counterexample, {})
        self.assertEqual(compare(_evaluator("sqrt(-1)", ()), _evaluator("1", ()), ()).verdict, UNDECIDED)


class TestNotEquivalent(unittest.TestCase):
    def test_readable_counterexample(self) -> None:
        report = _compare("(x+1)^2 - 1", "x^2 + 1 - 1")
        self.assertEqual(report.verdict, NOT_EQUIVALENT)
        self.assertEqual(report.counterexample, {"x": 1.0})
        self.assertEqual((report.first_value, report.second_value), (3.0, 1.0))

    def test_difference_beyond_the_first_tier(self) -> None:
        # abs(x - 5) is 5 - x only for x <= 5: the wider tiers see it
        report = _compare("abs(x-5)", "5-x")
        self.assertEqual(report.verdict, NOT_EQUIVALENT)
        self.assertEqual(report.counterexample, {"x": 6.0})

    def test_sqrt_of_a_square(self) -> None:
        report = _compare("sqrt(x^2)", "x")
        self.assertEqual(report.verdict, NOT_EQUIVALENT)
        self.assertEqual(report.counterexample, {"x": -1.0})
        self.assertEqual((report.first_value, report.second_value), (1.0, -1.0))

    def test_one_mismatch(self) -> None:
        """One mismatch decides when it is clear; a slight one leaves the verdict open."""
        odd = sample_points(["x"])[5]
        for factor, verdict in ((1 + 1e-3, NOT_EQUIVALENT), (1 + 1e-6, UNDECIDED)):
            with self.subTest(factor=factor):

                def second(point: Point, factor: float = factor) -> Optional[float]:
                    return float(point["x"] * factor if abs(point["x"] - odd["x"]) < 1e-9 else point["x"])

                report = compare(lambda point: point["x"], second, ("x",))
                self.assertEqual(report.verdict, verdict)
                self.assertEqual(report.mismatches, 1)
        # No readable point differs, so the counterexample is the sample point itself
        self.assertEqual(report.reason, "the values differ slightly at one sample point only, which may be rounding")
        clear = compare(
            lambda point: point["x"],
            lambda point: point["x"] * (1.01 if abs(point["x"] - odd["x"]) < 1e-9 else 1),
            ("x",),
        )
        self.assertEqual(clear.counterexample, odd)


class TestDomains(unittest.TestCase):
    def test_equivalent_where_both_are_defined(self) -> None:
        report = _compare("log(x^2)", "2*log(x)")
        self.assertEqual(report.verdict, EQUIVALENT)
        self.assertTrue(report.domain_note)
        self.assertEqual(report.undefined_point, {"x": -1.0})
        self.assertEqual(report.undefined_side, 2)
        report = _compare("sqrt(x)*sqrt(x)", "x")
        self.assertEqual(report.verdict, EQUIVALENT)
        self.assertEqual((report.undefined_point, report.undefined_side), ({"x": -1.0}, 1))

    def test_a_narrow_domain_widens_the_tiers(self) -> None:
        self.assertEqual(_compare("sqrt(x-10)", "sqrt(x-10)*1").verdict, EQUIVALENT)

    def test_undecided(self) -> None:
        report = _compare("sqrt(-x^2-1)", "sqrt(-x^2-2)")
        self.assertEqual(report.verdict, UNDECIDED)
        self.assertIn("neither expression", report.reason)
        report = _compare("sqrt(x-300)", "sqrt(x-300)")
        self.assertEqual(report.verdict, UNDECIDED)
        self.assertTrue(0 < report.agreeing < MIN_AGREEING, report.agreeing)
        self.assertIn("both defined at only", report.reason)
        report = compare(_evaluator("x", ("x",)), lambda point: None, ("x",))
        self.assertEqual(report.verdict, UNDECIDED)
        self.assertIn("second expression has no real value", report.reason)

    def test_overflow_is_unknown(self) -> None:
        # exp(2x) overflows near x = 355, where exp(x) does not: neither agreement nor mismatch
        report = _compare("exp(2*x)/exp(x)", "exp(x)")
        self.assertEqual(report.verdict, EQUIVALENT)
        self.assertFalse(report.domain_note)


class TestSamplesAndText(unittest.TestCase):
    def test_sample_points(self) -> None:
        points = sample_points(["x", "y"])
        self.assertEqual(len(points), POINTS_PER_TIER * len(TIERS))
        self.assertEqual(points, sample_points(["x", "y"]))
        for tier, reach in enumerate(TIERS):
            for point in points[tier * POINTS_PER_TIER : (tier + 1) * POINTS_PER_TIER]:
                for value in point.values():
                    self.assertLessEqual(abs(value), reach)
                    self.assertNotEqual(value, math.floor(value))
        self.assertNotEqual([p["x"] for p in points], [p["y"] for p in points])
        self.assertEqual(sample_points([]), [{}])

    def test_rounding_noise(self) -> None:
        expanded = _evaluator(EXPANDED_POWER, ("x",))
        self.assertGreater(rounding_noise(expanded, {"x": 2.2}), 1e-9)
        self.assertLess(rounding_noise(_evaluator("x*y", ("x", "y")), {"x": 2.0, "y": 0.0}), 1e-300)
        self.assertEqual(rounding_noise(_evaluator("log(x)", ("x",)), {"x": -1.0}), math.inf)

    def test_text(self) -> None:
        self.assertEqual(point_text({"x": 1.0, "y": -0.5}), "x = 1, y = -0.5")
        self.assertEqual(distinct_value_texts(3.0, 1.0), ("3", "1"))
        self.assertEqual(distinct_value_texts(1.0000001, 1.0000002), ("1.0000001", "1.0000002"))
        self.assertEqual(json_number(3.0), 3)
        self.assertIsInstance(json_number(3.0), int)
        self.assertEqual(json_number(1 / 3), 0.333333333333)
        self.assertEqual(json_number(math.inf), "Infinity")
        self.assertEqual(json_number(math.nan), "NaN")


if __name__ == "__main__":
    unittest.main()
