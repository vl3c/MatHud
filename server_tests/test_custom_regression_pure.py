"""Custom-model regression: reading models, the fit engine, its notes, and NIST's reference problems.

The NIST problems (server_tests/data/nist_strd_nonlinear.json) are the Statistical
Reference Datasets for nonlinear regression, with certified sums of squares and constants.
"""

from __future__ import annotations

import json
import math
import random
import unittest
from pathlib import Path
from typing import Callable, Dict, List, Optional

from server_tests import client_renderer  # noqa: F401  (browser stub, so MathUtils resolves)
from utils.statistics.custom_model import read_model
from utils.statistics.custom_regression import EVALUATION_LIMIT, fit_custom_model, CustomFit
from utils.statistics.variable_projection import linear_constants

NIST_PATH = Path(__file__).parent / "data" / "nist_strd_nonlinear.json"
# NIST problems the automatic search solves with no starting values
NIST_SOLVED_WITHOUT_STARTS = (
    "Misra1a",
    "Chwirut1",
    "Chwirut2",
    "Lanczos1",
    "Lanczos2",
    "Lanczos3",
    "Gauss1",
    "DanWood",
    "Misra1b",
    "Kirby2",
    "Hahn1",
    "MGH17",
    "Misra1c",
    "Misra1d",
    "Roszman1",
    "MGH09",
    "BoxBOD",
    "Rat42",
    "MGH10",
    "Eckerle4",
    "Rat43",
    "Bennett5",
)


def _grid(low: float, high: float, count: int) -> List[float]:
    return [low + (high - low) * i / (count - 1) for i in range(count)]


def _noisy(function: Callable[[float], float], xs: List[float], noise: float, seed: int) -> List[float]:
    rng = random.Random(seed)
    return [function(x) + noise * rng.gauss(0.0, 1.0) for x in xs]


def _fit(model: str, xs: List[float], ys: List[float], initial: Optional[Dict[str, float]] = None) -> CustomFit:
    return fit_custom_model(xs, ys, read_model(model), initial)


class TestReadModel(unittest.TestCase):
    def test_constants_in_order_of_appearance(self) -> None:
        self.assertEqual(read_model("a*exp(-b*x)*cos(c*x) + d").constants, ("a", "b", "c", "d"))
        self.assertEqual(read_model("Vmax*x/(Km + x)").constants, ("Vmax", "Km"))
        self.assertEqual(read_model("L/(1 + exp(-k*(x - x0)))").constants, ("L", "k", "x0"))
        self.assertEqual(read_model("a*x + pi*b + e").constants, ("a", "b"))

    def test_prefixes_are_dropped(self) -> None:
        for text in ("y = a*x + b", "y ~ a*x + b", "f(x) = a*x + b", "  a*x + b  "):
            with self.subTest(text=text):
                self.assertEqual(read_model(text).text, "a*x + b")

    def test_names_that_are_words_are_accepted(self) -> None:
        for text in ("ymax*(1 - exp(-k*x))", "a*x + xmax", "Lmax/(1 + exp(-k*x))", "beta*x + mu", "c1*x + c2"):
            with self.subTest(text=text):
                read_model(text)

    def test_refusals_carry_hints(self) -> None:
        cases = {
            "": "expression in x",
            "y = a*x = b": "without '='",
            "a*x + b*y": "write it in x only",
            "ax + b": "a*x",
            "a*sinx + b": "sin(x)",
            "a*xc": "x*c",
            "k*x + kx": "k*x",
            "A*exp(-gamma*x)": "'gamma' is a function",
            "a*x + ln": "'ln' is a function",
            "a*exp(-lambda*x)": "lambda is a reserved word",
            "a*t + b": "does not read x",
            "2*x + 1": "no constants",
            "a*x^2+b*x+c+d*x^3+f*x^4+g*x^5+h*x^6+k*x^7+m*x^8+p*x^9+q*x^10": "at most 10",
            "a(x + 1)": "write a*(x + 1), or use",
        }
        for text, hint in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as caught:
                    read_model(text)
                self.assertIn(hint, str(caught.exception))

    def test_with_values_writes_plain_numbers(self) -> None:
        model = read_model("a*exp(-b*x) + d")
        self.assertEqual(model.with_values({"a": 2.999999999, "b": -0.4, "d": 1e-7}, 6), "3*exp(-(-0.4)*x) + 0.0000001")
        self.assertEqual(read_model("2b*x").with_values({"b": 0.5}), "2(0.5)*x")


class TestLinearSplit(unittest.TestCase):
    def split(self, text: str, xs: Optional[List[float]] = None) -> List[str]:
        model = read_model(text)
        return [model.constants[k] for k in linear_constants(model, xs or _grid(0.0, 10.0, 20))]

    def test_affine_constants_are_linear(self) -> None:
        self.assertEqual(self.split("a*x + b"), ["a", "b"])
        self.assertEqual(self.split("a*exp(b*x)"), ["a"])
        self.assertEqual(self.split("a*exp(-b*x)*cos(c*x) + d"), ["a", "d"])

    def test_products_of_constants_are_not_jointly_linear(self) -> None:
        self.assertEqual(len(self.split("a*b*x")), 1)

    def test_curvature_far_out_is_seen(self) -> None:
        # atan(b3/(x - b4)) looks linear in b3 for small changes at large x
        self.assertEqual(self.split("b1 - b2*x - atan(b3/(x - b4))/pi", _grid(-4000.0, -1000.0, 25)), ["b1", "b2"])

    def test_an_underflowing_peak_is_not_called_linear(self) -> None:
        self.assertEqual(self.split("(b1/b2)*exp(-0.5*((x - b3)/b2)^2)", _grid(400.0, 500.0, 35)), [])


class TestTextbookFits(unittest.TestCase):
    def assertFits(self, fit: CustomFit, truth: Dict[str, float], tolerance: float) -> None:
        for name, value in truth.items():
            self.assertAlmostEqual(fit.values[name], value, delta=tolerance * max(1.0, abs(value)), msg=name)

    def test_common_models_are_found_without_starting_values(self) -> None:
        cases = [
            ("a*exp(-b*x)*cos(c*x) + d", _grid(0, 10, 40), lambda x: 3 * math.exp(-0.4 * x) * math.cos(2.5 * x) + 1),
            ("L/(1 + exp(-k*(x - x0)))", _grid(0, 20, 25), lambda x: 50 / (1 + math.exp(-0.6 * (x - 9)))),
            ("a*exp(-(x - m)^2/(2*s^2))", _grid(-10, 10, 60), lambda x: 5 * math.exp(-((x - 2.5) ** 2) / 2.88)),
            ("Vmax*x/(Km + x)", [0.02, 0.06, 0.11, 0.22, 0.56, 1.1], lambda x: 200 * x / (0.07 + x)),
            ("a*sin(b*x + c) + d", _grid(0, 20, 50), lambda x: 2 * math.sin(1.7 * x + 0.5) + 3),
            ("a*sin(b*x)", _grid(0, 10, 200), lambda x: math.sin(15 * x)),
            ("A*exp(-k*x)", _grid(2000, 2020, 21), lambda x: 1000 * math.exp(-0.05 * (x - 2000))),
        ]
        truths: List[Dict[str, float]] = [
            {"a": 3, "b": 0.4, "c": 2.5, "d": 1},
            {"L": 50, "k": 0.6, "x0": 9},
            {"a": 5, "m": 2.5, "s": 1.2},
            {"Vmax": 200, "Km": 0.07},
            {"a": 2, "b": 1.7, "d": 3},
            {"a": 1, "b": 15},
            {"k": 0.05},
        ]
        for (model, xs, function), truth in zip(cases, truths):
            with self.subTest(model=model):
                fit = _fit(model, xs, [function(x) for x in xs])
                self.assertGreater(fit.r_squared, 1 - 1e-9)
                self.assertTrue(fit.converged)
                self.assertFits(fit, truth, 1e-4)

    def test_noisy_data_gives_close_constants(self) -> None:
        xs = _grid(0, 10, 40)
        fit = _fit(
            "a*exp(-b*x)*cos(c*x) + d",
            xs,
            _noisy(lambda x: 3 * math.exp(-0.4 * x) * math.cos(2.5 * x) + 1, xs, 0.02, 1),
        )
        self.assertFits(fit, {"a": 3, "b": 0.4, "c": 2.5, "d": 1}, 0.02)
        self.assertAlmostEqual(fit.rmse, math.sqrt(fit.sse / len(xs)))
        self.assertEqual(len(fit.residuals), len(xs))

    def test_initial_values_reach_what_the_search_misses(self) -> None:
        xs = _grid(0, 10, 200)
        ys = _noisy(lambda x: math.sin(1.1 * x) + 0.5 * math.sin(3.3 * x) + 0.2 * math.sin(7 * x), xs, 0.05, 2)
        fit = _fit("a*sin(b*x) + c*sin(d*x) + g*sin(f*x)", xs, ys, {"b": 1, "d": 3, "f": 7})
        self.assertFits(fit, {"b": 1.1, "d": 3.3, "f": 7}, 0.01)

    def test_a_gaussian_width_is_reported_positive(self) -> None:
        xs = _grid(-5, 5, 30)
        fit = _fit("a*exp(-((x - m)/s)^2)", xs, [4 * math.exp(-(((x - 1) / 1.5) ** 2)) for x in xs])
        self.assertAlmostEqual(fit.values["s"], 1.5, places=6)

    def test_results_are_deterministic(self) -> None:
        xs = _grid(0, 10, 30)
        ys = _noisy(lambda x: 2 * math.sin(1.3 * x + 0.4) + 1, xs, 0.1, 3)
        first = _fit("a*sin(b*x + c) + d", xs, ys)
        second = _fit("a*sin(b*x + c) + d", xs, ys)
        self.assertEqual(first.values, second.values)
        self.assertEqual(first.evaluations, second.evaluations)

    def test_the_evaluation_limit_bounds_the_work(self) -> None:
        xs = _grid(0, 10, 1000)
        ys = _noisy(lambda x: 3 * math.exp(-((x - 3) ** 2) / 0.5) + 2 * math.exp(-((x - 6.5) ** 2)), xs, 0.05, 4)
        fit = _fit("a*exp(-(x - m)^2/s^2) + b*exp(-(x - n)^2/r^2)", xs, ys)
        self.assertLess(fit.evaluations, 1.3 * EVALUATION_LIMIT)
        self.assertGreater(fit.r_squared, 0.99)
        small = fit_custom_model(xs, ys, read_model("a*exp(-(x - m)^2/s^2) + b*exp(-(x - n)^2/r^2)"), None, 20_000)
        self.assertLess(small.evaluations, 60_000)


class TestNotes(unittest.TestCase):
    def test_an_undetermined_shift_is_held(self) -> None:
        xs = _grid(0, 10, 15)
        fit = _fit("A*exp(k*(x - x0))", xs, [0.6 * math.exp(0.3 * x) for x in xs])
        self.assertEqual(fit.held, {"x0": 0.0})
        self.assertAlmostEqual(fit.values["A"], 0.6, places=6)
        self.assertTrue(any("cannot determine x0" in note for note in fit.notes))

    def test_a_redundant_linear_constant_is_named(self) -> None:
        xs = _grid(-3, 3, 13)
        fit = _fit("a*x + b*x", xs, [2 * x for x in xs])
        self.assertEqual(fit.undetermined, ["b"])
        self.assertTrue(any("cannot determine b" in note for note in fit.notes))

    def test_a_product_of_constants_is_held(self) -> None:
        xs = _grid(0, 5, 10)
        fit = _fit("a*b*x", xs, [2 * x for x in xs])
        self.assertAlmostEqual(fit.values["a"] * fit.values["b"], 2.0, places=9)
        self.assertTrue(fit.held or fit.undetermined)

    def test_aliasing_prefers_the_slow_sine_and_names_the_other(self) -> None:
        fit = _fit("a*sin(b*x + c) + d", [0, 1, 2, 3, 4], [0.1, 0.9, 0.05, -1.1, 0.0])
        self.assertLess(abs(fit.values["b"]), math.pi)
        self.assertFalse(fit.swings)
        self.assertIsNotNone(fit.alternative)

    def test_as_many_points_as_constants(self) -> None:
        fit = _fit("a*exp(b*x) + c", [0, 1, 2], [1, 3, 7])
        self.assertTrue(fit.exact)
        self.assertTrue(any("passes through every point" in note for note in fit.notes))

    def test_x_far_from_zero_suggests_a_shift(self) -> None:
        xs = _grid(2000, 2020, 21)
        fit = _fit("a*exp(b*x)", xs, [5 * math.exp(0.6 * (x - 2000)) for x in xs])
        self.assertTrue(any("a*exp(b*(x - 2000))" in note for note in fit.notes), fit.notes)

    def test_a_work_limit_note(self) -> None:
        xs = _grid(0, 10, 300)
        ys = [3 * math.exp(-0.4 * x) * math.cos(2.5 * x) + 1 for x in xs]
        fit = fit_custom_model(xs, ys, read_model("a*exp(-b*x)*cos(c*x) + d"), None, 3_000)
        self.assertFalse(fit.converged)
        self.assertTrue(any("work limit" in note for note in fit.notes))


class TestDataRefusals(unittest.TestCase):
    def test_bad_data_is_refused(self) -> None:
        model = read_model("a*x + b")
        with self.assertRaises(ValueError):
            fit_custom_model([1, 2], [1], model)
        with self.assertRaises(ValueError):
            fit_custom_model([1], [1], model)
        with self.assertRaises(ValueError):
            fit_custom_model([1, 2, float("nan")], [1, 2, 3], model)
        with self.assertRaises(ValueError):
            fit_custom_model(list(range(1001)), list(range(1001)), model)

    def test_a_constant_power_of_negative_x_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _fit("a*x^n", _grid(-3, 3, 13), [2 * x * x for x in _grid(-3, 3, 13)])
        self.assertIn("whole n", str(caught.exception))

    def test_a_model_undefined_at_a_data_point_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _fit("a/x + b", [0, 1, 2, 3], [1, 2, 3, 4])
        self.assertIn("could not be evaluated", str(caught.exception))


class TestNist(unittest.TestCase):
    problems: Dict[str, Dict[str, object]] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.problems = json.loads(NIST_PATH.read_text())["problems"]

    def check(self, name: str, starts: bool) -> None:
        problem = self.problems[name]
        start = problem["start"] if starts else None
        fit = fit_custom_model(problem["x"], problem["y"], read_model(problem["model"]), start)  # type: ignore[arg-type]
        certified_sse = float(problem["certified_sse"])  # type: ignore[arg-type]
        # Lanczos1's certified sum (1.4e-25) is rounding: there an absolute margin applies
        self.assertLess(abs(fit.sse - certified_sse), 1e-6 * certified_sse + 1e-24, f"{name}: SSE {fit.sse}")

    def test_problems_solved_without_starting_values(self) -> None:
        for name in NIST_SOLVED_WITHOUT_STARTS:
            with self.subTest(name=name):
                self.check(name, starts=False)

    def test_every_problem_from_nists_first_starting_values(self) -> None:
        self.assertEqual(len(self.problems), 26)
        for name in self.problems:
            with self.subTest(name=name):
                self.check(name, starts=True)


if __name__ == "__main__":
    unittest.main()
