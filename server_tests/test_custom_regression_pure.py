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
        self.assertEqual(read_model("a*x + pi*b + e^c").constants, ("a", "b", "c"))

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
            "a sin(b x) + c": "'a sin' has no operator between its parts; write products with '*' (a*sin)",
            "a*exp(-b x)": "(b*x)",
            "a*x^4 + b*x^3 + c*x^2 + d*x + e": "e is Euler's number",
            "a*exp(-b*x) + c*exp(-d*x) + e": "such as e1 or k",
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

    def test_constants_hidden_at_small_values_are_found_far_from_0(self) -> None:
        # At x = 400..500 a peak or exp(b*x) underflows or overflows unless the constants are scaled to x
        self.assertEqual(self.split("(b1/b2)*exp(-0.5*((x - b3)/b2)^2)", _grid(400.0, 500.0, 35)), ["b1"])
        years = _grid(1960.0, 2020.0, 13)
        self.assertEqual(self.split("a*exp(b*x) + c", years), ["a", "c"])
        self.assertEqual(self.split("a*exp(-b*x) + c", years), ["a", "c"])
        self.assertEqual(self.split("c + a*exp(-x/tau)", years), ["c", "a"])
        self.assertEqual(self.split("a*b^x + c", years), ["a", "c"])


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

    def test_sines_in_shift_form_report_positive_amplitude_and_period_and_a_shift_within_it(self) -> None:
        months = [float(m) for m in range(1, 13)]
        temperatures = [-2.0, 0.0, 5.0, 11.0, 17.0, 21.0, 24.0, 23.0, 18.0, 11.0, 5.0, 0.0]
        for model, rate, shift in (
            ("A*cos(B*(x - C)) + D", "B", "C"),
            ("A*sin(B*(x - C)) + D", "B", "C"),
            ("A*cos(2*pi*(x - C)/T) + D", "T", "C"),
        ):
            with self.subTest(model=model):
                values = _fit(model, months, temperatures).values
                self.assertGreater(values["A"], 0.0)
                self.assertGreater(values[rate], 0.0)
                period = 2 * math.pi / values["B"] if rate == "B" else values["T"]
                self.assertGreaterEqual(values[shift], 0.0)
                self.assertLess(values[shift], period)
        xs = _grid(0, 20, 50)
        values = _fit("a*cos(b*x + c) + d", xs, [2.7 * math.cos(1.4 * x - 0.5) + 9 for x in xs]).values
        self.assertAlmostEqual(values["b"], 1.4, places=6)
        self.assertAlmostEqual(values["c"], -0.5, places=6)

    def test_damped_oscillations_are_found_on_well_sampled_data(self) -> None:
        xs = _grid(0, 30, 100)
        fit = _fit("A*exp(-k*x)*cos(w*x + p)", xs, [3 * math.exp(-0.05 * x) * math.cos(x + 2.8) for x in xs])
        self.assertGreater(fit.r_squared, 1 - 1e-9)
        self.assertEqual((fit.held, fit.undetermined, fit.alternative), ({}, [], None))
        xs = _grid(0, 20, 150)
        fit = _fit("a*exp(-b*x)*cos(c*x) + d", xs, [9.4 * math.exp(-0.0169 * x) * math.cos(3.82 * x) + 1 for x in xs])
        self.assertAlmostEqual(fit.values["c"], 3.82, places=6)

    def test_sines_over_many_cycles_are_found(self) -> None:
        months = [float(m) for m in range(1, 145)]
        monthly = [10 + 4 * math.sin(2 * math.pi * (x - 3) / 12) for x in months]
        for model, rate, truth in (
            ("A*sin(2*pi*(x - C)/T) + D", "T", 12.0),
            ("A*cos(2*pi*(x - h)/P) + k", "P", 12.0),
            ("A*sin(2*pi*f*x + p) + D", "f", 1 / 12),
            ("a*sin(b*x + c) + d", "b", math.pi / 6),
        ):
            with self.subTest(model=model):
                fit = _fit(model, months, monthly)
                self.assertGreater(fit.r_squared, 1 - 1e-9)
                self.assertAlmostEqual(fit.values[rate], truth, places=6)
        seconds = [i / 1000 for i in range(1000)]
        mains = _noisy(lambda x: 2 * math.sin(2 * math.pi * 50 * x + 0.7) + 0.5, seconds, 0.05, 1)
        self.assertAlmostEqual(_fit("A*sin(w*x + p) + c", seconds, mains).values["w"], 100 * math.pi, delta=0.1)
        hours = [float(h) for h in range(984)]
        daily = _noisy(lambda x: 15 + 5 * math.sin(2 * math.pi * (x - 8) / 24), hours, 0.5, 2)
        self.assertAlmostEqual(_fit("A*sin(2*pi*(x - C)/T) + D", hours, daily).values["T"], 24.0, delta=0.01)

    def test_noisy_sines_beyond_80_half_cycles_are_found(self) -> None:
        days = [i / 24 for i in range(1000)]
        hourly = _noisy(lambda x: 4 * math.sin(2 * math.pi * (x - 0.3)) + 15, days, 2.0, 5)
        for model, rate, truth in (("a*sin(b*x + c) + d", "b", 2 * math.pi), ("a*sin(2*pi*f*x + p) + d", "f", 1.0)):
            with self.subTest(model=model):
                self.assertAlmostEqual(_fit(model, days, hourly).values[rate], truth, delta=0.01 * truth)
        samples = [float(i) for i in range(1000)]
        near_limit = _noisy(lambda x: math.sin(2.9 * x), samples, 0.3, 3)
        fit = _fit("a*sin(b*x + c) + d", samples, near_limit)
        self.assertAlmostEqual(fit.values["b"], 2.9, places=4)
        self.assertLess(fit.evaluations, 1.8 * EVALUATION_LIMIT)

    def test_a_slow_sine_in_noisy_hourly_data_is_not_aliased(self) -> None:
        hours = [float(h) for h in range(480)]
        for seed in (3, 4):
            with self.subTest(seed=seed):
                temperatures = _noisy(lambda x: 15 + 6 * math.sin(2 * math.pi * (x - 5.7) / 24), hours, 3.0, seed)
                self.assertAlmostEqual(
                    _fit("A*sin(2*pi*(x - C)/T) + D", hours, temperatures).values["T"], 24, delta=0.2
                )

    def test_a_period_is_found_when_aliases_screen_better(self) -> None:
        # Two weeks of hourly data from 0: on the screened points an alias fits better, and the
        # curve that does not swing is refined with its own budget
        days = [i / 24 for i in range(336)]
        temperatures = _noisy(lambda x: 18 + 9.4 * math.sin(2 * math.pi * (x - 0.25)), days, 2.8, 10)
        fit = _fit("A*sin(2*pi*(x-C)/T)+D", days, temperatures)
        self.assertAlmostEqual(fit.values["T"], 1.0, delta=0.005)
        self.assertGreater(fit.r_squared, 0.85)

    def test_sines_over_x_far_from_zero_are_fitted_in_a_shift(self) -> None:
        # Decimal years and spreadsheet day numbers: frequency and phase are nearly collinear
        # in x, so the fit runs in x - shift and comes back in the model's own constants
        years = [2021 + i / 365.25 for i in range(1000)]
        daily = _noisy(lambda x: 12 + 9 * math.sin(2 * math.pi * (x - 2021.3)), years, 3.0, 7)
        days = [45000 + i / 24 for i in range(720)]
        hourly = _noisy(lambda x: 15 + 6 * math.sin(2 * math.pi * (x - 0.4)), days, 2.0, 8)
        for xs, ys, shift, model in (
            (years, daily, 2021, "a*sin(b*x+c)+d"),
            (years, daily, 2021, "A*sin(2*pi*(x-C)/T)+D"),
            (years, daily, 2021, "a*sin(b*x+c)+d+k*x"),
            (days, hourly, 45000, "a*sin(b*x+c)+d"),
            (days, hourly, 45000, "a*cos(b*(x-c))+d"),
        ):
            with self.subTest(model=model, shift=shift):
                fit = _fit(model, xs, ys)
                reference = _fit(model, [x - shift for x in xs], ys)
                self.assertLessEqual(fit.sse, reference.sse * (1 + 1e-6))
                self.assertTrue(fit.converged)
                self.assertFalse(any("work limit" in note or "far from 0" in note for note in fit.notes), fit.notes)
                self.assertLess(fit.evaluations, 450_000)
                curve = read_model(model).predictions(xs, [fit.values[name] for name in read_model(model).constants])
                self.assertAlmostEqual(curve[0], reference.predictions[0], delta=1e-3)

    def test_phases_written_back_from_a_shift_stay_plain(self) -> None:
        months = [2015 + i / 12 for i in range(120)]
        seasonal = _noisy(
            lambda x: 5 + 6 * math.sin(2 * math.pi * x + 1) + 2.4 * math.sin(4 * math.pi * x + 2), months, 0.6, 1
        )
        model = "a*sin(b*x+c)+d*sin(2*b*x+g)+f"
        fit = _fit(model, months, seasonal)
        reference = _fit(model, [x - 2015 for x in months], seasonal)
        self.assertLessEqual(fit.sse, reference.sse * (1 + 1e-6))
        self.assertLess(abs(fit.values["c"]), 2 * math.pi)
        self.assertLess(abs(fit.values["g"]), 2 * math.pi)

    def test_a_damped_cosine_over_years_is_fitted_in_a_shift(self) -> None:
        years = _grid(2000, 2020, 1000)
        ys = _noisy(lambda x: 5 * math.exp(-0.1 * (x - 2000)) * math.cos(math.pi * (x - 2000) + 0.3) + 1, years, 0.2, 3)
        fit = _fit("a*exp(-b*x)*cos(c*x+d)+g", years, ys)
        self.assertGreater(fit.r_squared, 0.98)
        self.assertAlmostEqual(fit.values["b"], 0.1, delta=0.005)
        self.assertLess(fit.evaluations, 450_000)
        # a is near 1e89 here, so the note gives the plain values in x - 2000
        self.assertTrue(any("written in x - 2000" in note and "a = 5.0" in note for note in fit.notes), fit.notes)

    def test_models_already_written_around_the_data_are_fitted_as_written(self) -> None:
        xs = [2003 + i / 100 for i in range(100)]
        ys = [19.21 + 4.109 * math.exp(-0.5839 * (x - 2003)) * math.cos(27.8 * (x - 2003) + 1.66) for x in xs]
        fit = _fit("a*exp(-b*(x-2003))*cos(c*(x-2003)+d)+g", xs, ys)
        self.assertGreater(fit.r_squared, 0.999999)
        self.assertEqual(fit.notes, [])
        months = [2000 + i / 12 for i in range(120)]
        seasonal = _noisy(lambda x: 20 * math.exp(0.1 * (x - 2000)) + 3 * math.sin(2 * math.pi * x + 1), months, 0.3, 5)
        fit = _fit("a*exp(k*(x-2000))+c*sin(b*(x-2000)+d)", months, seasonal)
        self.assertAlmostEqual(fit.values["b"], 2 * math.pi, delta=0.01)
        self.assertAlmostEqual(abs(fit.values["c"]), 3, delta=0.2)

    def test_a_model_whose_constants_overflow_far_from_zero_is_refused_with_its_shifted_fit(self) -> None:
        years = [2018 + i / 200 for i in range(1000)]
        ys = _noisy(
            lambda x: 17.7 + 7.6 * math.exp(-0.47 * (x - 2018)) * math.cos(10.3 * (x - 2018) + 1.8), years, 1.0, 4
        )
        with self.assertRaises(ValueError) as caught:
            _fit("a*exp(-b*x)*cos(c*x+d)+g", years, ys)
        message = str(caught.exception)
        self.assertIn("pass the float range", message)
        self.assertIn("as a*exp(-b*(x - 2018))*cos(c*(x - 2018)+d)+g, it fits with R² = ", message)
        self.assertIn("b = 0.4", message)

    def test_a_seasonal_cycle_on_a_trend_is_found_from_the_spectrum(self) -> None:
        # A trend hides the cycle from the mean crossings (20 to 30 cycles, beyond the 40 half
        # cycles always tried); the spectrum of the detrended data shows it
        months = [(i + 0.5) / 12 for i in range(360)]
        co2 = _noisy(lambda x: 300 + 1.5 * x + 3 * math.sin(2 * math.pi * x + 1), months, 0.3, 2)
        fit = _fit("m*x + c*sin(w*x + d) + k", months, co2)
        self.assertAlmostEqual(fit.values["w"], 2 * math.pi, delta=0.01)
        self.assertAlmostEqual(fit.values["c"], 3, delta=0.1)
        # Weekly data in years: no float-range refusal, and the exponential trend is found
        weeks = [2000 + (i + 0.5) / 52 for i in range(1000)]
        sales = _noisy(
            lambda x: 20 + 30 * math.exp(0.03 * (x - 2000)) + 5 * math.sin(2 * math.pi * x + 1), weeks, 0.5, 4
        )
        fit = _fit("a*exp(r*x) + c*sin(w*x + d) + k", weeks, sales)
        self.assertGreater(fit.r_squared, 0.995)
        self.assertAlmostEqual(fit.values["r"], 0.03, delta=0.002)
        self.assertAlmostEqual(fit.values["w"], 2 * math.pi, delta=0.01)

    def test_a_seasonal_cycle_is_found_past_the_leak_of_a_curved_trend(self) -> None:
        # Over 60 years the strongest spectral peak is the curvature's leak (2 half cycles)
        months = [1960 + (i + 0.5) / 12 for i in range(720)]
        co2 = _noisy(
            lambda x: 315 + 0.8 * (x - 1960) + 0.0125 * (x - 1960) ** 2 + 3 * math.sin(2 * math.pi * x + 1),
            months,
            0.3,
            1,
        )
        fit = _fit("a*exp(r*x) + c*sin(w*x + p) + k", months, co2)
        self.assertAlmostEqual(fit.values["w"], 2 * math.pi, delta=0.001)
        self.assertAlmostEqual(fit.values["c"], 3, delta=0.1)

    def test_two_sines_are_not_both_started_at_the_strongest_peak(self) -> None:
        xs = _grid(0, 20, 101)
        for curve in (
            lambda x: 3 * math.sin(5 * x + 1) + 2 * math.sin(2 * x) + 4,
            lambda x: 2 * x + 3 * math.sin(5 * x + 1) + 2 * math.sin(2 * x) + 4,
        ):
            ys = [curve(x) for x in xs]
            fit = _fit("m*x + a*sin(b*x + c) + h*sin(f*x + g) + d", xs, ys)
            self.assertGreater(fit.r_squared, 0.999999)
        harmonics = [math.sin(x) + 2 * math.sin(2 * x + 1) + 3 * math.sin(3 * x + 2) for x in xs]
        fit = _fit("a*sin(w*x + p) + b*sin(2*w*x + q) + c*sin(3*w*x + r) + d", xs, harmonics)
        self.assertGreater(fit.r_squared, 0.999999)

    def test_a_damped_cosine_with_a_time_constant_decays(self) -> None:
        # exp(-x/t) cannot pass from growth to decay but through t = infinity, so the decay is
        # screened with the cycle's frequency taken from the spectrum and its phase found
        for low, high, count, curve, decay in (
            (0, 50, 200, lambda x: -15.502 + 7.471 * math.exp(-x / 45.903) * math.cos(0.47586 * x - 2.168), 45.903),
            (0, 200, 100, lambda x: 16.353 + 7.215 * math.exp(-x / 218.36) * math.cos(0.27392 * x + 2.3449), 218.36),
        ):
            with self.subTest(decay=decay):
                xs = _grid(low, high, count)
                fit = _fit("a*exp(-x/t)*cos(w*x + p) + k", xs, [curve(x) for x in xs])
                self.assertGreater(fit.r_squared, 0.999999)
                self.assertAlmostEqual(fit.values["t"], decay, delta=0.01 * decay)

    def test_damped_oscillations_with_a_phase_are_found(self) -> None:
        xs = _grid(0, 5, 80)
        for rate in (10.0, 11.0):
            with self.subTest(rate=rate):
                ys = [3 * math.exp(-0.3 * x) * math.sin(rate * x - 1) + 1 for x in xs]
                fit = _fit("a*exp(-b*x)*cos(c*x + p) + d", xs, ys)
                self.assertGreater(fit.r_squared, 1 - 1e-9)
                self.assertAlmostEqual(fit.values["c"], rate, places=6)

    def test_sines_report_a_positive_amplitude_and_frequency_and_a_phase_within_pi(self) -> None:
        months = [float(m) for m in range(1, 13)]
        temperatures = [-5.0, -3.0, 2.0, 9.0, 15.0, 20.0, 23.0, 21.0, 16.0, 9.0, 3.0, -2.0]
        xs = _grid(0, 20, 21)
        cases = [
            ("a*sin(b*x + c) + d", months, temperatures),
            ("a*sin(b*x + c) + d", xs, [2 * math.sin(0.5 * x) + 1 for x in xs]),
            ("a*exp(-b*x)*cos(c*x + d)", xs, [3 * math.exp(-0.3 * x) * math.cos(2 * x + 0.2) for x in xs]),
        ]
        for model, x_data, y_data in cases:
            with self.subTest(model=model, points=len(x_data)):
                values = _fit(model, x_data, y_data).values
                amplitude, frequency = values["a"], values["b" if "sin" in model else "c"]
                phase = values["c" if "sin" in model else "d"]
                self.assertGreater(amplitude, 0.0)
                self.assertGreater(frequency, 0.0)
                self.assertLessEqual(abs(phase), math.pi + 1e-9)
        values = _fit(*cases[2]).values
        for name, truth in {"a": 3.0, "b": 0.3, "c": 2.0, "d": 0.2}.items():
            self.assertAlmostEqual(values[name], truth, places=6)

    def test_an_exponential_with_an_offset_over_years(self) -> None:
        years = _grid(1960, 2020, 13)
        ys = _noisy(lambda x: 280 + 37 * math.exp(0.0175 * (x - 1960)), years, 0.3, 4)
        for model, rate in (("a*exp(b*x) + c", "b"), ("c + a*exp(b*x)", "b"), ("a*b^x + c", "b")):
            with self.subTest(model=model):
                fit = _fit(model, years, ys)
                self.assertGreater(fit.r_squared, 0.9999)
                self.assertAlmostEqual(fit.values["c"], 280.0, delta=1.0)
                growth = fit.values[rate] if "exp" in model else math.log(fit.values[rate])
                self.assertAlmostEqual(growth, 0.0175, delta=2e-4)
        decay = _grid(1990, 2020, 16)
        fit = _fit("c + a*exp(-x/tau)", decay, _noisy(lambda x: 20 + 60 * math.exp(-0.1 * (x - 1990)), decay, 0.3, 5))
        self.assertAlmostEqual(fit.values["tau"], 10.0, delta=0.3)

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
        self.assertLess(small.evaluations, 80_000)


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

    def test_aliasing_prefers_the_slow_sine(self) -> None:
        fit = _fit("a*sin(b*x + c) + d", [0, 1, 2, 3, 4], [0.1, 0.9, 0.05, -1.1, 0.0])
        self.assertLess(abs(fit.values["b"]), math.pi)
        self.assertFalse(fit.swings)
        for step in (0.1, 0.25, 0.5, 1.0):
            xs = [step * i for i in range(int(10 / step) + 1)]
            with self.subTest(step=step):
                fit = _fit("a*sin(b*x + c) + d", xs, [2 * math.sin(0.5 * x + 0.3) + 1 for x in xs])
                self.assertAlmostEqual(fit.values["b"], 0.5, places=6)
                self.assertFalse(fit.swings)

    def test_as_many_points_as_constants(self) -> None:
        fit = _fit("a*exp(b*x) + c", [0, 1, 2], [1, 3, 7])
        self.assertTrue(fit.exact)
        self.assertTrue(any("passes through every point" in note for note in fit.notes))

    def test_x_far_from_zero_suggests_a_shift(self) -> None:
        xs = _grid(2000, 2020, 21)
        fit = _fit("a*exp(b*x)", xs, [5 * math.exp(0.6 * (x - 2000)) for x in xs])
        self.assertTrue(any("a*exp(b*(x - 2000))" in note for note in fit.notes), fit.notes)

    def test_x_far_from_zero_refits_a_model_the_shift_changes(self) -> None:
        years = [1900.0 + 10 * k for k in range(13)]
        population = [76.2, 92.2, 106.0, 123.2, 132.2, 151.3, 179.3, 203.2, 226.5, 248.7, 281.4, 308.7, 331.4]
        fit = _fit("K/(1+A*exp(-k*x))", years, population)
        notes = " ".join(fit.notes)
        self.assertIn("written in x - 1900, as K/(1+A*exp(-k*(x - 1900))), the model fits with R² = 0.9986", notes)
        self.assertIn("A = 7.48", notes)

    def test_x_far_from_zero_gives_the_shifted_fit_of_a_fast_rate(self) -> None:
        years = _grid(1990, 1995, 15)
        decay = _noisy(lambda x: 500 * math.exp(-0.5 * (x - 1990)) + 20, years, 1.0, 6)
        notes = " ".join(_fit("a*exp(-k*x) + c", years, decay).notes)
        self.assertIn("written in x - 1990, as a*exp(-k*(x - 1990)) + c, it fits with R² = 0.99", notes)
        self.assertIn("k = 0.49", notes)
        moore = _grid(1971, 2019, 25)
        fit = _fit("a*exp(b*x)", moore, _noisy(lambda x: 2300 * 2 ** ((x - 1971) / 2), moore, 0.0, 7))
        self.assertAlmostEqual(fit.values["b"], math.log(2) / 2, places=6)

    def test_x_far_from_zero_refits_textbook_forms(self) -> None:
        years = _grid(2000, 2020, 21)
        logistic = [round(500 / (1 + math.exp(-0.4 * (x - 2010))), 1) for x in years]
        fit = _fit("K/(1+((K-P0)/P0)*exp(-r*x))", years, logistic)
        notes = " ".join(fit.notes)
        self.assertIn("written in x - 2000, as K/(1+((K-P0)/P0)*exp(-r*(x - 2000))), the model fits with R² = 1", notes)
        self.assertNotIn("cannot determine", notes)
        rise = [50 + 300 * (1 - math.exp(-0.3 * (x - 2000))) for x in years]
        notes = " ".join(_fit("a*(1-exp(-k*x))+c", years, rise).notes)
        self.assertIn("k = 0.3", notes)

    def test_no_shift_is_suggested_when_it_would_change_the_model(self) -> None:
        years = _grid(2000, 2020, 21)
        wavy = [math.sin(x) + 0.01 * x for x in years]
        for model in ("a*ln(x) + b", "a/x + b", "a*x^b"):
            with self.subTest(model=model):
                fit = _fit(model, years, wavy)
                self.assertFalse(any("far from 0" in note for note in fit.notes), fit.notes)

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
