"""Pure Python tests for numeric roots, extrema and intersections (utils.function_features).

The Brython suite (static/client/client_tests/test_function_features.py) covers the same
module in the browser and the find_function_features tool on the canvas.
"""

from __future__ import annotations

import math
import random
import unittest

from utils.function_features import find_function_features, find_intersections


def _summary(report: dict) -> list:
    return [(f["x"], f["y"], f["kind"]) for f in report["features"]]


class TestFunctionFeaturesPure(unittest.TestCase):
    def test_polynomial_roots_and_extrema(self) -> None:
        report = find_function_features(lambda x: x**3 - 3 * x, -3, 3)
        self.assertEqual(
            _summary(report),
            [
                (-1.732050808, 0.0, "root"),
                (-1.0, 2.0, "local_max"),
                (0.0, 0.0, "root"),
                (1.0, -2.0, "local_min"),
                (1.732050808, 0.0, "root"),
            ],
        )

    def test_double_root_is_touching_root_and_minimum(self) -> None:
        report = find_function_features(lambda x: (x - 1) ** 2, -5, 4.3)
        self.assertEqual(_summary(report), [(1.0, 0.0, "root"), (1.0, 0.0, "local_min")])
        self.assertTrue(report["features"][0]["touching"])

    def test_poles_give_no_false_roots(self) -> None:
        self.assertEqual(find_function_features(lambda x: 1 / x, -5, 5)["features"], [])
        self.assertEqual(find_function_features(lambda x: 1 / x**2, -5, 5)["features"], [])
        tangent = find_function_features(math.tan, -5, 5)
        self.assertEqual([f["x"] for f in tangent["features"]], [-3.141592654, 0.0, 3.141592654])

    def test_breakpoints_split_the_interval(self) -> None:
        report = find_function_features(lambda x: 1 / (x - 1) + 1, -5, 5, breakpoints=[1.0])
        self.assertEqual(_summary(report), [(0.0, 0.0, "root")])

    def test_empty_result(self) -> None:
        report = find_function_features(lambda x: x * x + 1, -5, 5, features=["roots"])
        self.assertEqual(report["features"], [])
        self.assertFalse(report["truncated"])

    def test_truncation(self) -> None:
        report = find_function_features(math.sin, 0.5, 100.5, features=["roots"], max_results=5)
        self.assertEqual(len(report["features"]), 5)
        self.assertEqual(report["total_found"], 31)
        self.assertTrue(report["truncated"])

    def test_intersections(self) -> None:
        report = find_intersections(lambda x: x * x, lambda x: x + 2, -5, 5)
        self.assertEqual(_summary(report), [(-1.0, 1.0, "intersection"), (2.0, 4.0, "intersection")])

    def test_coinciding_functions_report_a_zero_interval(self) -> None:
        report = find_intersections(lambda x: 2 * x, lambda x: x + x, -1, 1)
        self.assertEqual(len(report["features"]), 1)
        self.assertEqual(report["features"][0]["zero_interval"], [-1.0, 1.0])

    # ---- roots at the interval ends that rounding moved off zero ----

    def test_endpoint_roots_rounded_off_zero_are_found(self) -> None:
        # sin(2*pi) = -2.4e-16, cos(pi/2) = 6.1e-17, sqrt(2)^2 - 2 = 4.4e-16: no sign change brackets them
        sine = find_function_features(math.sin, 0, 2 * math.pi, features=["roots"])
        self.assertEqual([f["x"] for f in sine["features"]], [0.0, 3.141592654, 6.283185307])
        cosine = find_function_features(math.cos, 0, math.pi / 2, features=["roots"])
        self.assertEqual([f["x"] for f in cosine["features"]], [1.570796327])
        square = find_function_features(lambda x: x * x - 2, -5, math.sqrt(2), features=["roots"])
        self.assertEqual([f["x"] for f in square["features"]], [-1.414213562, 1.414213562])

    def test_endpoint_root_is_not_duplicated_by_a_nearby_sign_change(self) -> None:
        # cos just past pi/2 is -9e-16: the last bracket changes sign and the end is near zero too
        report = find_function_features(math.cos, 0, math.pi / 2 + 1e-15, features=["roots"])
        self.assertEqual(len(report["features"]), 1)
        self.assertAlmostEqual(report["features"][0]["x"], math.pi / 2, places=9)

    def test_tiny_endpoint_value_that_is_not_a_root_is_ignored(self) -> None:
        # exp(-40) is 4e-18, far below 1e-12 of exp(0), but exp does not vanish there
        self.assertEqual(find_function_features(math.exp, -40, 0, features=["roots"])["features"], [])
        self.assertEqual(find_function_features(lambda x: x * x + 1, -5, 5, features=["roots"])["features"], [])

    def test_endpoint_intersection_rounded_off_zero(self) -> None:
        report = find_intersections(math.sin, math.cos, 0, math.pi / 4)
        self.assertEqual([(f["x"], f["y"]) for f in report["features"]], [(0.7853981634, 0.7071067812)])

    # ---- steep roots versus poles (no sample lands on 0 in [-1.37, 1.3]) ----

    def test_roots_with_a_vertical_or_near_vertical_tangent_are_kept(self) -> None:
        def cbrt(x: float) -> float:
            return math.copysign(abs(x) ** (1.0 / 3.0), x)

        self.assertEqual(_summary(find_function_features(cbrt, -1.37, 1.3)), [(0.0, 0.0, "root")])
        steep = find_function_features(lambda x: math.tanh(1e5 * x), -1.37, 1.3)
        self.assertEqual(_summary(steep), [(0.0, 0.0, "root")])

    def test_poles_and_jumps_still_give_no_roots(self) -> None:
        cases = {
            "1/x": lambda x: 1 / x,
            "-1/x^3": lambda x: -1 / x**3,
            "1/(x-1)^2": lambda x: 1 / (x - 1) ** 2,
            "jump -1 to 1": lambda x: -1.0 if x < 0.37 else 1.0,
            "jump -1e-4 to 1": lambda x: -1e-4 if x < 0.37 else 1.0,
            "jump through 0 at the step": lambda x: (x > 0.37) - (x < 0.37),
        }
        for name, f in cases.items():
            with self.subTest(name):
                self.assertEqual(find_function_features(f, -1.37, 1.3, features=["roots"])["features"], [])
        tangent = find_function_features(math.tan, -1.37 - math.pi, 1.3 + math.pi, features=["roots"])
        self.assertEqual([f["x"] for f in tangent["features"]], [-3.141592654, 0.0, 3.141592654])

    def test_invalid_interval(self) -> None:
        with self.assertRaises(ValueError):
            find_function_features(math.sin, 1, 1)


def _inflections(f, left: float, right: float) -> list:
    report = find_function_features(f, left, right, features=["inflections"])
    return [(feature["x"], feature["kind"]) for feature in report["features"]]


class TestInflectionPoints(unittest.TestCase):
    def test_default_features_do_not_include_inflections(self) -> None:
        kinds = {f["kind"] for f in find_function_features(lambda x: x**3 - 3 * x, -3, 3)["features"]}
        self.assertNotIn("inflection", kinds)

    def test_polynomial_inflections(self) -> None:
        self.assertEqual(_inflections(lambda x: x**3 - 3 * x, -3, 3), [(0.0, "inflection")])
        report = find_function_features(lambda x: (x - 1) ** 3 + 2, -2, 4, features=["inflections"])
        self.assertEqual(_summary(report), [(1.0, 2.0, "inflection")])
        self.assertEqual(_inflections(lambda x: x**5, -1.3, 1.1), [(0.0, "inflection")])

    def test_smooth_functions_are_located_accurately(self) -> None:
        sine = [x for x, _ in _inflections(math.sin, -7, 7)]
        for found, expected in zip(sine, [-2 * math.pi, -math.pi, 0.0, math.pi, 2 * math.pi]):
            self.assertAlmostEqual(found, expected, places=6)
        self.assertEqual(len(sine), 5)
        gauss = [x for x, _ in _inflections(lambda x: math.exp(-x * x), -3, 3)]
        self.assertEqual(len(gauss), 2)
        self.assertAlmostEqual(gauss[1], 1 / math.sqrt(2), places=5)
        self.assertAlmostEqual(gauss[0], -1 / math.sqrt(2), places=5)
        self.assertEqual(_inflections(lambda x: 1 / (1 + math.exp(-x)), -8, 8), [(0.0, "inflection")])

    def test_no_inflection_without_a_concavity_change(self) -> None:
        for f in (
            lambda x: x**4,
            lambda x: x**2,
            lambda x: 2 * x + 1,
            lambda x: 1e6 * x + 3,
            lambda x: 7.0,
            math.exp,
            abs,
        ):
            self.assertEqual(_inflections(f, -2, 2.1), [])

    def test_poles_and_jumps_are_not_inflections(self) -> None:
        self.assertEqual(_inflections(lambda x: 1 / x, -1.37, 1.3), [])
        self.assertEqual(_inflections(lambda x: 1 / x**2, -1.37, 1.3), [])
        self.assertEqual(_inflections(lambda x: 1.0 if x >= 0.3 else -1.0, -2, 2), [])
        # tan changes concavity at its roots, not at its poles
        tangent = [x for x, _ in _inflections(math.tan, -5, 5)]
        self.assertEqual(len(tangent), 3)
        for found, expected in zip(tangent, [-math.pi, 0.0, math.pi]):
            self.assertAlmostEqual(found, expected, places=5)

    def test_steep_and_non_smooth_inflections(self) -> None:
        cbrt = lambda x: math.copysign(abs(x) ** (1 / 3), x)  # noqa: E731
        self.assertEqual(_inflections(cbrt, -1.37, 1.3), [(0.0, "inflection")])
        self.assertEqual(_inflections(lambda x: x * abs(x), -2, 2.1), [(0.0, "inflection")])

    def test_scale_does_not_matter(self) -> None:
        self.assertEqual(_inflections(lambda x: 1e-9 * (x**3 - x), -2, 2), [(0.0, "inflection")])
        self.assertEqual(_inflections(lambda x: 1e9 * (x**3 - x), -2, 2), [(0.0, "inflection")])
        self.assertEqual(_inflections(lambda x: x**3 - 1e6 * x, -2, 2.3), [(0.0, "inflection")])
        self.assertEqual(_inflections(lambda x: (x - 1000) ** 3, 990, 1010), [(1000.0, "inflection")])

    def test_large_constant_offsets_are_found(self) -> None:
        # The curvature next to the inflection is below the rounding noise of a large offset
        # at a small step; the step grows until it is resolved
        self.assertEqual(_inflections(lambda x: 1e6 + x**3, -100, 100), [(0.0, "inflection")])
        self.assertEqual(_inflections(lambda x: 1e3 * x**3 + 1e8, -3, 3), [(0.0, "inflection")])
        offset_sine = [x for x, _ in _inflections(lambda x: 1e5 + math.sin(x), -10, 10)]
        self.assertEqual(len(offset_sine), 7)
        for found, k in zip(offset_sine, range(-3, 4)):
            self.assertAlmostEqual(found, k * math.pi, delta=0.01)

    def test_large_x_is_accurate_to_the_digits_reported(self) -> None:
        for centre in (1000.0, 1e4, 1e5):
            report = find_function_features(
                lambda x, c=centre: math.exp(-((x - c) ** 2)), centre - 5, centre + 5, features=["inflections"]
            )
            xs = [feature["x"] for feature in report["features"]]
            self.assertEqual(len(xs), 2)
            for found, expected in zip(xs, [centre - 1 / math.sqrt(2), centre + 1 / math.sqrt(2)]):
                # Rounded to the digits it is sure of: within one unit of the last digit
                digits = len(repr(found).split(".")[1]) if "." in repr(found) else 0
                self.assertLessEqual(abs(found - expected), 10.0**-digits)
            # y is f at the inflection: exp(-1/2) on both sides
            for feature in report["features"]:
                self.assertAlmostEqual(feature["y"], math.exp(-0.5), places=7)

    def test_y_is_zero_where_the_inflection_is_on_the_axis(self) -> None:
        report = find_function_features(math.sin, 0, 10, features=["inflections"])
        self.assertEqual([feature["y"] for feature in report["features"]], [0.0, 0.0, 0.0])
        for feature, k in zip(report["features"], (1, 2, 3)):
            self.assertAlmostEqual(feature["x"], k * math.pi, places=6)

    def test_underflowing_values_do_not_crash(self) -> None:
        report = find_function_features(lambda x: 1e-200 * x**3, -3, 3, features=["roots", "inflections"])
        self.assertEqual(_summary(report), [(0.0, 0.0, "root"), (0.0, 0.0, "inflection")])

    def test_unknown_feature_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            find_function_features(math.sin, 0, 1, features=["inflection_points"])


ALL_FEATURES = ["roots", "extrema", "inflections"]


def _xs_of(report: dict, kind: str) -> list:
    return [feature["x"] for feature in report["features"] if feature["kind"] == kind]


def _assert_within_last_digit(test: unittest.TestCase, found: float, expected: float) -> None:
    """``found`` is rounded to the digits it is sure of: within one unit of its last decimal."""
    text = repr(found)
    decimals = len(text.split(".")[1]) if "." in text and not text.endswith(".0") else 0
    test.assertLessEqual(abs(found - expected), 10.0**-decimals, f"{found} vs {expected}")


class TestFeaturesCloserThanTheSampling(unittest.TestCase):
    """Features within a sample step or two of each other (500 samples, steps of about 0.005)."""

    def test_two_roots_inside_one_sample_step(self) -> None:
        # Samples either side are positive; the minimum between them is below the axis
        report = find_function_features(lambda x: (x - 0.5) * (x - 0.5001), 0.0123, 1.1)
        self.assertEqual(
            _summary(report), [(0.5, 0.0, "root"), (0.50005, -2.5e-09, "local_min"), (0.5001, 0.0, "root")]
        )
        report = find_function_features(lambda x: x * x - 1e-10, -1.37, 1.3)
        self.assertEqual(_summary(report), [(-1e-05, 0.0, "root"), (0.0, -1e-10, "local_min"), (1e-05, 0.0, "root")])

    def test_close_inflections_of_a_sixth_power(self) -> None:
        # (x^2 - a^2)^3 changes concavity at -a, -a/sqrt(5), a/sqrt(5) and a, all within two steps
        for a in (0.01, 0.001):
            with self.subTest(a=a):
                report = find_function_features(lambda x, a=a: (x * x - a * a) ** 3, -1.37, 1.3, features=ALL_FEATURES)
                inflections = _xs_of(report, "inflection")
                expected = [-a, -a / math.sqrt(5), a / math.sqrt(5), a]
                self.assertEqual(len(inflections), 4)
                for found, value in zip(inflections, expected):
                    self.assertAlmostEqual(found, value, delta=1e-6 * a)
                self.assertEqual(_xs_of(report, "root"), [-a, a])
                self.assertEqual(_xs_of(report, "local_min"), [0.0])

    def test_steep_inflections_between_two_samples(self) -> None:
        # The samples are +/-1 (or 0 and 1) to the last bit beyond a transition 1e-4 wide
        report = find_function_features(lambda x: math.tanh(1e4 * (x - 0.123)), -1.37, 1.3, features=ALL_FEATURES)
        self.assertEqual(_summary(report), [(0.123, 0.0, "root"), (0.123, 0.0, "inflection")])

        def logistic(x: float) -> float:
            exponent = -1e4 * x
            return 0.0 if exponent > 700 else 1.0 / (1.0 + math.exp(exponent))

        report = find_function_features(logistic, -1.37, 1.3, features=["inflections"])
        self.assertEqual(_summary(report), [(0.0, 0.5, "inflection")])

    def test_cusp_on_the_axis_is_a_touching_root(self) -> None:
        report = find_function_features(lambda x: abs(x) ** (1 / 3), -0.8609, 3.948)
        self.assertEqual(_summary(report), [(0.0, 0.0, "root"), (0.0, 0.0, "local_min")])
        self.assertTrue(report["features"][0]["touching"])

    def test_wide_intervals_stay_within_the_evaluation_budget(self) -> None:
        calls = [0]

        def counted_tan(x: float) -> float:
            calls[0] += 1
            return math.tan(x)

        report = find_function_features(counted_tan, -1000, 1000, samples=10000)
        self.assertEqual(report["total_found"], 637)
        # Two scans' worth of samples at most, plus a few dozen evaluations per root
        self.assertLess(calls[0], 2 * 10001 + 637 * 200)


class TestFeaturesFarFromZero(unittest.TestCase):
    """Intervals about 10 wide around x = 1e6 and beyond, where x itself is rounded to ~1e-10."""

    def test_inflections_far_from_zero(self) -> None:
        for centre in (1e7, 1e9):
            report = find_function_features(
                lambda x, c=centre: (x - c) ** 3, centre - 5.3, centre + 4.1, features=["inflections"]
            )
            self.assertEqual(_xs_of(report, "inflection"), [centre])
        for centre in (1e6, 1e8):
            report = find_function_features(
                lambda x, c=centre: math.exp(-((x - c) ** 2)), centre - 5.3, centre + 4.1, features=["inflections"]
            )
            found = _xs_of(report, "inflection")
            self.assertEqual(len(found), 2)
            for x, expected in zip(found, (centre - 1 / math.sqrt(2), centre + 1 / math.sqrt(2))):
                _assert_within_last_digit(self, x, expected)
                self.assertNotEqual(x, round(x, 2), "more digits than eight significant ones of x")

    def test_sine_inflections_far_from_zero(self) -> None:
        report = find_function_features(math.sin, 1e6, 1e6 + 10, features=["inflections"])
        found = _xs_of(report, "inflection")
        expected = [k * math.pi for k in range(318310, 318314)]
        self.assertEqual(len(found), 4)
        for x, value in zip(found, expected):
            _assert_within_last_digit(self, x, value)

    def test_roots_and_extrema_far_from_zero(self) -> None:
        # Probing and merging at a fraction of |x| used to lose these (0.1 at 1e6, 10 at 1e8)
        c = 1e6
        report = find_function_features(lambda x: (x - c - 0.3) * (x - c - 1.4) * (x - c - 2.2), c - 1, c + 3)
        self.assertEqual(_xs_of(report, "root"), [c + 0.3, c + 1.4, c + 2.2])
        self.assertEqual(len(_xs_of(report, "local_max")), 1)
        self.assertEqual(len(_xs_of(report, "local_min")), 1)
        report = find_function_features(math.sin, 1e8, 1e8 + 20)
        roots = _xs_of(report, "root")
        expected = [k * math.pi for k in range(31830989, 31830995)]
        self.assertEqual(len(roots), len(expected))
        for x, value in zip(roots, expected):
            _assert_within_last_digit(self, x, value)
        self.assertEqual(len(_xs_of(report, "local_min")) + len(_xs_of(report, "local_max")), 6)


class TestNoFalseFeatures(unittest.TestCase):
    """Poles, jumps and rounding noise next to the refined sampling and the large-|x| handling."""

    def test_no_extrema_beside_poles(self) -> None:
        # Refined samples land close to the poles; none of them is an extremum
        report = find_function_features(math.tan, -91.90248507255576, 67.58290293488042, features=["extrema"])
        self.assertEqual(report["features"], [])
        report = find_function_features(lambda x: 1 / (x - 0.37) ** 2, -2.5, 2.5, features=ALL_FEATURES)
        self.assertEqual(report["features"], [])
        report = find_function_features(lambda x: math.log(abs(x)), -0.5911260420880353, 1.2789270879870989)
        self.assertEqual(_summary(report), [(1.0, 0.0, "root")])
        cosecant = find_function_features(
            lambda x: 1 / math.sin(x), 11.030146530938426, 69.14722590157838, features=["extrema"]
        )
        for feature in cosecant["features"]:
            # Only the true extrema, at odd multiples of pi/2
            self.assertAlmostEqual(feature["x"] / (math.pi / 2) % 2, 1.0, places=5)

    def test_no_features_at_poles_far_from_zero(self) -> None:
        report = find_function_features(math.tan, 29999998.4817316, 30000000.565277524, features=ALL_FEATURES)
        self.assertEqual(report["features"], [])
        report = find_function_features(lambda x: 1 / (x - 1e8), 1e8 - 0.5553, 1e8 + 1.7216, features=ALL_FEATURES)
        self.assertEqual(report["features"], [])

    def test_underflowing_values_are_not_inflections(self) -> None:
        # exp(-x^2) near |x| = 27 is a staircase of subnormal numbers
        report = find_function_features(lambda x: math.exp(-x * x), -1000, 1000, features=ALL_FEATURES)
        self.assertEqual(
            [feature["kind"] for feature in report["features"]],
            ["root", "inflection", "local_max", "inflection", "root"],
        )

    def test_steps_sampled_at_their_jumps_are_not_inflections(self) -> None:
        # Every jump of floor lands on a sample: the straight stretches between are not inflections
        report = find_function_features(math.floor, -1000, 1000, features=ALL_FEATURES)
        self.assertEqual(len(report["features"]), 1)
        self.assertEqual(report["features"][0]["zero_interval"][0], 0.0)
        self.assertEqual(find_function_features(math.floor, -5, 5, features=["inflections"])["features"], [])

    def test_noisy_nearly_straight_functions_have_no_inflections(self) -> None:
        # Rounding at 1e3 or 1e9 inside f, or random noise of a few eps, is far above the
        # modelled noise of the values; refined samples used to turn it into inflections
        for offset in (1e3, 1e6, 1e9):
            with self.subTest(offset=offset):
                report = find_function_features(lambda x, c=offset: (x + c) - c, 0, 10, features=ALL_FEATURES)
                self.assertEqual(_summary(report), [(0.0, 0.0, "root")])
        rng = random.Random(5)
        for size in (2, 8, 64):
            for _ in range(10):

                def noisy(x: float, size: float = size) -> float:
                    return x + rng.uniform(-1.0, 1.0) * size * 2.220446049250313e-16

                with self.subTest(size=size):
                    report = find_function_features(noisy, -1, 1, features=["inflections"])
                    self.assertEqual(report["features"], [])


class TestReviewRegressions(unittest.TestCase):
    """Behaviour of the original sampling kept where the refinements first broke it."""

    def test_noisy_smooth_extremum_is_kept(self) -> None:
        # sin's argument is about 700 here, so f's relative rounding noise is about 1e-13,
        # which once looked like a better neighbour of the true maximum
        def damped(x: float) -> float:
            return math.exp(-0.4696841252312553 * x) * math.sin(4.8928699595591265 * x)

        report = find_function_features(
            damped, 30.967068608091864, 143.42021819054372, features=["extrema"], max_results=1000
        )
        self.assertIn(141.55812, _xs_of(report, "local_max"))
        self.assertEqual(report["total_found"], 175)  # every extremum in the interval

    def test_root_and_extremum_at_a_continuous_breakpoint(self) -> None:
        # A corner where two pieces meet is declared as a breakpoint but is no discontinuity
        report = find_function_features(abs, -1.3, 1.7, breakpoints=[0.0])
        self.assertEqual(_summary(report), [(0.0, 0.0, "root"), (0.0, 0.0, "local_min")])
        self.assertTrue(report["features"][0]["touching"])
        report = find_function_features(lambda x: x - 0.3, -1.3, 1.7, breakpoints=[0.3])
        self.assertEqual(_summary(report), [(0.3, 0.0, "root")])
        intersections = find_intersections(abs, lambda x: 0.0, -1.3, 1.7, breakpoints=[0.0])
        self.assertEqual(_summary(intersections), [(0.0, 0.0, "intersection")])

    def test_discontinuous_breakpoints_still_split_the_interval(self) -> None:
        cases = {
            "pole": lambda x: 1 / x,
            "jump through 0": lambda x: x if x < 0 else x + 1,
            "sign": lambda x: float((x > 0) - (x < 0)),
            "hole": lambda x: x * x if x != 0 else math.nan,
        }
        for name, f in cases.items():
            with self.subTest(name):
                self.assertEqual(find_function_features(f, -1.3, 1.7, breakpoints=[0.0])["features"], [])
        tangent = find_function_features(
            math.tan, -5, 5, breakpoints=[-1.5 * math.pi, -0.5 * math.pi, 0.5 * math.pi, 1.5 * math.pi]
        )
        self.assertEqual(_xs_of(tangent, "root"), [-3.141592654, 0.0, 3.141592654])
        self.assertEqual(_xs_of(tangent, "local_min") + _xs_of(tangent, "local_max"), [])

    def test_roots_where_floats_are_coarse(self) -> None:
        # At 1e12 floats are 1.2e-4 apart: f is 5e-5 at the best x, more than 1e-3 of its samples
        c = 1e12
        report = find_function_features(lambda x: x - c - 0.3, c - 5, c + 5, features=["roots"])
        self.assertEqual(_xs_of(report, "root"), [c + 0.3])


if __name__ == "__main__":
    unittest.main()
