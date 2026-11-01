"""The special functions and distribution functions against mpmath at 40 digits.

``utils.statistics.special_functions`` (log-gamma, incomplete gamma and beta) and
``utils.statistics.distribution_functions`` run in the browser; these tests run the same
code under CPython. mpmath comes with SymPy.
"""

from __future__ import annotations

import math
import random
import unittest
from typing import Dict, List, Tuple

from utils.statistics import special_functions as sf
from utils.statistics.distributions import DISTRIBUTIONS

try:
    import mpmath as mp
except ImportError:  # pragma: no cover - SymPy brings mpmath
    mp = None

RELATIVE = 1e-10


def _assert_close(case: unittest.TestCase, got: float, want: float, label: str, relative: float = RELATIVE) -> None:
    if want == 0.0:
        case.assertEqual(got, 0.0, label)
        return
    error = abs(got - want) / abs(want)
    case.assertLessEqual(error, relative, f"{label}: got {got!r}, want {want!r} (relative error {error:.2e})")


@unittest.skipIf(mp is None, "mpmath not installed")
class TestSpecialFunctions(unittest.TestCase):
    def setUp(self) -> None:
        mp.mp.dps = 40
        self.rng = random.Random(20261101)

    def test_log_gamma(self) -> None:
        for x in (1e-8, 0.001, 0.5, 1.0, 1.5, 2.0, 7.3, 14.9, 15.0, 15.1, 100.0, 12345.6, 1e6):
            with self.subTest(x=x):
                want = float(mp.loggamma(x))
                self.assertLessEqual(abs(sf.log_gamma(x) - want), 1e-14 * max(1.0, abs(want)))

    def test_log_gamma_rejects_non_positive_x(self) -> None:
        for x in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                sf.log_gamma(x)

    def test_incomplete_gamma_on_a_random_grid(self) -> None:
        for _ in range(300):
            a = 10 ** self.rng.uniform(-2, 3)
            x = a * 10 ** self.rng.uniform(-2, 0.7)
            lower = float(mp.gammainc(a, 0, x, regularized=True))
            upper = float(mp.gammainc(a, x, mp.inf, regularized=True))
            with self.subTest(a=a, x=x):
                if lower > 1e-300:
                    _assert_close(self, sf.gamma_p(a, x), lower, "P")
                if upper > 1e-300:
                    _assert_close(self, sf.gamma_q(a, x), upper, "Q")

    def test_incomplete_gamma_near_the_centre_for_large_a(self) -> None:
        for a in (1e3, 1e4, 3e4):
            for offset in (-4.0, -1.0, 0.0, 1.0, 4.0):
                x = a + offset * math.sqrt(a)
                with self.subTest(a=a, x=x):
                    _assert_close(self, sf.gamma_p(a, x), float(mp.gammainc(a, 0, x, regularized=True)), "P")
                    _assert_close(self, sf.gamma_q(a, x), float(mp.gammainc(a, x, mp.inf, regularized=True)), "Q")

    def test_incomplete_gamma_edges(self) -> None:
        self.assertEqual(sf.gamma_p(2.0, 0.0), 0.0)
        self.assertEqual(sf.gamma_q(2.0, 0.0), 1.0)
        self.assertEqual(sf.gamma_p(2.0, math.inf), 1.0)
        self.assertEqual(sf.gamma_q(2.0, math.inf), 0.0)
        self.assertEqual(sf.gamma_p(0.5, 5e-324), sf.gamma_p(0.5, 5e-324))  # no domain error at a subnormal x
        for a, x in ((0.0, 1.0), (-1.0, 1.0), (1.0, -1.0)):
            with self.assertRaises(ValueError):
                sf.gamma_p(a, x)

    def test_incomplete_beta_on_a_random_grid(self) -> None:
        for _ in range(300):
            a = 10 ** self.rng.uniform(-2, 2.5)
            b = 10 ** self.rng.uniform(-2, 2.5)
            x = self.rng.random()
            lower = float(mp.betainc(a, b, 0, x, regularized=True))
            # The upper tail from its own side: mpmath's 1 - I loses it below 1e-40.
            upper = float(mp.betainc(b, a, 0, 1 - mp.mpf(x), regularized=True))
            with self.subTest(a=a, b=b, x=x):
                if lower > 1e-300:
                    _assert_close(self, sf.beta_inc(a, b, x), lower, "I")
                if upper > 1e-300:
                    _assert_close(self, sf.beta_inc_upper(a, b, x), upper, "1 - I")

    def test_incomplete_beta_edges(self) -> None:
        self.assertEqual(sf.beta_inc(2.0, 3.0, 0.0), 0.0)
        self.assertEqual(sf.beta_inc(2.0, 3.0, 1.0), 1.0)
        self.assertEqual(sf.beta_inc_upper(2.0, 3.0, 0.0), 1.0)
        self.assertEqual(sf.beta_inc_upper(2.0, 3.0, 1.0), 0.0)
        for a, b, x in ((0.0, 1.0, 0.5), (1.0, -1.0, 0.5), (1.0, 1.0, 1.5)):
            with self.assertRaises(ValueError):
                sf.beta_inc(a, b, x)

    def test_bisect_increasing(self) -> None:
        root = sf.bisect_increasing(lambda x: x * x * x, 2.0, (0.0, 2.0))
        self.assertAlmostEqual(root, 2.0 ** (1.0 / 3.0), places=14)


def _binomial_reference(n: int, p: float) -> List[float]:
    q = mp.mpf(p)
    return [mp.binomial(n, j) * q**j * (1 - q) ** (n - j) for j in range(n + 1)]


@unittest.skipIf(mp is None, "mpmath not installed")
class TestDistributionFunctions(unittest.TestCase):
    """cdf, sf and density of every row against mpmath at points across the body and both tails."""

    def setUp(self) -> None:
        mp.mp.dps = 40

    def check(self, name: str, raw: Dict[str, float], cases: List[Tuple[float, float, float, float]]) -> None:
        spec = DISTRIBUTIONS[name]
        params = spec.parse_params(raw)
        for x, cdf, upper, density in cases:
            with self.subTest(distribution=name, params=raw, x=x):
                for got, want, label in (
                    (spec.cdf(x, params), cdf, "cdf"),
                    (spec.sf(x, params), upper, "sf"),
                    (spec.pdf(x, params), density, "pdf"),
                ):
                    _assert_close(self, got, float(want), label)

    def test_normal(self) -> None:
        cases = []
        for x in (-38.0, -20.0, -3.0, -0.2, 0.0, 1.0, 5.0, 25.0):
            z = (mp.mpf(x) - 1) / 2
            cases.append((x, mp.ncdf(z), mp.ncdf(-z), mp.npdf(x, 1, 2)))
        self.check("normal", {"mean": 1, "sigma": 2}, cases)

    def test_t(self) -> None:
        for df in (1.0, 2.5, 5.0, 30.0, 1e5):
            nu = mp.mpf(df)

            def lower(t: float) -> float:
                tail = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, nu / (nu + mp.mpf(t) ** 2), regularized=True) / 2
                return tail if t <= 0 else 1 - tail

            def density(t: float) -> float:
                t = mp.mpf(t)
                return (
                    mp.gamma((nu + 1) / 2)
                    / (mp.sqrt(nu * mp.pi) * mp.gamma(nu / 2))
                    * (1 + t * t / nu) ** (-(nu + 1) / 2)
                )

            cases = [(t, lower(t), lower(-t), density(t)) for t in (-60.0, -4.0, -0.5, 0.0, 0.01, 2.1, 9.0)]
            self.check("t", {"df": df}, cases)

    def test_chi_squared(self) -> None:
        for df in (1.0, 2.0, 3.0, 10.0, 250.0):
            k = mp.mpf(df)

            def density(x: float) -> float:
                x = mp.mpf(x)
                return x ** (k / 2 - 1) * mp.exp(-x / 2) / (2 ** (k / 2) * mp.gamma(k / 2))

            cases = [
                (
                    x,
                    mp.gammainc(k / 2, 0, x / 2, regularized=True),
                    mp.gammainc(k / 2, x / 2, mp.inf, regularized=True),
                    density(x),
                )
                for x in (1e-4, 0.5, df, 3 * df + 20)
            ]
            self.check("chi_squared", {"df": df}, cases)

    def test_f(self) -> None:
        for d1, d2 in ((1.0, 1.0), (4.0, 9.0), (2.0, 30.0), (50.0, 3.0)):
            a, b = mp.mpf(d1) / 2, mp.mpf(d2) / 2
            cases = []
            for x in (1e-3, 0.4, 1.0, 7.5, 300.0):
                xm = mp.mpf(x)
                y = d1 * xm / (d1 * xm + d2)
                density = mp.sqrt((d1 * xm) ** d1 * mp.mpf(d2) ** d2 / (d1 * xm + d2) ** (d1 + d2)) / (
                    xm * mp.beta(a, b)
                )
                cases.append(
                    (
                        x,
                        mp.betainc(a, b, 0, y, regularized=True),
                        mp.betainc(b, a, 0, d2 / (d1 * xm + d2), regularized=True),
                        density,
                    )
                )
            self.check("f", {"df": d1, "df2": d2}, cases)

    def test_exponential_and_uniform(self) -> None:
        self.check(
            "exponential",
            {"rate": 2.0},
            [
                (x, 1 - mp.exp(-2 * mp.mpf(x)), mp.exp(-2 * mp.mpf(x)), 2 * mp.exp(-2 * mp.mpf(x)))
                for x in (1e-9, 0.3, 1.5, 300.0)
            ],
        )
        self.check("exponential", {"rate": 2.0}, [(-1.0, 0.0, 1.0, 0.0)])
        self.check(
            "uniform",
            {"lower": -1.0, "upper": 3.0},
            [(-2.0, 0.0, 1.0, 0.0), (0.0, 0.25, 0.75, 0.25), (3.0, 1.0, 0.0, 0.25), (4.0, 1.0, 0.0, 0.0)],
        )

    def test_binomial(self) -> None:
        for n, p in ((1, 0.5), (10, 0.3), (40, 0.3), (500, 0.02), (3000, 0.7)):
            terms = _binomial_reference(n, p)
            cases = []
            for k in sorted({0, 1, n // 3, int(n * p), n - 1, n}):
                cases.append((float(k), mp.fsum(terms[: k + 1]), mp.fsum(terms[k + 1 :]), terms[k]))
            cases.append((n // 2 + 0.5, mp.fsum(terms[: n // 2 + 1]), mp.fsum(terms[n // 2 + 1 :]), 0.0))
            self.check("binomial", {"n": n, "p": p}, cases)

    def test_poisson(self) -> None:
        for rate in (0.1, 4.0, 50.0, 1000.0):
            lam = mp.mpf(rate)
            cases = []
            for k in (0, 1, int(rate), int(rate + 5 * math.sqrt(rate)) + 3):
                body = mp.fsum(lam**j * mp.exp(-lam) / mp.factorial(j) for j in range(k + 1))
                upper = mp.gammainc(k + 1, 0, lam, regularized=True)
                cases.append((float(k), body, upper, lam**k * mp.exp(-lam) / mp.factorial(k)))
            self.check("poisson", {"rate": rate}, cases)

    def test_geometric(self) -> None:
        for p in (0.01, 0.2, 0.9):
            q = mp.mpf(p)
            self.check(
                "geometric",
                {"p": p},
                [(float(k), 1 - (1 - q) ** k, (1 - q) ** k, q * (1 - q) ** (k - 1)) for k in (1, 2, 7, 200)]
                + [(0.0, 0.0, 1.0, 0.0)],
            )

    def test_discrete_densities_are_zero_off_the_integers(self) -> None:
        for name, raw in (("binomial", {"n": 10, "p": 0.3}), ("poisson", {"rate": 3}), ("geometric", {"p": 0.4})):
            spec = DISTRIBUTIONS[name]
            params = spec.parse_params(raw)
            with self.subTest(distribution=name):
                self.assertEqual(spec.pdf(2.5, params), 0.0)
                self.assertEqual(spec.pdf(-1.0, params), 0.0)
                self.assertEqual(spec.pdf(3.0 + 1e-12, params), spec.pdf(3.0, params))
                self.assertEqual(spec.cdf(2.5, params), spec.cdf(2.0, params))

    def test_densities_at_zero(self) -> None:
        chi = DISTRIBUTIONS["chi_squared"]
        f = DISTRIBUTIONS["f"]
        self.assertEqual(chi.pdf(0.0, {"df": 1.0}), math.inf)
        self.assertEqual(chi.pdf(0.0, {"df": 2.0}), 0.5)
        self.assertEqual(chi.pdf(0.0, {"df": 3.0}), 0.0)
        self.assertEqual(f.pdf(0.0, {"df": 1.0, "df2": 5.0}), math.inf)
        self.assertEqual(f.pdf(0.0, {"df": 2.0, "df2": 5.0}), 1.0)
        self.assertEqual(f.pdf(0.0, {"df": 3.0, "df2": 5.0}), 0.0)


class TestQuantiles(unittest.TestCase):
    PROBABILITIES = (1e-10, 0.001, 0.025, 0.3, 0.5, 0.7, 0.975, 0.999, 1 - 1e-10)

    CONTINUOUS: Tuple[Tuple[str, Dict[str, float]], ...] = (
        ("normal", {"mean": 100.0, "sigma": 15.0}),
        ("uniform", {"lower": -2.0, "upper": 6.0}),
        ("exponential", {"rate": 0.5}),
        ("t", {"df": 1.0}),
        ("t", {"df": 7.0}),
        ("chi_squared", {"df": 1.0}),
        ("chi_squared", {"df": 12.0}),
        ("f", {"df": 1.0, "df2": 1.0}),
        ("f", {"df": 5.0, "df2": 20.0}),
    )

    def test_continuous_quantiles_invert_the_cdf(self) -> None:
        for name, raw in self.CONTINUOUS:
            spec = DISTRIBUTIONS[name]
            params = spec.parse_params(raw)
            for prob in self.PROBABILITIES:
                with self.subTest(distribution=name, params=raw, prob=prob):
                    x = spec.quantile(prob, params)
                    # The tail at the quantile matches to 1e-9, or to the rounding of x itself.
                    if prob <= 0.5:
                        got, want = spec.cdf(x, params), prob
                    else:
                        got, want = spec.sf(x, params), 1 - prob
                    self.assertLessEqual(abs(got - want), 1e-9 * want + 1e-15)

    def test_known_critical_values(self) -> None:
        cases: List[Tuple[str, Dict[str, float], float, float]] = [
            ("normal", {}, 0.975, 1.959963984540054),
            ("t", {"df": 10.0}, 0.975, 2.228138851986274),
            ("t", {"df": 1.0}, 0.995, 63.65674116287399),
            ("chi_squared", {"df": 5.0}, 0.95, 11.070497693516351),
            ("f", {"df": 3.0, "df2": 12.0}, 0.95, 3.4902948194976),
        ]
        for name, raw, prob, want in cases:
            spec = DISTRIBUTIONS[name]
            with self.subTest(distribution=name, prob=prob):
                _assert_close(self, spec.quantile(prob, spec.parse_params(raw)), want, "quantile", 1e-12)

    def test_the_median_of_a_symmetric_distribution_is_a_plain_zero(self) -> None:
        for name, raw in (("normal", {}), ("t", {"df": 3.0})):
            spec = DISTRIBUTIONS[name]
            value = spec.quantile(0.5, spec.parse_params(raw))
            self.assertEqual(value, 0.0)
            self.assertEqual(math.copysign(1.0, value), 1.0)

    def test_discrete_quantile_is_the_smallest_k_reaching_the_probability(self) -> None:
        cases: Tuple[Tuple[str, Dict[str, float]], ...] = (
            ("binomial", {"n": 10.0, "p": 0.3}),
            ("binomial", {"n": 1000.0, "p": 0.01}),
            ("poisson", {"rate": 0.1}),
            ("poisson", {"rate": 25.0}),
            ("geometric", {"p": 0.05}),
        )
        for name, raw in cases:
            spec = DISTRIBUTIONS[name]
            params = spec.parse_params(raw)
            for prob in self.PROBABILITIES:
                with self.subTest(distribution=name, params=raw, prob=prob):
                    k = spec.quantile(prob, params)
                    self.assertEqual(k, math.floor(k))
                    self.assertGreaterEqual(spec.cdf(k, params), prob * (1 - 1e-12))
                    if k > spec.support(params)[0]:
                        self.assertLess(spec.cdf(k - 1, params), prob)

    def test_a_probability_equal_to_a_cdf_value_lands_on_that_value(self) -> None:
        spec = DISTRIBUTIONS["binomial"]
        params = spec.parse_params({"n": 10, "p": 0.5})
        for k in range(10):
            with self.subTest(k=k):
                self.assertEqual(spec.quantile(spec.cdf(k, params), params), k)
        geometric = DISTRIBUTIONS["geometric"]
        gparams = geometric.parse_params({"p": 0.5})
        self.assertEqual(geometric.quantile(0.75, gparams), 2.0)


if __name__ == "__main__":
    unittest.main()
