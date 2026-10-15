"""Pure Python tests for the number theory and series helpers behind MathUtils."""

from __future__ import annotations

import math
import random
import unittest

from utils.number_theory import (
    MAX_FACTORIZATION_INPUT,
    arithmetic_sum,
    gcd,
    geometric_sum,
    is_prime,
    lcm,
    prime_factors,
)


def _is_prime_by_trial_division(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


class TestIsPrime(unittest.TestCase):
    def test_matches_trial_division_below_20000(self) -> None:
        for n in range(20000):
            self.assertEqual(is_prime(n), _is_prime_by_trial_division(n), n)

    def test_strong_pseudoprimes_are_composite(self) -> None:
        # Each passes Miller-Rabin for a prefix of the prime bases
        for n in (
            3215031751,
            2152302898747,
            3474749660383,
            341550071728321,
            3825123056546413051,
            318665857834031151167461,
            3317044064679887385961981,
        ):
            with self.subTest(n=n):
                self.assertFalse(is_prime(n))

    def test_large_primes_and_squares(self) -> None:
        for p in (2**31 - 1, 2**61 - 1, 2**89 - 1, 2**127 - 1, 1000000007):
            with self.subTest(p=p):
                self.assertTrue(is_prime(p))
        self.assertFalse(is_prime(100000007 * 100000007))
        self.assertFalse(is_prime((2**61 - 1) * (2**31 - 1)))


class TestPrimeFactors(unittest.TestCase):
    def test_product_and_primality_of_factors(self) -> None:
        rng = random.Random(1)
        for _ in range(200):
            n = rng.randrange(1, 10**12)
            factors = prime_factors(n)
            with self.subTest(n=n):
                self.assertEqual(math.prod(factors), n)
                self.assertEqual(factors, sorted(factors))
                self.assertTrue(all(is_prime(p) for p in factors))

    def test_known_factorizations(self) -> None:
        self.assertEqual(prime_factors(1), [])
        self.assertEqual(prime_factors(360), [2, 2, 2, 3, 3, 5])
        self.assertEqual(prime_factors(600851475143), [71, 839, 1471, 6857])
        self.assertEqual(prime_factors(2**64 - 1), [3, 5, 17, 257, 641, 65537, 6700417])
        self.assertEqual(prime_factors(1000000007 * 998244353), [998244353, 1000000007])
        self.assertEqual(prime_factors((10**9 + 7) ** 2 * 12), [2, 2, 3, 1000000007, 1000000007])
        self.assertEqual(prime_factors(10**18 + 9), [10**18 + 9])

    def test_refuses_inputs_above_limit(self) -> None:
        with self.assertRaises(ValueError):
            prime_factors(MAX_FACTORIZATION_INPUT + 1)

    def test_reports_factors_too_large_to_find(self) -> None:
        # Two 15-digit primes: far beyond the Pollard rho step budget
        with self.assertRaises(ValueError) as ctx:
            prime_factors(999999999999989 * 999999999999947)
        self.assertIn("could not factor", str(ctx.exception))


class TestGcdLcm(unittest.TestCase):
    def test_integers_and_integral_floats(self) -> None:
        self.assertEqual(gcd((48, 18)), 6)
        self.assertEqual(gcd((48.0, 18)), 6)
        self.assertEqual(gcd((-4, 6)), 2)
        self.assertEqual(lcm((4, 6)), 12)
        self.assertEqual(lcm((0, 5)), 0)

    def test_rejects_non_integers(self) -> None:
        for values in ((4.5, 6), (True, 2), (float("inf"), 2)):
            with self.subTest(values=values):
                with self.assertRaises(TypeError):
                    gcd(values)
                with self.assertRaises(TypeError):
                    lcm(values)


class TestSeriesSums(unittest.TestCase):
    def test_geometric_sum_exact_for_integers(self) -> None:
        self.assertEqual(geometric_sum(1, 2, 64), 2**64 - 1)
        self.assertEqual(geometric_sum(1, 10, 1000), (10**1000 - 1) // 9)
        self.assertEqual(geometric_sum(1, -2, 5), 11)
        self.assertEqual(geometric_sum(2, 0, 5), 2)
        self.assertEqual(geometric_sum(5, 1, 4), 20)

    def test_geometric_sum_floats(self) -> None:
        self.assertEqual(geometric_sum(1.5, 2, 3), 10.5)
        self.assertAlmostEqual(geometric_sum(3, 0.5, 15), 6 * (1 - 0.5**15), places=14)
        # Ratio near 1: no cancellation in (1 - r^n) / (1 - r)
        r = 1 + 1e-12
        self.assertAlmostEqual(geometric_sum(1, r, 10), 10.000000000045, places=13)

    def test_geometric_sum_overflow_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            geometric_sum(1, 10.5, 10**6)

    def test_arithmetic_sum(self) -> None:
        self.assertEqual(arithmetic_sum(1, 1, 10**9), 10**9 * (10**9 + 1) // 2)
        self.assertEqual(arithmetic_sum(1, 2, 20), 400)
        self.assertEqual(arithmetic_sum(0.5, 0.25, 4), 3.5)

    def test_term_count_must_be_a_positive_whole_number(self) -> None:
        for function in (arithmetic_sum, geometric_sum):
            with self.subTest(function=function.__name__):
                with self.assertRaises(TypeError):
                    function(1, 1, 2.5)
                with self.assertRaises(ValueError):
                    function(1, 1, 0)
                self.assertEqual(function(1, 2, 3.0), function(1, 2, 3))


if __name__ == "__main__":
    unittest.main()
