"""Pure-Python number theory and series helpers used by MathUtils.

No browser imports, so the algorithms run under pytest as well as Brython.

Key Features:
    - Integer argument validation that rejects non-integral values
    - Deterministic Miller-Rabin primality test (exact below 3.3e24)
    - Prime factorization by trial division plus Pollard's rho, with a size guard
    - Exact arithmetic and geometric series sums for integer inputs
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

Number = Union[int, float]

# The first 13 primes as Miller-Rabin bases decide primality exactly for every
# n below this bound (Sorenson and Webster, 2015)
MILLER_RABIN_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
MILLER_RABIN_EXACT_LIMIT = 3317044064679887385961981
# Above the bound these bases are added; a composite then passes with probability below 4^-25
MILLER_RABIN_EXTRA_BASES = (43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)

# Factorization: trial division by every candidate up to this bound, then Pollard's rho
# with at most this many polynomial steps per factor found
TRIAL_DIVISION_LIMIT = 10_000
POLLARD_RHO_MAX_STEPS = 100_000
# Larger inputs are refused instead of risking a browser hang
MAX_FACTORIZATION_INPUT = 10**30

# Exact integer geometric sums are used while ratio^n has at most this many bits
MAX_EXACT_POWER_BITS = 4096
# Float ratios closer to 1 than this use the expm1/log1p form of the geometric sum
NEAR_ONE_RATIO = 1e-3
# Integral float results are returned as int only below 2^53 (where they are exact)
MAX_EXACT_FLOAT_INT = 2**53


def require_integer(value: Number, requirement: str) -> int:
    """Return value as an int; raise TypeError unless it is an integer or an integral float.

    requirement starts the error message, e.g. "gcd arguments must be integers".
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise TypeError(f"{requirement}, got {value!r}")


def gcd(values: Sequence[Number]) -> int:
    return math.gcd(*[require_integer(v, "gcd arguments must be integers") for v in values])


def lcm(values: Sequence[Number]) -> int:
    return math.lcm(*[require_integer(v, "lcm arguments must be integers") for v in values])


def _passes_miller_rabin(n: int, base: int, d: int, s: int) -> bool:
    """Return True when n is a strong probable prime to the given base (n - 1 = d * 2^s, d odd)."""
    x = pow(base, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def is_prime(n: int) -> bool:
    """Primality test: trial division by small primes, then Miller-Rabin.

    Exact for n < 3.3e24; for larger n a composite is reported prime with
    probability below 4^-25.
    """
    if n < 2:
        return False
    for p in MILLER_RABIN_BASES:
        if n % p == 0:
            return n == p
    if n < 43 * 43:
        return True
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases: Tuple[int, ...] = MILLER_RABIN_BASES
    if n >= MILLER_RABIN_EXACT_LIMIT:
        bases = MILLER_RABIN_BASES + MILLER_RABIN_EXTRA_BASES
    return all(_passes_miller_rabin(n, base, d, s) for base in bases)


def _pollard_rho(n: int, max_steps: int) -> Optional[int]:
    """Return a nontrivial factor of the odd composite n (Brent's variant), or None after max_steps."""
    steps = 0
    for c in range(1, 20):
        y, power, factor = 2, 1, 1
        x = y
        product = 1
        saved_y = y
        while factor == 1:
            x = y
            for _ in range(power):
                y = (y * y + c) % n
            done = 0
            while done < power and factor == 1:
                saved_y = y
                batch = min(128, power - done)
                for _ in range(batch):
                    y = (y * y + c) % n
                    product = product * abs(x - y) % n
                factor = math.gcd(product, n)
                done += batch
            steps += 2 * power
            if steps > max_steps and factor == 1:
                return None
            power *= 2
        if factor == n:
            # The batch overshot: retrace it one step at a time
            factor = 1
            while factor == 1:
                saved_y = (saved_y * saved_y + c) % n
                factor = math.gcd(abs(x - saved_y), n)
        if factor != n:
            return factor
    return None


def _factor_large(n: int, factors: List[int]) -> None:
    """Append the prime factors of n (no factor below TRIAL_DIVISION_LIMIT) to factors."""
    if n == 1:
        return
    if is_prime(n):
        factors.append(n)
        return
    divisor = _pollard_rho(n, POLLARD_RHO_MAX_STEPS)
    if divisor is None:
        raise ValueError(
            f"prime_factors could not factor {n} quickly: its prime factors are all too large "
            "for in-browser factorization"
        )
    _factor_large(divisor, factors)
    _factor_large(n // divisor, factors)


def prime_factors(n: int) -> List[int]:
    """Return the prime factors of n >= 1 in ascending order, with multiplicity.

    Trial division removes factors below TRIAL_DIVISION_LIMIT; Pollard's rho splits
    the rest. Raises ValueError for n above MAX_FACTORIZATION_INPUT, or when the
    remaining factors are too large to find within POLLARD_RHO_MAX_STEPS.
    """
    if n > MAX_FACTORIZATION_INPUT:
        raise ValueError("prime_factors supports integers up to 10^30")
    factors: List[int] = []
    for d in (2, 3):
        while n % d == 0:
            factors.append(d)
            n //= d
    d = 5
    while d <= TRIAL_DIVISION_LIMIT and d * d <= n:
        for candidate in (d, d + 2):  # 6k - 1 and 6k + 1
            while n % candidate == 0:
                factors.append(candidate)
                n //= candidate
        d += 6
    if n > 1 and d * d > n:
        factors.append(n)  # no factor up to sqrt(n): n is prime
    else:
        _factor_large(n, factors)
    return sorted(factors)


def _as_exact_int(value: Number) -> Optional[int]:
    """Return value as an int when it is an exact integer (int, or integral float below 2^53), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer() and abs(value) < MAX_EXACT_FLOAT_INT:
        return int(value)
    return None


def _whole_number_result(result: float, name: str) -> Number:
    if not math.isfinite(result):
        raise ValueError(f"{name} is too large to represent")
    if result.is_integer() and abs(result) < MAX_EXACT_FLOAT_INT:
        return int(result)
    return result


def require_term_count(n: Number) -> int:
    """Return n as a positive int; reject a non-integer or n < 1."""
    count = require_integer(n, "The number of terms must be a whole number")
    if count < 1:
        raise ValueError("Number of terms must be at least 1")
    return count


def arithmetic_sum(first: Number, diff: Number, n: Number) -> Number:
    """Sum of the first n terms of first, first + diff, ...; exact for integer inputs."""
    count = require_term_count(n)
    a, d = _as_exact_int(first), _as_exact_int(diff)
    if a is not None and d is not None:
        return count * (2 * a + (count - 1) * d) // 2
    return _whole_number_result(count / 2 * (2 * first + (count - 1) * diff), "The arithmetic sum")


def geometric_sum(first: Number, ratio: Number, n: Number) -> Number:
    """Sum of the first n terms of first, first * ratio, ...; exact for integer inputs.

    For a float ratio within NEAR_ONE_RATIO of 1 the sum is computed as
    first * expm1(n * log1p(ratio - 1)) / (ratio - 1), which avoids the
    cancellation in (1 - ratio^n) / (1 - ratio).
    """
    count = require_term_count(n)
    if ratio == 1:
        return first * count
    a, r = _as_exact_int(first), _as_exact_int(ratio)
    if a is not None and r is not None:
        bits = count * math.log2(abs(r)) if abs(r) > 1 else 0.0
        if bits <= MAX_EXACT_POWER_BITS:
            power: int = r**count
            return a * (power - 1) // (r - 1)
    try:
        if 0 < ratio and abs(ratio - 1) < NEAR_ONE_RATIO:
            result = first * math.expm1(count * math.log1p(ratio - 1)) / (ratio - 1)
        else:
            result = first * (1 - float(ratio) ** count) / (1 - ratio)
    except OverflowError:
        result = math.inf
    return _whole_number_result(float(result), "The geometric sum")
