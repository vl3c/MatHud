"""The coefficients of a polynomial in one variable, read from an expression's syntax tree.

The asymptote analysis of every drawn function and the inequality sign chart's seeds need
the coefficients of polynomial denominators and differences, without asking the server.
MathUtils._math_tree turns math.js's parse tree into the tuples read here:

- ("num", text): a number literal, read exactly (0.1 is 1/10);
- ("var",): the variable;
- ("const", value): a part without the variable that math.js evaluates to a finite real
  number (pi, sqrt(2), a call such as log(2));
- ("op", name, [operands]): add, subtract, multiply, divide, pow, unaryMinus or unaryPlus;
- ("other",): anything else (another name, a call of the variable).

The tree is expanded as a ratio of two polynomials, exactly (Fraction) until a "const"
enters, then in floats: sums, products, quotients and whole powers. The result is a
polynomial when the denominator divides the numerator ((x^2-1)/(x-1) is x+1).
MathUtils._polynomial_coefficients confirms the coefficients by comparing values at sample
points, so a wrong reading can only lose a polynomial, never give a wrong one.

Pure Python (no browser imports), so server-side pytest covers it.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, List, Optional, Sequence, Tuple, Union

Coefficient = Union[Fraction, float]
Polynomial = List[Coefficient]
Ratio = Tuple[Polynomial, Polynomial]

# Polynomials of higher degree are not expanded (the numeric root finder takes degree 40).
MAX_DEGREE = 40
# A whole power of a number with a larger exponent is not computed exactly (10^10^10).
MAX_EXACT_EXPONENT = 4096
# A float remainder this small (relative to the numerator) counts as 0 in the final division.
REMAINDER_TOLERANCE = 1e-12


class _NotPolynomial(Exception):
    pass


def coefficients(tree: Any, max_degree: int = MAX_DEGREE) -> Optional[List[float]]:
    """[c0, c1, ...] of the polynomial the tree stands for (at least [0.0]), or None when it is
    not a polynomial of degree at most max_degree with finite real coefficients."""
    try:
        numerator, denominator = _expand(tree, max_degree)
        values = [float(coefficient) for coefficient in _quotient(numerator, denominator)]
    except (_NotPolynomial, ArithmeticError, ValueError, TypeError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def _expand(tree: Any, max_degree: int) -> Ratio:
    kind = tree[0]
    one: Polynomial = [Fraction(1)]
    if kind == "num":
        return [_exact_number(str(tree[1]))], one
    if kind == "var":
        return [Fraction(0), Fraction(1)], one
    if kind == "const":
        value = float(tree[1])
        if not math.isfinite(value):
            raise _NotPolynomial()
        return [value], one
    if kind != "op":
        raise _NotPolynomial()
    name, operands = tree[1], [_expand(operand, max_degree) for operand in tree[2]]
    if name == "unaryPlus" and len(operands) == 1:
        return operands[0]
    if name == "unaryMinus" and len(operands) == 1:
        return [-coefficient for coefficient in operands[0][0]], operands[0][1]
    if len(operands) != 2:
        raise _NotPolynomial()
    first, second = operands
    if name in ("add", "subtract"):
        sign = 1 if name == "add" else -1
        if _same(first[1], second[1]):
            return _sum(first[0], second[0], sign), first[1]
        top = _sum(_product(first[0], second[1], max_degree), _product(second[0], first[1], max_degree), sign)
        return top, _product(first[1], second[1], max_degree)
    if name == "multiply":
        return _product(first[0], second[0], max_degree), _product(first[1], second[1], max_degree)
    if name == "divide":
        if _trimmed(second[0]) == [0]:
            raise _NotPolynomial()
        return _product(first[0], second[1], max_degree), _product(first[1], second[0], max_degree)
    if name == "pow":
        return _power(first, _constant(second), max_degree)
    raise _NotPolynomial()


def _exact_number(text: str) -> Fraction:
    """A decimal literal ("1.001", "2.5e3", "1e-3") as an exact Fraction. Brython's Fraction()
    reads only integers and ratios from text."""
    mantissa, _, exponent_text = text.strip().lower().partition("e")
    whole, _, decimals = mantissa.partition(".")
    if not (whole + decimals).isdigit():
        raise _NotPolynomial()
    exponent = int(exponent_text) if exponent_text else 0
    if abs(exponent) > MAX_EXACT_EXPONENT:
        raise _NotPolynomial()
    value = Fraction(int(whole + decimals)) / Fraction(10 ** len(decimals))
    return value * Fraction(10**exponent) if exponent >= 0 else value / Fraction(10**-exponent)


def _trimmed(polynomial: Sequence[Coefficient]) -> Polynomial:
    trimmed = list(polynomial)
    while len(trimmed) > 1 and trimmed[-1] == 0:
        trimmed.pop()
    return trimmed


def _same(first: Polynomial, second: Polynomial) -> bool:
    return _trimmed(first) == _trimmed(second)


def _constant(ratio: Ratio) -> Coefficient:
    """The value of a ratio of numbers; raises _NotPolynomial for any other."""
    numerator, denominator = _trimmed(ratio[0]), _trimmed(ratio[1])
    if len(numerator) != 1 or len(denominator) != 1 or denominator[0] == 0:
        raise _NotPolynomial()
    return numerator[0] / denominator[0]


def _sum(first: Polynomial, second: Polynomial, sign: int) -> Polynomial:
    size = max(len(first), len(second))
    padded_first = list(first) + [Fraction(0)] * (size - len(first))
    padded_second = list(second) + [Fraction(0)] * (size - len(second))
    return _trimmed([a + sign * b for a, b in zip(padded_first, padded_second)])


def _product(first: Polynomial, second: Polynomial, max_degree: int) -> Polynomial:
    first, second = _trimmed(first), _trimmed(second)
    if len(first) + len(second) - 2 > max_degree:
        raise _NotPolynomial()
    result: Polynomial = [Fraction(0)] * (len(first) + len(second) - 1)
    for i, a in enumerate(first):
        if a == 0:
            continue
        for j, b in enumerate(second):
            result[i + j] = result[i + j] + a * b
    return _trimmed(result)


def _power(base: Ratio, exponent: Coefficient, max_degree: int) -> Ratio:
    numerator, denominator = _trimmed(base[0]), _trimmed(base[1])
    if len(numerator) == 1 and len(denominator) == 1:
        return [_number_power(numerator[0] / denominator[0], exponent)], [Fraction(1)]
    if exponent != int(exponent):
        raise _NotPolynomial()
    count = int(exponent)
    if count < 0:
        numerator, denominator, count = denominator, numerator, -count
    if max(len(numerator) - 1, len(denominator) - 1) * count > max_degree:
        raise _NotPolynomial()
    top: Polynomial = [Fraction(1)]
    bottom: Polynomial = [Fraction(1)]
    for _ in range(count):
        top = _product(top, numerator, max_degree)
        bottom = _product(bottom, denominator, max_degree)
    return top, bottom


def _number_power(base: Coefficient, exponent: Coefficient) -> Coefficient:
    """base^exponent for numbers: exact for a Fraction base and a whole exponent, else a real float."""
    if isinstance(base, Fraction) and isinstance(exponent, Fraction) and exponent.denominator == 1:
        if base == 0 and exponent < 0:
            raise _NotPolynomial()
        if abs(exponent.numerator) > MAX_EXACT_EXPONENT:
            raise _NotPolynomial()  # 10^10^10 must not exhaust memory
        return base**exponent.numerator
    if float(base) < 0 and float(exponent) != int(float(exponent)):
        raise _NotPolynomial()  # not real
    value = float(base) ** float(exponent)
    if isinstance(value, complex) or not math.isfinite(value):
        raise _NotPolynomial()
    return value


def _quotient(numerator: Polynomial, denominator: Polynomial) -> Polynomial:
    """numerator / denominator when the division leaves no remainder, else _NotPolynomial."""
    numerator, denominator = _trimmed(numerator), _trimmed(denominator)
    if denominator == [0]:
        raise _NotPolynomial()
    if len(denominator) == 1:
        return [coefficient / denominator[0] for coefficient in numerator]
    if len(numerator) < len(denominator):
        if numerator == [0]:
            return [Fraction(0)]
        raise _NotPolynomial()
    remainder = list(numerator)
    lead = denominator[-1]
    quotient: Polynomial = [Fraction(0)] * (len(numerator) - len(denominator) + 1)
    for shift in range(len(quotient) - 1, -1, -1):
        factor = remainder[shift + len(denominator) - 1] / lead
        quotient[shift] = factor
        for index, coefficient in enumerate(denominator):
            remainder[shift + index] = remainder[shift + index] - factor * coefficient
    size = max(abs(float(coefficient)) for coefficient in numerator)
    left = remainder[: len(denominator) - 1]
    exact = all(isinstance(value, Fraction) for value in left)
    if any(value != 0 if exact else abs(float(value)) > REMAINDER_TOLERANCE * size for value in left):
        raise _NotPolynomial()
    return _trimmed(quotient)
