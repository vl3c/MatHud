"""A regression battery for the numeric sign chart (utils.real_sets), as the browser runs it
without SymPy: every input the two reviews of the inequality solver found failing, and the
cases that were already right, each with its solution set written out.

Each case is (inequality, interval or None, expected set[, relative tolerance of its ends]).
The expected set is in interval notation; its ends may be expressions (sqrt(2), 3*pi/2).
ERROR means the sign chart must refuse (more than 40 intervals, or no reliable outer
interval). Every expected set must also pass verify_solution_set, which is how SymPy's
answer is checked ((x^2-1)/(x-1) >= 2 is x > 1, not refuted by the hole at 1).

real_sets_harness evaluates with compile_expression's Python functions where the browser
uses math.js; an answer that depends on the last ulp can differ between the two.
"""

from __future__ import annotations

import re
import warnings
from typing import List, Optional, Tuple, Union

import pytest

from server_tests import real_sets_harness as harness
from utils.inequality_solver import InequalitySolver
from utils.real_sets import verify_solution_set

ERROR = "error"

Case = Union[Tuple[str, Optional[str], str], Tuple[str, Optional[str], str, float]]

CASES: List[Case] = [
    # ----- The first review's failing inputs (rev126 on 9209f78)
    ("x*exp(x) < 0", None, "(-Infinity, 0)"),
    ("(x+1)*exp(x) < 0", None, "(-Infinity, -1)"),
    ("x*exp(-x) > 0", None, "(0, Infinity)"),
    ("exp(-x^2) > 0", None, "(-Infinity, Infinity)"),
    ("exp(x) > 0", None, "(-Infinity, Infinity)"),
    ("2^x > 0", None, "(-Infinity, Infinity)"),
    ("1/(1+exp(-x)) > 0", None, "(-Infinity, Infinity)"),
    ("(x-1)(x+2) < 0", None, "(-2, 1)"),
    ("(x-2)^2 (x+1) >= 0", None, "[-1, Infinity)"),
    ("|x-1|(x+1) > 0", None, "(-1, 1) ∪ (1, Infinity)"),
    ("(x+1)e^x < 0", None, "(-Infinity, -1)"),
    ("(x+1)x > 0", None, "(-Infinity, -1) ∪ (0, Infinity)"),
    ("(x+1)2 > 4", None, "(1, Infinity)"),
    ("sin(x)^2 + cos(x)^2 > 1", None, "{}"),
    ("sin(x)^2 + cos(x)^2 <= 1", "[0, 10]", "[0, 10]"),
    ("(x+1)^2 >= x^2 + 2x + 1", None, "(-Infinity, Infinity)"),
    ("cos(x) > 0", "[0, 2*pi)", "[0, pi/2) ∪ (3*pi/2, 2*pi)"),
    ("x^2 < 2", "(0, 5)", "(0, sqrt(2))"),
    ("x < 3", "(2, 5)", "(2, 3)"),
    ("x >= 1", "(1, 3)", "(1, 3)"),
    ("cos(6x) >= 1", "[0, pi]", "{0} ∪ {pi/3} ∪ {2*pi/3} ∪ {pi}"),
    ("sin(6x) >= 0", "[0, pi]", "[0, pi/6] ∪ [pi/3, pi/2] ∪ [2*pi/3, 5*pi/6] ∪ {pi}"),
    (
        "sin(6x) >= 0",
        None,
        "[-2*pi, -11*pi/6] ∪ [-5*pi/3, -3*pi/2] ∪ [-4*pi/3, -7*pi/6] ∪ [-pi, -5*pi/6] ∪ [-2*pi/3, -pi/2] ∪ [-pi/3, -pi/6] ∪ [0, pi/6] ∪ [pi/3, pi/2] ∪ [2*pi/3, 5*pi/6] ∪ [pi, 7*pi/6] ∪ [4*pi/3, 3*pi/2] ∪ [5*pi/3, 11*pi/6] ∪ {2*pi}",
    ),
    ("cos(x) >= 1", None, "{-2*pi} ∪ {0} ∪ {2*pi}"),
    ("cos(x) >= 1", "[0, 2*pi]", "{0} ∪ {2*pi}"),
    ("cos(x)^2 >= 1", "[0, 3*pi]", "{0} ∪ {pi} ∪ {2*pi} ∪ {3*pi}"),
    ("sqrt(2 - x^2) >= 0", None, "[-sqrt(2), sqrt(2)]"),
    ("sqrt(x^2 - 3) >= 0", None, "(-Infinity, -sqrt(3)] ∪ [sqrt(3), Infinity)"),
    ("x > 2000000", None, "(2000000, Infinity)"),
    ("x > 1e20", None, "(100000000000000000000, Infinity)"),
    ("x^2 - 5x + 6 < 0", "[-10000, 10000]", "(2, 3)"),
    ("(x-1)*(x-1.0000001) < 0", None, "(1, 1.0000001)"),
    ("x >= 1.23456789", None, "[1.23456789, Infinity)"),
    ("x > 0", "[1e-300, 1]", "[1e-300, 1]"),
    ("x < 3", "(1+1, 5)", "(2, 3)"),
    ("x < 3", "[-inf, 5]", "(-Infinity, 3)"),
    ("x^2 > 1e-12", None, "(-Infinity, -1e-06) ∪ (1e-06, Infinity)"),
    ("x^2 + 1 < 0", "[-2, 2]", "{}"),
    # ----- The second review's batteries (rev126b on e0a177e): its failing inputs B1-B3, S1,
    # tiny coefficients, and the probes that were already right
    ("(x-1)*(x-1.000001) < 0", None, "(1, 1.000001)"),
    ("(x-1)*(x-1.00000001) < 0", None, "(1, 1.00000001)"),
    ("(x-1)*(x-1.000000001) < 0", None, "(1, 1.000000001)"),
    ("(x-1000)*(x-1000.001) < 0", None, "(1000, 1000.001)"),
    ("(x-1000)*(x-1000.000001) < 0", None, "(1000, 1000.000001)"),
    ("x^2 < 1e-10", None, "(-1e-05, 1e-05)"),
    ("x^2 < 1e-20", None, "(-1e-10, 1e-10)"),
    ("x^2 <= 1e-30", None, "[-1e-15, 1e-15]"),
    ("1e-8*x > 1e-16", None, "(1e-08, Infinity)"),
    ("1e-8*x > 0", None, "(0, Infinity)"),
    ("1e-12*x^2 - 1e-12 < 0", None, "(-1, 1)"),
    ("1e12*x^2 - 1e12 < 0", None, "(-1, 1)"),
    ("1e-20*(x-3) > 0", None, "(3, Infinity)"),
    ("x^2 + 1e-10 > 0", None, "(-Infinity, Infinity)"),
    ("(x-1)^2 + 1e-10 > 0", None, "(-Infinity, Infinity)"),
    ("(x-1)^2 + 1e-12 > 0", None, "(-Infinity, Infinity)"),
    ("(x-1)^2 + 1e-7 > 0", None, "(-Infinity, Infinity)"),
    ("(x-1)^2 - 1e-20 < 0", None, "(0.9999999999, 1.0000000001)"),
    ("(x-1)^2 < 1e-12", None, "(0.999999, 1.000001)"),
    ("x^2 - 2x + 1.0000000001 > 0", None, "(-Infinity, Infinity)"),
    ("(x-2)/(x-2.0001) < 0", None, "(2, 2.0001)"),
    ("(x-2)/(x-2.000001) > 0", None, "(-Infinity, 2) ∪ (2.000001, Infinity)"),
    ("(x-1)/((x-1)^2 + 1e-10) > 0", None, "(1, Infinity)"),
    ("1/(x^2 + 1e-10) > 1e9", None, "(-3e-05, 3e-05)"),
    ("abs(x-1) + abs(x+1) <= 2", None, "[-1, 1]"),
    ("abs(x-1) + abs(x+1) > 2", None, "(-Infinity, -1) ∪ (1, Infinity)"),
    ("abs(x^2 - 4) <= 0", None, "{-2} ∪ {2}"),
    ("abs(x^2 - 4) > 0", None, "(-Infinity, -2) ∪ (-2, 2) ∪ (2, Infinity)"),
    ("abs(x-1)*abs(x-3) <= 0", None, "{1} ∪ {3}"),
    ("abs(sin(x)) <= 0", "[0, 10]", "{0} ∪ {pi} ∪ {2*pi} ∪ {3*pi}"),
    ("-1 <= x^2 - 2 <= 2", None, "[-2, -1] ∪ [1, 2]"),
    ("0 < 1/x < 1", None, "(1, Infinity)"),
    ("x < x^2 < 4", None, "(-2, 0) ∪ (1, 2)"),
    ("1 < 2x+1 <= 5", None, "(0, 2]"),
    ("x > -Infinity", None, "(-Infinity, Infinity)"),
    ("x < 5", "(-Infinity, 10]", "(-Infinity, 5)"),
    ("x^2 > 4", "[0, Infinity)", "(2, Infinity)"),
    ("x^2 > 4", "(-inf, 0]", "(-Infinity, -2)"),
    ("exp(x) > 2", "[0, oo)", "(log(2), Infinity)"),
    ("1/x > 0", "(-Infinity, Infinity)", "(0, Infinity)"),
    ("x^100 < 1", None, "(-1, 1)"),
    ("x^101 > 1", None, "(1, Infinity)"),
    ("x^4 - 1e-8 < 0", None, "(-0.01, 0.01)"),
    ("exp(-x) < 1e-300", None, "(300*log(10), Infinity)"),
    ("exp(x) < 1e-300", None, "(-Infinity, -(300*log(10)))"),
    ("x^3 - 3x^2 + 3x - 1 < 0", None, "(-Infinity, 1)"),
    ("x^3 - 3x^2 + 3x - 1 <= 0", None, "(-Infinity, 1]"),
    ("(x-1)^3 >= 0", None, "[1, Infinity)"),
    ("(x-1)^4 > 0", None, "(-Infinity, 1) ∪ (1, Infinity)"),
    ("(x-1)^4 <= 0", None, "{1}"),
    ("(x-1)^6 > 0", None, "(-Infinity, 1) ∪ (1, Infinity)"),
    ("(x - 1/3)^2 > 0", None, "(-Infinity, 1/3) ∪ (1/3, Infinity)"),
    ("(x - sqrt(2))^2 <= 0", None, "{sqrt(2)}"),
    ("(x - pi)^2 > 0", None, "(-Infinity, pi) ∪ (pi, Infinity)"),
    ("sin(x) >= 1", "[0, 10]", "{pi/2} ∪ {5*pi/2}"),
    ("sin(x) > 0.9999999", "[0, 3]", "(1.5703491132, 1.57124354039)"),
    # 1 - cos(x) is 1e-12 to within a few ulps of 1, so its end is good to about 1e-5 (relative)
    ("1 - cos(x) <= 1e-12", "[-1, 1]", "[-sqrt(2)*1e-6, sqrt(2)*1e-6]", 2e-5),
    ("1 - cos(x) < 1e-6", "[-1, 1]", "(-acos(1 - 1e-6), acos(1 - 1e-6))"),
    ("cos(x) - 1 >= 0", None, "{-2*pi} ∪ {0} ∪ {2*pi}"),
    ("tan(x) > 0", "[0, 2*pi]", "(0, pi/2) ∪ (pi, 3*pi/2)"),
    ("tan(x) >= 1", "(-pi/2, pi/2)", "[pi/4, pi/2)"),
    ("1/(x-1) + 1/(x-1.001) > 0", None, "(1, 1.0005) ∪ (1.001, Infinity)"),
    ("x*sin(1/x) > 0", "[0.1, 1]", "(0.106103295395, 0.159154943092) ∪ (0.318309886184, 1]"),
    ("log(x) > 0", None, "(1, Infinity)"),
    ("log(x, 10) > 2", None, "(100, Infinity)"),
    ("ln(x) < -700", None, "(0, exp(-700))"),
    ("x*ln(x) < 0", None, "(0, 1)"),
    ("x^x > 2", "(0, 5)", "(1.55961046946, 5)"),
    ("sqrt(x^2) >= x", None, "(-Infinity, Infinity)"),
    ("sqrt(x)*sqrt(x) >= x", None, "[0, Infinity)"),
    ("floor(x) >= 2.5", None, "[3, Infinity)"),
    ("x - floor(x) < 0.5", "[0, 3]", "[0, 1/2) ∪ [1, 3/2) ∪ [2, 5/2) ∪ {3}"),
    ("1e6*x^2 - 1 < 0", None, "(-0.001, 0.001)"),
    ("x^2 - 1e6 < 0", None, "(-1000, 1000)"),
    ("x^2 - 1e12 < 0", None, "(-1000000, 1000000)"),
    ("(x-1e6)^2 < 1", None, "(999999, 1000001)"),
    ("1e300*x > 1", None, "(1e-300, Infinity)"),
    ("x/1e300 > 1e-300", None, "(1, Infinity)"),
    ("2 <= floor(x) < 3", None, "[2, 3)"),
    ("2 <= floor(x) <= 2", None, "[2, 3)"),
    ("floor(x) <= 2", "[2, 5]", "[2, 3)"),
    ("ceil(x) <= 1", "[0, 3]", "[0, 1]"),
    ("(floor(x) - 2)^2 <= 0", None, "[2, 3)"),
    ("abs(floor(x) - 2) < 1", None, "[2, 3)"),
    ("floor(x) >= 2", "[0, 5]", "[2, 5]"),
    ("floor(x) < 2", "[0, 5]", "[0, 2)"),
    ("|x-1| + |x+1| <= 2", None, "[-1, 1]"),
    ("abs(x-1) + abs(x-3) <= 2", "[0, 5]", "[1, 3]"),
    ("abs(x) + abs(x-2) <= 2", "[1, 5]", "[1, 2]"),
    ("abs(x) + abs(x-2) > 2", "[1, 5]", "(2, 5]"),
    ("abs(x) - x <= 0", None, "[0, Infinity)"),
    ("abs(x-1) + abs(x+1) <= 2 + floor(x/100)", None, "[0, 1]"),
    ("6.67e-11*x > 1e-10", "[1.5, 3]", "[1.5, 3]"),
    ("6.67e-11*x >= 1e-10", "[1, 3]", "[1e-10/6.67e-11, 3]"),
    ("1e-10*x*(x-2) < 0", "[1, 3]", "[1, 2)"),
    ("1e-10*x*(x-2) <= 0", "[1, 3]", "[1, 2]"),
    ("1e-10*(x-2) < 0", "[1, 3]", "[1, 2)"),
    ("x^3 - 3x^2 + 3x - 1 > 0", None, "(1, Infinity)"),
    ("x^4 - 4x^3 + 6x^2 - 4x + 1 > 0", None, "(-Infinity, 1) ∪ (1, Infinity)"),
    ("x^2 - 2x + 1 > 0", None, "(-Infinity, 1) ∪ (1, Infinity)"),
    ("x^2 - 2x + 1 <= 0", None, "{1}"),
    ("x^2 - 6x + 9 <= 0", None, "{3}"),
    ("4x^2 - 12x + 9 > 0", None, "(-Infinity, 3/2) ∪ (3/2, Infinity)"),
    ("x^3 - x^2 - x + 1 <= 0", None, "(-Infinity, -1] ∪ {1}"),
    ("(x^2 - 1)^2 <= 0", None, "{-1} ∪ {1}"),
    ("sin(x) - x < 0", None, "(0, 2*pi]"),
    ("sin(x) < x", None, "(0, 2*pi]"),
    ("tan(x) > x", "(0, pi/2)", "(0, pi/2)"),
    ("cos(x) > x", None, "[-2*pi, 0.739085133215)"),
    ("x - sin(x) >= 0", None, "[0, 2*pi]"),
    ("1 - cos(x) > 0", "[-1, 1]", "[-1, 0) ∪ (0, 1]"),
    ("sin(x)^2 <= 0", "[0, 10]", "{0} ∪ {pi} ∪ {2*pi} ∪ {3*pi}"),
    ("x*exp(-x^2) > 0.1", None, "(0.101025848316, 1.67963061043)"),
    ("exp(x) > x + 1", None, "(-Infinity, 0) ∪ (0, Infinity)"),
    ("exp(x) >= x + 1", None, "(-Infinity, Infinity)"),
    ("log(x) <= x - 1", None, "(0, Infinity)"),
    ("log(x) < x - 1", None, "(0, 1) ∪ (1, Infinity)"),
    ("(x+1)/(x-1) > (x+2)/(x-2)", None, "(-Infinity, 0) ∪ (1, 2)"),
    ("x^2 >= 0", "[-1, 1]", "[-1, 1]"),
    ("x >= 0", "[0, 1]", "[0, 1]"),
    ("x > 0", "[0, 1]", "(0, 1]"),
    ("x < 1", "[0, 1]", "[0, 1)"),
    ("x^2 < 1", "[-1, 1]", "(-1, 1)"),
    ("sqrt(1 - x^2) > 0", None, "(-1, 1)"),
    ("sqrt(1 - x^2) >= 0", None, "[-1, 1]"),
    ("sqrt(4 - x^2) >= 1", None, "[-sqrt(3), sqrt(3)]"),
    ("1/sqrt(x-1) > 1", None, "(1, 2)"),
    ("x^(1/3) > 1", None, "(1, Infinity)"),
    ("cbrt(x) < 1", None, "(-Infinity, 1)"),
    ("(x^2-1)/(x-1) > 2", None, "(1, Infinity)"),
    ("(x^2-1)/(x-1) >= 2", None, "(1, Infinity)"),
    ("x/x > 0", None, "(-Infinity, 0) ∪ (0, Infinity)"),
    ("(x-1)/(x-1) >= 1", None, "(-Infinity, 1) ∪ (1, Infinity)"),
    ("(x^2-4)/(x-2) <= 4", None, "(-Infinity, 2)"),
    ("cos(x) <= 1 - x^2/2", None, "{0}"),
    (
        "sin(1/x) > 0",
        "[0.01, 1]",
        "(0.0102680608446, 0.0106103295395) ∪ (0.0109762029719, 0.0113682102208) ∪ (0.0117892550438, 0.0122426879301) ∪ (0.0127323954474, 0.0132629119243) ∪ (0.0138395602689, 0.0144686311902) ∪ (0.0151576136278, 0.0159154943092) ∪ (0.0167531519044, 0.0176838825658) ∪ (0.018724110952, 0.0198943678865) ∪ (0.0212206590789, 0.0227364204417) ∪ (0.0244853758603, 0.0265258238486) ∪ (0.0289372623803, 0.0318309886184) ∪ (0.0353677651315, 0.039788735773) ∪ (0.0454728408834, 0.0530516476973) ∪ (0.0636619772368, 0.0795774715459) ∪ (0.106103295395, 0.159154943092) ∪ (0.318309886184, 1]",
    ),
    (
        "sin(x^2) > 0",
        "[0, 10]",
        "(0, sqrt(pi)) ∪ (2.50662827463, 3.06998012384) ∪ (3.54490770181, 3.96332729761) ∪ (4.34160752735, 4.68947209983) ∪ (5.01325654926, 5.31736155272) ∪ (5.6049912164, 5.87856438167) ∪ (6.13996024768, 6.39067324283) ∪ (6.63191504396, 6.86468424648) ∪ (7.08981540362, 7.30801444382) ∪ (7.51988482389, 7.72594721819) ∪ (7.92665459521, 8.12240393759) ∪ (8.31354547585, 8.50039005179) ∪ (8.6832150547, 8.86226925453) ∪ (9.03777677271, 9.20994037152) ∪ (9.37894419967, 9.54495610017) ∪ (9.70812956278, 9.86860538583)",
    ),
    ("sin(x^2) > 0", "[0, 30]", ERROR),
    (
        "tan(x) > 0",
        "[0, 100]",
        "(0, pi/2) ∪ (pi, 3*pi/2) ∪ (2*pi, 5*pi/2) ∪ (3*pi, 7*pi/2) ∪ (4*pi, 9*pi/2) ∪ (5*pi, 11*pi/2) ∪ (6*pi, 13*pi/2) ∪ (7*pi, 15*pi/2) ∪ (8*pi, 17*pi/2) ∪ (9*pi, 19*pi/2) ∪ (10*pi, 21*pi/2) ∪ (11*pi, 23*pi/2) ∪ (12*pi, 25*pi/2) ∪ (13*pi, 27*pi/2) ∪ (14*pi, 29*pi/2) ∪ (15*pi, 31*pi/2) ∪ (16*pi, 33*pi/2) ∪ (17*pi, 35*pi/2) ∪ (18*pi, 37*pi/2) ∪ (19*pi, 39*pi/2) ∪ (20*pi, 41*pi/2) ∪ (21*pi, 43*pi/2) ∪ (22*pi, 45*pi/2) ∪ (23*pi, 47*pi/2) ∪ (24*pi, 49*pi/2) ∪ (25*pi, 51*pi/2) ∪ (26*pi, 53*pi/2) ∪ (27*pi, 55*pi/2) ∪ (28*pi, 57*pi/2) ∪ (29*pi, 59*pi/2) ∪ (30*pi, 61*pi/2) ∪ (31*pi, 63*pi/2)",
    ),
    (
        "sin(x) > 0",
        "[0, 120]",
        "(0, pi) ∪ (2*pi, 3*pi) ∪ (4*pi, 5*pi) ∪ (6*pi, 7*pi) ∪ (8*pi, 9*pi) ∪ (10*pi, 11*pi) ∪ (12*pi, 13*pi) ∪ (14*pi, 15*pi) ∪ (16*pi, 17*pi) ∪ (18*pi, 19*pi) ∪ (20*pi, 21*pi) ∪ (22*pi, 23*pi) ∪ (24*pi, 25*pi) ∪ (26*pi, 27*pi) ∪ (28*pi, 29*pi) ∪ (30*pi, 31*pi) ∪ (32*pi, 33*pi) ∪ (34*pi, 35*pi) ∪ (36*pi, 37*pi) ∪ (38*pi, 120]",
    ),
    ("sin(x) > 0", None, "(-2*pi, -pi) ∪ (0, pi)"),
    ("sin(2x) > cos(x)", None, "(-11*pi/6, -3*pi/2) ∪ (-7*pi/6, -pi/2) ∪ (pi/6, pi/2) ∪ (5*pi/6, 3*pi/2)"),
    ("sin(x) + cos(x) > 1", None, "(-2*pi, -3*pi/2) ∪ (0, pi/2)"),
    ("2sin(x)^2 - sin(x) - 1 < 0", "[0, 2*pi)", "[0, pi/2) ∪ (pi/2, 7*pi/6) ∪ (11*pi/6, 2*pi)"),
    ("sin(x) > 0.5", "[0, 4*pi]", "(pi/6, 5*pi/6) ∪ (13*pi/6, 17*pi/6)"),
    ("sin(x) >= 1/2", "(pi/6, 5*pi/6)", "(pi/6, 5*pi/6)"),
    ("cos(x)^2 > 1/4", "[0, pi]", "[0, pi/3) ∪ (2*pi/3, pi]"),
    ("abs(sin(x)) < 1/2", "[0, 2*pi]", "[0, pi/6) ∪ (5*pi/6, 7*pi/6) ∪ (11*pi/6, 2*pi]"),
    (
        "x*sin(x) > 1",
        "[0, 20]",
        "(1.11415714087, 2.77260470827) ∪ (6.43911723842, 9.31724294141) ∪ (12.6455325788, 15.6439973748) ∪ (18.9024837303, 20]",
    ),
    (
        "x^5 - 3x^3 + x - 0.1 > 0",
        None,
        "(-1.60931373204, -0.671337294182) ∪ (0.103294633999, 0.550949017763) ∪ (1.62640737446, Infinity)",
    ),
    ("(x^2 - 2)^3 * (x - 5)^2 >= 0", None, "(-Infinity, -sqrt(2)] ∪ [sqrt(2), Infinity)"),
    ("(x+2)*(x-3)/(x^2-1) <= 0", None, "[-2, -1) ∪ (1, 3]"),
    ("1/(x-1)^2 > 100", None, "(9/10, 1) ∪ (1, 11/10)"),
    ("x/(x^2-4) >= 0", None, "(-2, 0] ∪ (2, Infinity)"),
    ("x^2*exp(-x) > 0.1", None, "(-Infinity, -0.275530294694) ∪ (0.382965772684, 5.82789779587)"),
    ("ln(x+1) < x", None, "(-1, 0) ∪ (0, Infinity)"),
    ("sqrt(x) + sqrt(x-1) > 2", None, "(25/16, Infinity)"),
    ("x^(1/2) < 3", None, "[0, 9)"),
    ("ln(x+1) <= x", None, "(-1, Infinity)"),
    ("ln(1+x) >= x", None, "{0}"),
    ("cosh(x) > 1", None, "(-Infinity, 0) ∪ (0, Infinity)"),
    ("x*sin(x) > 0", "[-1, 1]", "[-1, 0) ∪ (0, 1]"),
    ("exp(x) - 1 - x > 0", None, "(-Infinity, 0) ∪ (0, Infinity)"),
    ("exp(x) > 1 + x + x^2/2", "[-1, 1]", "(0, 1]"),
    ("x^2 - sin(x)^2 > 0", "[-1, 1]", "[-1, 0) ∪ (0, 1]"),
    ("tan(x) > x", "(-pi/2, pi/2)", "(0, pi/2)"),
    ("sin(x)^2 < x^2", "[-2, 2]", "[-2, 0) ∪ (0, 2]"),
    # (sin(1/x) > 0 on [0.001, 1] is in KNOWN_WRONG)
    ("sin(x) > 0", "[0, 1e6]", ERROR),
    ("floor(x) > x/2", "[0, 100]", "[1, 100]"),
    (
        "sin(x)*cos(10x) > 0.1",
        "[0, 10]",
        "(0.492547160462, 0.770998189182) ∪ (1.11074109317, 1.40355773681) ∪ (1.73803491678, 2.03085156042) ∪ (2.37059446441, 2.64904549313) ∪ (3.34930010411, 3.58953247834) ∪ (3.94098476207, 4.2298369078) ∪ (4.56543560144, 4.85934235933) ∪ (5.19494105297, 5.4837931987) ∪ (5.83524548242, 6.07547785666) ∪ (6.77573246764, 7.05418349636) ∪ (7.39392640035, 7.68674304399) ∪ (8.02122022396, 8.3140368676) ∪ (8.65377977159, 8.93223080031) ∪ (9.63248541129, 9.87271778552)",
    ),
    ("abs(sin(x)) > 0", None, "(-2*pi, -pi) ∪ (-pi, 0) ∪ (0, pi) ∪ (pi, 2*pi)"),
    ("1/sin(x) > 1", "[0, 2*pi]", "(0, pi/2) ∪ (pi/2, pi)"),
    ("tan(x)^2 < 3", "[0, pi]", "[0, pi/3) ∪ (2*pi/3, pi]"),
    ("sin(sin(sin(x))) > 0.5", "[0, 2*pi]", "(0.583645464044, 2.55794718955)"),
    ("x^x^x > 2", "(0, 3)", "(1.47668433736, 3)"),
    ("gamma(x) > 1", "(0, 5)", "(0, 1) ∪ (2, 5)"),
    (
        "-1 < sin(x) + cos(3x) - x/10 < 0.5",
        "[-10, 10]",
        "(-9.68014636407, -8.84717742312) ∪ (-8.09496791957, -7.5437015024) ∪ (-7.20112957759, -6.72171734818) ∪ (-5.46250057217, -5.11257854947) ∪ (-3.57238673753, -1.51950562043) ∪ (-0.676142830259, -0.253175566699) ∪ (0.500073473784, 1.4524029028) ∪ (2.55135774061, 2.97130987564) ∪ (3.59838002955, 4.55897916508) ∪ (5.77973911199, 7.95581194799) ∪ (8.65621132154, 9.05786812348)",
    ),
    # ----- The first review's battery: mostly cases already right at 9209f78
    ("(x-1)^2*(x-2) < 0", None, "(-Infinity, 1) ∪ (1, 2)"),
    ("(x-1)^2*(x-2) <= 0", None, "(-Infinity, 2]"),
    ("(x-1)^2*(x-2) >= 0", None, "{1} ∪ [2, Infinity)"),
    ("x^3 - 2x^2 + x > 0", None, "(0, 1) ∪ (1, Infinity)"),
    ("x^4 - 2x^2 + 1 > 0", None, "(-Infinity, -1) ∪ (-1, 1) ∪ (1, Infinity)"),
    ("x^4 - 2x^2 + 1 <= 0", None, "{-1} ∪ {1}"),
    ("(x^2 - 4)/(x - 1) <= 0", None, "(-Infinity, -2] ∪ (1, 2]"),
    ("(x+3)/(x^2 - 9) > 0", None, "(3, Infinity)"),
    ("(x+3)/(x^2 - 9) >= 0", None, "(3, Infinity)"),
    ("x/(x^2 + 1) > 1/3", None, "((3-sqrt(5))/2, (3+sqrt(5))/2)"),
    ("(2x - 1)/(x + 3) <= 1", None, "(-3, 4]"),
    ("1/(x^2 - 1) < 0", None, "(-1, 1)"),
    ("abs(2x - 3) <= 5", None, "[-1, 4]"),
    ("abs(x) + abs(x - 2) > 3", None, "(-Infinity, -1/2) ∪ (5/2, Infinity)"),
    ("abs(x^2 - 1) > x + 1", None, "(-Infinity, -1) ∪ (-1, 0) ∪ (2, Infinity)"),
    ("abs(x - 1) < abs(x + 1)", None, "(0, Infinity)"),
    ("sqrt(x + 4) > x - 2", None, "[-4, 5)"),
    ("sqrt(x) < x", None, "(1, Infinity)"),
    ("sqrt(1 - x) >= x", None, "(-Infinity, (sqrt(5)-1)/2]"),
    ("ln(x) > ln(2 - x)", None, "(1, 2)"),
    ("ln(x^2 - 1) > 0", None, "(-Infinity, -sqrt(2)) ∪ (sqrt(2), Infinity)"),
    ("log(x, 2) <= 3", None, "(0, 8]"),
    ("exp(x) < 3", None, "(-Infinity, log(3))"),
    ("exp(2x) - 3exp(x) + 2 < 0", None, "(0, log(2))"),
    ("2^x < 8", None, "(-Infinity, 3)"),
    ("x^(2) > 3x", None, "(-Infinity, 0) ∪ (3, Infinity)"),
    ("cos(x) < 1/2", "[0, 2*pi]", "(pi/3, 5*pi/3)"),
    ("cos(x) <= -1", "[0, 4*pi]", "{pi} ∪ {3*pi}"),
    ("sin(x) <= 0", "[0, 2*pi]", "{0} ∪ [pi, 2*pi]"),
    ("sin(x) < 0", "(0, 2*pi)", "(pi, 2*pi)"),
    ("sin(2x) >= 1/2", "[0, pi]", "[pi/12, 5*pi/12]"),
    ("cos(2x) > 0", None, "[-2*pi, -7*pi/4) ∪ (-5*pi/4, -3*pi/4) ∪ (-pi/4, pi/4) ∪ (3*pi/4, 5*pi/4) ∪ (7*pi/4, 2*pi]"),
    ("tan(x) < 0", "[0, pi]", "(pi/2, pi)"),
    ("tan(x) >= -1", "(-pi/2, pi/2)", "[-pi/4, pi/2)"),
    ("sin(x) > cos(x)", "[0, 2*pi]", "(pi/4, 5*pi/4)"),
    ("sin(x) + cos(x) >= 1", "[0, 2*pi]", "[0, pi/2] ∪ {2*pi}"),
    ("2cos(x)^2 - 1 < 0", "[0, 2*pi]", "(pi/4, 3*pi/4) ∪ (5*pi/4, 7*pi/4)"),
    (
        "sin(x)^2 < 1/2",
        None,
        "[-2*pi, -7*pi/4) ∪ (-5*pi/4, -3*pi/4) ∪ (-pi/4, pi/4) ∪ (3*pi/4, 5*pi/4) ∪ (7*pi/4, 2*pi]",
    ),
    ("cos(x) >= 0", "[-pi, 3*pi]", "[-pi/2, pi/2] ∪ [3*pi/2, 5*pi/2]"),
    ("sin(x/2) > 0", "[0, 4*pi]", "(0, 2*pi)"),
    ("sin(x) >= -1/2", None, "[-2*pi, -5*pi/6] ∪ [-pi/6, 7*pi/6] ∪ [11*pi/6, 2*pi]"),
    ("sin(3x) > 0", "[0, pi]", "(0, pi/3) ∪ (2*pi/3, pi)"),
    ("1 <= 3 - 2x < 5", None, "(-1, 1]"),
    ("-2 < 1/x <= 1", None, "(-Infinity, -1/2) ∪ [1, Infinity)"),
    ("0 <= x^2 - 4 <= 5", None, "[-3, -2] ∪ [2, 3]"),
    ("x^2 < x < 2", None, "(0, 1)"),
    ("x^2 - 5x + 6 < 0", None, "(2, 3)"),
    (
        "2sin(x)^2 - sin(x) - 1 < 0",
        None,
        "[-2*pi, -3*pi/2) ∪ (-3*pi/2, -5*pi/6) ∪ (-pi/6, pi/2) ∪ (pi/2, 7*pi/6) ∪ (11*pi/6, 2*pi]",
    ),
    ("sin(x) > 1/2", "[0, 400]", ERROR),
    ("cos(x) <= 0", "[-1000, 1000]", ERROR),
    ("sin(x) > 0", "[0, Infinity)", ERROR),
    ("2^x > x^2", None, "(-0.766664695962, 2) ∪ (4, Infinity)"),
    ("x*sin(x) > 0", None, "(-pi, 0) ∪ (0, pi)"),
    ("sin(x) + x > 0", None, "(0, 2*pi]"),
    ("floor(x) > 1", None, "[2, Infinity)"),
    ("x^3 - 3x + 1 > 0", None, "(-1.87938524157, 0.347296355334) ∪ (1.53208888624, Infinity)"),
    ("1/(x-1)^2 > 0", None, "(-Infinity, 1) ∪ (1, Infinity)"),
    ("(x^2-1)/(x-1) > 0", None, "(-1, 1) ∪ (1, Infinity)"),
    ("sin(x) >= 1", None, "{-3*pi/2} ∪ {pi/2}"),
    ("cos(x) >= 1", "[0, 2*pi)", "{0}"),
    ("tan(x) > 1", None, "(-7*pi/4, -3*pi/2) ∪ (-3*pi/4, -pi/2) ∪ (pi/4, pi/2) ∪ (5*pi/4, 3*pi/2)"),
    ("x > 1e-20", None, "(1e-20, Infinity)"),
    ("x > 0.1", None, "(1/10, Infinity)"),
    ("0.1x > 0.3", None, "(3, Infinity)"),
    ("x^2 < 2", None, "(-sqrt(2), sqrt(2))"),
    ("exp(x) > 2", None, "(log(2), Infinity)"),
    ("x^3 + x + 1 > 0", None, "(-0.682327803828, Infinity)"),
    ("sqrt(x - 1) >= x - 3", None, "[1, 5]"),
    ("ln(x - 1) <= 0", None, "(1, 2]"),
    ("log(x) > 1", None, "(e, Infinity)"),
    ("log(x, 10) > 1", None, "(10, Infinity)"),
    ("log10(x) > 1", None, "(10, Infinity)"),
    ("|x^2 - 4| < 3", None, "(-sqrt(7), -1) ∪ (1, sqrt(7))"),
    ("x(x-1) > 0", None, "(-Infinity, 0) ∪ (1, Infinity)"),
    ("sin(pi*x) > 0", "[0, 200]", ERROR),
    ("x^2 > 1", "(0, 5]", "(1, 5]"),
    ("tan(3x) > 1", "[0, pi]", "(pi/12, pi/6) ∪ (5*pi/12, pi/2) ∪ (3*pi/4, 5*pi/6)"),
    ("(x-1)^2/(x+1) >= 0", None, "(-1, Infinity)"),
    ("|x-1| + |x+1| < 4", None, "(-2, 2)"),
    ("|x| > x", None, "(-Infinity, 0)"),
    ("1/x < 1/(x-1)", None, "(-Infinity, 0) ∪ (1, Infinity)"),
    ("(x^2+1)/(x^2-4) <= 0", None, "(-2, 2)"),
    ("cbrt(x) > -1", None, "(-1, Infinity)"),
    ("sqrt(x^2) > 1", None, "(-Infinity, -1) ∪ (1, Infinity)"),
    ("x^x > 2", None, "(1.55961046946, Infinity)"),
    ("-1 <= sin(x) <= 1", None, "[-2*pi, 2*pi]"),
    ("0 < x^2 - 1 < 3", None, "(-2, -1) ∪ (1, 2)"),
    ("1 < 2x + 1 <= 5", "[0, 1]", "(0, 1]"),
    ("x > 2.5", None, "(5/2, Infinity)"),
    ("x^2 < 2.25", None, "(-3/2, 3/2)"),
    ("x^2 - 2x + 1 < 0", None, "{}"),
    ("(x-1)^3 > 0", None, "(1, Infinity)"),
    ("x^4 > 0", None, "(-Infinity, 0) ∪ (0, Infinity)"),
    ("x^2 >= 4", "[-2, 2]", "{-2} ∪ {2}"),
    ("x^2 > 4", "[-2, 2]", "{}"),
    ("x^2 > 4", "(-2, 2)", "{}"),
    ("tan(x) >= 0", "[0, pi]", "[0, pi/2) ∪ {pi}"),
    ("1/(x-2) >= 0", "[0, 4]", "(2, 4]"),
    ("1/(x-2) <= 0", "[2, 4]", "{}"),
    ("x < 3", "[0, 2pi]", "[0, 3)"),
    ("x < 3", "0, 5", "[0, 3)"),
    ("sin(x) > 0", "[0, 1e9]", ERROR),
    ("exp(x) > 1e300", None, "(300*log(10), Infinity)"),
    ("exp(-x^2) > 0.5", None, "(-sqrt(log(2)), sqrt(log(2)))"),
    ("e^x > 0", None, "(-Infinity, Infinity)"),
    ("x > pi", None, "(pi, Infinity)"),
    ("x > -pi/2", "[-2pi, 2pi]", "(-pi/2, 2*pi]"),
    ("log(x^2) < 0", None, "(-1, 0) ∪ (0, 1)"),
    ("ln(abs(x)) > 0", None, "(-Infinity, -1) ∪ (1, Infinity)"),
    ("1/ln(x) > 0", None, "(1, Infinity)"),
    ("sec(x) > 2", None, "(-5*pi/3, -3*pi/2) ∪ (-pi/2, -pi/3) ∪ (pi/3, pi/2) ∪ (3*pi/2, 5*pi/3)"),
    ("2(x+1) > 4", None, "(1, Infinity)"),
    ("x^2(x-1) > 0", None, "(1, Infinity)"),
    ("1/x^2 > 0", None, "(-Infinity, 0) ∪ (0, Infinity)"),
    ("x^2 + exp(x) > 0", None, "(-Infinity, Infinity)"),
    ("e^x >= 1", None, "[0, Infinity)"),
    ("exp(x) < 1", None, "(-Infinity, 0)"),
    ("x < 3", "[1+1, 5]", "[2, 3)"),
    ("sin(x) > 1/2", "(pi/6, 5pi/6)", "(pi/6, 5*pi/6)"),
    ("sin(x) >= 1/2", "[pi/6, 5*pi/6]", "[pi/6, 5*pi/6]"),
    ("sin(x) > 1/2", None, "(-11*pi/6, -7*pi/6) ∪ (pi/6, 5*pi/6)"),
    ("sin(x) > 1/2", "[0, 2*pi)", "(pi/6, 5*pi/6)"),
    ("x^2 <= 1e-10", None, "[-1e-05, 1e-05]"),
    ("(x-1)^2*(x-3) <= 0", None, "(-Infinity, 3]"),
    ("x^3 >= 0", None, "[0, Infinity)"),
    ("1/(x-1) > 0", None, "(1, Infinity)"),
    ("1/(x-1) >= 0", None, "(1, Infinity)"),
    ("tan(x) > 0", None, "(-2*pi, -3*pi/2) ∪ (-pi, -pi/2) ∪ (0, pi/2) ∪ (pi, 3*pi/2)"),
    ("tan(x) > 0", "[0, 2pi]", "(0, pi/2) ∪ (pi, 3*pi/2)"),
    ("abs(x - 0.1) < 0.05", None, "(0.05, 0.15)"),
    ("x^2 - 0.04 < 0", None, "(-1/5, 1/5)"),
    ("x^2 > 1/9", None, "(-Infinity, -1/3) ∪ (1/3, Infinity)"),
    ("x > 1/7", None, "(1/7, Infinity)"),
    ("x > sqrt(2)", None, "(sqrt(2), Infinity)"),
    ("x^2 - 3 < 0", None, "(-sqrt(3), sqrt(3))"),
    ("x^2 - 2x - 1 < 0", None, "(-(sqrt(2)-1), 1+sqrt(2))"),
    ("(x-1)(x-1.0000001) < 0", None, "(1, 1.0000001)"),
    ("sin(x)^2 + cos(x)^2 >= 1", None, "[-2*pi, 2*pi]"),
    ("(x+1)^2 > x^2 + 2x + 1", None, "{}"),
    ("sqrt(x)^2 <= x", None, "[0, Infinity)"),
    ("x > 1", "", "(1, Infinity)"),
    ("x^2 < 4", "[0, 2π]", "[0, 2)"),
    ("x² < 4", None, "(-2, 2)"),
    ("x ≤ 2", None, "(-Infinity, 2]"),
    ("2 ≥ x", None, "(-Infinity, 2]"),
    ("x =< 2", None, "(-Infinity, 2]"),
    ("x => 2", None, "[2, Infinity)"),
    ("x<-1", None, "(-Infinity, -1)"),
    ("x>-1", None, "(-1, Infinity)"),
    ("x>=-1", None, "[-1, Infinity)"),
    ("-x > 1", None, "(-Infinity, -1)"),
    ("x^-1 > 1", None, "(0, 1)"),
    ("(1 - x^2)*exp(-x^2) > 0", None, "(-1, 1)"),
    ("exp(-x) > 0", None, "(-Infinity, Infinity)"),
    ("sqrt(x^2 - 2) >= 0", None, "(-Infinity, -sqrt(2)] ∪ [sqrt(2), Infinity)"),
    ("sqrt(3 - x^2) <= 1", None, "[-sqrt(3), -sqrt(2)] ∪ [sqrt(2), sqrt(3)]"),
    ("sqrt(5 - x^2) >= 0", None, "[-sqrt(5), sqrt(5)]"),
    ("x*exp(x) > 1", None, "(0.56714329041, Infinity)"),
    ("asin(x) > 0", None, "(0, 1]"),
    ("acos(x) <= pi/2", None, "[0, 1]"),
    ("sqrt(x) + sqrt(2 - x) >= 0", None, "[0, 2]"),
    ("1/x > 1e-7", None, "(0, 10000000)"),
    ("x^2 - 3 >= 0", None, "(-Infinity, -sqrt(3)] ∪ [sqrt(3), Infinity)"),
    ("(x-500)*(x-501) < 0", None, "(500, 501)"),
    ("(x-3)^2 < 1", "[-1e9, 1e9]", "(2, 4)"),
    ("sin^2(x) < 1/4", "[0, pi]", "[0, pi/6) ∪ (5*pi/6, pi]"),
    ("√x > 2", None, "(4, Infinity)"),
    ("|x| < 2", None, "(-2, 2)"),
    ("x·(x-1) > 0", None, "(-Infinity, 0) ∪ (1, Infinity)"),
    ("2|x| > 1", None, "(-Infinity, -1/2) ∪ (1/2, Infinity)"),
    ("3 > x > 1", None, "(1, 3)"),
    ("x² - 5x + 6 ≤ 0", None, "[2, 3]"),
]

# Wrong answers the sign chart still gives, kept so that a fix shows up (strict xfail)
KNOWN_WRONG: List[Case] = [
    # sin(1/x) turns faster than the grid samples near 0.001 (about 160 intervals): the
    # answer should be the 40-interval error, but the grid misses most crossings
    ("sin(1/x) > 0", "[0.001, 1]", ERROR),
]


def _ids(cases: List[Case]) -> List[str]:
    return [f"{case[0]} on {case[1]}" if case[1] else case[0] for case in cases]


def _check(case: Case) -> None:
    text, interval, expected = case[0], case[1], case[2]
    tolerance = case[3] if len(case) > 3 else 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pieces, answer = harness.solve(text, interval)
    if expected == ERROR:
        assert pieces is None, answer
        return
    assert pieces is not None, answer
    assert harness.same_set(pieces, harness.parse_set(expected), tolerance), f"{answer}; expected {expected}"


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_sign_chart_finds_the_set(case: Case) -> None:
    _check(case)


@pytest.mark.parametrize("case", KNOWN_WRONG, ids=_ids(KNOWN_WRONG))
@pytest.mark.xfail(strict=True, reason="a known wrong answer")
def test_known_wrong_answers(case: Case) -> None:
    _check(case)


@pytest.mark.parametrize(
    "case", [case for case in CASES if case[2] != ERROR], ids=_ids([c for c in CASES if c[2] != ERROR])
)
def test_the_set_passes_the_check(case: Case) -> None:
    """The check SymPy's answers go through accepts the right set: the expected one, or where
    its ends are 12-digit decimals, the sign chart's (equal to them, with every digit)."""
    text, interval, expected = case[0], case[1], case[2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        relations, window, seeds, _ = harness.prepared(text, interval)
        if re.search(r"\d\.\d{6}", expected):
            pieces, _ = harness.solve(text, interval)
            assert pieces is not None
        else:
            pieces = harness.pieces_of(harness.parse_set(expected))
        assert verify_solution_set(relations, pieces, window, seeds) != "refuted"


def test_balanced_sides_put_each_term_where_it_is_added() -> None:
    """The sides whose magnitude measures the rounding of lhs - rhs (x^3 - 2x^2 + x near 1)."""
    assert InequalitySolver._balanced_sides("x^3 - 2x^2 + x", "0") == ("(x^3) + (x)", "(2x^2) + (0)")
    assert InequalitySolver._balanced_sides("exp(2x) - 3exp(x) + 2", "0") == ("(exp(2x)) + (2)", "(3exp(x)) + (0)")
    assert InequalitySolver._balanced_sides("1", "x - 2") == ("(1) + (2)", "(x)")
    assert InequalitySolver._balanced_sides("x^2 + 1", "2x") is None
