"""Plot correctness: function curves drawn by FunctionRenderable match the true curve.

Each case is drawn on an 800 x 600 view with the analysis lists the client gives for it
(asymptotes, point discontinuities, corners, periodicity) and compared with the true curve
by plot_checker: no drawn line farther than 2 px from the curve (connectors across jumps
and poles, false waves) and no stretch of the curve left undrawn (branches cut short).
"""

from __future__ import annotations

from server_tests.client_renderer import plot_checker  # noqa: F401  (path setup, browser stub)

import math
import unittest
from typing import Callable, Dict, List, Optional, Tuple

from server_tests.client_renderer.plot_checker import (
    CurveModel,
    View,
    draw,
    periodic_asymptotes,
    plot_errors,
    safe,
)

from rendering.renderables.adaptive_sampler import AdaptiveSampler

STANDARD: View = (-10.0, 10.0, 7.5, -7.5)
HALF_PI = math.pi / 2
TAN_POLES = periodic_asymptotes(HALF_PI, math.pi)


def _case(
    f: Callable[[float], float],
    view: View = STANDARD,
    asymptotes: Optional[List[float]] = None,
    discontinuities: Optional[List[float]] = None,
    corners: Optional[List[float]] = None,
    period: Optional[float] = None,
) -> Tuple[Callable[[float], float], View, CurveModel]:
    model = CurveModel(
        safe(f),
        vertical_asymptotes=asymptotes or [],
        point_discontinuities=discontinuities or [],
        corner_points=corners or [],
        is_periodic=period is not None,
        estimated_period=period,
    )
    return f, view, model


# Analysis lists and periodicity flags as the client computes them for each expression.
CASES: Dict[str, Tuple[Callable[[float], float], View, CurveModel]] = {
    # Jumps the expression analysis does not list
    "floor(x)": _case(math.floor, period=1.0),
    "round(x)": _case(lambda x: math.floor(x + 0.5), period=1.0),
    "floor(x+0.5)": _case(lambda x: math.floor(x + 0.5), period=1.0),
    "mod(x, 2)": _case(lambda x: x % 2, period=2.0),
    "sign(x)": _case(lambda x: (x > 0) - (x < 0)),
    # Poles, listed or not
    "tan(x)": _case(math.tan, asymptotes=TAN_POLES, period=1.1111111111111112),
    "tan(x) on [-100, 100]": _case(math.tan, (-100.0, 100.0, 75.0, -75.0), TAN_POLES, period=1.1111111111111112),
    "tan(10x)": _case(
        lambda x: math.tan(10 * x), asymptotes=periodic_asymptotes(HALF_PI / 10, math.pi / 10), period=1.0
    ),
    "cot(x) (poles not listed)": _case(lambda x: math.cos(x) / math.sin(x), period=1.25),
    "1/cos(x)": _case(lambda x: 1 / math.cos(x), asymptotes=periodic_asymptotes(HALF_PI, math.pi, 785), period=1.0),
    "1/x": _case(lambda x: 1 / x, asymptotes=[0.0], period=2.5),
    "1/(x-0.3)": _case(lambda x: 1 / (x - 0.3), asymptotes=[0.3], period=2.0),
    "x/(x^2-4)": _case(lambda x: x / (x * x - 4), asymptotes=[-2.0, 2.0], period=2.5),
    "1/((x-1)(x-1.001))": _case(lambda x: 1 / ((x - 1) * (x - 1.001)), asymptotes=[1.0, 1.001], period=2.5),
    "exp(1/x)": _case(lambda x: math.exp(1 / x), asymptotes=[0.0], period=2.857142857142857),
    "sin(x)/x^2": _case(lambda x: math.sin(x) / (x * x), asymptotes=[0.0], period=2.5),
    "(x^2-1)/(x-1)": _case(lambda x: (x * x - 1) / (x - 1), discontinuities=[1.0]),
    # Steep but continuous: one path, no break
    "tanh(100x)": _case(lambda x: math.tanh(100 * x)),
    "atan(1000x)": _case(lambda x: math.atan(1000 * x)),
    "1/(1+exp(-50x))": _case(lambda x: 1 / (1 + math.exp(-50 * x))),
    "x^10": _case(lambda x: x**10, period=1.0),
    "cbrt(x)": _case(lambda x: math.copysign(abs(x) ** (1 / 3), x)),
    "1000(x-1.234)": _case(lambda x: 1000 * (x - 1.234)),
    "10000(x-0.3)": _case(lambda x: 10000 * (x - 0.3)),
    # Corners and domain edges
    "abs(x)": _case(abs, corners=[0.0]),
    "abs(x-1)+abs(x+2)": _case(lambda x: abs(x - 1) + abs(x + 2), corners=[-2.0, 1.0]),
    "sqrt(abs(x)) zoomed": _case(lambda x: math.sqrt(abs(x)), (-1e-4, 1e-4, 7.5e-5, -7.5e-5), corners=[0.0]),
    "sqrt(1-x^2)": _case(lambda x: math.sqrt(1 - x * x), (-2.0, 2.0, 1.5, -1.5)),
    "sqrt(x) zoomed": _case(math.sqrt, (-1e-3, 1e-3, 7.5e-4, -7.5e-4)),
    "ln(x)": _case(math.log, asymptotes=[0.0]),
    "x*ln(x)": _case(lambda x: x * math.log(x), (-1.0, 1.0, 0.75, -0.75), asymptotes=[0.0]),
    # Fast oscillation: refined evenly across the view
    "sin(20x)": _case(lambda x: math.sin(20 * x), period=1.0),
    "sin(25x)": _case(lambda x: math.sin(25 * x)),
}


class TestPlotCorrectness(unittest.TestCase):
    def test_curves_match_the_true_curve(self) -> None:
        for name, (f, view, model) in CASES.items():
            with self.subTest(name):
                spurious, missing = plot_errors(model, f, view)
                self.assertLessEqual(spurious, 2.0, f"{name}: {spurious} px drawn off the curve")
                self.assertLessEqual(missing, 2.0, f"{name}: {missing} px of the curve not drawn")

    def test_steep_line_crossing_the_view_between_two_samples_is_drawn(self) -> None:
        _, view, model = CASES["1000(x-1.234)"]
        paths = draw(model, view)
        self.assertEqual(len(paths), 1)
        y1, y2 = paths[0][0][1], paths[0][-1][1]
        self.assertLess(min(y1, y2), 0.0)
        self.assertGreater(max(y1, y2), 600.0)

    def test_corner_is_a_sample_and_not_a_break(self) -> None:
        _, view, model = _case(lambda x: abs(x - 0.123), corners=[0.123])
        paths = draw(model, view)
        self.assertEqual(len(paths), 1)
        corner_sy = max(y for _, y in paths[0])
        self.assertAlmostEqual(corner_sy, 300.0, places=6)


def _screen(scale: float = 40.0) -> Callable[[float, float], Tuple[float, float]]:
    """1 math unit = scale px, origin at (400, 300)."""
    return lambda x, y: (400 + x * scale, 300 - y * scale)


BAND = (-300.0, 900.0)


def _breaks(f: Callable[[float], float], left: float = -10.0, right: float = 10.0) -> List[Tuple[float, float]]:
    found: List[Tuple[float, float]] = []
    AdaptiveSampler.generate_samples(left, right, safe(f), _screen(), max_samples=800, viewport_band=BAND, breaks=found)
    return found


class TestJumpDetection(unittest.TestCase):
    def assert_breaks_at(self, f: Callable[[float], float], expected: List[float]) -> None:
        found = _breaks(f)
        self.assertEqual(len(found), len(expected), found)
        for (x_left, x_right), x in zip(sorted(found), expected):
            self.assertLessEqual(x_left, x)
            self.assertGreaterEqual(x_right, x)
            self.assertLess(x_right - x_left, 1e-4)  # 0.004 px

    def test_floor_breaks_at_each_integer(self) -> None:
        # The last pair ends on x = 10, where floor jumps too
        self.assert_breaks_at(math.floor, [float(k) for k in range(-9, 11)])

    def test_sign_breaks_at_zero(self) -> None:
        found = _breaks(lambda x: (x > 0) - (x < 0))
        self.assertTrue(found)
        self.assertTrue(all(abs(a) < 1e-4 and abs(b) < 1e-4 for a, b in found), found)

    def test_mod_breaks_at_each_multiple(self) -> None:
        self.assert_breaks_at(lambda x: x % 2, [float(k) for k in range(-8, 11, 2)])

    def test_unlisted_poles_break(self) -> None:
        self.assert_breaks_at(lambda x: 1 / (x - 0.3), [0.3])
        self.assert_breaks_at(math.tan, [HALF_PI + k * math.pi for k in range(-3, 3)])

    def test_steep_continuous_curves_do_not_break(self) -> None:
        curves = {
            "atan(1000x)": lambda x: math.atan(1000 * x),
            "tanh(100x)": lambda x: math.tanh(100 * x),
            "1000x": lambda x: 1000 * x,
            "x^10": lambda x: x**10,
            "cbrt(x)": lambda x: math.copysign(abs(x) ** (1 / 3), x),
            "sin(25x)": lambda x: math.sin(25 * x),
        }
        for name, f in curves.items():
            with self.subTest(name):
                self.assertEqual(_breaks(f), [])

    def test_cbrt_zoomed_in_does_not_break(self) -> None:
        found: List[Tuple[float, float]] = []
        AdaptiveSampler.generate_samples(
            -1e-4,
            1e-4,
            safe(lambda x: math.copysign(abs(x) ** (1 / 3), x)),
            _screen(4e6),
            max_samples=800,
            viewport_band=BAND,
            breaks=found,
        )
        self.assertEqual(found, [])

    def test_pairs_both_beyond_the_band_are_not_tested(self) -> None:
        # floor(x) + 100 is far above the band: nothing is drawn, so nothing is tested
        self.assertEqual(_breaks(lambda x: math.floor(x) + 100), [])


class TestBreadthFirstRefinement(unittest.TestCase):
    def test_fast_oscillation_is_refined_alike_across_the_range(self) -> None:
        samples, _ = AdaptiveSampler.generate_samples(
            -10, 10, lambda x: math.sin(50 * x), _screen(), max_samples=800, viewport_band=BAND
        )
        left = sum(1 for x in samples if x < 0)
        right = sum(1 for x in samples if x > 0)
        self.assertLess(abs(left - right), 0.1 * len(samples))

    def test_worst_first_takes_the_worst_and_thins_a_group_evenly(self) -> None:
        point = (0.0, 0.0)
        level = [(float(i), i + 1.0, point, point, 3, 1.0) for i in range(10)]
        level[7] = (7.0, 8.0, point, point, 3, 5.0)
        chosen = AdaptiveSampler._worst_first(level, 4)
        self.assertEqual(chosen[0][0], 7.0)
        self.assertEqual([item[0] for item in chosen[1:]], [0.0, 3.0, 6.0])


if __name__ == "__main__":
    unittest.main()
