"""Pure Python tests for arc length (static/client/utils/arc_length.py, utils/arc_length_tool.py
and managers/plotted_curves.py).

Lengths are checked against references computed with mpmath (the speed integrated to 20
digits), on ranges whose sharp points fall between the start grids' nodes as well as on them;
every result's error estimate must cover its true error. Refusals cover breaks (jumps and
vertical asymptotes, also poles that do not change sign), undefined points, unclear input and
curves traced more than once. The Brython suite (client_tests/test_arc_length.py) runs the
tool through FunctionRegistry and measures plotted curves by name.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import mpmath

from server_tests import client_renderer  # noqa: F401  (browser stub, so MathUtils resolves)
from managers.plotted_curves import resolve_plotted_curve
from utils.arc_length import arc_length, arc_length_in_pieces, traced_times
from utils.arc_length_tool import arc_length_tool

mpmath.mp.dps = 25


def _reference(x: Callable[[Any], Any], y: Callable[[Any], Any], a: float, b: float, cuts: List[float] = []) -> float:
    """The length of (x(s), y(s)) on [a, b] to about 20 digits (mpmath), split at the cuts."""

    def speed(s: Any) -> Any:
        return mpmath.sqrt(mpmath.diff(x, s) ** 2 + mpmath.diff(y, s) ** 2)

    edges = [a] + sorted(cuts) + [b]
    return float(sum(mpmath.quad(speed, [lo, hi]) for lo, hi in zip(edges, edges[1:])))


def _graph(f: Callable[[Any], Any], a: float, b: float, cuts: List[float] = []) -> float:
    return _reference(lambda s: s, f, a, b, cuts)


def _polar(r: Callable[[Any], Any], a: float, b: float, cuts: List[float] = []) -> float:
    return _reference(lambda t: r(t) * mpmath.cos(t), lambda t: r(t) * mpmath.sin(t), a, b, cuts)


class ArcLengthCase(unittest.TestCase):
    def assertLength(self, result: Dict[str, Any], expected: float, digits_required: int = 9) -> None:
        """The length matches to digits_required significant digits and the estimate covers the true error."""
        error = abs(result["length"] - expected)
        self.assertLessEqual(error, 10 ** (-digits_required) * max(1.0, abs(expected)), result)
        # The length is rounded to what the estimate supports: allow half a unit of that rounding
        self.assertLessEqual(error, 2 * result["error_estimate"] + 1e-12 * max(1.0, abs(expected)), result)

    def assertRefused(self, fragment: str, **arguments: Any) -> str:
        with self.assertRaises(ValueError) as caught:
            arc_length_tool(**arguments)
        self.assertIn(fragment, str(caught.exception))
        return str(caught.exception)


class TestFunctionLengths(ArcLengthCase):
    def test_parabola(self) -> None:
        result = arc_length_tool(function_string="x^2", lower_bound=0, upper_bound=1)
        self.assertLength(result, math.sqrt(5) / 2 + math.asinh(2) / 4, 11)
        self.assertEqual(result["curve_type"], "function")
        self.assertEqual(result["variable"], "x")
        self.assertNotIn("warning", result)

    def test_catenary_and_sine(self) -> None:
        self.assertLength(arc_length_tool(function_string="cosh(x)", lower_bound=0, upper_bound=1), math.sinh(1), 11)
        self.assertLength(
            arc_length_tool(function_string="sin(x)", lower_bound=0, upper_bound="pi"), _graph(mpmath.sin, 0, math.pi)
        )

    def test_leading_y_equals_is_read(self) -> None:
        result = arc_length_tool(function_string="y = x^2", lower_bound=0, upper_bound=1)
        self.assertLength(result, math.sqrt(5) / 2 + math.asinh(2) / 4)

    def test_corners_on_and_between_grid_nodes(self) -> None:
        self.assertLength(arc_length_tool(function_string="|x|", lower_bound=-1, upper_bound=1), 2 * math.sqrt(2))
        self.assertLength(
            arc_length_tool(function_string="abs(x - 0.3)", lower_bound=0, upper_bound=1), math.sqrt(2), 9
        )
        expected = _graph(lambda x: abs(x**2 - 2), 0, 3, [math.sqrt(2)])
        self.assertLength(arc_length_tool(function_string="abs(x^2 - 2)", lower_bound=0, upper_bound=3), expected)

    def test_vertical_tangents(self) -> None:
        self.assertLength(
            arc_length_tool(function_string="sqrt(x)", lower_bound=0, upper_bound=1),
            math.sqrt(5) / 2 + math.asinh(2) / 4,
        )
        cube_root = _graph(lambda x: mpmath.cbrt(x) if x >= 0 else -mpmath.cbrt(-x), -1, 1, [0])
        self.assertLength(arc_length_tool(function_string="cbrt(x)", lower_bound=-1, upper_bound=1), cube_root)

    def test_vertical_cusp_between_nodes_is_honest(self) -> None:
        def half(a: float) -> float:  # length of sqrt(u) on [0, a]
            return math.sqrt(a * (a + 0.25)) + 0.25 * math.asinh(2 * math.sqrt(a))

        result = arc_length_tool(function_string="sqrt(abs(x - 0.3))", lower_bound=0, upper_bound=1)
        self.assertLength(result, half(0.3) + half(0.7), 6)

    def test_textbook_cusp_two_thirds_power(self) -> None:
        expected = _graph(lambda x: mpmath.power(abs(x), mpmath.mpf(2) / 3), -1, 8, [0])
        self.assertLength(arc_length_tool(function_string="cbrt(x)^2", lower_bound=-1, upper_bound=8), expected)

    def test_removable_points_and_endpoint_limits(self) -> None:
        sinc = _graph(lambda x: mpmath.sin(x) / x if x != 0 else mpmath.mpf(1), -1, 1, [0])
        self.assertLength(arc_length_tool(function_string="sin(x)/x", lower_bound=-1, upper_bound=1), sinc)
        x_log_x = _graph(lambda x: x * mpmath.log(x), 1e-30, 1)
        self.assertLength(arc_length_tool(function_string="x*ln(x)", lower_bound=0, upper_bound=1), x_log_x, 8)

    def test_steep_and_oscillating(self) -> None:
        self.assertLength(
            arc_length_tool(function_string="sin(10*x)", lower_bound=0, upper_bound=10),
            _graph(lambda x: mpmath.sin(10 * x), 0, 10, [k * math.pi / 10 for k in range(1, 32)]),
        )
        self.assertLength(
            arc_length_tool(function_string="tanh(1000*x)", lower_bound=-1, upper_bound=1),
            _graph(lambda x: mpmath.tanh(1000 * x), -1, 1, [-0.01, 0, 0.01]),
        )

    def test_large_values_are_not_rounding_noise(self) -> None:
        result = arc_length_tool(function_string="10^8 + x^2", lower_bound=0, upper_bound=1)
        self.assertLength(result, math.sqrt(5) / 2 + math.asinh(2) / 4, 7)

    def test_bounds_reversed_equal_and_expressions(self) -> None:
        forward = arc_length_tool(function_string="x^2", lower_bound=0, upper_bound=1)
        backward = arc_length_tool(function_string="x^2", lower_bound=1, upper_bound=0)
        self.assertEqual(forward["length"], backward["length"])
        self.assertEqual(arc_length_tool(function_string="x^2", lower_bound=2, upper_bound=2)["length"], 0.0)
        result = arc_length_tool(function_string="sin(x)", lower_bound="-pi/2", upper_bound="pi/2")
        self.assertAlmostEqual(result["lower_bound"], -math.pi / 2)
        self.assertLength(result, _graph(mpmath.sin, -math.pi / 2, math.pi / 2))


class TestParametricAndPolarLengths(ArcLengthCase):
    def test_circle_ellipse_and_astroid(self) -> None:
        circle = arc_length_tool(x_expression="cos(t)", y_expression="sin(t)", lower_bound=0, upper_bound="2*pi")
        self.assertLength(circle, 2 * math.pi, 11)
        self.assertEqual(circle["curve_type"], "parametric")
        ellipse = _reference(lambda t: 3 * mpmath.cos(t), mpmath.sin, 0, 2 * math.pi)
        self.assertLength(arc_length_tool(x_expression="3*cos(t)", y_expression="sin(t)"), ellipse)
        astroid = arc_length_tool(x_expression="cos(t)^3", y_expression="sin(t)^3", lower_bound=0, upper_bound="2*pi")
        self.assertLength(astroid, 6.0)

    def test_cusps_between_grid_nodes(self) -> None:
        astroid = arc_length_tool(
            x_expression="cos(t)^3", y_expression="sin(t)^3", lower_bound=0.3, upper_bound="2*pi+0.3"
        )
        self.assertLength(astroid, 6.0)
        cardioid = arc_length_tool(r_expression="1 + cos(θ)", lower_bound=0.3, upper_bound="2*pi + 0.3")
        self.assertLength(cardioid, 8.0)

    def test_polar_curves(self) -> None:
        cardioid = arc_length_tool(r_expression="1 + cos(theta)")
        self.assertLength(cardioid, 8.0)
        self.assertEqual(cardioid["curve_type"], "polar")
        self.assertEqual(cardioid["variable"], "θ")
        self.assertNotIn("traced_times", cardioid)
        spiral = 0.5 * (2 * math.pi * math.sqrt(1 + 4 * math.pi**2) + math.asinh(2 * math.pi))
        self.assertLength(arc_length_tool(r_expression="θ", lower_bound=0, upper_bound="2*pi"), spiral)

    def test_traced_twice_reports_one_trace(self) -> None:
        circle = arc_length_tool(r_expression="cos(θ)")
        self.assertLength(circle, 2 * math.pi)
        self.assertEqual(circle["traced_times"], 2)
        self.assertAlmostEqual(circle["length_of_one_trace"], math.pi, places=9)
        self.assertIn("2 times", circle["note"])
        rose = arc_length_tool(r_expression="2*sin(3*θ)")
        self.assertEqual(rose["traced_times"], 2)
        self.assertAlmostEqual(rose["length_of_one_trace"] * 2, rose["length"], places=8)
        self.assertNotIn("traced_times", arc_length_tool(r_expression="cos(2*θ)"))
        doubled = arc_length_tool(x_expression="cos(2*t)", y_expression="sin(2*t)")
        self.assertEqual(doubled["traced_times"], 2)
        self.assertEqual(traced_times(lambda t: (math.cos(t), math.sin(t)), 0, 6 * math.pi), 3)

    def test_constant_curve_has_zero_length(self) -> None:
        self.assertEqual(
            arc_length_tool(x_expression="1", y_expression="2", lower_bound=0, upper_bound=1)["length"], 0.0
        )


class TestRefusals(ArcLengthCase):
    def test_jumps(self) -> None:
        self.assertRefused("breaks near x = 1", function_string="floor(x)", lower_bound=0, upper_bound=3)
        self.assertRefused(
            "breaks near x = 0.3", function_string="x + 0.01*floor(x + 0.7)", lower_bound=0, upper_bound=1
        )
        self.assertRefused("breaks near x = 1", function_string="e^x + 5*floor(x)", lower_bound=0.5, upper_bound=1.5)

    def test_vertical_asymptotes_with_and_without_sign_change(self) -> None:
        for text, a, b, where in (
            ("tan(x)", 0, 2, "1.5708"),
            ("1/x", -1, 1, "0"),
            ("1/x^2", -1, 1, "0"),
            ("1/x^2", -0.7, 1.3, "0"),
            ("1/(x - 0.5)^2", 0, 1, "0.5"),
            ("ln(abs(x))", -1, 2, "0"),
            ("ln(abs(x - 0.3))", 0.1, 0.9, "0.3"),
            ("sec(x)^2", 0, 3, "1.5708"),
            ("1/sqrt(abs(x))", -1, 1, "0"),
        ):
            with self.subTest(text=text, a=a, b=b):
                message = self.assertRefused("x = ", function_string=text, lower_bound=a, upper_bound=b)
                self.assertIn(f"x = {where}", message)
        self.assertRefused("t = 0", x_expression="t", y_expression="1/t^2", lower_bound=-1, upper_bound=1)

    def test_undefined_ranges(self) -> None:
        self.assertRefused("no real point at x = -1", function_string="ln(x)", lower_bound=-1, upper_bound=1)
        self.assertRefused("no real point at x = 0", function_string="ln(x)", lower_bound=0, upper_bound=1)
        message = self.assertRefused("no real point", function_string="x^(1/3)", lower_bound=-1, upper_bound=1)
        self.assertIn("cbrt(x)", message)

    def test_infinitely_long_curve_does_not_converge(self) -> None:
        self.assertRefused("did not converge", function_string="x*sin(1/x)", lower_bound=0, upper_bound=1)

    def test_curve_fields(self) -> None:
        self.assertRefused("Give exactly one curve", lower_bound=0, upper_bound=1)
        self.assertRefused(
            "Give exactly one curve", function_string="x", r_expression="θ", lower_bound=0, upper_bound=1
        )
        self.assertRefused("both x_expression and y_expression", x_expression="cos(t)")
        self.assertRefused("Give lower_bound and upper_bound", function_string="x^2")
        self.assertRefused("only on the canvas", curve_name="f")

    def test_unclear_spellings_are_refused_with_a_hint(self) -> None:
        self.assertRefused("Write sin(x)", function_string="sinx", lower_bound=0, upper_bound=1)
        self.assertRefused("'a' is not defined", function_string="a*x", lower_bound=0, upper_bound=1)
        self.assertRefused("has an '='", function_string="x^2 = 3", lower_bound=0, upper_bound=1)
        self.assertRefused("Write cos(t)", x_expression="cost", y_expression="sin(t)")

    def test_bounds(self) -> None:
        self.assertRefused("radians", function_string="sin(x)", lower_bound=0, upper_bound="90°")
        self.assertRefused("constant expression", function_string="x", lower_bound=0, upper_bound="x + 1")
        self.assertRefused("50π", r_expression="θ", lower_bound=0, upper_bound=200)


class TestPieces(ArcLengthCase):
    def test_pieces_must_meet(self) -> None:
        joined = arc_length_in_pieces(lambda x: (x, x * x if x < 1 else 2 * x - 1), 0, 2, [1], "x")
        expected = math.sqrt(5) / 2 + math.asinh(2) / 4 + math.sqrt(5)
        self.assertAlmostEqual(joined["length"], expected, places=9)
        with self.assertRaises(ValueError) as caught:
            arc_length_in_pieces(lambda x: (x, x * x if x < 1 else 2 * x), 0, 2, [1], "x")
        self.assertIn("breaks near x = 1", str(caught.exception))

    def test_budget_gives_a_warning_or_a_refusal(self) -> None:
        try:
            result = arc_length(lambda x: (x, math.sin(x)), 0, 200, "x", max_evaluations=300)
        except ValueError as exc:
            self.assertIn("did not converge", str(exc))
            return
        if "warning" not in result:
            self.assertLessEqual(result["error_estimate"], 1e-8 * result["length"])


def _interval(
    expression: str, evaluate: Callable[[float], float], left: Optional[float], right: Optional[float]
) -> Any:
    return SimpleNamespace(expression=expression, evaluator=evaluate, left=left, right=right, undefined_at=[])


def _piecewise(name: str, pieces: List[Any]) -> Any:
    def function(x: float) -> float:
        for piece in pieces:
            if (piece.left is None or x >= piece.left) and (piece.right is None or x < piece.right):
                return float(piece.evaluator(x))
        return float("nan")

    lefts = [p.left for p in pieces]
    rights = [p.right for p in pieces]
    return SimpleNamespace(
        name=name,
        function=function,
        intervals=pieces,
        left_bound=None if None in lefts else min(lefts),
        right_bound=None if None in rights else max(rights),
        point_discontinuities=[],
        vertical_asymptotes=[],
    )


class TestPlottedCurves(ArcLengthCase):
    def setUp(self) -> None:
        point = SimpleNamespace
        self.drawables = SimpleNamespace(
            Functions=[
                SimpleNamespace(
                    name="f",
                    function=lambda x: x * x,
                    function_string="x^2",
                    left_bound=0.0,
                    right_bound=1.0,
                    point_discontinuities=[],
                    undefined_at=[],
                    vertical_asymptotes=[],
                ),
                SimpleNamespace(
                    name="g",
                    function=lambda x: 1 / x,
                    function_string="1/x",
                    left_bound=None,
                    right_bound=None,
                    point_discontinuities=[],
                    undefined_at=[],
                    vertical_asymptotes=[0.0],
                ),
            ],
            PiecewiseFunctions=[
                _piecewise(
                    "p",
                    [_interval("x^2", lambda x: x * x, -1.0, 1.0), _interval("2*x - 1", lambda x: 2 * x - 1, 1.0, 2.0)],
                ),
                _piecewise("q", [_interval("x", lambda x: x, 0.0, 1.0), _interval("x + 1", lambda x: x + 1, 1.0, 2.0)]),
            ],
            ParametricFunctions=[
                SimpleNamespace(
                    name="c",
                    evaluate=lambda t: (math.cos(t), math.sin(t)),
                    t_min=0.0,
                    t_max=math.pi,
                    x_expression="cos(t)",
                    y_expression="sin(t)",
                )
            ],
            PolarFunctions=[],
            Circles=[SimpleNamespace(name="A(2)", center=point(x=1.0, y=1.0), radius=2.0)],
            CircleArcs=[
                # A clockwise quarter of the circle of radius 2 about (1, 1), from angle π/2
                SimpleNamespace(
                    name="arc_P_Q",
                    center_x=1.0,
                    center_y=1.0,
                    radius=2.0,
                    start_and_sweep=lambda: (math.pi / 2, -math.pi / 2),
                ),
            ],
            Ellipses=[
                SimpleNamespace(name="E", center=point(x=0.0, y=0.0), radius_x=3.0, radius_y=1.0, rotation_angle=30.0)
            ],
        )

    def _measure(self, **arguments: Any) -> Dict[str, Any]:
        return dict(arc_length_tool(resolve=lambda name: resolve_plotted_curve(self.drawables, name), **arguments))

    def test_function_uses_its_own_bounds(self) -> None:
        result = self._measure(curve_name="f")
        self.assertLength(result, math.sqrt(5) / 2 + math.asinh(2) / 4)
        self.assertEqual((result["lower_bound"], result["upper_bound"]), (0.0, 1.0))
        wider = self._measure(curve_name="f", lower_bound=0, upper_bound=2)
        self.assertIn("beyond the curve's plotted range", wider["note"])

    def test_known_asymptote_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._measure(curve_name="g", lower_bound=-1, upper_bound=1)
        self.assertIn("vertical asymptote at x = 0", str(caught.exception))

    def test_piecewise_functions(self) -> None:
        joined = self._measure(curve_name="p")
        expected = _graph(lambda x: x * x, -1, 1) + math.sqrt(5)
        self.assertLength(joined, expected)
        with self.assertRaises(ValueError) as caught:
            self._measure(curve_name="q")
        self.assertIn("breaks near x = 1", str(caught.exception))

    def test_parametric_circle_and_ellipse(self) -> None:
        self.assertLength(self._measure(curve_name="c"), math.pi)
        self.assertLength(self._measure(curve_name="A(2)"), 4 * math.pi)
        self.assertLength(self._measure(curve_name="arc_P_Q"), math.pi)
        arc = resolve_plotted_curve(self.drawables, "arc_P_Q")
        end = arc.point(math.pi / 2)
        assert end is not None
        self.assertAlmostEqual(end[0], 3.0)  # clockwise from the top (1, 3) to the right (3, 1)
        self.assertAlmostEqual(end[1], 1.0)
        ellipse = _reference(lambda t: 3 * mpmath.cos(t), mpmath.sin, 0, 2 * math.pi)
        self.assertLength(self._measure(curve_name="E"), ellipse)

    def test_unknown_name(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._measure(curve_name="nope")
        self.assertIn("No function, parametric or polar curve", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
