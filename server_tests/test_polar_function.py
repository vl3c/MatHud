"""Pure Python tests for polar curves r(θ) (static/client/drawables/polar_function.py).

Parsing (θ, the word theta, a leading "r ="), refusals, the calculator convention for
negative r, the θ range, the "has a real value" guard, saved state, and polar curves in
utils.object_intersections. The Brython suite (client_tests/test_polar_function_manager.py)
covers the manager, undo, the tools, tangents and workspace restore.
"""

from __future__ import annotations

import copy
import math
import unittest
from typing import Any, Dict, List, Tuple

from server_tests import client_renderer  # noqa: F401  (browser stub, so MathUtils resolves)
from drawables.polar_function import PolarFunction
from utils.object_intersections import ParametricShape, circle_shape, find_object_intersections


def _shape(curve: PolarFunction) -> ParametricShape:
    return ParametricShape(
        curve.name,
        curve.evaluate_x,
        curve.evaluate_y,
        curve.t_min,
        curve.t_max,
        pixels_per_unit=32.0,
        parameter_name="theta",
    )


def _points(report: Dict[str, Any]) -> List[Tuple[float, float]]:
    return sorted((round(p["x"], 6) + 0.0, round(p["y"], 6) + 0.0) for p in report["points"])


class TestParsing(unittest.TestCase):
    def test_stored_forms(self) -> None:
        cases = {
            "1+cos(θ)": "1+cos(θ)",
            "2θ": "2*θ",
            "sinθ": "sin(θ)",
            "ϑ/2": "θ/2",
            "3theta": "3*θ",
            "cos 2theta": "cos(2*θ)",
            "1 - cos theta": "1 - cos(θ)",
            "2*sin(3*theta)": "2*sin(3*θ)",
            "e^(0.1θ)": "e^(0.1*θ)",
            "-1": "-1",
        }
        for text, stored in cases.items():
            with self.subTest(text=text):
                self.assertEqual(PolarFunction(text).r_expression, stored)

    def test_a_leading_r_equals_is_dropped(self) -> None:
        for text in ("r = 1 + cos(θ)", "r=1+cos(θ)", "r(θ) = 1 + cos(θ)", "r(theta)=1+cos(theta)", " r = 1+cos θ"):
            with self.subTest(text=text):
                curve = PolarFunction(text)
                self.assertAlmostEqual(curve.radius(0.0), 2.0)

    def test_other_equations_are_refused(self) -> None:
        for text in ("θ = 1", "r == 2", "y = 1 + cos(θ)", "r = 1 = 2"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "right-hand side"):
                    PolarFunction(text)

    def test_other_names_are_refused_with_a_hint(self) -> None:
        with self.assertRaisesRegex(ValueError, "may use only θ.*'t' is not defined"):
            PolarFunction("2*t")
        with self.assertRaisesRegex(ValueError, "'x' is not defined"):
            PolarFunction("x + θ")
        with self.assertRaisesRegex(ValueError, r"Write cos\(theta\)"):
            PolarFunction("costheta")
        with self.assertRaisesRegex(ValueError, "'a' is not defined"):
            PolarFunction("a*θ")

    def test_empty_and_unparsable_text_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            PolarFunction("  ")
        with self.assertRaisesRegex(ValueError, "Could not read"):
            PolarFunction("1 + (θ")

    def test_function_power_spelling_gets_a_hint(self) -> None:
        with self.assertRaisesRegex(ValueError, r"could not be evaluated.*sin\(θ\)\^2"):
            PolarFunction("sin^2θ")
        self.assertAlmostEqual(PolarFunction("sin(θ)^2").radius(math.pi / 2), 1.0)


class TestEvaluation(unittest.TestCase):
    def test_cardioid_points(self) -> None:
        curve = PolarFunction("1 + cos(θ)")
        x, y = curve.evaluate(math.pi / 2)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertEqual(curve.evaluate(0.0), (2.0, 0.0))
        self.assertAlmostEqual(curve.evaluate_x(math.pi / 3), 1.5 * 0.5)
        self.assertAlmostEqual(curve.evaluate_y(math.pi / 3), 1.5 * math.sin(math.pi / 3))

    def test_negative_r_lies_on_the_opposite_ray(self) -> None:
        unit = PolarFunction("-1")
        x, y = unit.evaluate(0.0)
        self.assertAlmostEqual(x, -1.0)
        self.assertAlmostEqual(y, 0.0)
        rose = PolarFunction("cos(2θ)")
        x, y = rose.evaluate(math.pi / 2)  # r = -1 at θ = π/2: the point (0, -1)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, -1.0)

    def test_circle_through_the_pole(self) -> None:
        circle = PolarFunction("2cos(θ)")
        for theta in (0.3, 1.0, 2.5):
            x, y = circle.evaluate(theta)
            self.assertAlmostEqual(math.hypot(x - 1.0, y), 1.0)

    def test_non_real_r_raises(self) -> None:
        lemniscate = PolarFunction("sqrt(cos(2θ))")
        self.assertAlmostEqual(lemniscate.radius(0.0), 1.0)
        for evaluate in (lemniscate.evaluate, lemniscate.evaluate_x, lemniscate.evaluate_y, lemniscate.radius):
            with self.assertRaisesRegex(ValueError, "not real at θ=1.5"):
                evaluate(1.5)
        with self.assertRaises(ValueError):
            PolarFunction("1/θ", theta_min=-1, theta_max=1).evaluate(0.0)


class TestRange(unittest.TestCase):
    def test_default_range_is_one_turn(self) -> None:
        curve = PolarFunction("1")
        self.assertEqual((curve.theta_min, curve.theta_max), (0.0, 2 * math.pi))
        self.assertEqual((curve.t_min, curve.t_max), (0.0, 2 * math.pi))

    def test_reversed_range_is_ordered(self) -> None:
        curve = PolarFunction("θ", theta_min=4.0, theta_max=1.0)
        self.assertEqual((curve.theta_min, curve.theta_max), (1.0, 4.0))

    def test_bad_ranges_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "theta_min must be less than theta_max"):
            PolarFunction("1", theta_min=2.0, theta_max=2.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            PolarFunction("1", theta_min=0.0, theta_max=float("inf"))
        with self.assertRaisesRegex(ValueError, "must be numbers"):
            PolarFunction("1", theta_min=None, theta_max=1.0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "longer than 50π"):
            PolarFunction("θ", theta_min=0.0, theta_max=200.0)
        PolarFunction("θ", theta_min=0.0, theta_max=50 * math.pi)

    def test_samples_grow_with_the_number_of_turns(self) -> None:
        self.assertEqual(PolarFunction("1").min_samples, 240)
        self.assertEqual(PolarFunction("θ", theta_max=20 * math.pi).min_samples, 2400)
        self.assertEqual(PolarFunction("θ", theta_max=50 * math.pi).min_samples, 6000)

    def test_update_range_checks_before_changing(self) -> None:
        curve = PolarFunction("sqrt(θ)", theta_min=0.0, theta_max=4.0)
        with self.assertRaisesRegex(ValueError, "no real value"):
            curve.update_theta_range(-3.0, -1.0)
        with self.assertRaisesRegex(ValueError, "longer than 50π"):
            curve.update_theta_range(0.0, 400.0)
        self.assertEqual((curve.theta_min, curve.theta_max), (0.0, 4.0))
        curve.update_theta_range(6.0, 1.0)
        self.assertEqual((curve.theta_min, curve.theta_max), (1.0, 6.0))


class TestRealValueGuard(unittest.TestCase):
    def test_a_curve_with_no_real_point_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "has no real value for θ in"):
            PolarFunction("sqrt(-1-θ^2)")
        with self.assertRaisesRegex(ValueError, "has no real value"):
            PolarFunction("ln(cos(θ) - 2)")

    def test_partial_domains_are_accepted(self) -> None:
        PolarFunction("sqrt(cos(2θ))")
        PolarFunction("1/(1 - cos(θ))")
        PolarFunction("sqrt(θ)", theta_min=-1.0, theta_max=1.0)


class TestState(unittest.TestCase):
    def test_saved_state_schema(self) -> None:
        curve = PolarFunction("r = 1 + cos(theta)", name="card", theta_min=0.0, theta_max=4.0, color="red")
        self.assertEqual(curve.get_class_name(), "PolarFunction")
        self.assertEqual(
            curve.get_state(),
            {
                "name": "card",
                "args": {"r_expression": "1 + cos(θ)", "theta_min": 0.0, "theta_max": 4.0, "color": "red"},
            },
        )

    def test_state_rebuilds_the_same_curve(self) -> None:
        curve = PolarFunction("2*sin(3θ)", name="rose", theta_min=0.5, theta_max=3.0)
        args = curve.get_state()["args"]
        rebuilt = PolarFunction(args["r_expression"], "rose", args["theta_min"], args["theta_max"], args["color"])
        self.assertEqual(rebuilt.get_state(), curve.get_state())
        self.assertEqual(rebuilt.evaluate(1.2), curve.evaluate(1.2))

    def test_deepcopy(self) -> None:
        curve = PolarFunction("θ", name="spiral", theta_max=6 * math.pi, color="blue")
        copied = copy.deepcopy(curve)
        self.assertIsNot(copied, curve)
        self.assertEqual(copied.get_state(), curve.get_state())
        self.assertEqual(copied.evaluate(2.0), curve.evaluate(2.0))

    def test_a_polar_curve_cannot_be_translated_or_rotated(self) -> None:
        # translate_object finds targets by a callable translate; rotate_object excludes the class
        curve = PolarFunction("1")
        self.assertFalse(callable(getattr(curve, "translate", None)))
        with self.assertRaises(NotImplementedError):
            curve.rotate(1.0)


class TestPolarIntersections(unittest.TestCase):
    def test_cardioid_and_circle_meet_at_two_points_and_the_pole(self) -> None:
        # 1 + cos θ = 3 cos θ at θ = ±π/3, r = 3/2; both curves also pass through the pole
        cardioid = PolarFunction("1 + cos(θ)", name="c")
        circle = PolarFunction("3cos(θ)", name="k")
        report = find_object_intersections(_shape(cardioid), _shape(circle))
        half = round(1.5 * math.sin(math.pi / 3), 6)
        self.assertEqual(_points(report), [(0.0, 0.0), (0.75, -half), (0.75, half)])

    def test_hits_report_theta(self) -> None:
        unit = PolarFunction("1", name="u")
        report = find_object_intersections(_shape(unit), circle_shape("c", (1.0, 0.0), 1.0))
        self.assertEqual(_points(report), [(0.5, -round(math.sqrt(3) / 2, 6)), (0.5, round(math.sqrt(3) / 2, 6))])
        thetas = sorted(p["params"]["u"]["theta"] for p in report["points"])
        self.assertAlmostEqual(thetas[0], math.pi / 3, places=6)
        self.assertAlmostEqual(thetas[1], 5 * math.pi / 3, places=6)

    def test_a_circle_traced_twice_gives_each_point_once(self) -> None:
        circle = PolarFunction("2cos(θ)", name="k")  # over [0, 2π] the circle is traced twice
        report = find_object_intersections(_shape(circle), circle_shape("c", (1.0, 1.0), 1.0))
        # (x - 1)^2 + y^2 = 1 and (x - 1)^2 + (y - 1)^2 = 1 meet at y = 1/2, x = 1 ± √3/2
        offset = math.sqrt(3) / 2
        self.assertEqual(_points(report), [(round(1 - offset, 6), 0.5), (round(1 + offset, 6), 0.5)])


if __name__ == "__main__":
    unittest.main()
