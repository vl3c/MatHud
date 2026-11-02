"""Pure Python tests for parameters (sliders): the live namespace and the Parameter drawable.

compiled_expression's parameter namespace (bind(parameters=True), set_parameter_values,
with_parameter_values, plain_number_text) and drawables/parameter.py (name rules, range
rules, saved state). The Brython suite (client_tests/test_parameter_manager.py) covers the
manager, undo, the tools, the panel and workspace restore.
"""

from __future__ import annotations

import copy
import math
import random
import unittest

from server_tests import client_renderer  # noqa: F401  (browser stub, so MathUtils resolves)
import compiled_expression as ce
from utils.parameter_expressions import called_name_hint, read_definition, undefined_names_hint
from drawables.parameter import Parameter, check_parameter_name, format_value, resolve_range, snap_to_step


class _CleanNamespace(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = ce.parameter_values()
        ce.set_parameter_values({})

    def tearDown(self) -> None:
        ce.set_parameter_values(self._saved)


class TestLiveNamespace(_CleanNamespace):
    def test_bound_callable_reads_values_at_call_time(self) -> None:
        ce.set_parameter_values({"a": 2.0})
        f = ce.compile_expression("a*x^2").bind(parameters=True)
        self.assertEqual(f(3.0), 18.0)
        ce.set_parameter_values({"a": -1.0})
        self.assertEqual(f(3.0), -9.0)

    def test_curve_function_set_reads_values(self) -> None:
        ce.set_parameter_values({"ω": 2.0})
        f = ce.compile_expression("cos(ω*t)", ("t",), ce.FUNCTIONS_CURVE).bind(parameters=True)
        self.assertAlmostEqual(f(math.pi / 2), -1.0)

    def test_removed_parameter_raises_name_error_again(self) -> None:
        ce.set_parameter_values({"a": 1.0, "b": 2.0})
        f = ce.compile_expression("a + b").bind(parameters=True)
        self.assertEqual(f(0.0), 3.0)
        ce.set_parameter_values({"a": 1.0})
        with self.assertRaises(NameError):
            f(0.0)

    def test_plain_bind_never_sees_parameters(self) -> None:
        ce.set_parameter_values({"a": 2.0})
        plain = ce.compile_expression("a*x").bind()
        with self.assertRaises(NameError):
            plain(1.0)
        # The live and plain lambdas of one text are cached separately
        live = ce.compile_expression("a*x").bind(parameters=True)
        self.assertIsNot(plain, live)
        self.assertEqual(live(1.0), 2.0)

    def test_variables_shadow_parameters(self) -> None:
        ce.set_parameter_values({"a": 2.0})
        f = ce.compile_expression("a*x", ("x", "a")).bind(parameters=True)
        self.assertEqual(f(1.0, 5.0), 5.0)

    def test_parameter_and_undefined_names(self) -> None:
        ce.set_parameter_values({"a": 1.0})
        compiled = ce.compile_expression("a*x + b")
        self.assertEqual(compiled.parameter_names, ("a",))
        self.assertEqual(compiled.undefined_names, ("b",))

    def test_bad_names_and_values_are_refused(self) -> None:
        for name in ("sin", "pi", "e", "i", "_a", "2a", "lambda", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ce.set_parameter_values({name: 1.0})
        for value in (True, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ce.set_parameter_values({"a": value})  # type: ignore[dict-item]
        self.assertEqual(ce.parameter_values(), {})

    def test_namespace_never_replaces_functions(self) -> None:
        ce.set_parameter_values({"a": 1.0})
        f = ce.compile_expression("sin(a*x)").bind(parameters=True)
        self.assertAlmostEqual(f(math.pi / 2), 1.0)

    def test_with_parameter_values(self) -> None:
        ce.set_parameter_values({"a": -1.5, "b": 1e-05})
        self.assertEqual(ce.with_parameter_values("1/(x-a)"), "1/(x-(-1.5))")
        self.assertEqual(ce.with_parameter_values("b*x + max(a, ab)"), "(0.00001)*x + max((-1.5), ab)")
        self.assertEqual(ce.with_parameter_values("a*x", names=("b",)), "a*x")
        self.assertEqual(ce.with_parameter_values("c*x"), "c*x")

    def test_plain_number_text_round_trips_without_exponent(self) -> None:
        rng = random.Random(7)
        for _ in range(5000):
            value = rng.choice((-1, 1)) * rng.random() * 10 ** rng.randint(-30, 30)
            text = ce.plain_number_text(value)
            self.assertNotIn("e", text)
            self.assertEqual(float(text), value)
        self.assertEqual(ce.plain_number_text(0.0), "0.0")
        self.assertEqual(ce.plain_number_text(2.5e-7), "0.00000025")


class TestDefinitionsAndHints(_CleanNamespace):
    def test_read_definition(self) -> None:
        ce.set_parameter_values({"a": 2.0})
        self.assertEqual((read_definition(3, "x").value, read_definition(3, "x").expression), (3.0, None))
        self.assertEqual(read_definition("2*pi", "x").expression, None)
        defined = read_definition("a^2", "y")
        self.assertEqual((defined.value, defined.expression), (4.0, "a^2"))
        for bad in ("sqrt(-a)", "", float("inf"), True, "b"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                read_definition(bad, "x")

    def test_one_pair_of_surrounding_quotes_is_dropped(self) -> None:
        ce.set_parameter_values({"a": 2.0, "b": 3.0})
        for quoted, expected in (('"b"', (3.0, "b")), ("'2*a'", (4.0, "2*a")), ('" a^2 "', (4.0, "a^2"))):
            with self.subTest(quoted=quoted):
                defined = read_definition(quoted, "x")
                self.assertEqual((defined.value, defined.expression), expected)
        self.assertEqual(read_definition('"3"', "x").value, 3.0)
        for bad in ('"b"*2', "'a", "\"a'", '""'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                read_definition(bad, "x")
        with self.assertRaisesRegex(ValueError, "without quotes"):
            read_definition('"a" + "b"', "x")

    def test_glued_names_get_the_product_hint(self) -> None:
        message = undefined_names_hint("ax^2 + bx + c", ["ax", "bx", "c"], ("x",))
        self.assertIn("For a times x write a*x", message)
        self.assertIn("search_tools 'slider'", message)

    def test_other_variables_and_glued_calls_get_their_hint(self) -> None:
        self.assertIn("write sin(x)", undefined_names_hint("sinx", ["sinx"], ("x",)))
        self.assertNotIn("write pi(x)", undefined_names_hint("pix", ["pix"], ("x",)))
        self.assertIn("plot_sequence", undefined_names_hint("1/n", ["n"], ("x",)))
        self.assertIn("draw_parametric_function", undefined_names_hint("t^2", ["t"], ("x",)))
        polar = undefined_names_hint("theta", ["theta"], ("x",))
        self.assertIn("draw_polar_function", polar)
        self.assertNotIn("set_parameter", polar)

    def test_unknown_name_gets_the_create_hint(self) -> None:
        message = undefined_names_hint("k*x", ["k"], ("x",))
        self.assertIn("set_parameter(name='k'", message)

    def test_called_names(self) -> None:
        ce.set_parameter_values({"a": 1.0})
        self.assertIn("write a*(x+1)", called_name_hint("a(x+1)^2") or "")
        self.assertIn("write g*(x)", called_name_hint("g(x)") or "")
        variable = called_name_hint("x(x-1)") or ""
        self.assertIn("x is a variable", variable)
        self.assertNotIn("set_parameter", variable)
        for fine in ("sin(x) + 2", "f(x) = x^2", "2(x+1)", "x.real(1)", "log10(x)"):
            with self.subTest(fine=fine):
                self.assertIsNone(called_name_hint(fine))


class TestParameterNames(unittest.TestCase):
    def test_accepted_names(self) -> None:
        for name in ("a", "b2", "k_1", "ω", "alpha", "A", "speed", "t", "r"):
            with self.subTest(name=name):
                self.assertEqual(check_parameter_name(name), name)

    def test_refused_names(self) -> None:
        for name in ("", " ", "2a", "_a", "a b", "sin", "pi", "e", "i", "j", "for", "x", "y", "θ", "theta", "π"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_parameter_name(name)

    def test_name_in_use_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "another object"):
            check_parameter_name("f", names_in_use={"f", "A"})


class TestRanges(unittest.TestCase):
    def test_defaults_for_a_new_parameter(self) -> None:
        self.assertEqual(resolve_range(1.5, None, None, None)[:4], (1.5, -10.0, 10.0, 0.1))
        self.assertEqual(resolve_range(None, None, None, None)[:4], (1.0, -10.0, 10.0, 0.1))
        self.assertEqual(resolve_range(15, None, None, None)[:4], (15.0, -10.0, 15.0, 0.1))
        self.assertEqual(resolve_range(None, 0, 5, None)[:4], (1.0, 0.0, 5.0, 0.1))
        self.assertEqual(resolve_range(None, 2, 5, None)[:4], (2.0, 2.0, 5.0, 0.1))
        self.assertEqual(resolve_range(None, 20, None, None)[:4], (20.0, 20.0, 40.0, 0.1))
        self.assertEqual(resolve_range(None, None, -20, None)[:4], (-20.0, -40.0, -20.0, 0.1))

    def test_value_outside_the_stored_range_widens_it(self) -> None:
        current = Parameter("a", 2, -5, 5, 1)
        value, low, high, step, notes = resolve_range(12, None, None, None, current)
        self.assertEqual((value, low, high, step), (12.0, -5.0, 12.0, 1.0))
        self.assertEqual(notes, ["range widened to [-5, 12] to include 12"])

    def test_value_outside_the_range_given_with_it_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "above the max 5 given with it"):
            resolve_range(12, -5, 5, 1)
        with self.assertRaisesRegex(ValueError, "below the min 0 given with it"):
            resolve_range(-1, 0, None, None, Parameter("a", 2, -5, 5, 1))

    def test_nulls_keep_the_current_parameter(self) -> None:
        current = Parameter("a", 2, 0, 5, 0.5)
        self.assertEqual(resolve_range(None, None, None, None, current)[:4], (2.0, 0.0, 5.0, 0.5))
        self.assertEqual(resolve_range(3, None, None, None, current)[:4], (3.0, 0.0, 5.0, 0.5))
        value, low, high, step, notes = resolve_range(None, None, 1, None, current)
        self.assertEqual((value, low, high, step), (1.0, 0.0, 1.0, 0.5))
        self.assertEqual(notes, ["value clamped to 1 to stay in [0, 1]"])

    def test_refusals(self) -> None:
        for args in (
            (1, 5, 5, None),
            (1, 6, 5, None),
            (1, None, None, 0),
            (1, None, None, -1),
            (math.nan, None, None, None),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_range(*args)
        with self.assertRaises(ValueError):
            resolve_range(True, None, None, None)

    def test_long_step_becomes_the_range(self) -> None:
        self.assertEqual(resolve_range(0, 0, 1, 5)[3], 1.0)


class TestSliderHelpers(unittest.TestCase):
    def test_snap_to_step(self) -> None:
        self.assertEqual(snap_to_step(0.333, -10, 10, 0.1), 0.3)
        self.assertEqual(snap_to_step(9.97, -10, 10, 0.3), 10.0)  # max stays reachable
        self.assertEqual(snap_to_step(3.14159, 0, 6.283185, 0.1), 3.1)
        self.assertEqual(snap_to_step(-20, -10, 10, 0.1), -10.0)
        self.assertEqual(snap_to_step(0.7, 0.5, 1.5, 0.25), 0.75)

    def test_format_value(self) -> None:
        self.assertEqual(format_value(1.5, 0.1), "1.5")
        self.assertEqual(format_value(0.30000000000000004, 0.1), "0.3")
        self.assertEqual(format_value(2, 1), "2")
        self.assertEqual(format_value(-0.0001, 0.1), "0")
        self.assertEqual(format_value(0.0012, 0.0005), "0.0012")
        self.assertEqual(format_value(123456.78, 0.01), "123457")


class TestParameterDrawable(unittest.TestCase):
    def test_state_and_copy(self) -> None:
        parameter = Parameter("a", 1.5, -5, 5, 0.1)
        self.assertEqual(parameter.get_class_name(), "Parameter")
        self.assertFalse(parameter.is_renderable)
        self.assertEqual(
            parameter.get_state(), {"name": "a", "args": {"value": 1.5, "min": -5.0, "max": 5.0, "step": 0.1}}
        )
        parameter.animation = {"start": 0.0, "end": 1.0, "duration": 4.0, "repeat": "loop"}
        clone = copy.deepcopy(parameter)
        self.assertEqual(clone.get_state(), parameter.get_state())
        clone.animation["repeat"] = "once"  # type: ignore[index]
        self.assertEqual(parameter.animation["repeat"], "loop")

    def test_value_is_clamped(self) -> None:
        parameter = Parameter("a", 1, 0, 2)
        parameter.set_value(5)
        self.assertEqual(parameter.value, 2.0)
        parameter.set_value(-1)
        self.assertEqual(parameter.value, 0.0)


class TestCanvasText(unittest.TestCase):
    def test_parameters_and_definitions_in_the_canvas_text(self) -> None:
        from static.canvas_state_formatter import render_text

        state = {
            "Parameters": [
                {"name": "a", "args": {"value": 1.5, "min": -5, "max": 5, "step": 0.1}},
                {
                    "name": "k",
                    "args": {
                        "value": 0,
                        "min": 0,
                        "max": 5,
                        "step": 0.1,
                        "animation": {"start": 0, "end": 5, "duration": 3, "repeat": "loop"},
                    },
                },
            ],
            "Points": [
                {"name": "P", "args": {"position": {"x": 1.5, "y": 2.25}, "x_expression": "a", "y_expression": "a^2"}},
                {"name": "Q", "args": {"position": {"x": 1, "y": 0}, "x_expression": "sqrt(a)", "undefined": True}},
            ],
        }
        text = render_text(state)
        self.assertIn("a = 1.5  slider [-5, 5] step 0.1", text)
        self.assertIn("animating loop from 0 to 5 over 3 s", text)
        self.assertIn("P = (1.5, 2.25)  defined by x = a, y = a^2", text)
        self.assertIn("not drawn", text)
        self.assertLess(text.index("a = 1.5"), text.index("P = "))


if __name__ == "__main__":
    unittest.main()
