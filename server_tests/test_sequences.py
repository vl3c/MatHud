"""Sequences and cobweb diagrams: parsing, terms, orbits, drawables, the manager and the canvas text.

The math (``utils/sequences.py``), the drawables and ``SequencePlotManager`` are pure Python,
so they run under CPython with a stub canvas; rendering is tested in the browser
(``client_tests/test_sequence_plots.py``).
"""

from __future__ import annotations

import copy
import json
import math
import re
import time
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List

from server_tests import client_renderer  # noqa: F401  (browser stub, so MathUtils resolves)
from drawables.cobweb_plot import CobwebPlot
from drawables.sequence_plot import SequencePlot
from managers.drawables_container import DrawablesContainer
from managers.sequence_plot_manager import SequencePlotManager
from plot_kinds import build_saved_plot, plot_kind
from static.canvas_state_formatter import render_state
from static.canvas_view_note import measure_shapes
from static.tool_search_service import ToolSearchService
from utils.sequences import (
    CLASSIFY_STEPS,
    MAX_SEQUENCE_TERMS,
    Orbit,
    check_steps,
    classify_orbit,
    drawn_orbit,
    iterate,
    orbit_until_settled,
    orbit_x_range,
    parse_iteration_function,
    parse_sequence,
    sequence_terms,
    trend,
)


def _terms(text: str, start: Any, end: Any, initial: Any = None) -> List[float]:
    terms: List[float] = sequence_terms(parse_sequence(text), start, end, initial).terms
    return terms


def _behaviour(text: str, x0: float, steps: int = 20) -> Dict[str, Any]:
    g = parse_iteration_function(text).evaluate
    behaviour: Dict[str, Any] = orbit_until_settled(g, x0, steps)[1]
    return behaviour


class TestParseSequence(unittest.TestCase):
    def test_explicit_and_recursive_orders(self) -> None:
        cases = [
            ("1/n", 0),
            ("(-1)^n/n", 0),
            ("u(n-1) + 1", 1),
            ("u(n - 1) + u( n-2 )", 2),
            ("n*u(n-1)", 1),
            ("u(n-2)", 2),
            ("2u(n-1)", 1),
        ]
        for text, order in cases:
            with self.subTest(text=text):
                self.assertEqual(parse_sequence(text).order, order)

    def test_leading_u_of_n_is_dropped(self) -> None:
        self.assertEqual(parse_sequence("U(n) = 1/n").expression, "1/n")
        self.assertEqual(parse_sequence("u(n\u22121) + 1").order, 1)  # the Unicode minus
        self.assertEqual(parse_sequence("u(n) = 1/n").expression, "1/n")
        self.assertEqual(parse_sequence("u (n)= u(n-1)/2").expression, "u(n-1)/2")

    def test_refusals_name_the_accepted_form(self) -> None:
        cases = [
            ("", "empty"),
            ("k^2", "write the index as n"),
            ("i/2", "imaginary unit"),
            ("j + 1", "imaginary unit"),
            ("a(n-1) + 1", "Name the sequence u"),
            ("u_{n-1} + 1", "u(n-1) and u(n-2)"),
            ("u[n-1]*2", "u(n-1) and u(n-2)"),
            ("u(n+1) = 2u(n)", "Shift the index"),
            ("u(n) + 1", "not a previous term"),
            ("u(n-3)", "not a previous term"),
            ("r*u(n-1)", "write a number in place"),
            ("a*n + b", "'a', 'b' are not defined"),
            ("a_n = 1/n", "has an '='"),
            ("x = 1/n", "has an '='"),
            ("3.2u(n-1)(1-u(n-1))", "Write every product with *"),
            ("n(n+1)/2", "Write every product with *"),
            ("2n(n-1)", "Write every product with *"),
            ("a_n = a_{n-1} + 2", "not 'a_{n-1"),
            ("summation(1/k^2, k, 1, n)", "u(n-1) + 1/n^2 with initial_terms [1]"),
        ]
        for text, hint in cases:
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as caught:
                    parse_sequence(text)
                self.assertIn(hint, str(caught.exception))
                self.assertNotIn("u_prev", str(caught.exception))
                self.assertNotIn("Call(func", str(caught.exception))


class TestSequenceTerms(unittest.TestCase):
    def test_explicit_terms(self) -> None:
        self.assertEqual(_terms("1/n", 1, 4), [1.0, 0.5, 1 / 3, 0.25])
        self.assertEqual(_terms("(-1)^n", 1, 4), [-1.0, 1.0, -1.0, 1.0])
        self.assertEqual(_terms("factorial(n)", 0, 5), [1.0, 1.0, 2.0, 6.0, 24.0, 120.0])
        self.assertEqual(_terms("n!", 3, 4), [6.0, 24.0])

    def test_recursions(self) -> None:
        self.assertEqual(_terms("u(n-1) + u(n-2)", 1, 10, [1, 1]), [1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
        self.assertEqual(_terms("u(n-1) + u(n-2)", 0, 4, [2, 1]), [2, 1, 3, 4, 7])  # u(start) first
        self.assertEqual(_terms("0.5*u(n-1) + 1", 0, 3, [0]), [0.0, 1.0, 1.5, 1.75])
        self.assertEqual(_terms("n*u(n-1)", 1, 5, [1]), [1, 2, 6, 24, 120])
        # A range shorter than the order draws the initial terms in range
        self.assertEqual(_terms("u(n-1) + u(n-2)", 1, 1, [1, 1]), [1.0])

    def test_undefined_explicit_terms_are_left_out(self) -> None:
        terms = sequence_terms(parse_sequence("1/(n-3)"), 1, 5)
        self.assertEqual(terms.n_values, [1, 2, 4, 5])
        self.assertEqual(terms.undefined_at, [3])
        complex_terms = sequence_terms(parse_sequence("sqrt(n)"), -2, 1)
        self.assertEqual(complex_terms.undefined_at, [-2, -1])

    def test_a_recursion_stops_at_its_first_undefined_term(self) -> None:
        terms = sequence_terms(parse_sequence("1/(u(n-1) - 1)"), 0, 5, [2])
        self.assertEqual(terms.terms, [2.0, 1.0])
        self.assertEqual(terms.stopped_at, 2)

    def test_overflow_is_fast_and_undefined(self) -> None:
        started = time.time()
        towers = sequence_terms(parse_sequence("2^(2^n)"), 1, 40)
        self.assertEqual(towers.terms[:4], [4.0, 16.0, 256.0, 65536.0])
        self.assertEqual(len(towers.undefined_at), 31)
        with self.assertRaises(ValueError):
            sequence_terms(parse_sequence("factorial(n)"), 999001, 1000000)
        sequence_terms(parse_sequence("n^n"), 1, 1000)
        sequence_terms(parse_sequence("u(n-1)^2"), 1, 1000, [3])
        self.assertLess(time.time() - started, 2.0)

    def test_bounds_and_initial_terms(self) -> None:
        rule = parse_sequence("u(n-1) + u(n-2)")
        bad = [
            (parse_sequence("1/n"), 1, 0, None, "must not be less than start"),
            (parse_sequence("1/n"), 1, MAX_SEQUENCE_TERMS + 1, None, "at most"),
            (parse_sequence("1/n"), 1.5, 3, None, "whole number"),
            (parse_sequence("1/n"), True, 3, None, "whole number"),
            (parse_sequence("1/n"), 1, 3, [1], "takes no initial_terms"),
            (rule, 1, 3, [1], "[u(start), u(start+1)] in that order"),
            (rule, 1, 3, None, "initial_terms"),
            (parse_sequence("u(n-1)"), 1, 3, [math.inf], "finite"),
            (parse_sequence("1/(n-n)"), 1, 3, None, "not defined"),
        ]
        for case_rule, start, end, initial, hint in bad:
            with self.subTest(start=start, end=end, initial=initial, hint=hint):
                with self.assertRaises(ValueError) as caught:
                    sequence_terms(case_rule, start, end, initial)
                self.assertIn(hint, str(caught.exception))
        self.assertEqual(len(_terms("n", 5.0, 6.0)), 2)

    def test_trend(self) -> None:
        self.assertEqual(trend([1, 2, 2, 3]), "increasing")
        self.assertEqual(trend([3, 2, 1]), "decreasing")
        self.assertEqual(trend([1, 1]), "constant")
        self.assertEqual(trend([1, 2, 1]), "neither")
        self.assertIsNone(trend([]))


class TestOrbits(unittest.TestCase):
    def test_fixed_points_at_the_default_steps(self) -> None:
        cases = [("cos(x)", 1.0, 0.739085133215, True), ("exp(-x)", 0.0, 0.567143290410, True)]
        cases += [("2.8*x*(1-x)", 0.1, 0.642857142857, True), ("x/2 + 1/x", 1.0, math.sqrt(2), True)]
        for text, x0, point, stable in cases:
            with self.subTest(g=text):
                behaviour = _behaviour(text, x0)
                self.assertEqual(behaviour["behaviour"], "fixed_point")
                self.assertAlmostEqual(behaviour["fixed_point"], point, places=10)
                self.assertIs(behaviour["stable"], stable)
                self.assertLessEqual(behaviour["iterations"], CLASSIFY_STEPS)

    def test_a_fixed_point_at_zero_is_snapped(self) -> None:
        self.assertEqual(_behaviour("x/2", 1.0)["fixed_point"], 0.0)

    def test_an_unstable_fixed_point_hit_exactly(self) -> None:
        behaviour = _behaviour("x^2", 1.0)
        self.assertEqual(
            (behaviour["behaviour"], behaviour["fixed_point"], behaviour["stable"]), ("fixed_point", 1.0, False)
        )
        self.assertAlmostEqual(behaviour["slope"], 2.0, places=5)

    def test_cycles(self) -> None:
        two = _behaviour("3.2*x*(1-x)", 0.1)
        self.assertEqual((two["behaviour"], two["period"]), ("cycle", 2))
        self.assertAlmostEqual(two["cycle"][0], 0.513044509533, places=9)
        self.assertAlmostEqual(two["multiplier"], 0.16, places=6)
        self.assertTrue(two["stable"])
        self.assertEqual(_behaviour("3.5*x*(1-x)", 0.3)["period"], 4)
        self.assertEqual(_behaviour("3.83*x*(1-x)", 0.1)["period"], 3)

    def test_escapes_diverges_and_undefined(self) -> None:
        self.assertEqual(_behaviour("2x", 1.0)["behaviour"], "escapes")
        self.assertEqual(_behaviour("1.5x", 1.0)["behaviour"], "escapes")
        self.assertEqual(_behaviour("x + 1", 0.0)["behaviour"], "diverges")
        undefined = _behaviour("ln(x)", 0.5)
        self.assertEqual((undefined["behaviour"], undefined["undefined_at"]), ("undefined", 2))
        self.assertAlmostEqual(undefined["undefined_input"], math.log(0.5))
        self.assertEqual(_behaviour("sqrt(x - 2)", 3.0)["behaviour"], "undefined")
        # A large start converges rather than "escaping": the bound is relative to x0
        self.assertEqual(_behaviour("x/2", 5e12)["behaviour"], "fixed_point")

    def test_unsettled_orbits_say_what_they_are_doing(self) -> None:
        self.assertEqual(_behaviour("sin(x)", 1.0)["pattern"], "monotone")
        self.assertEqual(_behaviour("x + 1/x", 1.0)["pattern"], "monotone")
        chaotic = _behaviour("3.9*x*(1-x)", 0.2)
        self.assertEqual((chaotic["behaviour"], chaotic["pattern"]), ("none", "irregular"))
        self.assertEqual(chaotic["iterations"], CLASSIFY_STEPS)

    def test_chaos_and_slow_convergence_are_told_apart(self) -> None:
        # Review of #137: comparing two changes of a chaotic orbit called a third of starts
        # "shrinking", and the slow approach at the r = 3 bifurcation "irregular".
        for r in (3.7, 3.85, 3.9):
            g = parse_iteration_function(f"{r}*x*(1-x)").evaluate
            for start in (0.05, 0.2, 0.37, 0.61, 0.88):
                with self.subTest(r=r, x0=start):
                    behaviour = orbit_until_settled(g, start, 20)[1]
                    self.assertEqual((behaviour["behaviour"], behaviour["pattern"]), ("none", "irregular"))
        for text, x0 in (("3*x*(1-x)", 0.2), ("2.999*x*(1-x)", 0.2), ("-0.999*x+1", 0.0)):
            with self.subTest(g=text):
                behaviour = _behaviour(text, x0)
                self.assertEqual((behaviour["behaviour"], behaviour["pattern"]), ("none", "shrinking"))

    def test_a_slow_approach_to_a_cycle_is_shrinking_with_its_period(self) -> None:
        for r, period in ((3.45, 4), (3.4495, 4), (3.449, 2), (3.544, 4), (3.564, 8)):
            with self.subTest(r=r):
                behaviour = _behaviour(f"{r}*x*(1-x)", 0.2)
                self.assertEqual((behaviour["behaviour"], behaviour["pattern"]), ("none", "shrinking"))
                self.assertEqual(behaviour["period"], period)
        for r in (3.57, 3.6, 3.7, 3.9):
            g = parse_iteration_function(f"{r}*x*(1-x)").evaluate
            for start in (0.03, 0.2, 0.41, 0.77):
                with self.subTest(r=r, x0=start):
                    self.assertEqual(orbit_until_settled(g, start, 20)[1]["pattern"], "irregular")

    def test_a_large_start_moving_by_one_is_not_settled(self) -> None:
        self.assertEqual(_behaviour("x + 1", 1e10)["behaviour"], "diverges")

    def test_a_slow_approach_is_a_fixed_point_not_a_cycle(self) -> None:
        for text, x0, point in (("2.97*x*(1-x)", 0.2, 1 - 1 / 2.97), ("1-0.99*x", 0.0, 1 / 1.99)):
            with self.subTest(g=text):
                behaviour = _behaviour(text, x0)
                self.assertEqual(behaviour["behaviour"], "fixed_point")
                self.assertAlmostEqual(behaviour["fixed_point"], point, places=10)
                self.assertTrue(behaviour["stable"])

    def test_slow_drift_is_not_divergence_and_zero_slopes_are_exact(self) -> None:
        self.assertEqual(_behaviour("tan(x)", 1.0)["behaviour"], "none")
        self.assertEqual(_behaviour("x + 1", 0.0)["behaviour"], "diverges")
        self.assertEqual(_behaviour("2*x*(1-x)", 0.1)["slope"], 0.0)
        self.assertEqual(_behaviour("x^3", 0.5)["slope"], 0.0)
        multiplier = _behaviour("x^2 - 1", 0.5)["multiplier"]
        self.assertEqual((multiplier, math.copysign(1.0, multiplier)), (0.0, 1.0))

    def test_drawn_orbit_and_range(self) -> None:
        g = parse_iteration_function("2x").evaluate
        orbit = iterate(g, 1.0, 20)
        self.assertEqual(drawn_orbit(orbit.values)[-1], 512.0)  # the next value passes 1000
        self.assertEqual(orbit_x_range([0.1, 0.5], [0.2]), (-0.1, 0.7))
        self.assertEqual(orbit_x_range([2.0, 2.0]), (1.0, 3.0))

    def test_function_parsing(self) -> None:
        self.assertEqual(parse_iteration_function("g(x) = cos(x)").expression, "cos(x)")
        self.assertEqual(parse_iteration_function("y=x/2").expression, "x/2")
        for text, hint in [("3.2x(1-x)", "3.2*x*(1-x)"), ("r*x*(1-x)", "write a number"), ("", "empty")]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as caught:
                    parse_iteration_function(text)
                self.assertIn(hint, str(caught.exception))
        self.assertEqual(check_steps(None), 20)
        for steps in (0, 501, 2.5):
            with self.assertRaises(ValueError):
                check_steps(steps)

    def test_classify_reports_iterations(self) -> None:
        g = parse_iteration_function("x").evaluate
        self.assertEqual(classify_orbit(g, Orbit(values=[1.0, 1.0]))["iterations"], 1)


class TestToolSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ToolSearchService.__new__(ToolSearchService)

    def _top(self, query: str) -> List[str]:
        return [tool["function"]["name"] for tool in self.service.search_tools_local(query, 5)]

    def test_root_finder_iterations_are_not_cobwebs(self) -> None:
        for query in (
            "apply newton's method to x^2 - 2 starting at 1 for 5 iterations",
            "do 5 iterations of the bisection method on x^3 - x - 2 on [1, 2]",
            "find the root of x^3 - 2x - 5 to 3 decimal places by iteration",
            "approximate the zero of cos(x) - x with 10 iterations",
            "find the root of x^3 - 2x - 5 = 0 by iteration starting at 2",
            "approximate the solution of x^3 + 2x = 5 by iteration",
            "plot the orbit x(t) = 2cos(t), y(t) = sin(t)",
            "plot the orbit of a satellite x = 3cos(t), y = 2sin(t)",
        ):
            with self.subTest(query=query):
                self.assertNotEqual(self._top(query)[:1], ["plot_cobweb"])
        for query in (
            "Fixed-point iteration for x = cos(x) from x0 = 0.5",
            "iterate g(x) = sqrt(x + 2) from x0 = 0 to find the root",
            "iterate x -> cos(x) from x0 = 1 until it converges to 4 decimal places",
            "iterate x^2 - 1 starting from zero",
            "iterate cos(x) from 1 and find the solution",
            "iterate x = 1 + 1/x from 1 and find the solution",
            "plot the orbit of the logistic map x(t+1) = 4x(t)(1-x(t))",
        ):
            with self.subTest(query=query):
                self.assertEqual(self._top(query)[:1], ["plot_cobweb"])

    def test_the_index_n_is_not_a_sequence_name(self) -> None:
        cases = {
            "evaluate n(n-1)/2 for n = 10": "evaluate_expression",
            "find n such that n(n+1) = 132": "solve",
            "plot n(n-1)": "draw_function",
            "plot the staircase function floor(x)": "draw_function",
            "check the 3 steps of my derivation": "check_steps",
        }
        for query, tool in cases.items():
            with self.subTest(query=query):
                self.assertEqual(self._top(query)[:1], [tool])

    def test_plot_requests_keep_plot_sequence(self) -> None:
        for query in (
            "graph u(n) = 2u(n-1) + 1 with u(0) = 1",
            "plot the convergent sequence 1/n",
            "graph the sequence 1/n to see whether it converges",
            "plot u(n) = 0.5*u(n-1) + 1 with u(0) = 0 for 20 steps",
        ):
            with self.subTest(query=query):
                self.assertEqual(self._top(query)[:1], ["plot_sequence"])
        self.assertEqual(self._top("delete the sequence")[:1], ["delete_plot"])


class TestDrawables(unittest.TestCase):
    def plots(self) -> List[Any]:
        return [
            SequencePlot("S", rule=parse_sequence("1/(n-2)"), start=0, end=4, color="red"),
            SequencePlot("F", rule=parse_sequence("u(n-1) + u(n-2)"), start=1, end=6, initial_terms=[1, 1]),
            CobwebPlot("W", function=parse_iteration_function("cos(x)"), x0=1.0, steps=10),
        ]

    def test_states_rebuild_through_the_plot_kinds(self) -> None:
        for plot in self.plots():
            with self.subTest(plot=plot.name):
                state = plot.get_state()
                kind = plot_kind(plot.get_class_name())
                assert kind is not None
                rebuilt, name = build_saved_plot(kind, json.loads(json.dumps(state)))
                self.assertEqual((name, rebuilt.get_state()), (plot.name, state))

    def test_large_terms_are_saved_without_an_exponent_brython_cannot_read(self) -> None:
        # Brython's json.loads refuses "1e+21"; 2^n to 100 used to break the workspace and backup.
        plot = SequencePlot("S", rule=parse_sequence("2^n"), start=1, end=100)
        text = json.dumps(plot.get_state())
        self.assertIsNone(re.search(r"e\+\d+[,\]} ]", text))  # only inside strings
        saved = json.loads(text)["args"]["summary"]
        self.assertEqual(saved["terms"][9], 1024.0)
        self.assertEqual(saved["max"], "1.26765060023e+30")
        kind = plot_kind("SequencePlot")
        assert kind is not None
        rebuilt, _ = build_saved_plot(kind, json.loads(text))
        self.assertEqual(rebuilt.summary["max"], 2.0**100)
        shape = next(iter(measure_shapes({"SequencePlots": [json.loads(text)]}).values()))
        self.assertAlmostEqual(shape.box.top / 2.0**100, 1.0, places=9)

    def test_bad_saved_items_are_refused(self) -> None:
        for class_name, args in [
            ("SequencePlot", {"expression": "k^2", "start": 1, "end": 3}),
            ("SequencePlot", {"expression": "u(n-1)", "start": 1, "end": 3, "initial_terms": None}),
            ("CobwebPlot", {"function": "cos(x)", "x0": None, "steps": 5}),
        ]:
            kind = plot_kind(class_name)
            assert kind is not None
            with self.subTest(args=args):
                with self.assertRaises((TypeError, ValueError)):
                    build_saved_plot(kind, {"name": "N", "args": args})

    def test_copies_share_the_computed_parts(self) -> None:
        sequence, _, cobweb = self.plots()
        copied = copy.deepcopy(cobweb)
        self.assertIs(copied.curve, cobweb.curve)
        self.assertIs(copied.summary, cobweb.summary)
        self.assertIs(copy.deepcopy(sequence).rule, sequence.rule)
        self.assertEqual(copied.render_signature(), cobweb.render_signature())

    def test_points_web_and_extents(self) -> None:
        sequence, fib, cobweb = self.plots()
        self.assertEqual(sequence.points(), [(0.0, -0.5), (1.0, -1.0), (3.0, 1.0), (4.0, 0.5)])
        self.assertEqual(fib.extent_points(), [(1.0, 1.0), (6.0, 8.0)])
        web = cobweb.web_points()
        self.assertEqual(web[0], (1.0, 0.0))
        self.assertEqual(web[1], (1.0, math.cos(1.0)))
        self.assertEqual(web[2], (math.cos(1.0), math.cos(1.0)))
        (x_min, y_min), (x_max, y_max) = cobweb.extent_points()
        self.assertEqual((x_min, x_max), cobweb.x_range)
        self.assertLessEqual(y_min, 0.0)
        self.assertEqual(cobweb.curve.evaluate(0.0), (0.0, 1.0))


class _UndoRecorder:
    def __init__(self) -> None:
        self.archives = 0

    def archive(self) -> None:
        self.archives += 1


class _NameGenerator:
    def filter_string(self, value: str) -> str:
        return "".join(ch for ch in value if ch.isalnum() or ch == "_")


class TestSequencePlotManager(unittest.TestCase):
    def setUp(self) -> None:
        self.drawables = DrawablesContainer()
        self.undo = _UndoRecorder()
        self.canvas = SimpleNamespace(draw_enabled=False, undo_redo_manager=self.undo)
        self.manager = SequencePlotManager(self.canvas, self.drawables, _NameGenerator())

    def test_sequence_result(self) -> None:
        result = self.manager.plot_sequence(expression="u(n-1) + u(n-2)", start=1, end=10, initial_terms=[1, 1])
        self.assertEqual(result["plot_name"], "sequence")
        self.assertEqual((result["kind"], result["order"]), ("recursive", 2))
        self.assertEqual(result["terms"], [1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
        self.assertEqual((result["min"], result["max"], result["trend"]), (1, 55, "increasing"))
        self.assertNotIn("note", result)
        self.assertEqual(self.undo.archives, 1)

    def test_long_sequences_list_their_ends(self) -> None:
        result = self.manager.plot_sequence(expression="1/(n-50)", start=1, end=200, name="harmonic")
        self.assertEqual(len(result["first_terms"]), 20)
        self.assertEqual(len(result["last_terms"]), 10)
        self.assertEqual((result["undefined_at"], result["undefined_count"]), ([50], 1))
        self.assertIn("undefined", result["note"])
        self.assertEqual(result["terms" if "terms" in result else "first_terms"][0], -0.0204081632653)

    def test_cobweb_results(self) -> None:
        fixed = self.manager.plot_cobweb(function="cos(x)", x0=1)
        self.assertEqual(
            (fixed["behaviour"], fixed["fixed_point"], fixed["stable"]), ("fixed_point", 0.739085133215, True)
        )
        self.assertEqual((fixed["steps"], len(fixed["orbit"])), (20, 21))
        self.assertIn("attracts", fixed["note"])
        cycle = self.manager.plot_cobweb(function="3.2*x*(1-x)", x0=0.1, steps=5)
        self.assertEqual((cycle["behaviour"], cycle["period"]), ("cycle", 2))
        escapes = self.manager.plot_cobweb(function="2x", x0=1)
        self.assertEqual(escapes["behaviour"], "escapes")
        self.assertIn("the web is drawn for 9 of the 20 steps", escapes["note"])
        chaotic = self.manager.plot_cobweb(function="3.9*x*(1-x)", x0=0.2)
        self.assertIn("may be chaotic", chaotic["note"])
        self.assertEqual(
            [d.name for d in self.drawables.get_by_class_name("CobwebPlot")],
            ["cobweb", "cobweb_1", "cobweb_2", "cobweb_3"],
        )

    def test_an_unknown_slope_is_worded_as_unknown(self) -> None:
        result = self.manager.plot_cobweb(function="sqrt(x)", x0=0)
        self.assertEqual((result["fixed_point"], result["slope"], result["stable"]), (0.0, None, None))
        self.assertIn("slope of g cannot be computed there", result["note"])
        self.assertNotIn("None", result["note"])

    def test_errors_add_nothing(self) -> None:
        for call in (
            lambda: self.manager.plot_cobweb(function="ln(x)", x0=0),
            lambda: self.manager.plot_cobweb(function="cos(x)", x0=math.nan),
            lambda: self.manager.plot_sequence(expression="k^2", start=1, end=3),
            lambda: self.manager.plot_sequence(expression="1/n", start=3, end=1),
        ):
            with self.assertRaises(ValueError):
                call()
        self.assertEqual(self.undo.archives, 0)
        self.assertEqual(self.drawables.get_by_class_name("CobwebPlot"), [])


class TestCanvasText(unittest.TestCase):
    def state(self, *plots: Any) -> Dict[str, Any]:
        buckets: Dict[str, Any] = {}
        for plot in plots:
            buckets.setdefault(plot.get_class_name() + "s", []).append(plot.get_state())
        return buckets

    def test_text_lines(self) -> None:
        fib = SequencePlot("fib", rule=parse_sequence("u(n-1) + u(n-2)"), start=1, end=12, initial_terms=[1, 1])
        inverse = SequencePlot("inv", rule=parse_sequence("1/(n-3)"), start=1, end=40, color="red")
        web = CobwebPlot("web", function=parse_iteration_function("3.2*x*(1-x)"), x0=0.1, steps=50)
        text = render_state(self.state(fib, inverse, web), "text", None)
        self.assertIn(
            "fib = Sequence(u(n) = u(n-1) + u(n-2), u(1) = 1, u(2) = 1; n 1..12; "
            "terms 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144; min 1, max 144; increasing)",
            text,
        )
        self.assertIn("inv = Sequence(u(n) = 1/(n-3); n 1..40; terms -0.5, -1, 1,", text)
        self.assertIn("... (27 more); min -1, max 1; neither; undefined at n = 3) color red", text)
        self.assertIn("web = Cobweb(g(x) = 3.2*x*(1-x); x0 0.1; 50 steps; orbit 0.1, 0.288,", text)
        self.assertIn("2-cycle 0.513045, 0.799455 (stable))", text)
        self.assertNotIn("summary", text)

    def test_unknown_stability_and_slow_orbits_are_worded(self) -> None:
        root = CobwebPlot("r", function=parse_iteration_function("sqrt(x)"), x0=0.0, steps=5)
        slow = CobwebPlot("s", function=parse_iteration_function("sin(x)"), x0=1.0, steps=5)
        text = render_state(self.state(root, slow), "text", None)
        self.assertIn("fixed point 0 (stability unknown)", text)
        self.assertIn("no fixed point or cycle yet, still moving one way after 2000 iterations", text)

    def test_min_json_shortens_long_lists(self) -> None:
        plot = SequencePlot("S", rule=parse_sequence("n"), start=1, end=500)
        rendered = json.loads(render_state(self.state(plot), "min_json", 1500))
        summary = rendered["SequencePlots"][0]["summary"]
        self.assertEqual(summary["terms"][-1], "... 500 values in all")
        self.assertEqual(len(summary["terms"]), 13)

    def test_view_note_boxes(self) -> None:
        sequence = SequencePlot("S", rule=parse_sequence("n^2"), start=-2, end=3)
        web = CobwebPlot("W", function=parse_iteration_function("x/2"), x0=4.0, steps=3)
        state: Dict[str, Any] = {"Points": []}
        state.update(self.state(sequence, web))
        boxes = {key[1]: shape.box for key, shape in measure_shapes(state).items()}
        self.assertEqual((boxes["S"].left, boxes["S"].right, boxes["S"].bottom, boxes["S"].top), (-2, 3, 0, 9))
        self.assertEqual((boxes["W"].left, boxes["W"].right), web.x_range)
        self.assertEqual((boxes["W"].bottom, boxes["W"].top), (0.0, 4.0))


if __name__ == "__main__":
    unittest.main()
