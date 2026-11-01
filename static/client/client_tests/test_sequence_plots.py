"""Sequence plots and cobweb diagrams in the browser: rendering helpers, plans, the tool path and undo.

The parsing, terms, orbit classification, drawables' state and manager results are tested
with pytest (``server_tests/test_sequences.py``); these tests check the same code runs under
Brython (float and int arithmetic, the character scan of u(n-1)) and the canvas wiring.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Tuple

from canvas import Canvas
from constants import successful_call_message
from drawables.cobweb_plot import CobwebPlot
from drawables.sequence_plot import SequencePlot
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from rendering import shared_drawable_renderers as shared
from rendering.cached_render_plan import build_plan_for_drawable, has_plan_helper
from scenario_hooks import content_extent
from utils.sequences import parse_iteration_function, parse_sequence, sequence_terms
from workspace_manager import WorkspaceManager


class _Recorder(shared.RendererPrimitives):
    def __init__(self) -> None:
        self.calls: List[Tuple[str, tuple, dict]] = []

    def stroke_polyline(self, points, stroke, **kwargs):
        self.calls.append(("stroke_polyline", (list(points), stroke), dict(kwargs)))

    def fill_circle(self, center, radius, fill, stroke=None, **kwargs):
        self.calls.append(("fill_circle", (center, radius, fill), dict(kwargs)))

    def draw_text(self, text, position, font, color, alignment, style_overrides=None, **kwargs):
        self.calls.append(("draw_text", (text,), dict(kwargs)))

    def named(self, name: str) -> List[Tuple[str, tuple, dict]]:
        return [call for call in self.calls if call[0] == name]


STYLE: Dict[str, Any] = {
    "point_radius": 3,
    "point_color": "black",
    "function_color": "blue",
    "function_stroke_width": 1,
    "segment_stroke_width": 1,
    "label_color": "#000",
    "label_font_family": "Arial",
    "label_font_size": 12,
}


def _cobweb(text: str, x0: float, steps: int, color: Any = None) -> CobwebPlot:
    return CobwebPlot("W", function=parse_iteration_function(text), x0=x0, steps=steps, color=color)


class TestSequencePlotRendering(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.mapper = self.canvas.coordinate_mapper
        self.recorder = _Recorder()

    def test_sequence_draws_one_screen_space_dot_per_term(self) -> None:
        plot = SequencePlot("S", rule=parse_sequence("1/n"), start=1, end=4, color="red")
        shared.render_sequence_plot_helper(self.recorder, plot, self.mapper, STYLE)
        dots = self.recorder.named("fill_circle")
        expected = [self.mapper.math_to_screen(n, 1.0 / n) for n in (1, 2, 3, 4)]
        self.assertEqual([call[1][0] for call in dots], expected)
        self.assertEqual({call[1][2].color for call in dots}, {"red"})
        self.assertTrue(all(call[2].get("screen_space") for call in dots))

    def test_cobweb_draws_curve_diagonal_web_and_start_dot(self) -> None:
        plot = _cobweb("cos(x)", 1.0, 3)
        shared.render_cobweb_plot_helper(self.recorder, plot, self.mapper, STYLE)
        lines = self.recorder.named("stroke_polyline")
        colors = [call[1][1].color for call in lines]
        self.assertIn("blue", colors)  # the curve, sampled by the parametric helper
        self.assertIn("#888888", colors)  # the diagonal
        web = [call for call in lines if call[1][1].color == "#d62728"]
        self.assertEqual(len(web), 1)
        self.assertEqual(len(web[0][1][0]), 1 + 2 * 3)
        self.assertEqual(web[0][1][0][0], self.mapper.math_to_screen(1.0, 0.0))
        dots = self.recorder.named("fill_circle")
        self.assertEqual([call[1][0] for call in dots], [self.mapper.math_to_screen(1.0, 0.0)])
        self.assertEqual(self.recorder.named("draw_text"), [])  # the curve has no label

    def test_curve_breaks_where_g_is_undefined(self) -> None:
        plot = _cobweb("1/x", 2.0, 4)
        shared.render_cobweb_plot_helper(self.recorder, plot, self.mapper, STYLE)
        curve = [call for call in self.recorder.named("stroke_polyline") if call[1][1].color == "blue"]
        self.assertGreaterEqual(len(curve), 1)

    def test_both_kinds_have_a_plan(self) -> None:
        plots = [
            SequencePlot("S", rule=parse_sequence("u(n-1) + u(n-2)"), start=1, end=10, initial_terms=[1, 1]),
            _cobweb("3.2*x*(1-x)", 0.1, 30),
        ]
        for plot in plots:
            with self.subTest(plot=plot.get_class_name()):
                self.assertTrue(has_plan_helper(plot.get_class_name()))
                self.assertIsNotNone(build_plan_for_drawable(plot, self.mapper, STYLE))


class TestSequenceArithmetic(unittest.TestCase):
    """The parts that differ between Brython and CPython: number types and the text scan."""

    def test_previous_terms_with_an_implied_factor(self) -> None:
        terms = sequence_terms(parse_sequence("2u(n-1) + 1"), 1, 4, [1]).terms
        self.assertEqual(terms, [1.0, 3.0, 7.0, 15.0])

    def test_fibonacci_and_factorial_and_signs(self) -> None:
        fib = sequence_terms(parse_sequence("u(n-1) + u(n-2)"), 1, 10, [1, 1]).terms
        self.assertEqual(fib[-1], 55.0)
        self.assertEqual(sequence_terms(parse_sequence("factorial(n)"), 1, 5).terms, [1.0, 2.0, 6.0, 24.0, 120.0])
        self.assertEqual(sequence_terms(parse_sequence("(-1)^n/n"), 1, 3).terms[:2], [-1.0, 0.5])

    def test_refusals_carry_hints(self) -> None:
        for text, hint in (("3.2u(n-1)(1-u(n-1))", "Write every product with *"), ("i/2", "imaginary unit")):
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as caught:
                    parse_sequence(text)
                self.assertIn(hint, str(caught.exception))

    def test_cobweb_of_cos_settles_on_its_fixed_point(self) -> None:
        plot = _cobweb("cos(x)", 1.0, 20)
        self.assertEqual(plot.summary["behaviour"], "fixed_point")
        self.assertAlmostEqual(plot.summary["fixed_point"], 0.739085133215, places=9)
        self.assertTrue(plot.summary["stable"])


class TestSequencePlotTools(unittest.TestCase):
    """The tools through the model's tool path on a real canvas."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.workspace_manager = WorkspaceManager(self.canvas)
        self.functions = FunctionRegistry.get_available_functions(self.canvas, self.workspace_manager)
        self.undoable = FunctionRegistry.get_undoable_functions()

    def run_batch(self, *calls: Tuple[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        batch = [{"function_name": name, "arguments": dict(args)} for name, args in calls]
        _, traced = ProcessFunctionCalls.get_results_traced(batch, self.functions, self.undoable, self.canvas)
        return list(traced)

    def plot_names(self) -> List[str]:
        names: List[str] = []
        for class_name in ("SequencePlot", "CobwebPlot"):
            names.extend(str(d.name) for d in self.canvas.drawable_manager.drawables.get_by_class_name(class_name))
        return names

    def test_results_reach_the_model_in_full(self) -> None:
        traced = self.run_batch(
            (
                "plot_sequence",
                {"expression": "u(n-1) + u(n-2)", "start": 1, "end": 10, "initial_terms": [1, 1], "name": "Fib"},
            ),
            ("plot_cobweb", {"function": "cos(x)", "x0": 1, "steps": None, "name": None, "color": None}),
        )
        for call in traced:
            with self.subTest(tool=call.get("function_name")):
                self.assertFalse(call["is_error"], call["result"])
                self.assertNotEqual(call["result"], successful_call_message)
                self.assertIsInstance(call["result"], dict)
        self.assertEqual(traced[0]["result"]["terms"][-1], 55)
        self.assertEqual(traced[1]["result"]["behaviour"], "fixed_point")
        self.assertEqual(traced[1]["result"]["steps"], 20)
        self.assertEqual(self.plot_names(), ["Fib", "cobweb"])

    def test_one_batch_is_one_undo_step_and_redo_restores_it(self) -> None:
        self.run_batch(
            ("plot_sequence", {"expression": "1/n", "start": 1, "end": 5}),
            ("plot_cobweb", {"function": "x/2", "x0": 4}),
        )
        self.assertEqual(sorted(self.plot_names()), ["cobweb", "sequence"])
        self.assertEqual(len(self.canvas.undo_redo_manager.undo_stack), 1)
        self.canvas.undo()
        self.assertEqual(self.plot_names(), [])
        self.canvas.redo()
        self.assertEqual(sorted(self.plot_names()), ["cobweb", "sequence"])

    def test_errors_are_flagged_and_add_nothing(self) -> None:
        traced = self.run_batch(
            ("plot_sequence", {"expression": "u(n-1) + 1", "start": 1, "end": 5}),
            ("plot_cobweb", {"function": "ln(x)", "x0": -1}),
        )
        self.assertTrue(traced[0]["is_error"])
        self.assertIn("initial_terms", str(traced[0]["result"]))
        self.assertTrue(traced[1]["is_error"])
        self.assertIn("nothing to iterate", str(traced[1]["result"]))
        self.assertEqual(self.plot_names(), [])

    def test_delete_plot_and_transform_refusal(self) -> None:
        self.run_batch(("plot_cobweb", {"function": "cos(x)", "x0": 1, "name": "W"}))
        traced = self.run_batch(("translate_object", {"name": "W", "x_offset": 1, "y_offset": 0}))
        self.assertIn("plot_cobweb", str(traced[0]["result"]))
        self.run_batch(("delete_plot", {"name": "W"}))
        self.assertEqual(self.plot_names(), [])

    def test_content_extent_frames_the_plots(self) -> None:
        manager = self.canvas.drawable_manager.sequence_plot_manager
        manager.plot_sequence(expression="n", start=-2, end=6)
        self.assertEqual(content_extent(self.canvas), [-2.0, -2.0, 6.0, 6.0])

    def test_saved_workspace_restores_the_plots(self) -> None:
        manager = self.canvas.drawable_manager.sequence_plot_manager
        manager.plot_sequence(expression="u(n-1) + u(n-2)", start=0, end=8, initial_terms=[0, 1], name="Fib")
        manager.plot_cobweb(function="3.2*x*(1-x)", x0=0.1, steps=40, name="Web", color="green")
        state = self.workspace_manager._snapshot_persistable_canvas_state()
        self.canvas.clear()
        self.workspace_manager._restore_workspace_state(state)
        self.assertEqual(sorted(self.plot_names()), ["Fib", "Web"])
        drawables = self.canvas.drawable_manager.drawables
        fib = drawables.get_by_class_name("SequencePlot")[0]
        self.assertEqual(fib.summary["terms"][-1], 21.0)
        web = drawables.get_by_class_name("CobwebPlot")[0]
        self.assertEqual((web.plot_color, web.steps, web.summary["behaviour"]), ("green", 40, "cycle"))
