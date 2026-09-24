"""Tests for static/canvas_state_formatter.py (canvas state rendering for LLM prompts)."""

from __future__ import annotations

import copy
import json
import os
import random
import unittest
from typing import Any, Dict

from static.canvas_state_formatter import (
    CHANGES_HEADER,
    CURRENT_HEADER,
    OMITTED_NOTE,
    format_expression,
    format_number,
    parse_canvas_format,
    render_delta,
    render_min_json,
    render_state,
    render_text,
    render_update,
)
from static.token_estimation import estimate_tokens_from_text

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "canvas_states")


def load_scene(name: str) -> Dict[str, Any]:
    """Load a captured get_canvas_state() payload from server_tests/fixtures/canvas_states."""
    with open(os.path.join(FIXTURES, f"{name}.json"), encoding="utf-8") as handle:
        return json.load(handle)


VIEW = {
    "Cartesian_System_Visibility": {"left_bound": -10, "right_bound": 10, "top_bound": 5, "bottom_bound": -5},
    "current_tick_spacing": 1,
    "default_tick_spacing": 100,
    "current_tick_spacing_repr": "1",
    "min_tick_spacing": 1e-06,
    "visible": True,
    "coordinate_system": {"mode": "cartesian"},
}


def with_view(**buckets: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = dict(buckets)
    state.update(copy.deepcopy(VIEW))
    return state


def point(name: str, x: float, y: float) -> Dict[str, Any]:
    return {"name": name, "args": {"position": {"x": x, "y": y}}}


def segment(p1: str, p2: str, **extra: Any) -> Dict[str, Any]:
    args: Dict[str, Any] = {"p1": p1, "p2": p2}
    args.update(extra)
    return {"name": p1 + p2, "args": args, "_p1_coords": [0, 0], "_p2_coords": [0, 0]}


class TestFormatNumber(unittest.TestCase):
    def test_snaps_float_noise(self) -> None:
        self.assertEqual(format_number(199.20000000000002), "199.2")
        self.assertEqual(format_number(1.2197482119507639e-14), "0")
        self.assertEqual(format_number(-3.659244635852291e-14), "0")
        self.assertEqual(format_number(2.9999999999999996), "3")

    def test_limits_significant_digits(self) -> None:
        self.assertEqual(format_number(3.3333333333), "3.33333")
        self.assertEqual(format_number(140.8556708123603), "140.856")
        self.assertEqual(format_number(1234567.891), "1.23457e6")
        self.assertEqual(format_number(0.000012345678), "1.23457e-5")
        self.assertEqual(format_number(3.0427698574338087, 4), "3.043")

    def test_integers_and_non_numbers(self) -> None:
        self.assertEqual(format_number(5), "5")
        self.assertEqual(format_number(-4.0), "-4")
        self.assertEqual(format_number(None), "None")
        self.assertEqual(format_number(True), "True")
        self.assertEqual(format_number(float("inf")), "inf")
        self.assertEqual(format_number("x"), "x")

    def test_expression_long_decimals_are_shortened(self) -> None:
        self.assertEqual(format_expression("(1.8595995550611797)*x + 1.806609936967"), "(1.8596)*x + 1.80661")
        self.assertEqual(format_expression("x^2 - 1.5*x + 0.25"), "x^2 - 1.5*x + 0.25")
        self.assertEqual(format_expression("a1.12345678"), "a1.12345678")


class TestRenderTextGolden(unittest.TestCase):
    """Golden outputs for real get_canvas_state() captures."""

    maxDiff = None

    def test_triangle_and_circle(self) -> None:
        expected = "\n".join(
            [
                "view x [-491, 491] y [-249, 249]; grid 100",
                "A = (0, 0)",
                "B = (4, 0)",
                "C = (1, 3)",
                "AB = Segment(A, B)  len 4",
                "BC = Segment(B, C)  len 4.24264",
                "CA = Segment(C, A)  len 3.16228",
                "ABC = Triangle(A, B, C)  scalene; sides AB=4 BC=4.24264 CA=3.16228; area 6",
                "A(3) = Circle(center A, r 3)  area 28.2743",
            ]
        )
        self.assertEqual(render_text(load_scene("triangle_circle")), expected)

    def test_mixed_medium_scene(self) -> None:
        expected = "\n".join(
            [
                "view x [-6, 6] y [-3.043, 3.043]; grid 1",
                "A = (0, 0)",
                "B = (4, 0)",
                "C = (4, 3)",
                "D = (0, 3)",
                "E = (-2, 1.5)",
                "F = (2, -1)",
                "G = (-3, -2)",
                "H = (1.5, 2.25)",
                "I = (-1, 0.5)",
                "J = (3.33333, 1.1)",
                "AB = Segment(A, B)  len 4",
                "BC = Segment(B, C)  len 3",
                'AC = Segment(A, C)  len 5 label "diag"',
                "EF = Segment(E, F)  len 4.71699",
                "GH = Segment(G, H)  len 6.18971",
                "ABC = Triangle(A, B, C)  scalene right; sides AB=4 BC=3 CA=5; area 6",
                "angle_BAC = Angle(AB, AC)  vertex A, 36.8699 deg",
                "f(x) = x^2 - 1  on [-5, 5]",
                "g(x) = sin(x)*2",
                "e1 = Curve(x(t) = 3*cos(t), y(t) = 2*sin(t), t in [0, 2pi]) color purple",
                "area_between_f_and_g = AreaBetween(f, g, x in [-1, 1])",
                'label_A = Text("Area between f and g" at (-4, 4))',
                'label_B = Text("rectangle corner" at (2, 3.5))',
            ]
        )
        self.assertEqual(render_text(load_scene("mixed_medium")), expected)

    def test_weighted_graph_edges_replace_segments(self) -> None:
        expected = "\n".join(
            [
                "view x [-491, 491] y [-249, 249]; grid 100",
                "A = (199.2, 0)",
                "B = (140.856, 140.856)",
                "C = (0, 199.2)",
                "D = (-140.856, 140.856)",
                "E = (-199.2, 0)",
                "F = (-140.856, -140.856)",
                "G = (0, -199.2)",
                "H = (140.856, -140.856)",
                "G1 = Graph(undirected weighted; vertices A B C D E F G H)",
                "  edges (12): A-B 4, A-C 2, B-C 5, B-D 10, C-E 3, E-D 4, D-F 11, E-F 7, F-G 1, G-H 6, D-H 8, C-G 9",
            ]
        )
        self.assertEqual(render_text(load_scene("weighted_graph")), expected)

    def test_duplicate_names_are_flagged_once(self) -> None:
        text = render_text(load_scene("regression_duplicates"))
        lines = text.splitlines()
        self.assertEqual(lines[0], "view x [-15, 25] y [-0.1426, 20.14]; grid 5")
        self.assertEqual(lines[1], "! duplicate names, tools cannot tell these apart: F x25 (points)")
        self.assertIn("F = (14.6, 29.505)", lines)
        self.assertIn("fit1(x) = (1.8596)*x + 1.80661  on [-1.35, 16.05]", lines)
        self.assertIn("sales = BarChart(Mon 12, Tue 19, Wed 7, Thu 15, Fri 22)  x_start -12", lines)
        self.assertEqual(len(lines), 2 + 30 + 2)

    def test_text_is_much_smaller_than_raw_json(self) -> None:
        for name in ("triangle_circle", "mixed_medium", "weighted_graph", "regression_duplicates"):
            state = load_scene(name)
            with self.subTest(scene=name):
                self.assertLess(
                    estimate_tokens_from_text(render_text(state)) * 2,
                    estimate_tokens_from_text(json.dumps(state)),
                )


class TestRenderTextBuckets(unittest.TestCase):
    """Each drawable bucket produced by the client's get_state() renders a readable line."""

    def render(self, **buckets: Any) -> str:
        return render_text(with_view(**buckets))

    def test_empty_canvas(self) -> None:
        self.assertEqual(self.render(), "view x [-10, 10] y [-5, 5]; grid 1\n(empty canvas)")

    def test_view_line_mentions_polar_and_hidden_axes(self) -> None:
        state = with_view()
        state["coordinate_system"] = {"mode": "polar"}
        state["visible"] = False
        self.assertTrue(
            render_text(state).startswith("view x [-10, 10] y [-5, 5]; grid 1; polar coordinates; axes hidden")
        )

    def test_vector(self) -> None:
        text = self.render(
            Points=[point("A", 0, 0), point("B", 3, 4)],
            Vectors=[
                {"name": "v1", "args": {"origin": "A", "tip": "B"}, "_origin_coords": [0, 0], "_tip_coords": [3, 4]}
            ],
        )
        self.assertIn("v1 = Vector(A -> B)  <3, 4> len 5", text)

    def test_rectangle_vertices_are_put_in_cyclic_order(self) -> None:
        # Rectangle.get_state lists vertex names alphabetically, not around the shape.
        text = self.render(
            Points=[point("A", 0, 0), point("B", 4, 3), point("C", 4, 0), point("D", 0, 3)],
            Rectangles=[{"name": "ACBD", "args": {"p1": "A", "p2": "B", "p3": "C", "p4": "D"}, "types": ["rectangle"]}],
        )
        self.assertIn("ACBD = Rectangle(A, C, B, D); sides AC=4 CB=3 BD=4 DA=3; area 12", text)

    def test_generic_polygon_with_many_vertices(self) -> None:
        names = [f"P{i}" for i in range(12)]
        args = {f"p{i + 1}": name for i, name in enumerate(names)}
        text = self.render(
            Points=[point(n, i, i % 2) for i, n in enumerate(names)],
            GenericPolygons=[{"name": "poly", "args": args, "types": []}],
        )
        self.assertIn("poly = Polygon(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11)", text)

    def test_polygon_types_and_missing_points_skip_facts(self) -> None:
        text = self.render(
            Quadrilaterals=[
                {
                    "name": "q",
                    "args": {"p1": "A", "p2": "B", "p3": "C", "p4": "D"},
                    "types": ["quadrilateral", "square"],
                }
            ]
        )
        self.assertIn("q = Quadrilateral(A, B, C, D)  square", text)
        self.assertNotIn("area", text)

    def test_circle_passes_through_points(self) -> None:
        text = self.render(
            Points=[point("O", 1, 1), point("P", 4, 5), point("Q", 2, 2)],
            Circles=[{"name": "c", "args": {"center": "O", "radius": 5, "circle_formula": "..."}}],
        )
        self.assertIn("c = Circle(center O, r 5)  area 78.5398; passes through P", text)
        self.assertNotIn("formula", text)

    def test_ellipse_with_rotation(self) -> None:
        text = self.render(
            Points=[point("O", 0, 0)],
            Ellipses=[
                {
                    "name": "e",
                    "args": {
                        "center": "O",
                        "radius_x": 3,
                        "radius_y": 2,
                        "rotation_angle": 30,
                        "ellipse_formula": "...",
                    },
                }
            ],
        )
        self.assertIn("e = Ellipse(center O, rx 3, ry 2, rotation 30 deg)  area 18.8496", text)

    def test_circle_arc_sweep_and_length(self) -> None:
        arc_args = {
            "point1_name": "P",
            "point2_name": "Q",
            "center_x": 0,
            "center_y": 0,
            "radius": 2,
            "circle_name": "c",
            "use_major_arc": True,
            "color": "black",
        }
        text = self.render(
            Points=[point("P", 2, 0), point("Q", 0, 2)],
            CircleArcs=[{"name": "arc1", "type": "circle_arc", "args": arc_args}],
        )
        self.assertIn("arc1 = Arc(P to Q, on c, center (0, 0), r 2) major  sweep 270 deg, length 9.42478", text)

    def test_reflex_angle(self) -> None:
        text = self.render(
            Points=[point("A", 0, 0), point("B", 1, 0), point("C", 0, 1)],
            Segments=[segment("A", "B"), segment("A", "C")],
            Angles=[
                {
                    "name": "ang",
                    "type": "angle",
                    "args": {"segment1_name": "AB", "segment2_name": "AC", "color": "red", "is_reflex": True},
                }
            ],
        )
        self.assertIn("ang = Angle(AB, AC) reflex  vertex A, 270 deg color red", text)

    def test_function_features_and_unknown_args(self) -> None:
        text = self.render(
            Functions=[
                {
                    "name": "t",
                    "args": {
                        "function_string": "tan(x)",
                        "left_bound": None,
                        "right_bound": 3,
                        "vertical_asymptotes": [1.5707963267948966, -1.5707963267948966],
                        "future_field": {"a": 0.30000000000000004},
                    },
                }
            ]
        )
        self.assertIn(
            't(x) = tan(x)  on [-inf, 3]; vertical asymptotes x = 1.5708, -1.5708  [future_field {"a":0.3}]', text
        )

    def test_piecewise_function(self) -> None:
        pieces = [
            {"expression": "x^2", "left": None, "right": 0, "left_inclusive": False, "right_inclusive": False},
            {"expression": "x", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": False},
        ]
        text = self.render(PiecewiseFunctions=[{"name": "pw", "args": {"pieces": pieces}}])
        self.assertIn("pw(x) = piecewise { x^2 on (-inf, 0); x on [0, inf) }", text)

    def test_colored_areas(self) -> None:
        text = self.render(
            FunctionSegmentBoundedColoredAreas=[
                {"name": "a1", "args": {"color": "lightblue", "opacity": 0.3, "func": "f", "segment": "AB"}}
            ],
            SegmentsBoundedColoredAreas=[
                {"name": "a2", "args": {"color": "red", "opacity": 0.5, "segment1": "AB", "segment2": "x_axis"}}
            ],
            ClosedShapeColoredAreas=[
                {
                    "name": "a3",
                    "args": {
                        "color": "lightblue",
                        "opacity": 0.3,
                        "shape_type": "circle",
                        "segments": [],
                        "circle": "c1",
                        "ellipse": None,
                        "chord_segment": None,
                        "arc_clockwise": False,
                        "resolution": 64,
                        "expression": None,
                        "points": [[0, 0], [1, 0]],
                        "geometry_snapshot": {"big": [1, 2, 3]},
                    },
                }
            ],
        )
        self.assertIn("a1 = AreaBetween(f, segment AB)", text)
        self.assertIn("a2 = AreaBetween(segment AB, x_axis)  color red opacity 0.5", text)
        self.assertIn("a3 = ShadedRegion(circle; circle c1)", text)
        self.assertNotIn("geometry_snapshot", text)

    def test_directed_graph_and_tree(self) -> None:
        text = self.render(
            Points=[point("A", 0, 0), point("B", 1, 0), point("C", 2, 0)],
            Vectors=[
                {"name": "AB", "args": {"origin": "A", "tip": "B"}},
                {"name": "BC", "args": {"origin": "B", "tip": "C", "label": {"text": "7", "visible": True}}},
            ],
            Segments=[segment("A", "C")],
            DirectedGraphs=[{"name": "D1", "args": {"vectors": ["AB", "BC"]}}],
            Trees=[{"name": "T1", "args": {"segments": ["AC"], "root": "A"}}],
        )
        self.assertIn("D1 = Graph(directed weighted; vertices A B C)\n  edges (2): A->B, B->C 7", text)
        self.assertIn("T1 = Graph(tree root A; vertices A C)\n  edges (1): A-C", text)
        self.assertNotIn("Vector(", text)
        self.assertNotIn("Segment(", text)

    def test_plots(self) -> None:
        text = self.render(
            ContinuousPlots=[
                {
                    "name": "n1",
                    "args": {
                        "plot_type": "distribution",
                        "distribution_type": "normal",
                        "distribution_params": {"mean": 0, "sigma": 1},
                        "bounds": {"left": -4, "right": 4},
                        "metadata": None,
                        "function_name": "n1_pdf",
                        "fill_area_name": "n1_fill",
                    },
                }
            ],
            DiscretePlots=[
                {
                    "name": "b1",
                    "args": {
                        "plot_type": "distribution",
                        "distribution_type": "binomial",
                        "distribution_params": {"n": 4, "p": 0.5},
                        "bounds": {"left": 0, "right": 4},
                        "metadata": {"bar_count": 5},
                        "bar_count": 5,
                        "bar_labels": None,
                        "curve_color": None,
                        "fill_color": None,
                        "fill_opacity": None,
                        "rectangle_names": ["r1"],
                        "fill_area_names": [],
                    },
                }
            ],
            BarsPlots=[
                {
                    "name": "bars",
                    "args": {
                        "plot_type": "bars",
                        "values": [1, 2.5],
                        "labels_below": ["a", "b"],
                        "labels_above": ["1", "2.5"],
                        "bar_spacing": 0.2,
                        "bar_width": 2,
                        "x_start": 0,
                        "y_base": 0,
                        "stroke_color": None,
                        "fill_color": "lightblue",
                        "fill_opacity": 0.3,
                    },
                }
            ],
        )
        self.assertIn("n1 = Distribution(normal; mean 0, sigma 1; on [-4, 4])  curve n1_pdf, fill n1_fill", text)
        self.assertIn("b1 = DiscreteDistribution(binomial; n 4, p 0.5; on [0, 4])  5 bars", text)
        self.assertIn("bars = BarChart(a 1, b 2.5)  bar_width 2 labels above 1/2.5", text)

    def test_label_non_default_style(self) -> None:
        label_args = {
            "position": {"x": 1, "y": 2},
            "text": "hello",
            "color": "red",
            "font_size": 20,
            "rotation_degrees": 45,
            "reference_scale_factor": 1,
            "visible": False,
            "render_mode": {"kind": "world"},
        }
        text = self.render(Labels=[{"name": "lab", "args": label_args}])
        self.assertIn('lab = Text("hello" at (1, 2)) rotation 45 deg font 20 hidden color red', text)

    def test_unknown_bucket_falls_back_to_compact_json(self) -> None:
        text = self.render(Spirals=[{"name": "s1", "args": {"turns": 3.0000000001, "_p1_coords": [1, 2]}}])
        self.assertIn('s1 = Spiral{"args":{"turns":3}}', text)

    def test_computations_are_listed(self) -> None:
        state = with_view()
        state["computations"] = [
            {"expression": "2+2", "result": 4},
            {"expression": "sqrt(2)", "result": 1.4142135623730951},
        ]
        text = render_text(state)
        self.assertIn("calc 2+2 = 4", text)
        self.assertIn("calc sqrt(2) = 1.41421", text)


class TestRenderTextBudget(unittest.TestCase):
    def _large_scene(self) -> Dict[str, Any]:
        rng = random.Random(3)
        state = load_scene("mixed_medium")
        names = []
        for index in range(120):
            name = f"P{index}"
            names.append(name)
            state["Points"].append(point(name, rng.uniform(-50, 50), rng.uniform(-30, 30)))
        for index in range(60):
            state["Segments"].append(segment(names[2 * index], names[2 * index + 1]))
        for index in range(6):
            state["Functions"].append(
                {
                    "name": f"h{index}",
                    "args": {"function_string": f"{index}*x^3 - x", "left_bound": None, "right_bound": None},
                }
            )
        return state

    def test_no_budget_renders_everything(self) -> None:
        text = render_text(self._large_scene())
        self.assertNotIn(OMITTED_NOTE, text)
        self.assertIn("P119 = (", text)

    def test_budget_is_respected_and_scene_defining_objects_are_kept(self) -> None:
        state = self._large_scene()
        text = render_text(state, budget_tokens=800)
        self.assertLessEqual(estimate_tokens_from_text(text), 800)
        self.assertIn(OMITTED_NOTE, text)
        # Points are packed several per line before anything is dropped.
        self.assertIn("A=(0, 0); B=(4, 0)", text)
        self.assertRegex(text, r"\.\.\. \d+ more points omitted")
        self.assertRegex(text, r"\.\.\. \d+ more segments omitted")
        for kept in ("ABC = Triangle(", "f(x) = x^2 - 1", "h5(x) =", "e1 = Curve("):
            self.assertIn(kept, text)

    def test_budget_that_fits_after_packing_drops_nothing(self) -> None:
        state = self._large_scene()
        full = render_text(state)
        text = render_text(state, budget_tokens=estimate_tokens_from_text(full) - 20)
        self.assertNotIn(OMITTED_NOTE, text)
        self.assertIn("P119=(", text)

    def test_zero_budget_means_unlimited(self) -> None:
        state = self._large_scene()
        self.assertEqual(render_text(state, budget_tokens=0), render_text(state))


class TestRenderMinJson(unittest.TestCase):
    def test_noise_and_defaults_removed(self) -> None:
        payload = json.loads(render_min_json(load_scene("weighted_graph")))
        self.assertEqual(payload["view"], [-491, 491, -249, 249])
        self.assertEqual(payload["grid"], 100)
        self.assertEqual(payload["Points"][0], {"name": "A", "position": [199.2, 0]})
        self.assertEqual(payload["Segments"][0], {"name": "AB", "p1": "A", "p2": "B", "label": "4"})
        self.assertNotIn("_p1_coords", json.dumps(payload))

    def test_labels_drop_default_style(self) -> None:
        payload = json.loads(render_min_json(load_scene("mixed_medium")))
        self.assertEqual(payload["Labels"][0], {"name": "label_A", "position": [-4, 4], "text": "Area between f and g"})
        self.assertEqual(payload["Triangles"][0]["types"], ["triangle", "scalene", "right"])


class TestRenderDelta(unittest.TestCase):
    def test_added_changed_removed(self) -> None:
        delta = render_delta(load_scene("mixed_medium"), load_scene("mixed_medium_after"))
        self.assertEqual(
            delta,
            "\n".join(
                [
                    "+ A(5) = Circle(center A, r 5)  area 78.5398; passes through C",
                    "~ I = (-1, 0.5)  ->  (-1, -0.5)",
                    "- EF (segment) removed",
                ]
            ),
        )

    def test_unchanged_state_gives_empty_delta(self) -> None:
        state = load_scene("mixed_medium")
        self.assertEqual(render_delta(state, copy.deepcopy(state)), "")

    def test_float_noise_is_not_a_change(self) -> None:
        before = with_view(Points=[point("A", 1, 2)])
        after = with_view(Points=[point("A", 1.0000000000001, 2)])
        self.assertEqual(render_delta(before, after), "")

    def test_moved_point_reports_dependent_facts(self) -> None:
        before = with_view(Points=[point("A", 0, 0), point("B", 3, 4)], Segments=[segment("A", "B")])
        after = with_view(Points=[point("A", 0, 0), point("B", 6, 8)], Segments=[segment("A", "B")])
        self.assertEqual(
            render_delta(before, after),
            "~ B = (3, 4)  ->  (6, 8)\n~ AB = Segment(A, B)  len 5  ->  Segment(A, B)  len 10",
        )

    def test_view_change(self) -> None:
        before = with_view()
        after = with_view()
        after["Cartesian_System_Visibility"] = {
            "left_bound": -20,
            "right_bound": 20,
            "top_bound": 10,
            "bottom_bound": -10,
        }
        self.assertEqual(render_delta(before, after), "~ view x [-20, 20] y [-10, 10]; grid 1")

    def test_new_computation(self) -> None:
        before = with_view()
        after = with_view()
        after["computations"] = [{"expression": "1+1", "result": 2}]
        self.assertEqual(render_delta(before, after), "+ calc 1+1 = 2")


class TestRenderUpdate(unittest.TestCase):
    def test_small_change_sends_delta(self) -> None:
        update = render_update(load_scene("mixed_medium"), load_scene("mixed_medium_after"), "text")
        self.assertTrue(update.startswith(CHANGES_HEADER + "\n+ A(5) = Circle"))

    def test_no_change_sends_nothing(self) -> None:
        state = load_scene("mixed_medium")
        self.assertEqual(render_update(state, copy.deepcopy(state), "text"), "")

    def test_unknown_previous_state_sends_full_state(self) -> None:
        state = load_scene("triangle_circle")
        self.assertEqual(render_update(None, state, "text"), f"{CURRENT_HEADER}\n{render_text(state)}")

    def test_cleared_canvas_sends_full_state_when_smaller(self) -> None:
        cleared = with_view()
        update = render_update(load_scene("weighted_graph"), cleared, "text")
        self.assertEqual(update, f"{CURRENT_HEADER}\n{render_text(cleared)}")

    def test_min_json_full_state(self) -> None:
        state = load_scene("triangle_circle")
        self.assertEqual(render_update(None, state, "min_json"), f"{CURRENT_HEADER}\n{render_min_json(state)}")


class TestFormatDispatch(unittest.TestCase):
    def test_render_state(self) -> None:
        state = load_scene("triangle_circle")
        self.assertEqual(render_state(state, "json"), json.dumps(state))
        self.assertEqual(render_state(state, "min_json"), render_min_json(state))
        self.assertEqual(render_state(state, "text"), render_text(state))

    def test_parse_canvas_format(self) -> None:
        self.assertEqual(parse_canvas_format(" Text "), "text")
        self.assertEqual(parse_canvas_format("MIN_JSON"), "min_json")
        self.assertEqual(parse_canvas_format("json"), "json")
        self.assertIsNone(parse_canvas_format("yaml"))
        self.assertIsNone(parse_canvas_format(None))


class TestTokenEstimation(unittest.TestCase):
    def test_digits_count_as_one_token_each(self) -> None:
        self.assertEqual(estimate_tokens_from_text("1234567890"), 10)
        self.assertGreater(estimate_tokens_from_text("199.20000000000002"), 15)

    def test_words_and_symbols(self) -> None:
        self.assertEqual(estimate_tokens_from_text(""), 0)
        self.assertEqual(estimate_tokens_from_text("hello world"), 2)
        self.assertEqual(estimate_tokens_from_text('{"a":'), 3)


if __name__ == "__main__":
    unittest.main()
