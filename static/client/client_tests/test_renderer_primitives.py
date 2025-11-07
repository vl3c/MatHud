from __future__ import annotations

import math
import unittest
import unittest.mock
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from coordinate_mapper import CoordinateMapper
from drawables.point import Point
from drawables.segment import Segment
from drawables.circle import Circle
from drawables.vector import Vector
from drawables.angle import Angle
from drawables.function import Function

from rendering.canvas2d_renderer import Canvas2DRenderer
from rendering.svg_renderer import SvgRenderer
from rendering.svg_primitive_adapter import SvgPrimitiveAdapter
from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
from rendering.webgl_primitive_adapter import WebGLPrimitiveAdapter
from rendering.shared_drawable_renderers import (
    RendererPrimitives,
    FillStyle,
    render_vector_helper,
)
from .simple_mock import SimpleMock


Point2D = Tuple[float, float]


# ---------------------------------------------------------------------------
# SVG recording infrastructure
# ---------------------------------------------------------------------------


class MockSvgModule:
    def circle(self, **attrs: Any) -> SimpleMock:
        return SimpleMock(element_type="circle", attrs=dict(attrs), style={})

    def line(self, **attrs: Any) -> SimpleMock:
        return SimpleMock(element_type="line", attrs=dict(attrs), style={})

    def ellipse(self, **attrs: Any) -> SimpleMock:
        return SimpleMock(element_type="ellipse", attrs=dict(attrs), style={})

    def polygon(self, **attrs: Any) -> SimpleMock:
        return SimpleMock(element_type="polygon", attrs=dict(attrs), style={})

    def path(self, **attrs: Any) -> SimpleMock:
        return SimpleMock(element_type="path", attrs=dict(attrs), style={})

    def text(self, *args: Any, **attrs: Any) -> SimpleMock:
        if args:
            attrs = dict(attrs)
            attrs["__text"] = args[0]
        return SimpleMock(element_type="text", attrs=dict(attrs), style={})


class MockSvgSurface:
    def __init__(self, log: List[Any]) -> None:
        self.log = log

    def clear(self) -> None:
        self.log.append(("clear",))

    def __le__(self, other: SimpleMock) -> "MockSvgSurface":
        self.log.append(
            (
                other.element_type,
                dict(other.attrs),
                dict(getattr(other, "style", {})),
            )
        )
        return self


class MockDocument:
    def __init__(self, surface: MockSvgSurface) -> None:
        self.surface = surface

    def getElementById(self, element_id: str) -> Optional[MockSvgSurface]:
        if element_id == "math-svg":
            return self.surface
        return None

    def __le__(self, element: Any) -> "MockDocument":
        return self

    def __getitem__(self, item: str) -> Any:
        if item == "math-svg":
            return self.surface
        raise KeyError(item)


# ---------------------------------------------------------------------------
# Canvas recording infrastructure
# ---------------------------------------------------------------------------


class RecordingCanvasContext:
    def __init__(self) -> None:
        self.log: List[Any] = []
        self._strokeStyle = "#000"
        self._lineWidth = 1.0
        self._fillStyle = "#000"
        self._globalAlpha = 1.0
        self._font = ""
        self._textAlign = "left"
        self._textBaseline = "alphabetic"

    # Properties to log style changes
    @property
    def strokeStyle(self) -> str:
        return self._strokeStyle

    @strokeStyle.setter
    def strokeStyle(self, value: str) -> None:
        self._strokeStyle = value
        self.log.append(("strokeStyle", value))

    @property
    def lineWidth(self) -> float:
        return self._lineWidth

    @lineWidth.setter
    def lineWidth(self, value: float) -> None:
        self._lineWidth = value
        self.log.append(("lineWidth", value))

    @property
    def fillStyle(self) -> str:
        return self._fillStyle

    @fillStyle.setter
    def fillStyle(self, value: str) -> None:
        self._fillStyle = value
        self.log.append(("fillStyle", value))

    @property
    def globalAlpha(self) -> float:
        return self._globalAlpha

    @globalAlpha.setter
    def globalAlpha(self, value: float) -> None:
        self._globalAlpha = value
        self.log.append(("globalAlpha", value))

    @property
    def font(self) -> str:
        return self._font

    @font.setter
    def font(self, value: str) -> None:
        self._font = value
        self.log.append(("font", value))

    @property
    def textAlign(self) -> str:
        return self._textAlign

    @textAlign.setter
    def textAlign(self, value: str) -> None:
        self._textAlign = value
        self.log.append(("textAlign", value))

    @property
    def textBaseline(self) -> str:
        return self._textBaseline

    @textBaseline.setter
    def textBaseline(self, value: str) -> None:
        self._textBaseline = value
        self.log.append(("textBaseline", value))

    # Canvas state methods
    def save(self) -> None:
        self.log.append(("save",))

    def restore(self) -> None:
        self.log.append(("restore",))

    def beginPath(self) -> None:
        self.log.append(("beginPath",))

    def moveTo(self, x: float, y: float) -> None:
        self.log.append(("moveTo", x, y))

    def lineTo(self, x: float, y: float) -> None:
        self.log.append(("lineTo", x, y))

    def arc(self, x: float, y: float, radius: float, start: float, end: float, anticlockwise: bool = False) -> None:
        self.log.append(("arc", x, y, radius, start, end, anticlockwise))

    def ellipse(self, x: float, y: float, radius_x: float, radius_y: float, rotation: float, start: float, end: float) -> None:
        self.log.append(("ellipse", x, y, radius_x, radius_y, rotation, start, end))

    def closePath(self) -> None:
        self.log.append(("closePath",))

    def stroke(self) -> None:
        self.log.append(("stroke", self._strokeStyle, self._lineWidth))

    def fill(self) -> None:
        self.log.append(("fill", self._fillStyle, self._globalAlpha))

    def clearRect(self, x: float, y: float, width: float, height: float) -> None:
        self.log.append(("clearRect", x, y, width, height))

    def fillText(self, text: str, x: float, y: float) -> None:
        self.log.append(("fillText", text, x, y, self._fillStyle, self._font, self._textAlign, self._textBaseline))


class MockCanvasElement:
    def __init__(self) -> None:
        self.width = 400
        self.height = 300
        self.attrs: Dict[str, Any] = {}
        self.style: Dict[str, Any] = {}
        self._ctx = RecordingCanvasContext()

    def getContext(self, context_type: str) -> RecordingCanvasContext:
        return self._ctx


# ---------------------------------------------------------------------------
# Parity helper utilities
# ---------------------------------------------------------------------------


def reset_svg_environment(log: List[Any]) -> Tuple[MockSvgModule, MockDocument]:
    log.clear()
    surface = MockSvgSurface(log)
    mock_svg = MockSvgModule()
    document = MockDocument(surface)
    return mock_svg, document


def reset_canvas_environment(renderer: Canvas2DRenderer, mock_canvas: MockCanvasElement) -> RecordingCanvasContext:
    renderer.canvas_el = mock_canvas
    renderer.ctx = mock_canvas.getContext("2d")
    renderer._shared_primitives = Canvas2DPrimitiveAdapter(mock_canvas)
    mock_canvas._ctx.log.clear()
    return mock_canvas._ctx


def normalize_svg_log(log: Sequence[Any]) -> List[Any]:
    return list(log)


def normalize_canvas_log(log: Sequence[Any]) -> List[Any]:
    return list(log)


# ---------------------------------------------------------------------------
# Primitive recorder for cartesian helper verification
# ---------------------------------------------------------------------------


class RecordingPrimitives(RendererPrimitives):
    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []

    def begin_shape(self) -> None:
        self.calls.append(("begin_shape",))

    def end_shape(self) -> None:
        self.calls.append(("end_shape",))

    def stroke_line(self, start: Point2D, end: Point2D, stroke: Any, *, include_width: bool = True) -> None:
        self.calls.append(("stroke_line", start, end, stroke.color, stroke.width))

    def draw_text(
        self,
        text: str,
        position: Point2D,
        font: Any,
        color: str,
        alignment: Any,
        style_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.calls.append(("draw_text", text, position, color))

    # The remaining primitives are not expected in cartesian rendering; raise if invoked.
    def stroke_polyline(self, points: List[Point2D], stroke: Any) -> None:
        raise AssertionError("Unexpected stroke_polyline call")

    def stroke_circle(self, center: Point2D, radius: float, stroke: Any) -> None:
        raise AssertionError("Unexpected stroke_circle call")

    def fill_circle(self, center: Point2D, radius: float, fill: Any, stroke: Optional[Any] = None) -> None:
        raise AssertionError("Unexpected fill_circle call")

    def stroke_ellipse(self, center: Point2D, radius_x: float, radius_y: float, rotation_rad: float, stroke: Any) -> None:
        raise AssertionError("Unexpected stroke_ellipse call")

    def fill_polygon(self, points: List[Point2D], fill: Any, stroke: Optional[Any] = None) -> None:
        raise AssertionError("Unexpected fill_polygon call")

    def fill_joined_area(self, forward: List[Point2D], reverse: List[Point2D], fill: Any) -> None:
        raise AssertionError("Unexpected fill_joined_area call")

    def stroke_arc(
        self,
        center: Point2D,
        radius: float,
        start_angle_rad: float,
        end_angle_rad: float,
        sweep_clockwise: bool,
        stroke: Any,
        css_class: Optional[str] = None,
    ) -> None:
        raise AssertionError("Unexpected stroke_arc call")

    def clear_surface(self) -> None:
        raise AssertionError("Unexpected clear_surface call")

    def resize_surface(self, width: float, height: float) -> None:
        raise AssertionError("Unexpected resize_surface call")


class MockWebGLRenderer:
    def __init__(self) -> None:
        self.log: List[Tuple[str, Any]] = []

    def _draw_lines(self, points: Sequence[Point2D], color: Any) -> None:
        self.log.append(("lines", list(points), color))

    def _draw_line_strip(self, points: Sequence[Point2D], color: Any) -> None:
        self.log.append(("line_strip", list(points), color))

    def _draw_points(self, points: Sequence[Point2D], color: Any, size: float) -> None:
        self.log.append(("points", list(points), color, size))

    def _parse_color(self, color: str) -> str:
        return color

    def clear(self) -> None:
        self.log.append(("clear",))

    def _resize_viewport(self) -> None:
        self.log.append(("resize",))


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestRendererPrimitives(unittest.TestCase):
    maxDiff = None

    def _assert_line_present(
        self,
        calls: List[Tuple[Any, ...]],
        color: str,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> None:
        for entry in calls:
            if entry[0] != "stroke_line":
                continue
            _, call_start, call_end, call_color, _ = entry
            if call_color != color:
                continue
            if all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(call_start, start)) and all(
                math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(call_end, end)
            ):
                return
        self.fail(f"Expected stroke_line from {start} to {end} with color {color}")

    def setUp(self) -> None:
        self.mapper = CoordinateMapper(400, 300)
        self.point_a = Point(1, 2, name="A", color="red")
        self.point_b = Point(3, -1, name="B", color="blue")
        self.point_c = Point(-2, 4, name="C", color="green")
        self.segment_ab = Segment(self.point_a, self.point_b, color="purple")
        self.circle = Circle(self.point_a, radius=2.5, color="orange")
        self.vector_ab = Vector(self.point_a, self.point_b, color="teal")
        self.angle_abc = Angle(self.point_b, self.point_a, self.point_c, color="magenta")
        self.function = Function("x", name="f", color="brown")

    def test_svg_primitives_render_shapes(self) -> None:
        log: List[Any] = []
        mock_svg, mock_document = reset_svg_environment(log)

        renderer = SvgRenderer()

        with unittest.mock.patch("rendering.svg_renderer.svg", mock_svg), \
                unittest.mock.patch("rendering.svg_renderer.document", mock_document), \
                unittest.mock.patch("rendering.svg_primitive_adapter.svg", mock_svg), \
                unittest.mock.patch("rendering.svg_primitive_adapter.document", mock_document):
            renderer._shared_primitives = SvgPrimitiveAdapter("math-svg")
            renderer._render_point(self.point_a, self.mapper)
            renderer._render_segment(self.segment_ab, self.mapper)
            renderer._render_circle(self.circle, self.mapper)
            renderer._render_vector(self.vector_ab, self.mapper)
            renderer._render_angle(self.angle_abc, self.mapper)
            renderer._render_function(self.function, self.mapper)

        element_types = [entry[0] for entry in normalize_svg_log(log)]
        self.assertIn("circle", element_types)
        self.assertIn("line", element_types)
        self.assertIn("polygon", element_types)
        self.assertTrue(any(et == "path" for et in element_types))

        text_entries = [entry[1].get("__text") for entry in log if len(entry) > 1 and entry[0] == "text"]
        self.assertTrue(any(text and text.startswith("A(") for text in text_entries))
        self.assertIn("f", text_entries)
        self.assertTrue(
            any(
                entry[0] == "path" and entry[1].get("class") == "angle-arc"
                for entry in log
                if len(entry) > 1
            )
        )

    def test_canvas_primitives_render_shapes(self) -> None:
        renderer = Canvas2DRenderer()
        mock_canvas = MockCanvasElement()
        ctx = reset_canvas_environment(renderer, mock_canvas)
        renderer._render_point(self.point_a, self.mapper)
        renderer._render_segment(self.segment_ab, self.mapper)
        renderer._render_circle(self.circle, self.mapper)
        renderer._render_vector(self.vector_ab, self.mapper)
        renderer._render_angle(self.angle_abc, self.mapper)
        renderer._render_function(self.function, self.mapper)

        operations = [entry[0] for entry in normalize_canvas_log(ctx.log)]
        self.assertIn("fill", operations)
        self.assertIn("stroke", operations)
        self.assertTrue(any(op == "arc" for op in operations))
        text_operations = [entry for entry in ctx.log if entry[0] == "fillText"]
        self.assertTrue(any(entry[1] == "f" for entry in text_operations))

    def test_canvas_cartesian_uses_shared_primitives(self) -> None:
        renderer = Canvas2DRenderer()
        mock_canvas = MockCanvasElement()
        reset_canvas_environment(renderer, mock_canvas)
        recorder = RecordingPrimitives()
        renderer._shared_primitives = recorder
        cartesian = SimpleNamespace(current_tick_spacing=50.0, default_tick_spacing=50.0)

        renderer.render_cartesian(cartesian, self.mapper)

        self.assertEqual(cartesian.width, mock_canvas.width)
        self.assertEqual(cartesian.height, mock_canvas.height)

        ox, oy = self.mapper.math_to_screen(0, 0)
        axis_color = renderer.style["cartesian_axis_color"]
        grid_color = renderer.style["cartesian_grid_color"]

        self._assert_line_present(recorder.calls, axis_color, (0.0, oy), (float(mock_canvas.width), oy))
        self._assert_line_present(recorder.calls, axis_color, (ox, 0.0), (ox, float(mock_canvas.height)))
        self.assertTrue(any(call[0] == "stroke_line" and call[3] == grid_color for call in recorder.calls))

        text_calls = [call for call in recorder.calls if call[0] == "draw_text"]
        self.assertTrue(any(call[1] == "O" for call in text_calls))
        self.assertTrue(any(call[1] != "O" for call in text_calls))
        self.assertIn(("begin_shape",), recorder.calls)
        self.assertIn(("end_shape",), recorder.calls)

    def test_svg_cartesian_uses_shared_primitives(self) -> None:
        renderer = SvgRenderer()
        recorder = RecordingPrimitives()
        renderer._shared_primitives = recorder
        cartesian = SimpleNamespace(width=400.0, height=300.0, current_tick_spacing=50.0, default_tick_spacing=50.0)

        renderer.render_cartesian(cartesian, self.mapper)

        ox, oy = self.mapper.math_to_screen(0, 0)
        axis_color = renderer.style["cartesian_axis_color"]
        grid_color = renderer.style["cartesian_grid_color"]

        self._assert_line_present(recorder.calls, axis_color, (0.0, oy), (cartesian.width, oy))
        self._assert_line_present(recorder.calls, axis_color, (ox, 0.0), (ox, cartesian.height))
        self.assertTrue(any(call[0] == "stroke_line" and call[3] == grid_color for call in recorder.calls))

        text_calls = [call for call in recorder.calls if call[0] == "draw_text"]
        self.assertTrue(any(call[1] == "O" for call in text_calls))
        self.assertTrue(any(call[1] != "O" for call in text_calls))

    def test_webgl_fill_polygon_and_joined_area_do_not_raise(self) -> None:
        mock_renderer = MockWebGLRenderer()
        adapter = WebGLPrimitiveAdapter(mock_renderer)
        triangle = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        adapter.fill_polygon(triangle, FillStyle("orange"))
        forward = [(0.0, 0.0), (1.0, 0.0)]
        reverse = [(1.0, 1.0), (0.0, 1.0)]
        adapter.fill_joined_area(forward, reverse, FillStyle("purple"))

        line_strip_calls = [entry for entry in mock_renderer.log if entry[0] == "line_strip"]
        self.assertGreaterEqual(len(line_strip_calls), 2)
        first_path = line_strip_calls[0][1]
        self.assertEqual(first_path[0], first_path[-1], "fill_polygon should close the path")
        second_path = line_strip_calls[1][1]
        self.assertEqual(second_path[0], second_path[-1], "fill_joined_area should produce a closed outline")

    def test_webgl_vector_helper_uses_polygon_fallback(self) -> None:
        mock_renderer = MockWebGLRenderer()
        adapter = WebGLPrimitiveAdapter(mock_renderer)
        style = {
            "segment_stroke_width": 1,
            "vector_tip_size": 8,
            "point_radius": 2,
            "vector_color": "teal",
        }

        render_vector_helper(adapter, self.vector_ab, self.mapper, style)

        line_ops = [entry for entry in mock_renderer.log if entry[0] == "lines"]
        self.assertTrue(line_ops, "vector helper should draw the segment body")
        tip_calls = [entry for entry in mock_renderer.log if entry[0] == "line_strip"]
        self.assertTrue(tip_calls, "vector helper should approximate arrowhead with line strip fallback")

