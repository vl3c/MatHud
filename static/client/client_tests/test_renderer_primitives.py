from __future__ import annotations

import unittest
import unittest.mock
from dataclasses import dataclass
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
# Test cases
# ---------------------------------------------------------------------------


class TestRendererPrimitives(unittest.TestCase):
    maxDiff = None
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
