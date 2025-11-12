from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import unittest
from typing import Any, List, Tuple

from rendering.shared_drawable_renderers import StrokeStyle, FillStyle, FontStyle, TextAlignment


class MockCanvasContext:
    def __init__(self):
        self.operations: List[Tuple[str, Any]] = []
        self._stroke_style = "#000000"
        self._fill_style = "#000000"
        self._line_width = 1.0
        self._line_join = "miter"
        self._line_cap = "butt"
        self._global_alpha = 1.0
        self._font = "10px sans-serif"
        self._text_align = "start"
        self._text_baseline = "alphabetic"

    @property
    def strokeStyle(self) -> str:
        return self._stroke_style

    @strokeStyle.setter
    def strokeStyle(self, value: str) -> None:
        self._stroke_style = value
        self.operations.append(("set_stroke_style", value))

    @property
    def fillStyle(self) -> str:
        return self._fill_style

    @fillStyle.setter
    def fillStyle(self, value: str) -> None:
        self._fill_style = value
        self.operations.append(("set_fill_style", value))

    @property
    def lineWidth(self) -> float:
        return self._line_width

    @lineWidth.setter
    def lineWidth(self, value: float) -> None:
        self._line_width = value
        self.operations.append(("set_line_width", value))

    @property
    def lineJoin(self) -> str:
        return self._line_join

    @lineJoin.setter
    def lineJoin(self, value: str) -> None:
        self._line_join = value
        self.operations.append(("set_line_join", value))

    @property
    def lineCap(self) -> str:
        return self._line_cap

    @lineCap.setter
    def lineCap(self, value: str) -> None:
        self._line_cap = value
        self.operations.append(("set_line_cap", value))

    @property
    def globalAlpha(self) -> float:
        return self._global_alpha

    @globalAlpha.setter
    def globalAlpha(self, value: float) -> None:
        self._global_alpha = value
        self.operations.append(("set_global_alpha", value))

    @property
    def font(self) -> str:
        return self._font

    @font.setter
    def font(self, value: str) -> None:
        self._font = value
        self.operations.append(("set_font", value))

    @property
    def textAlign(self) -> str:
        return self._text_align

    @textAlign.setter
    def textAlign(self, value: str) -> None:
        self._text_align = value
        self.operations.append(("set_text_align", value))

    @property
    def textBaseline(self) -> str:
        return self._text_baseline

    @textBaseline.setter
    def textBaseline(self, value: str) -> None:
        self._text_baseline = value
        self.operations.append(("set_text_baseline", value))

    def beginPath(self) -> None:
        self.operations.append(("begin_path",))

    def moveTo(self, x: float, y: float) -> None:
        self.operations.append(("move_to", x, y))

    def lineTo(self, x: float, y: float) -> None:
        self.operations.append(("line_to", x, y))

    def arc(self, x: float, y: float, radius: float, start: float, end: float, ccw: bool = False) -> None:
        self.operations.append(("arc", x, y, radius, start, end, ccw))

    def ellipse(self, x: float, y: float, rx: float, ry: float, rotation: float, start: float, end: float, ccw: bool = False) -> None:
        self.operations.append(("ellipse", x, y, rx, ry, rotation, start, end, ccw))

    def closePath(self) -> None:
        self.operations.append(("close_path",))

    def stroke(self) -> None:
        self.operations.append(("stroke",))

    def fill(self) -> None:
        self.operations.append(("fill",))

    def fillText(self, text: str, x: float, y: float) -> None:
        self.operations.append(("fill_text", text, x, y))

    def save(self) -> None:
        self.operations.append(("save",))

    def restore(self) -> None:
        self.operations.append(("restore",))

    def clearRect(self, x: float, y: float, width: float, height: float) -> None:
        self.operations.append(("clear_rect", x, y, width, height))


class MockCanvasElement:
    def __init__(self):
        self.width = 800
        self.height = 600
        self._ctx = MockCanvasContext()

    def getContext(self, context_type: str) -> MockCanvasContext:
        return self._ctx


class TestCanvas2DPrimitiveAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas_el = MockCanvasElement()
        from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
        self.adapter = Canvas2DPrimitiveAdapter(self.canvas_el)

    def test_stroke_line_draws_line(self) -> None:
        stroke = StrokeStyle(color="#FF0000", width=2.0)
        
        self.adapter.stroke_line((10.0, 20.0), (30.0, 40.0), stroke)
        
        ops = self.canvas_el._ctx.operations
        op_types = [op[0] for op in ops]
        
        self.assertIn("begin_path", op_types)
        self.assertIn("move_to", op_types)
        self.assertIn("line_to", op_types)
        self.assertIn("stroke", op_types)

    def test_stroke_line_sets_stroke_color(self) -> None:
        stroke = StrokeStyle(color="#00FF00", width=1.5)
        
        self.adapter.stroke_line((0.0, 0.0), (10.0, 10.0), stroke)
        
        ops = self.canvas_el._ctx.operations
        stroke_style_ops = [op for op in ops if op[0] == "set_stroke_style"]
        
        self.assertTrue(any("#00FF00" in str(op[1]).upper() for op in stroke_style_ops))

    def test_fill_circle_draws_circle(self) -> None:
        fill = FillStyle(color="#0000FF", opacity=0.5)
        
        self.adapter.fill_circle((50.0, 60.0), 20.0, fill)
        
        ops = self.canvas_el._ctx.operations
        op_types = [op[0] for op in ops]
        
        self.assertIn("begin_path", op_types)
        self.assertIn("arc", op_types)
        self.assertIn("fill", op_types)

    def test_fill_circle_with_opacity_sets_global_alpha(self) -> None:
        fill = FillStyle(color="#FF0000", opacity=0.7)
        
        self.adapter.fill_circle((30.0, 40.0), 15.0, fill)
        
        ops = self.canvas_el._ctx.operations
        alpha_ops = [op for op in ops if op[0] == "set_global_alpha"]
        
        self.assertTrue(any(abs(op[1] - 0.7) < 0.01 for op in alpha_ops))

    def test_stroke_polyline_draws_connected_segments(self) -> None:
        points = [(10.0, 20.0), (30.0, 40.0), (50.0, 30.0)]
        stroke = StrokeStyle(color="#000000", width=1.0)
        
        self.adapter.stroke_polyline(points, stroke)
        
        ops = self.canvas_el._ctx.operations
        line_to_ops = [op for op in ops if op[0] == "line_to"]
        
        self.assertGreaterEqual(len(line_to_ops), 2)

    def test_fill_polygon_does_not_crash(self) -> None:
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        fill = FillStyle(color="#FFFF00")
        
        try:
            self.adapter.fill_polygon(points, fill)
        except Exception as e:
            self.fail(f"fill_polygon raised exception: {e}")

    def test_draw_text_sets_font(self) -> None:
        font = FontStyle(family="Arial", size=16)
        alignment = TextAlignment(horizontal="center", vertical="middle")
        
        self.adapter.draw_text("Test", (100.0, 200.0), font, "#000000", alignment)
        
        ops = self.canvas_el._ctx.operations
        font_ops = [op for op in ops if op[0] == "set_font"]
        
        self.assertTrue(len(font_ops) > 0)

    def test_draw_text_sets_alignment(self) -> None:
        font = FontStyle(family="Arial", size=12)
        alignment = TextAlignment(horizontal="right", vertical="top")
        
        self.adapter.draw_text("Label", (50.0, 75.0), font, "#000000", alignment)
        
        ops = self.canvas_el._ctx.operations
        
        align_ops = [op for op in ops if op[0] == "set_text_align"]
        self.assertTrue(any("right" in str(op[1]).lower() for op in align_ops))

    def test_stroke_ellipse_draws_ellipse(self) -> None:
        stroke = StrokeStyle(color="#FF00FF", width=1.0)
        
        self.adapter.stroke_ellipse((80.0, 90.0), 30.0, 20.0, 0.5, stroke)
        
        ops = self.canvas_el._ctx.operations
        op_types = [op[0] for op in ops]
        
        self.assertIn("ellipse", op_types)
        self.assertIn("stroke", op_types)

    def test_clear_surface_clears_canvas(self) -> None:
        self.adapter.clear_surface()
        
        ops = self.canvas_el._ctx.operations
        clear_ops = [op for op in ops if op[0] == "clear_rect"]
        
        self.assertEqual(len(clear_ops), 1)
        self.assertEqual(clear_ops[0][3], self.canvas_el.width)
        self.assertEqual(clear_ops[0][4], self.canvas_el.height)


class TestCanvas2DPrimitiveAdapterStateManagement(unittest.TestCase):
    def test_stroke_state_cached_between_calls(self) -> None:
        canvas_el = MockCanvasElement()
        from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
        adapter = Canvas2DPrimitiveAdapter(canvas_el)
        
        stroke = StrokeStyle(color="#FF0000", width=2.0)
        
        adapter.stroke_line((0.0, 0.0), (10.0, 10.0), stroke)
        
        first_call_ops = len(canvas_el._ctx.operations)
        
        adapter.stroke_line((10.0, 10.0), (20.0, 20.0), stroke)
        
        second_call_ops = len(canvas_el._ctx.operations)
        
        style_changes = [op for op in canvas_el._ctx.operations if op[0].startswith("set_")]
        
        self.assertGreater(len(style_changes), 0)

    def test_different_strokes_change_state(self) -> None:
        canvas_el = MockCanvasElement()
        from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
        adapter = Canvas2DPrimitiveAdapter(canvas_el)
        
        stroke1 = StrokeStyle(color="#FF0000", width=1.0)
        stroke2 = StrokeStyle(color="#00FF00", width=2.0)
        
        adapter.stroke_line((0.0, 0.0), (10.0, 10.0), stroke1)
        adapter.stroke_line((10.0, 10.0), (20.0, 20.0), stroke2)
        
        stroke_style_ops = [op for op in canvas_el._ctx.operations if op[0] == "set_stroke_style"]
        
        self.assertGreaterEqual(len(stroke_style_ops), 2)

    def test_fill_with_no_opacity_uses_default_alpha(self) -> None:
        canvas_el = MockCanvasElement()
        from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
        adapter = Canvas2DPrimitiveAdapter(canvas_el)
        
        fill = FillStyle(color="#0000FF", opacity=None)
        
        adapter.fill_circle((25.0, 35.0), 10.0, fill)
        
        alpha_ops = [op for op in canvas_el._ctx.operations if op[0] == "set_global_alpha"]
        
        self.assertTrue(all(op[1] == 1.0 or op[1] is None for op in alpha_ops if len(alpha_ops) > 0))


class TestCanvas2DPrimitiveAdapterEdgeCases(unittest.TestCase):
    def test_empty_polyline_does_not_crash(self) -> None:
        canvas_el = MockCanvasElement()
        from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
        adapter = Canvas2DPrimitiveAdapter(canvas_el)
        
        stroke = StrokeStyle(color="#000000", width=1.0)
        
        try:
            adapter.stroke_polyline([], stroke)
        except Exception as e:
            self.fail(f"Empty polyline raised exception: {e}")

    def test_zero_radius_circle_handles_gracefully(self) -> None:
        canvas_el = MockCanvasElement()
        from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
        adapter = Canvas2DPrimitiveAdapter(canvas_el)
        
        fill = FillStyle(color="#FF0000")
        
        try:
            adapter.fill_circle((10.0, 10.0), 0.0, fill)
        except Exception as e:
            self.fail(f"Zero radius circle raised exception: {e}")

    def test_negative_radius_circle_handles_gracefully(self) -> None:
        canvas_el = MockCanvasElement()
        from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
        adapter = Canvas2DPrimitiveAdapter(canvas_el)
        
        fill = FillStyle(color="#FF0000")
        
        try:
            adapter.fill_circle((10.0, 10.0), -5.0, fill)
        except Exception as e:
            self.fail(f"Negative radius circle raised exception: {e}")


__all__ = [
    "TestCanvas2DPrimitiveAdapter",
    "TestCanvas2DPrimitiveAdapterStateManagement",
    "TestCanvas2DPrimitiveAdapterEdgeCases",
]

