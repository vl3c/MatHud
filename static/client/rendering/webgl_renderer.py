from __future__ import annotations

import math
from typing import Any, Callable, Dict, Sequence, Tuple

from browser import document, html, window

from constants import default_color
from rendering.interfaces import RendererProtocol


class WebGLRenderer(RendererProtocol):
    """Experimental renderer backed by WebGL (line and point support)."""

    def __init__(self, canvas_id: str = "math-webgl") -> None:
        self.canvas_el = self._ensure_canvas(canvas_id)
        self.gl = self.canvas_el.getContext("webgl")
        if self.gl is None:
            raise RuntimeError("WebGL context unavailable")

        self._program = self._create_program()
        self.gl.useProgram(self._program)
        self._position_attrib = self.gl.getAttribLocation(self._program, "a_position")
        self._color_uniform = self.gl.getUniformLocation(self._program, "u_color")
        self._point_size_uniform = self.gl.getUniformLocation(self._program, "u_point_size")

        self._buffer = self.gl.createBuffer()
        self.gl.bindBuffer(self.gl.ARRAY_BUFFER, self._buffer)
        self.gl.enableVertexAttribArray(self._position_attrib)
        self.gl.vertexAttribPointer(self._position_attrib, 2, self.gl.FLOAT, False, 0, 0)

        self.gl.clearColor(0.0, 0.0, 0.0, 0.0)
        self._handlers_by_type: Dict[type, Callable[[Any, Any], None]] = {}
        self.register_default_drawables()

    def clear(self) -> None:
        self._resize_viewport()
        self.gl.clear(self.gl.COLOR_BUFFER_BIT)

    def render(self, drawable: Any, coordinate_mapper: Any) -> bool:
        handler = self._handlers_by_type.get(type(drawable))
        if handler is None:
            return False
        handler(drawable, coordinate_mapper)
        return True

    def render_cartesian(self, cartesian: Any, coordinate_mapper: Any) -> None:
        ox, oy = coordinate_mapper.math_to_screen(0, 0)
        width = self.canvas_el.width
        height = self.canvas_el.height
        color = self._parse_color(default_color)
        self._draw_lines([(0, oy), (width, oy)], color)
        self._draw_lines([(ox, 0), (ox, height)], color)

    def register(self, cls: type, handler: Callable[[Any, Any], None]) -> None:
        self._handlers_by_type[cls] = handler

    def register_default_drawables(self) -> None:
        try:
            from drawables.point import Point as PointDrawable
            self.register(PointDrawable, self._render_point)
        except Exception:
            pass
        try:
            from drawables.segment import Segment as SegmentDrawable
            self.register(SegmentDrawable, self._render_segment)
        except Exception:
            pass
        try:
            from drawables.circle import Circle as CircleDrawable
            self.register(CircleDrawable, self._render_circle)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Handlers

    def _render_point(self, point: Any, coordinate_mapper: Any) -> None:
        sx, sy = coordinate_mapper.math_to_screen(point.x, point.y)
        color = self._parse_color(getattr(point, "color", default_color))
        ndc = [self._to_ndc(sx, sy)]
        self._draw_points(ndc, color, getattr(point, "size", 6.0))

    def _render_segment(self, segment: Any, coordinate_mapper: Any) -> None:
        p1 = coordinate_mapper.math_to_screen(segment.point1.x, segment.point1.y)
        p2 = coordinate_mapper.math_to_screen(segment.point2.x, segment.point2.y)
        color = self._parse_color(getattr(segment, "color", default_color))
        self._draw_lines([p1, p2], color)

    def _render_circle(self, circle: Any, coordinate_mapper: Any) -> None:
        cx, cy = coordinate_mapper.math_to_screen(circle.center.x, circle.center.y)
        radius = coordinate_mapper.scale_value(circle.radius)
        color = self._parse_color(getattr(circle, "color", default_color))
        segments = 32
        points: list[Tuple[float, float]] = []
        for i in range(segments + 1):
            theta = 2 * math.pi * i / segments
            points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
        self._draw_line_strip(points, color)

    # ------------------------------------------------------------------
    # Drawing helpers

    def _draw_points(self, points: Sequence[Tuple[float, float]], color: Tuple[float, float, float, float], size: float) -> None:
        ndc = [self._to_ndc(x, y) for x, y in points]
        flat = [coord for vertex in ndc for coord in vertex]
        self.gl.uniform4fv(self._color_uniform, color)
        self.gl.uniform1f(self._point_size_uniform, size)
        buffer_data = window.Float32Array(flat)
        self.gl.bufferData(self.gl.ARRAY_BUFFER, buffer_data, self.gl.STATIC_DRAW)
        self.gl.drawArrays(self.gl.POINTS, 0, len(points))

    def _draw_lines(self, points: Sequence[Tuple[float, float]], color: Tuple[float, float, float, float]) -> None:
        ndc = [self._to_ndc(x, y) for x, y in points]
        flat = [coord for vertex in ndc for coord in vertex]
        self.gl.uniform4fv(self._color_uniform, color)
        self.gl.uniform1f(self._point_size_uniform, 1.0)
        buffer_data = window.Float32Array(flat)
        self.gl.bufferData(self.gl.ARRAY_BUFFER, buffer_data, self.gl.STATIC_DRAW)
        self.gl.drawArrays(self.gl.LINES, 0, len(points))

    def _draw_line_strip(self, points: Sequence[Tuple[float, float]], color: Tuple[float, float, float, float]) -> None:
        ndc = [self._to_ndc(x, y) for x, y in points]
        flat = [coord for vertex in ndc for coord in vertex]
        self.gl.uniform4fv(self._color_uniform, color)
        self.gl.uniform1f(self._point_size_uniform, 1.0)
        buffer_data = window.Float32Array(flat)
        self.gl.bufferData(self.gl.ARRAY_BUFFER, buffer_data, self.gl.STATIC_DRAW)
        self.gl.drawArrays(self.gl.LINE_STRIP, 0, len(points))

    # ------------------------------------------------------------------
    # Utility

    def _to_ndc(self, x: float, y: float) -> Tuple[float, float]:
        width = self.canvas_el.width or 1
        height = self.canvas_el.height or 1
        ndc_x = (x / width) * 2 - 1
        ndc_y = 1 - (y / height) * 2
        return ndc_x, ndc_y

    def _parse_color(self, color: str) -> Tuple[float, float, float, float]:
        ctx = getattr(self, "_scratch_ctx", None)
        if ctx is None:
            scratch_canvas = html.CANVAS()
            ctx = scratch_canvas.getContext("2d")
            ctx.fillStyle = color
            self._scratch_ctx = ctx
        ctx.fillStyle = color
        computed = ctx.fillStyle
        if computed.startswith("#") and len(computed) == 7:
            r = int(computed[1:3], 16) / 255.0
            g = int(computed[3:5], 16) / 255.0
            b = int(computed[5:7], 16) / 255.0
            return r, g, b, 1.0
        return 1.0, 1.0, 1.0, 1.0

    def _resize_viewport(self) -> None:
        container = getattr(self.canvas_el, "parentElement", None)
        if container is not None and hasattr(container, "getBoundingClientRect"):
            rect = container.getBoundingClientRect()
            if rect.width != self.canvas_el.width or rect.height != self.canvas_el.height:
                self.canvas_el.width = rect.width
                self.canvas_el.height = rect.height
        self.gl.viewport(0, 0, self.canvas_el.width, self.canvas_el.height)

    def _create_program(self):
        vertex_src = """
            attribute vec2 a_position;
            uniform float u_point_size;
            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
                gl_PointSize = u_point_size;
            }
        """
        fragment_src = """
            precision mediump float;
            uniform vec4 u_color;
            void main() {
                gl_FragColor = u_color;
            }
        """
        vertex_shader = self._compile_shader(vertex_src, self.gl.VERTEX_SHADER)
        fragment_shader = self._compile_shader(fragment_src, self.gl.FRAGMENT_SHADER)
        program = self.gl.createProgram()
        self.gl.attachShader(program, vertex_shader)
        self.gl.attachShader(program, fragment_shader)
        self.gl.linkProgram(program)
        if not self.gl.getProgramParameter(program, self.gl.LINK_STATUS):
            raise RuntimeError("Failed to link WebGL program")
        return program

    def _compile_shader(self, source: str, shader_type: int):
        shader = self.gl.createShader(shader_type)
        self.gl.shaderSource(shader, source)
        self.gl.compileShader(shader)
        if not self.gl.getShaderParameter(shader, self.gl.COMPILE_STATUS):
            raise RuntimeError("Failed to compile WebGL shader")
        return shader

    def _ensure_canvas(self, canvas_id: str):
        canvas_el = document.getElementById(canvas_id)
        if canvas_el is None:
            canvas_el = html.CANVAS(id=canvas_id)
            container = document.getElementById("math-container")
            if container is None:
                document <= canvas_el
            else:
                container <= canvas_el
        container = getattr(canvas_el, "parentElement", None)
        rect = container.getBoundingClientRect() if hasattr(container, "getBoundingClientRect") else None
        if rect:
            canvas_el.width = rect.width
            canvas_el.height = rect.height
        canvas_el.style.position = "absolute"
        canvas_el.style.top = "0"
        canvas_el.style.left = "0"
        canvas_el.style.pointerEvents = "none"
        canvas_el.style.display = "none"
        return canvas_el


