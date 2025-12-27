from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from browser import document, html, window, console

from rendering.interfaces import RendererProtocol
from rendering.style_manager import get_renderer_style
from rendering.webgl_primitive_adapter import WebGLPrimitiveAdapter
from rendering.cached_render_plan import (
    build_plan_for_cartesian,
    build_plan_for_drawable,
)


class WebGLRenderer(RendererProtocol):
    """Experimental renderer backed by WebGL that delegates shape drawing to primitives."""

    def __init__(self, canvas_id: str = "math-webgl") -> None:
        self.gl = self._prepare_canvas_and_context(canvas_id)
        self.style: Dict[str, Any] = get_renderer_style()
        self._initialize_program_state()
        self._initialize_handler_registry()
        self._shared_primitives: WebGLPrimitiveAdapter = WebGLPrimitiveAdapter(self)

    def clear(self) -> None:
        self._resize_viewport()
        self.gl.clear(self.gl.COLOR_BUFFER_BIT)

    def render(self, drawable: Any, coordinate_mapper: Any) -> bool:
        # Handlers perform the actual drawing; this method only dispatches.
        handler = self._handlers_by_type.get(type(drawable))
        if handler is None:
            return False
        handler(drawable, coordinate_mapper)
        return True

    def render_cartesian(self, cartesian: Any, coordinate_mapper: Any) -> None:
        width, height = self._prepare_cartesian_dimensions(cartesian)
        plan = self._build_cartesian_plan(cartesian, coordinate_mapper)
        if plan is None:
            return
        if not self._should_apply_plan(plan, width, height):
            return
        plan.apply(self._shared_primitives)

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
        try:
            from drawables.circle_arc import CircleArc as CircleArcDrawable
            self.register(CircleArcDrawable, self._render_circle_arc)
        except Exception:
            pass
        try:
            from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea as FunctionsAreaDrawable
            self.register(FunctionsAreaDrawable, self._render_drawable)
        except Exception:
            pass
        try:
            from drawables.function_segment_bounded_colored_area import FunctionSegmentBoundedColoredArea as FunctionSegmentAreaDrawable
            self.register(FunctionSegmentAreaDrawable, self._render_drawable)
        except Exception:
            pass
        try:
            from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea as SegmentsAreaDrawable
            self.register(SegmentsAreaDrawable, self._render_drawable)
        except Exception:
            pass
        try:
            from drawables.closed_shape_colored_area import ClosedShapeColoredArea as ClosedShapeAreaDrawable
            self.register(ClosedShapeAreaDrawable, self._render_drawable)
        except Exception:
            pass
        try:
            from drawables.label import Label as LabelDrawable
            self.register(LabelDrawable, self._render_label)
        except Exception:
            pass
        try:
            from drawables.bar import Bar as BarDrawable
            self.register(BarDrawable, self._render_drawable)
        except Exception:
            pass

    def begin_frame(self) -> None:
        self._shared_primitives.begin_frame()

    def end_frame(self) -> None:
        self._shared_primitives.end_frame()

    # ------------------------------------------------------------------
    # Handlers

    def _render_point(self, point: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(point, coordinate_mapper)

    def _render_segment(self, segment: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(segment, coordinate_mapper)

    def _render_circle(self, circle: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(circle, coordinate_mapper)

    def register_circle_arc(self, arc_cls: type) -> None:
        self.register(arc_cls, self._render_circle_arc)

    def _render_circle_arc(self, circle_arc: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(circle_arc, coordinate_mapper)

    def _render_label(self, label: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(label, coordinate_mapper)

    def _render_drawable(self, drawable: Any, coordinate_mapper: Any) -> None:
        renderable_attr = getattr(drawable, "is_renderable", True)
        try:
            if not bool(renderable_attr):
                return
        except Exception:
            return
        plan = self._resolve_drawable_plan(drawable, coordinate_mapper)
        if plan is None:
            return
        if not self._should_apply_plan(plan, self.canvas_el.width, self.canvas_el.height):
            return
        plan.apply(self._shared_primitives)

    # ------------------------------------------------------------------
    # Drawing helpers

    def _draw_points(self, points: Sequence[Tuple[float, float]], color: Tuple[float, float, float, float], size: float) -> None:
        self._use_program_for_draw()
        flat = self._prepare_vertices(points)
        self._set_color_uniform(color)
        self._set_point_size_uniform(float(size))
        self._upload_vertices(flat)
        self.gl.drawArrays(self.gl.POINTS, 0, len(points))

    def _draw_lines(self, points: Sequence[Tuple[float, float]], color: Tuple[float, float, float, float]) -> None:
        self._use_program_for_draw()
        flat = self._prepare_vertices(points)
        self._set_color_uniform(color)
        self._set_point_size_uniform(1.0)
        self._upload_vertices(flat)
        self.gl.drawArrays(self.gl.LINES, 0, len(points))

    def _draw_line_strip(self, points: Sequence[Tuple[float, float]], color: Tuple[float, float, float, float]) -> None:
        self._use_program_for_draw()
        flat = self._prepare_vertices(points)
        self._set_color_uniform(color)
        self._set_point_size_uniform(1.0)
        self._upload_vertices(flat)
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
        ctx = self._ensure_scratch_context(color)
        ctx.fillStyle = color
        computed = ctx.fillStyle
        hex_color = self._parse_hex_color(computed)
        if hex_color is not None:
            return hex_color
        rgb_color = self._parse_rgb_color(computed)
        if rgb_color is not None:
            return rgb_color
        return 1.0, 1.0, 1.0, 1.0

    def _resize_viewport(self) -> None:
        container = getattr(self.canvas_el, "parentElement", None)
        rect = self._read_container_rect(container)
        if rect is not None:
            self._apply_canvas_dimensions(rect.width, rect.height)
        self._apply_canvas_size_styles()
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
        canvas_el = self._obtain_canvas_element(canvas_id)
        container = getattr(canvas_el, "parentElement", None)
        rect = self._read_container_rect(container)
        if rect is not None:
            self._apply_canvas_rect(canvas_el, rect.width, rect.height)
        self._apply_canvas_style_defaults(canvas_el)
        return canvas_el

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_canvas_and_context(self, canvas_id: str):
        self.canvas_el = self._ensure_canvas(canvas_id)
        gl = self.canvas_el.getContext("webgl")
        if gl is None:
            raise RuntimeError("WebGL context unavailable")
        return gl

    def _initialize_program_state(self) -> None:
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

    def _initialize_handler_registry(self) -> None:
        self._handlers_by_type = {}
        self.register_default_drawables()

    def _prepare_cartesian_dimensions(self, cartesian: Any) -> Tuple[int, int]:
        width = self.canvas_el.width
        height = self.canvas_el.height
        cartesian.width = width
        cartesian.height = height
        return width, height

    def _build_cartesian_plan(self, cartesian: Any, coordinate_mapper: Any):
        return build_plan_for_cartesian(cartesian, coordinate_mapper, self.style, supports_transform=False)

    def _resolve_drawable_plan(self, drawable: Any, coordinate_mapper: Any):
        return build_plan_for_drawable(drawable, coordinate_mapper, self.style, supports_transform=False)

    def _should_apply_plan(self, plan: Any, width: int, height: int) -> bool:
        return bool(plan.is_visible(width, height))

    def _use_program_for_draw(self) -> None:
        self.gl.useProgram(self._program)
        self.gl.bindBuffer(self.gl.ARRAY_BUFFER, self._buffer)
        self.gl.enableVertexAttribArray(self._position_attrib)
        self.gl.vertexAttribPointer(self._position_attrib, 2, self.gl.FLOAT, False, 0, 0)

    def _prepare_vertices(self, points: Sequence[Tuple[float, float]]) -> list[float]:
        ndc = [self._to_ndc(x, y) for x, y in points]
        return [coord for vertex in ndc for coord in vertex]

    def _set_color_uniform(self, color: Tuple[float, float, float, float]) -> None:
        color_vec = window.Float32Array.new(color)
        self.gl.uniform4fv(self._color_uniform, color_vec)

    def _set_point_size_uniform(self, size: float) -> None:
        self.gl.uniform1f(self._point_size_uniform, size)

    def _upload_vertices(self, flat: Sequence[float]) -> None:
        buffer_data = window.Float32Array.new(flat)
        self.gl.bufferData(self.gl.ARRAY_BUFFER, buffer_data, self.gl.STATIC_DRAW)

    def _ensure_scratch_context(self, color: str):
        ctx = getattr(self, "_scratch_ctx", None)
        if ctx is None:
            scratch_canvas = html.CANVAS()
            ctx = scratch_canvas.getContext("2d")
            ctx.fillStyle = color
            self._scratch_ctx = ctx
        return ctx

    def _parse_hex_color(self, computed: str) -> Optional[Tuple[float, float, float, float]]:
        if computed.startswith("#") and len(computed) == 7:
            r = int(computed[1:3], 16) / 255.0
            g = int(computed[3:5], 16) / 255.0
            b = int(computed[5:7], 16) / 255.0
            return r, g, b, 1.0
        return None

    def _parse_rgb_color(self, computed: str) -> Optional[Tuple[float, float, float, float]]:
        if not computed.startswith("rgb"):
            return None
        inside = computed[computed.index("(") + 1 : computed.rindex(")")]
        parts = [part.strip() for part in inside.split(",")]
        if len(parts) < 3:
            return None
        try:
            r = float(parts[0]) / 255.0
            g = float(parts[1]) / 255.0
            b = float(parts[2]) / 255.0
            a = float(parts[3]) if len(parts) > 3 else 1.0
            if a > 1.0:
                a /= 255.0
            return r, g, b, a
        except Exception:
            return None

    def _read_container_rect(self, container: Any):
        if container is None or not hasattr(container, "getBoundingClientRect"):
            return None
        try:
            return container.getBoundingClientRect()
        except Exception:
            return None

    def _apply_canvas_dimensions(self, width: float, height: float) -> None:
        pixel_width = int(width)
        pixel_height = int(height)
        self.canvas_el.width = pixel_width
        self.canvas_el.height = pixel_height
        self.canvas_el.attrs["width"] = pixel_width
        self.canvas_el.attrs["height"] = pixel_height

    def _apply_canvas_size_styles(self) -> None:
        self.canvas_el.style.width = f"{int(self.canvas_el.width)}px"
        self.canvas_el.style.height = f"{int(self.canvas_el.height)}px"

    def _obtain_canvas_element(self, canvas_id: str):
        canvas_el = document.getElementById(canvas_id)
        if canvas_el is not None:
            return canvas_el
        canvas_el = html.CANVAS(id=canvas_id)
        container = document.getElementById("math-container")
        if container is None:
            document <= canvas_el
        else:
            container <= canvas_el
        return canvas_el

    def _apply_canvas_rect(self, canvas_el: Any, width: float, height: float) -> None:
        pixel_width = int(width)
        pixel_height = int(height)
        canvas_el.width = pixel_width
        canvas_el.height = pixel_height
        canvas_el.attrs["width"] = pixel_width
        canvas_el.attrs["height"] = pixel_height

    def _apply_canvas_style_defaults(self, canvas_el: Any) -> None:
        canvas_el.style.width = f"{int(canvas_el.width)}px"
        canvas_el.style.height = f"{int(canvas_el.height)}px"
        canvas_el.style.position = "absolute"
        canvas_el.style.top = "0"
        canvas_el.style.left = "0"
        canvas_el.style.pointerEvents = "none"
        canvas_el.style.display = "block"
        canvas_el.style.zIndex = "20"


