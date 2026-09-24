"""Canvas 2D renderer implementation using HTML5 Canvas API.

This module provides a renderer that draws mathematical graphics using the
Canvas 2D context. It uses cached render plans for efficient redrawing during
pan/zoom operations and supports optional layer compositing for complex scenes.

Key Features:
    - Canvas 2D API rendering for all drawable types
    - Cached render plans avoid recomputation during view changes
    - Optional offscreen compositing for improved performance
    - Telemetry tracking for performance analysis
    - Automatic canvas sizing to match container, with a devicePixelRatio-scaled
      bitmap for sharp output on HiDPI screens (drawing stays in CSS pixels)

Architecture:
    1. Canvas2DRenderer manages the canvas element and rendering pipeline
    2. Canvas2DPrimitiveAdapter translates drawing commands to Canvas 2D calls
    3. Cached plans from cached_render_plan.py are reprojected and applied
    4. Canvas2DTelemetry collects timing and usage metrics
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Set, Tuple

from browser import document, html, window

from rendering.base_telemetry import BaseRendererTelemetry
from rendering.style_manager import get_renderer_style
from rendering.interfaces import RendererProtocol
from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
from rendering.cached_render_plan import (
    OptimizedPrimitivePlan,
    build_plan_for_cartesian,
    build_plan_for_polar,
    build_plan_for_drawable,
)

# Fraction of the viewport that function plots also sample beyond each edge.
# Function plans are then only rebuilt when a pan leaves its half-screen bucket;
# smaller pans reproject the cached plan instead of resampling every frame.
FUNCTION_VIEW_MARGIN: float = 0.5

# Attributes holding points whose coordinates shape a drawable but are not
# part of its serialized state (which only stores the point names).
_DEPENDENT_POINT_ATTRS: Tuple[str, ...] = (
    "center",
    "point1",
    "point2",
    "vertex_point",
    "arm1_point",
    "arm2_point",
)


class Canvas2DTelemetry(BaseRendererTelemetry):
    """Performance telemetry collector for Canvas 2D rendering.

    Inherits common timing, counting, and snapshot logic from
    ``BaseRendererTelemetry``.  Overrides ``_new_drawable_bucket`` to include
    legacy-render counters specific to the Canvas 2D pipeline.
    """

    def _new_drawable_bucket(self) -> Dict[str, float]:
        """Return a fresh per-drawable counters dict with legacy-render keys."""
        bucket = super()._new_drawable_bucket()
        bucket["legacy_render_ms"] = 0.0
        bucket["legacy_render_count"] = 0
        return bucket


class Canvas2DRenderer(RendererProtocol):
    """Renderer using the HTML5 Canvas 2D API.

    Implements RendererProtocol to draw mathematical graphics using the browser's
    Canvas 2D context. Uses cached render plans to avoid recomputing drawing
    commands during pan/zoom operations.

    Attributes:
        canvas_el: The canvas DOM element.
        ctx: The 2D rendering context.
        style: Style configuration dictionary.

    Key Features:
        - Automatic canvas sizing to match container dimensions
        - Cached render plans for efficient redrawing
        - Optional offscreen compositing for complex scenes
        - Visibility culling to skip off-screen elements
    """

    def __init__(self, canvas_id: str = "math-canvas-2d") -> None:
        """Initialize the Canvas 2D renderer.

        Args:
            canvas_id: HTML id attribute for the canvas element.
        """
        self.canvas_el, self.ctx = self._initialize_canvas_context(canvas_id)
        self._resize_to_container()
        self.style: Dict[str, Any] = get_renderer_style()
        self.style["function_view_margin"] = FUNCTION_VIEW_MARGIN
        self._background_color: Optional[str] = self.style.get("canvas_background_color")
        self._handlers_by_type: Dict[type, Callable[[Any, Any], None]] = {}
        self._telemetry = Canvas2DTelemetry()
        target_canvas = self._initialize_layer_state()
        self._shared_primitives: Canvas2DPrimitiveAdapter = Canvas2DPrimitiveAdapter(
            target_canvas, telemetry=self._telemetry
        )
        self._apply_device_transform()
        self._apply_background()
        self.register_default_drawables()
        self._initialize_plan_caches()

    def clear(self) -> None:
        """Clear the canvas and apply the background color."""
        self._resize_to_container()
        self._sync_offscreen_size()
        self._shared_primitives.clear_surface()
        if not self._primitives_draw_on_main_canvas():
            self.ctx.clearRect(0, 0, self.canvas_el.width, self.canvas_el.height)
        self._apply_background()

    def render(self, drawable: Any, coordinate_mapper: Any) -> bool:
        """Render a drawable object using its registered handler.

        Args:
            drawable: The drawable object to render.
            coordinate_mapper: Mapper for coordinate transformations.

        Returns:
            True if a handler was found and invoked, False otherwise.
        """
        handler = self._handlers_by_type.get(type(drawable))
        if handler is None:
            return False
        handler(drawable, coordinate_mapper)
        return True

    def render_cartesian(self, cartesian: Any, coordinate_mapper: Any) -> None:
        """Render the Cartesian grid coordinate system.

        Args:
            cartesian: The Cartesian2Axis grid object.
            coordinate_mapper: Mapper for coordinate transformations.
        """
        self._resize_to_container()
        self._sync_offscreen_size()
        width, height = self._viewport_size()
        self._assign_cartesian_dimensions(cartesian, width, height)
        drawable_name = "Cartesian2Axis"
        map_state = self._capture_map_state(coordinate_mapper)
        signature = self._compute_drawable_signature(cartesian, coordinate_mapper)
        plan = self._resolve_cartesian_plan(cartesian, coordinate_mapper, map_state, signature, drawable_name)
        if plan is None:
            return
        apply_start = self._telemetry.mark_time()
        if not plan.is_visible(width, height):
            self._telemetry.record_plan_skip(drawable_name)
            return
        plan.apply(self._shared_primitives)
        apply_elapsed = self._telemetry.elapsed_since(apply_start)
        self._telemetry.record_plan_apply(drawable_name, apply_elapsed, cartesian=True)

    def render_polar(self, polar_grid: Any, coordinate_mapper: Any) -> None:
        """Render a polar coordinate grid.

        Args:
            polar_grid: The PolarGrid object.
            coordinate_mapper: Mapper for coordinate transformations.
        """
        self._resize_to_container()
        self._sync_offscreen_size()
        width, height = self._viewport_size()
        self._assign_polar_dimensions(polar_grid, width, height)
        drawable_name = "PolarGrid"
        map_state = self._capture_map_state(coordinate_mapper)
        signature = self._compute_drawable_signature(polar_grid, coordinate_mapper)
        plan = self._resolve_polar_plan(polar_grid, coordinate_mapper, map_state, signature, drawable_name)
        if plan is None:
            return
        apply_start = self._telemetry.mark_time()
        if not plan.is_visible(width, height):
            self._telemetry.record_plan_skip(drawable_name)
            return
        plan.apply(self._shared_primitives)
        apply_elapsed = self._telemetry.elapsed_since(apply_start)
        self._telemetry.record_plan_apply(drawable_name, apply_elapsed, cartesian=True)

    def _assign_polar_dimensions(self, polar_grid: Any, width: int, height: int) -> None:
        setattr(polar_grid, "width", width)
        setattr(polar_grid, "height", height)

    def _resolve_polar_plan(
        self,
        polar_grid: Any,
        coordinate_mapper: Any,
        map_state: Dict[str, float],
        signature: Optional[Any],
        drawable_name: str,
    ) -> Optional[OptimizedPrimitivePlan]:
        plan_entry = self._cartesian_cache
        if self._is_cached_plan_valid(plan_entry, signature):
            plan = plan_entry["plan"]
            plan.update_map_state(map_state)
            self._mark_screen_space_plan_dirty(plan)
        else:
            if plan_entry is not None:
                self._drop_plan_group(plan_entry.get("plan"))
            plan = self._build_polar_plan_with_metrics(polar_grid, coordinate_mapper, map_state, drawable_name)
            if plan is None:
                self._cartesian_cache = None
                return None
            self._cartesian_cache = {"plan": plan, "signature": signature}
        return plan

    def _build_polar_plan_with_metrics(
        self,
        polar_grid: Any,
        coordinate_mapper: Any,
        map_state: Dict[str, float],
        drawable_name: str,
    ) -> Optional[OptimizedPrimitivePlan]:
        build_start = self._telemetry.mark_time()
        try:
            plan = build_plan_for_polar(polar_grid, coordinate_mapper, self.style, supports_transform=False)
        except Exception:
            raise
        build_elapsed = self._telemetry.elapsed_since(build_start)
        if plan is None:
            self._telemetry.record_plan_miss(drawable_name)
            return None
        self._telemetry.record_plan_build(drawable_name, build_elapsed)
        plan.update_map_state(map_state)
        self._mark_screen_space_plan_dirty(plan)
        return plan

    def begin_frame(self) -> None:
        """Begin a new rendering frame."""
        self._telemetry.begin_frame()
        self._shared_primitives.begin_frame()
        self._frame_seen_plan_keys.clear()

    def end_frame(self) -> None:
        """End the current frame and flush any buffered content."""
        self._shared_primitives.end_frame()
        self._flush_offscreen_to_main()
        self._prune_unused_plan_entries()
        self._telemetry.end_frame()

    def invalidate_drawable_cache(self, drawable: Any) -> None:
        """Drop the cached plan for a drawable so it is rebuilt on next render."""
        cache_key = self._plan_cache_key(drawable, self._resolve_drawable_name(drawable))
        self._plan_cache.pop(cache_key, None)

    def invalidate_all_drawable_caches(self) -> None:
        """Drop every cached drawable plan."""
        self._plan_cache.clear()

    def _prune_unused_plan_entries(self) -> None:
        """Remove cached plans for drawables that were not rendered this frame."""
        if self._plan_cache:
            stale_keys = [key for key in self._plan_cache if key not in self._frame_seen_plan_keys]
            for key in stale_keys:
                self._plan_cache.pop(key, None)
        self._frame_seen_plan_keys.clear()

    def register(self, cls: type, handler: Callable[[Any, Any], None]) -> None:
        """Register a handler function for a drawable type.

        Args:
            cls: The drawable class to handle.
            handler: Function taking (drawable, coordinate_mapper) to render it.
        """
        self._handlers_by_type[cls] = handler

    def register_default_drawables(self) -> None:
        """Register handlers for all standard drawable types."""
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
            from drawables.ellipse import Ellipse as EllipseDrawable

            self.register(EllipseDrawable, self._render_ellipse)
        except Exception:
            pass
        try:
            from drawables.circle_arc import CircleArc as CircleArcDrawable

            self.register(CircleArcDrawable, self._render_circle_arc)
        except Exception:
            pass
        try:
            from drawables.vector import Vector as VectorDrawable

            self.register(VectorDrawable, self._render_vector)
        except Exception:
            pass
        try:
            from drawables.angle import Angle as AngleDrawable

            self.register(AngleDrawable, self._render_angle)
        except Exception:
            pass
        try:
            from drawables.function import Function as FunctionDrawable

            self.register(FunctionDrawable, self._render_function)
        except Exception:
            pass
        try:
            from drawables.piecewise_function import PiecewiseFunction as PiecewiseFunctionDrawable

            self.register(PiecewiseFunctionDrawable, self._render_function)
        except Exception:
            pass
        try:
            from drawables.parametric_function import ParametricFunction as ParametricFunctionDrawable

            self.register(ParametricFunctionDrawable, self._render_function)
        except Exception:
            pass
        try:
            from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea as FunctionsAreaDrawable

            self.register(FunctionsAreaDrawable, self._render_functions_bounded_colored_area)
        except Exception:
            pass
        try:
            from drawables.function_segment_bounded_colored_area import (
                FunctionSegmentBoundedColoredArea as FunctionSegmentAreaDrawable,
            )

            self.register(FunctionSegmentAreaDrawable, self._render_function_segment_bounded_colored_area)
        except Exception:
            pass
        try:
            from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea as SegmentsAreaDrawable

            self.register(SegmentsAreaDrawable, self._render_segments_bounded_colored_area)
        except Exception:
            pass
        try:
            from drawables.closed_shape_colored_area import ClosedShapeColoredArea as ClosedShapeAreaDrawable

            self.register(ClosedShapeAreaDrawable, self._render_closed_shape_colored_area)
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

    # ------------------------------------------------------------------
    # Handlers

    def _render_point(self, point: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(point, coordinate_mapper)

    def _render_segment(self, segment: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(segment, coordinate_mapper)

    def _render_circle(self, circle: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(circle, coordinate_mapper)

    def _render_ellipse(self, ellipse: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(ellipse, coordinate_mapper)

    def _render_circle_arc(self, circle_arc: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(circle_arc, coordinate_mapper)

    def _render_vector(self, vector: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(vector, coordinate_mapper)

    def _render_angle(self, angle: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(angle, coordinate_mapper)

    def _render_function(self, func: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(func, coordinate_mapper)

    def _render_functions_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(area, coordinate_mapper)

    def _render_function_segment_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(area, coordinate_mapper)

    def _render_segments_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(area, coordinate_mapper)

    def _render_closed_shape_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(area, coordinate_mapper)

    def _render_label(self, label: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(label, coordinate_mapper)

    def _ensure_canvas(self, canvas_id: str):
        canvas_el = document.getElementById(canvas_id)
        if canvas_el is None:
            canvas_el = html.CANVAS(id=canvas_id)
            container = document.getElementById("math-container")
            if container is None:
                document <= canvas_el
            else:
                container <= canvas_el
        size = self._container_client_size(getattr(canvas_el, "parentElement", None))
        if size is not None:
            pixel_width, pixel_height = size
            canvas_el.width = pixel_width
            canvas_el.height = pixel_height
            canvas_el.attrs["width"] = str(pixel_width)
            canvas_el.attrs["height"] = str(pixel_height)
        canvas_el.style.width = f"{int(canvas_el.width)}px"
        canvas_el.style.height = f"{int(canvas_el.height)}px"
        canvas_el.style.position = "absolute"
        canvas_el.style.top = "0"
        canvas_el.style.left = "0"
        canvas_el.style.pointerEvents = "none"
        canvas_el.style.display = "block"
        canvas_el.style.zIndex = "10"
        return canvas_el

    def _container_client_size(self, container: Any) -> Optional[Tuple[int, int]]:
        """Return the container's inner (border-excluded) size in whole CSS pixels.

        clientWidth/clientHeight match the 100%-sized SVG surface the coordinate
        mapper uses; the bounding rect would include the container border.
        """
        if container is None:
            return None
        width = getattr(container, "clientWidth", None)
        height = getattr(container, "clientHeight", None)
        if width and height:
            return int(width), int(height)
        if hasattr(container, "getBoundingClientRect"):
            rect = container.getBoundingClientRect()
            return int(round(rect.width)), int(round(rect.height))
        return None

    def _device_pixel_ratio(self) -> float:
        """Current window.devicePixelRatio (1.0 when unavailable)."""
        try:
            ratio = float(getattr(window, "devicePixelRatio", 1.0) or 1.0)
        except Exception:
            return 1.0
        if not math.isfinite(ratio) or ratio <= 0:
            return 1.0
        return ratio

    def _viewport_size(self) -> Tuple[int, int]:
        """Canvas size in CSS pixels: the space all drawing and culling uses."""
        size = getattr(self, "_css_size", None)
        if size is not None:
            return size
        return int(self.canvas_el.width), int(self.canvas_el.height)

    def _resize_to_container(self) -> None:
        """Match the canvas to its container, with a devicePixelRatio-scaled bitmap.

        The element keeps the container's CSS size while its bitmap is
        CSS size x devicePixelRatio; the context transform scales drawing so
        callers keep working in CSS pixels.
        """
        size = self._container_client_size(getattr(self.canvas_el, "parentElement", None))
        if size is None:
            # No container to measure: the element's own size is the viewport.
            self._css_size = None
            return
        css_width, css_height = size
        ratio = self._device_pixel_ratio()
        pixel_width = int(round(css_width * ratio))
        pixel_height = int(round(css_height * ratio))
        self._css_size = (css_width, css_height)
        current = (int(self.canvas_el.width), int(self.canvas_el.height))
        # Compare whole pixels: assigning width/height clears the bitmap, so
        # fractional container sizes must not trigger a reset every frame. A
        # bitmap resized by other code (its context state was reset) is
        # resized again so the device transform is restored.
        if (
            current != (pixel_width, pixel_height)
            or current != getattr(self, "_bitmap_size", current)
            or ratio != getattr(self, "_pixel_ratio", 1.0)
        ):
            self.canvas_el.width = pixel_width
            self.canvas_el.height = pixel_height
            self.canvas_el.attrs["width"] = str(pixel_width)
            self.canvas_el.attrs["height"] = str(pixel_height)
            self._bitmap_size = (pixel_width, pixel_height)
            self._pixel_ratio = ratio
            self._reset_context_state()
        self.canvas_el.style.width = f"{css_width}px"
        self.canvas_el.style.height = f"{css_height}px"

    def _apply_device_transform(self) -> None:
        """Scale drawing contexts by the device pixel ratio (CSS px -> bitmap px)."""
        ratio = getattr(self, "_pixel_ratio", 1.0)
        contexts = [self.ctx]
        offscreen = getattr(self, "_offscreen_canvas", None)
        if offscreen is not None:
            try:
                contexts.append(offscreen.getContext("2d"))
            except Exception:
                pass
        for ctx in contexts:
            try:
                ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
            except Exception:
                pass

    def _reset_context_state(self) -> None:
        """Re-establish context state after the bitmap (and so the context) was reset.

        Resizing a canvas resets its 2D context, so the device transform is
        reapplied and the primitive adapter is recreated to drop its cached
        stroke/fill/font state.
        """
        self._apply_device_transform()
        primitives = getattr(self, "_shared_primitives", None)
        target_canvas = getattr(primitives, "canvas_el", None)
        if target_canvas is None:
            return
        try:
            self._shared_primitives = Canvas2DPrimitiveAdapter(target_canvas, telemetry=self._telemetry)
        except Exception:
            pass

    def _render_drawable(self, drawable: Any, coordinate_mapper: Any) -> None:
        renderable_attr = getattr(drawable, "is_renderable", True)
        try:
            if not bool(renderable_attr):
                return
        except Exception:
            return
        drawable_name = self._resolve_drawable_name(drawable)
        map_state = self._capture_map_state(coordinate_mapper)
        signature = self._compute_drawable_signature(drawable, coordinate_mapper)
        cache_key = self._plan_cache_key(drawable, drawable_name)
        plan = self._resolve_drawable_plan(drawable, coordinate_mapper, map_state, signature, drawable_name, cache_key)
        if plan is None:
            return
        self._frame_seen_plan_keys.add(cache_key)
        apply_start = self._telemetry.mark_time()
        viewport_width, viewport_height = self._viewport_size()
        if not plan.is_visible(viewport_width, viewport_height):
            self._telemetry.record_plan_skip(drawable_name)
            return
        plan.apply(self._shared_primitives)
        apply_elapsed = self._telemetry.elapsed_since(apply_start)
        self._telemetry.record_plan_apply(drawable_name, apply_elapsed)

    def _capture_map_state(self, mapper: Any) -> Dict[str, float]:
        origin = getattr(mapper, "origin", None)
        offset = getattr(mapper, "offset", None)
        return {
            "scale": float(getattr(mapper, "scale_factor", 1.0)),
            "offset_x": float(getattr(offset, "x", 0.0)) if offset is not None else 0.0,
            "offset_y": float(getattr(offset, "y", 0.0)) if offset is not None else 0.0,
            "origin_x": float(getattr(origin, "x", 0.0)) if origin is not None else 0.0,
            "origin_y": float(getattr(origin, "y", 0.0)) if origin is not None else 0.0,
        }

    def _plan_cache_key(self, drawable: Any, drawable_name: str) -> str:
        name = getattr(drawable, "name", None)
        if isinstance(name, str) and name:
            return f"{drawable_name}:{name}"
        identifier = getattr(drawable, "id", None)
        if isinstance(identifier, str) and identifier:
            return f"{drawable_name}:{identifier}"
        return f"{drawable_name}:{id(drawable)}"

    def _compute_drawable_signature(self, drawable: Any, coordinate_mapper: Any = None) -> Tuple[Any, ...]:
        state_func = getattr(drawable, "get_state", None)
        state: Any = None
        if callable(state_func):
            try:
                state = state_func()
            except Exception:
                state = None
        fallback: Dict[str, Any] = {}
        for attr in ("name", "color"):
            if hasattr(drawable, attr):
                fallback[attr] = getattr(drawable, attr)
        if state is None:
            snapshot = fallback
        else:
            snapshot = {**fallback, "__state__": state}
        dependent_coords = self._collect_dependent_coordinates(drawable)
        if dependent_coords:
            snapshot["_dependent_coords"] = dependent_coords
        if self._needs_scale_in_signature(drawable) and coordinate_mapper is not None:
            scale = getattr(coordinate_mapper, "scale_factor", None)
            if scale is not None:
                snapshot["_view_scale"] = round(float(scale), 4)
            offset = getattr(coordinate_mapper, "offset", None)
            canvas_width = getattr(coordinate_mapper, "canvas_width", None)
            canvas_height = getattr(coordinate_mapper, "canvas_height", None)
            drawable_name = self._resolve_drawable_name(drawable)
            # Parametric samples do not depend on the viewport, so a pan only
            # reprojects the cached plan instead of resampling the curve.
            if offset is not None and drawable_name != "ParametricFunction":
                snapshot["_view_offset"] = self._offset_signature(drawable_name, offset, canvas_width, canvas_height)
            if canvas_width is not None and canvas_height is not None:
                snapshot["_view_size"] = (round(float(canvas_width), 2), round(float(canvas_height), 2))
        return self._freeze_signature(snapshot)

    def _offset_signature(self, drawable_name: str, offset: Any, canvas_width: Any, canvas_height: Any) -> Tuple:
        """Pan offset as it affects a view-dependent plan.

        Function plots are sampled FUNCTION_VIEW_MARGIN of a screen beyond each
        edge, so only the half-screen bucket of the offset matters; the cached
        plan is reprojected for pans within the bucket.
        """
        offset_x = float(offset.x)
        offset_y = float(offset.y)
        margin = float(self.style.get("function_view_margin", 0.0) or 0.0) if hasattr(self, "style") else 0.0
        if drawable_name in ("Function", "PiecewiseFunction") and margin > 0 and canvas_width and canvas_height:
            bucket_x = float(canvas_width) * margin
            bucket_y = float(canvas_height) * margin
            return ("bucket", math.floor(offset_x / bucket_x), math.floor(offset_y / bucket_y))
        return (round(offset_x, 2), round(offset_y, 2))

    def _collect_dependent_coordinates(self, drawable: Any) -> list:
        """Collect coordinates of referenced points so moving them invalidates the plan."""
        coords: list = []
        for attr in _DEPENDENT_POINT_ATTRS:
            try:
                point = getattr(drawable, attr, None)
                x = getattr(point, "x", None)
                y = getattr(point, "y", None)
            except Exception:
                continue
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                coords.append((attr, x, y))
        return coords

    def _needs_scale_in_signature(self, drawable: Any) -> bool:
        class_name = getattr(drawable, "get_class_name", lambda: "")()
        return class_name in (
            "Function",
            "PiecewiseFunction",
            "ParametricFunction",
            "FunctionsBoundedColoredArea",
            "FunctionSegmentBoundedColoredArea",
            "PolarGrid",
        )

    def _freeze_signature(self, value: Any) -> Tuple[Any, ...]:
        if isinstance(value, dict):
            items = []
            for key in sorted(value.keys()):
                items.append((key, self._freeze_signature(value[key])))
            return tuple(items)
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_signature(item) for item in value)
        if isinstance(value, float):
            try:
                return (round(value, 6),)
            except Exception:
                return (value,)
        if isinstance(value, (int, str, bool)) or value is None:
            return (value,)
        return (repr(value),)

    def _resolve_drawable_name(self, drawable: Any) -> str:
        try:
            candidate = getattr(drawable, "get_class_name", None)
            if callable(candidate):
                name = candidate()
                if isinstance(name, str) and name:
                    return name
        except Exception:
            pass
        return drawable.__class__.__name__

    def drain_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data and reset all counters.

        Returns:
            Dictionary with frames, phase timings, per-drawable stats,
            adapter events, and cache metrics.
        """
        snapshot = self._telemetry.drain()
        snapshot.update(self._collect_cache_metrics())
        return snapshot

    def peek_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data without resetting counters.

        Returns:
            Dictionary with current telemetry and cache metrics.
        """
        snapshot = self._telemetry.snapshot()
        snapshot.update(self._collect_cache_metrics())
        return snapshot

    def _collect_cache_metrics(self) -> Dict[str, Any]:
        """Collect metrics about cache usage."""
        metrics: Dict[str, Any] = {}
        try:
            primitives = getattr(self, "_shared_primitives", None)
            font_cache = getattr(primitives, "_font_cache", None) if primitives is not None else None
            if isinstance(font_cache, dict):
                metrics["font_cache_entries"] = int(len(font_cache))
        except Exception:
            pass

        plan_cache = getattr(self, "_plan_cache", None)
        if isinstance(plan_cache, dict):
            try:
                metrics["plan_cache_entries"] = int(len(plan_cache))
            except Exception:
                pass
            try:
                total_commands = 0
                for entry in plan_cache.values():
                    plan = entry.get("plan") if isinstance(entry, dict) else None
                    commands = getattr(plan, "commands", None)
                    try:
                        total_commands += int(len(commands)) if commands is not None else 0
                    except Exception:
                        pass
                metrics["plan_cache_total_commands"] = int(total_commands)
            except Exception:
                pass

        return metrics

    def _should_use_layer_compositing(self) -> bool:
        try:
            flag = getattr(window, "MatHudCanvas2DOffscreen", None)
            if isinstance(flag, bool):
                return flag
        except Exception:
            pass
        try:
            stored = window.localStorage.getItem("mathud.canvas2d.offscreen")
            if stored and stored.lower() in {"1", "true", "yes", "on"}:
                return True
        except Exception:
            pass
        return False

    def _create_offscreen_canvas(self):
        offscreen = html.CANVAS()
        offscreen.width = self.canvas_el.width
        offscreen.height = self.canvas_el.height
        offscreen.attrs["width"] = str(offscreen.width)
        offscreen.attrs["height"] = str(offscreen.height)
        offscreen.style.display = "none"
        return offscreen

    def _sync_offscreen_size(self) -> None:
        if not self._use_layer_compositing or self._offscreen_canvas is None:
            return
        target_width = self.canvas_el.width
        target_height = self.canvas_el.height
        if self._offscreen_canvas.width != target_width or self._offscreen_canvas.height != target_height:
            self._shared_primitives.resize_surface(target_width, target_height)
            self._reset_context_state()
            self._apply_background()

    def _flush_offscreen_to_main(self) -> None:
        if not self._use_layer_compositing or self._offscreen_canvas is None:
            return
        try:
            # Both bitmaps are in device pixels: copy 1:1 without the DPR transform.
            self.ctx.save()
            try:
                self.ctx.setTransform(1, 0, 0, 1, 0, 0)
                self.ctx.clearRect(0, 0, self.canvas_el.width, self.canvas_el.height)
                self.ctx.drawImage(self._offscreen_canvas, 0, 0)
            finally:
                self.ctx.restore()
        except Exception:
            try:
                off_ctx = self._offscreen_canvas.getContext("2d")
                if off_ctx is not None:
                    image = off_ctx.getImageData(0, 0, self._offscreen_canvas.width, self._offscreen_canvas.height)
                    self.ctx.putImageData(image, 0, 0)
            except Exception:
                pass

    def _initialize_canvas_context(self, canvas_id: str) -> Tuple[Any, Any]:
        canvas_el = self._ensure_canvas(canvas_id)
        ctx = canvas_el.getContext("2d")
        if ctx is None:
            raise RuntimeError("Canvas 2D context unavailable")
        return canvas_el, ctx

    def _initialize_layer_state(self) -> Any:
        self._use_layer_compositing = self._should_use_layer_compositing()
        self._offscreen_canvas = self._create_offscreen_canvas() if self._use_layer_compositing else None
        return self._offscreen_canvas or self.canvas_el

    def _primitives_draw_on_main_canvas(self) -> bool:
        primitives = getattr(self, "_shared_primitives", None)
        return primitives is not None and getattr(primitives, "canvas_el", None) is self.canvas_el

    def _apply_background(self) -> None:
        color = self._background_color
        if not color:
            return
        try:
            self._shared_primitives.fill_background(color)
        except Exception:
            pass
        if self._primitives_draw_on_main_canvas():
            # The adapter already filled the main canvas.
            return
        try:
            self.ctx.save()
            self.ctx.setTransform(1, 0, 0, 1, 0, 0)
            self.ctx.fillStyle = color
            self.ctx.fillRect(0, 0, self.canvas_el.width, self.canvas_el.height)
        except Exception:
            pass
        finally:
            try:
                self.ctx.restore()
            except Exception:
                pass

    def _initialize_plan_caches(self) -> None:
        self._plan_cache = {}
        self._cartesian_cache = None
        self._frame_seen_plan_keys: Set[str] = set()

    def _assign_cartesian_dimensions(self, cartesian: Any, width: int, height: int) -> None:
        cartesian.width = width
        cartesian.height = height

    def _resolve_cartesian_plan(
        self,
        cartesian: Any,
        coordinate_mapper: Any,
        map_state: Dict[str, float],
        signature: Optional[Any],
        drawable_name: str,
    ) -> Optional[OptimizedPrimitivePlan]:
        plan_entry = self._cartesian_cache
        if self._is_cached_plan_valid(plan_entry, signature):
            plan = plan_entry["plan"]
            plan.update_map_state(map_state)
            return plan
        build_start = self._telemetry.mark_time()
        plan = build_plan_for_cartesian(cartesian, coordinate_mapper, self.style, supports_transform=False)
        build_elapsed = self._telemetry.elapsed_since(build_start)
        self._telemetry.record_plan_build(drawable_name, build_elapsed, cartesian=True)
        if plan is None:
            self._telemetry.record_plan_miss(drawable_name)
            self._cartesian_cache = None
            return None
        plan.update_map_state(map_state)
        self._cartesian_cache = {"plan": plan, "signature": signature}
        return plan

    def _resolve_drawable_plan(
        self,
        drawable: Any,
        coordinate_mapper: Any,
        map_state: Dict[str, float],
        signature: Optional[Any],
        drawable_name: str,
        cache_key: str,
    ) -> Optional[OptimizedPrimitivePlan]:
        cached_entry = self._plan_cache.get(cache_key)
        if self._is_cached_plan_valid(cached_entry, signature):
            plan = cached_entry["plan"]
            plan.update_map_state(map_state)
            return plan
        build_start = self._telemetry.mark_time()
        plan = build_plan_for_drawable(drawable, coordinate_mapper, self.style, supports_transform=False)
        build_elapsed = self._telemetry.elapsed_since(build_start)
        if plan is not None:
            self._telemetry.record_plan_build(drawable_name, build_elapsed)
            plan.update_map_state(map_state)
            if signature is not None:
                self._plan_cache[cache_key] = {"plan": plan, "signature": signature}
            return plan
        self._telemetry.record_plan_miss(drawable_name)
        self._plan_cache.pop(cache_key, None)
        return None

    def _is_cached_plan_valid(self, cache_entry: Optional[Dict[str, Any]], signature: Optional[Any]) -> bool:
        return bool(
            cache_entry
            and cache_entry.get("signature") == signature
            and isinstance(cache_entry.get("plan"), OptimizedPrimitivePlan)
        )

    def _drop_plan_group(self, plan_obj: Optional[OptimizedPrimitivePlan]) -> None:
        """No-op for Canvas2D since there are no DOM groups to manage."""
        pass

    def _mark_screen_space_plan_dirty(self, plan: OptimizedPrimitivePlan) -> None:
        """Mark a screen-space plan as needing reapplication."""
        if getattr(plan, "uses_screen_space", lambda: False)():
            plan.mark_dirty()
