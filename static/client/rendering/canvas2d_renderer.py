from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

from browser import document, html, window

from rendering.style_manager import get_renderer_style
from rendering.interfaces import RendererProtocol
from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
from rendering.optimized_drawable_renderers import (
    OptimizedPrimitivePlan,
    build_plan_for_cartesian,
    build_plan_for_drawable,
)


class Canvas2DTelemetry:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phase_totals: Dict[str, float] = {
            "plan_build_ms": 0.0,
            "plan_apply_ms": 0.0,
            "cartesian_plan_build_ms": 0.0,
            "cartesian_plan_apply_ms": 0.0,
        }
        self._phase_counts: Dict[str, int] = {
            "plan_build_count": 0,
            "plan_apply_count": 0,
            "cartesian_plan_count": 0,
            "plan_miss_count": 0,
            "plan_skip_count": 0,
        }
        self._per_drawable: Dict[str, Dict[str, float]] = {}
        self._adapter_events: Dict[str, int] = {}
        self._frames: int = 0
        self._max_batch_depth: int = 0

    def begin_frame(self) -> None:
        self._frames += 1

    def end_frame(self) -> None:
        pass

    def _now(self) -> float:
        try:
            perf = getattr(window, "performance", None)
            if perf is not None:
                return float(perf.now())
        except Exception:
            pass
        return time.time() * 1000.0

    def mark_time(self) -> float:
        return self._now()

    def elapsed_since(self, start: float) -> float:
        return max(self._now() - start, 0.0)

    def _drawable_bucket(self, name: str) -> Dict[str, float]:
        bucket = self._per_drawable.get(name)
        if bucket is None:
            bucket = {
                "plan_build_ms": 0.0,
                "plan_apply_ms": 0.0,
                "legacy_render_ms": 0.0,
                "plan_build_count": 0,
                "plan_apply_count": 0,
                "legacy_render_count": 0,
                "plan_miss_count": 0,
                "plan_skip_count": 0,
            }
            self._per_drawable[name] = bucket
        return bucket

    def record_plan_build(self, name: str, duration_ms: float, *, cartesian: bool = False) -> None:
        self._phase_totals["plan_build_ms"] += duration_ms
        self._phase_counts["plan_build_count"] += 1
        bucket = self._drawable_bucket(name)
        bucket["plan_build_ms"] += duration_ms
        bucket["plan_build_count"] += 1
        if cartesian:
            self._phase_totals["cartesian_plan_build_ms"] += duration_ms
            self._phase_counts["cartesian_plan_count"] += 1

    def record_plan_apply(self, name: str, duration_ms: float, *, cartesian: bool = False) -> None:
        self._phase_totals["plan_apply_ms"] += duration_ms
        self._phase_counts["plan_apply_count"] += 1
        bucket = self._drawable_bucket(name)
        bucket["plan_apply_ms"] += duration_ms
        bucket["plan_apply_count"] += 1
        if cartesian:
            self._phase_totals["cartesian_plan_apply_ms"] += duration_ms

    def record_plan_miss(self, name: str) -> None:
        self._phase_counts["plan_miss_count"] += 1
        bucket = self._drawable_bucket(name)
        bucket["plan_miss_count"] += 1

    def record_plan_skip(self, name: str) -> None:
        self._phase_counts["plan_skip_count"] += 1
        bucket = self._drawable_bucket(name)
        bucket["plan_skip_count"] += 1

    def record_adapter_event(self, name: str, amount: int = 1) -> None:
        self._adapter_events[name] = self._adapter_events.get(name, 0) + amount

    def track_batch_depth(self, depth: int) -> None:
        if depth > self._max_batch_depth:
            self._max_batch_depth = depth

    def snapshot(self) -> Dict[str, Any]:
        adapter_events = dict(self._adapter_events)
        if self._max_batch_depth:
            adapter_events["max_batch_depth"] = self._max_batch_depth
        per_drawable = {name: dict(bucket) for name, bucket in self._per_drawable.items()}
        phase = dict(self._phase_totals)
        phase.update(self._phase_counts)
        return {
            "frames": self._frames,
            "phase": phase,
            "per_drawable": per_drawable,
            "adapter_events": adapter_events,
        }

    def drain(self) -> Dict[str, Any]:
        snapshot = self.snapshot()
        self.reset()
        return snapshot


class Canvas2DRenderer(RendererProtocol):
    """Experimental renderer backed by the Canvas 2D API using shared primitives."""

    def __init__(self, canvas_id: str = "math-canvas-2d") -> None:
        self.canvas_el = self._ensure_canvas(canvas_id)
        self.ctx = self.canvas_el.getContext("2d")
        if self.ctx is None:
            raise RuntimeError("Canvas 2D context unavailable")
        self.style: Dict[str, Any] = get_renderer_style()
        self._handlers_by_type: Dict[type, Callable[[Any, Any], None]] = {}
        self._telemetry = Canvas2DTelemetry()
        self._use_layer_compositing: bool = self._should_use_layer_compositing()
        self._offscreen_canvas = self._create_offscreen_canvas() if self._use_layer_compositing else None
        target_canvas = self._offscreen_canvas or self.canvas_el
        self._shared_primitives: Canvas2DPrimitiveAdapter = Canvas2DPrimitiveAdapter(
            target_canvas, telemetry=self._telemetry
        )
        self.register_default_drawables()
        self._plan_cache: Dict[str, Dict[str, Any]] = {}
        self._cartesian_cache: Optional[Dict[str, Any]] = None

    def clear(self) -> None:
        width = self.canvas_el.width
        height = self.canvas_el.height
        self._shared_primitives.clear_surface()
        self.ctx.clearRect(0, 0, width, height)

    def render(self, drawable: Any, coordinate_mapper: Any) -> bool:
        handler = self._handlers_by_type.get(type(drawable))
        if handler is None:
            return False
        handler(drawable, coordinate_mapper)
        return True

    def render_cartesian(self, cartesian: Any, coordinate_mapper: Any) -> None:
        self._resize_to_container()
        self._sync_offscreen_size()
        width = self.canvas_el.width
        height = self.canvas_el.height
        cartesian.width = width
        cartesian.height = height
        drawable_name = "Cartesian2Axis"
        map_state = self._capture_map_state(coordinate_mapper)
        signature = self._compute_drawable_signature(cartesian)
        plan_entry = self._cartesian_cache
        plan: Optional[OptimizedPrimitivePlan] = None
        if (
            plan_entry is not None
            and plan_entry.get("signature") == signature
            and isinstance(plan_entry.get("plan"), OptimizedPrimitivePlan)
        ):
            plan = plan_entry["plan"]
            plan.update_map_state(map_state)
        else:
            build_start = self._telemetry.mark_time()
            plan = build_plan_for_cartesian(cartesian, coordinate_mapper, self.style, supports_transform=False)
            build_elapsed = self._telemetry.elapsed_since(build_start)
            self._telemetry.record_plan_build(drawable_name, build_elapsed, cartesian=True)
            if plan is None:
                self._telemetry.record_plan_miss(drawable_name)
                self._cartesian_cache = None
                return
            plan.update_map_state(map_state)
            self._cartesian_cache = {"plan": plan, "signature": signature}
        apply_start = self._telemetry.mark_time()
        if not plan.is_visible(width, height):
            self._telemetry.record_plan_skip(drawable_name)
            return
        plan.apply(self._shared_primitives)
        apply_elapsed = self._telemetry.elapsed_since(apply_start)
        self._telemetry.record_plan_apply(drawable_name, apply_elapsed, cartesian=True)

    def begin_frame(self) -> None:
        self._telemetry.begin_frame()
        self._shared_primitives.begin_frame()

    def end_frame(self) -> None:
        self._shared_primitives.end_frame()
        self._flush_offscreen_to_main()
        self._telemetry.end_frame()

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
            from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea as FunctionsAreaDrawable
            self.register(FunctionsAreaDrawable, self._render_functions_bounded_colored_area)
        except Exception:
            pass
        try:
            from drawables.function_segment_bounded_colored_area import FunctionSegmentBoundedColoredArea as FunctionSegmentAreaDrawable
            self.register(FunctionSegmentAreaDrawable, self._render_function_segment_bounded_colored_area)
        except Exception:
            pass
        try:
            from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea as SegmentsAreaDrawable
            self.register(SegmentsAreaDrawable, self._render_segments_bounded_colored_area)
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
            pixel_width = int(rect.width)
            pixel_height = int(rect.height)
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

    def _resize_to_container(self) -> None:
        container = getattr(self.canvas_el, "parentElement", None)
        if container is None or not hasattr(container, "getBoundingClientRect"):
            return
        rect = container.getBoundingClientRect()
        if rect.width != self.canvas_el.width or rect.height != self.canvas_el.height:
            pixel_width = int(rect.width)
            pixel_height = int(rect.height)
            self.canvas_el.width = pixel_width
            self.canvas_el.height = pixel_height
            self.canvas_el.attrs["width"] = str(pixel_width)
            self.canvas_el.attrs["height"] = str(pixel_height)
        self.canvas_el.style.width = f"{int(self.canvas_el.width)}px"
        self.canvas_el.style.height = f"{int(self.canvas_el.height)}px"

    def _render_drawable(self, drawable: Any, coordinate_mapper: Any) -> None:
        drawable_name = self._resolve_drawable_name(drawable)
        map_state = self._capture_map_state(coordinate_mapper)
        signature = self._compute_drawable_signature(drawable)
        cache_key = self._plan_cache_key(drawable, drawable_name)
        cached_entry = self._plan_cache.get(cache_key)
        plan: Optional[OptimizedPrimitivePlan] = None
        if (
            cached_entry is not None
            and cached_entry.get("signature") == signature
            and isinstance(cached_entry.get("plan"), OptimizedPrimitivePlan)
        ):
            plan = cached_entry["plan"]
            plan.update_map_state(map_state)
        else:
            build_start = self._telemetry.mark_time()
            plan = build_plan_for_drawable(drawable, coordinate_mapper, self.style, supports_transform=False)
            build_elapsed = self._telemetry.elapsed_since(build_start)
            if plan is not None:
                self._telemetry.record_plan_build(drawable_name, build_elapsed)
                plan.update_map_state(map_state)
                if signature is not None:
                    self._plan_cache[cache_key] = {"plan": plan, "signature": signature}
            else:
                self._telemetry.record_plan_miss(drawable_name)
                self._plan_cache.pop(cache_key, None)
                return
        apply_start = self._telemetry.mark_time()
        if not plan.is_visible(self.canvas_el.width, self.canvas_el.height):
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

    def _compute_drawable_signature(self, drawable: Any) -> Tuple[Any, ...]:
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
        return self._freeze_signature(snapshot)

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
        return self._telemetry.drain()

    def peek_telemetry(self) -> Dict[str, Any]:
        return self._telemetry.snapshot()

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

    def _flush_offscreen_to_main(self) -> None:
        if not self._use_layer_compositing or self._offscreen_canvas is None:
            return
        try:
            self.ctx.clearRect(0, 0, self.canvas_el.width, self.canvas_el.height)
            self.ctx.drawImage(self._offscreen_canvas, 0, 0)
        except Exception:
            try:
                off_ctx = self._offscreen_canvas.getContext("2d")
                if off_ctx is not None:
                    image = off_ctx.getImageData(0, 0, self._offscreen_canvas.width, self._offscreen_canvas.height)
                    self.ctx.putImageData(image, 0, 0)
            except Exception:
                pass


