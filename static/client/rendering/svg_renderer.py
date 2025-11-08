from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Set, Tuple

from browser import document, svg, window

from rendering.optimized_drawable_renderers import (
    OptimizedPrimitivePlan,
    build_plan_for_cartesian,
    build_plan_for_drawable,
)
from rendering.interfaces import RendererProtocol
from rendering.style_manager import get_renderer_style
from rendering.svg_primitive_adapter import SvgPrimitiveAdapter
from rendering.shared_drawable_renderers import (
    render_point_helper,
    render_segment_helper,
    render_circle_helper,
    render_vector_helper,
    render_angle_helper,
    render_function_helper,
    render_functions_bounded_area_helper,
    render_function_segment_area_helper,
    render_segments_bounded_area_helper,
    render_triangle_helper,
    render_rectangle_helper,
    render_ellipse_helper,
    render_cartesian_helper,
)


class SvgTelemetry:
    def __init__(self) -> None:
        self._mode: str = "legacy"
        self.reset()

    def reset(self) -> None:
        mode = getattr(self, "_mode", "legacy")
        self._phase_totals: Dict[str, float] = {
            "plan_build_ms": 0.0,
            "plan_apply_ms": 0.0,
            "cartesian_plan_build_ms": 0.0,
            "cartesian_plan_apply_ms": 0.0,
            "legacy_render_ms": 0.0,
        }
        self._phase_counts: Dict[str, int] = {
            "plan_build_count": 0,
            "plan_apply_count": 0,
            "cartesian_plan_count": 0,
            "legacy_render_count": 0,
            "plan_miss_count": 0,
            "plan_skip_count": 0,
        }
        self._per_drawable: Dict[str, Dict[str, float]] = {}
        self._adapter_events: Dict[str, int] = {}
        self._frames: int = 0
        self._max_batch_depth: int = 0
        self._mode = mode

    def set_mode(self, mode: str) -> None:
        self._mode = mode

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

    def record_legacy_render(self, name: str, duration_ms: float, *, cartesian: bool = False) -> None:
        self._phase_totals["legacy_render_ms"] += duration_ms
        self._phase_counts["legacy_render_count"] += 1
        bucket = self._drawable_bucket(name)
        bucket["legacy_render_ms"] += duration_ms
        bucket["legacy_render_count"] += 1

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
            "mode": self._mode,
            "frames": self._frames,
            "phase": phase,
            "per_drawable": per_drawable,
            "adapter_events": adapter_events,
        }

    def drain(self) -> Dict[str, Any]:
        snapshot = self.snapshot()
        self.reset()
        return snapshot


class SvgRenderer(RendererProtocol):
    """SVG renderer with caching, culling, and optional offscreen staging."""

    def __init__(self, style_config: Optional[Dict[str, Any]] = None, surface_id: str = "math-svg") -> None:
        self.style: Dict[str, Any] = get_renderer_style(style_config)
        self._handlers_by_type: Dict[type, Callable[[Any, Any], None]] = {}
        self._render_mode: str = "legacy"
        self._primary_surface_id: str = surface_id
        self._offscreen_surface_id: str = f"{surface_id}-offscreen"
        self._telemetry = SvgTelemetry()
        self._plan_cache: Dict[str, Dict[str, Any]] = {}
        self._cartesian_cache: Optional[Dict[str, Any]] = None
        self._use_offscreen_surface: bool = self._should_use_offscreen_surface()
        self._offscreen_surface: Optional[Any] = None
        if self._use_offscreen_surface:
            self._offscreen_surface = self._ensure_offscreen_surface()
            adapter_surface_id = self._offscreen_surface_id
        else:
            adapter_surface_id = self._primary_surface_id
        self._shared_primitives: SvgPrimitiveAdapter = SvgPrimitiveAdapter(
            adapter_surface_id, telemetry=self._telemetry
        )
        self._telemetry.set_mode(self._render_mode)
        self._frame_seen_plan_keys: Set[str] = set()
        self._cartesian_rendered_this_frame: bool = False

    def register_default_drawables(self) -> None:
        self._register_shape("drawables.point", "Point", self._render_point)
        self._register_shape("drawables.segment", "Segment", self._render_segment)
        self._register_shape("drawables.circle", "Circle", self._render_circle)
        self._register_shape("drawables.ellipse", "Ellipse", self._render_ellipse)
        self._register_shape("drawables.vector", "Vector", self._render_vector)
        self._register_shape("drawables.angle", "Angle", self._render_angle)
        self._register_shape("drawables.function", "Function", self._render_function)
        self._register_shape("drawables.triangle", "Triangle", self._render_triangle)
        self._register_shape("drawables.rectangle", "Rectangle", self._render_rectangle)
        self._register_shape(
            "drawables.functions_bounded_colored_area",
            "FunctionsBoundedColoredArea",
            self._render_functions_bounded_colored_area,
        )
        self._register_shape(
            "drawables.function_segment_bounded_colored_area",
            "FunctionSegmentBoundedColoredArea",
            self._render_function_segment_bounded_colored_area,
        )
        self._register_shape(
            "drawables.segments_bounded_colored_area",
            "SegmentsBoundedColoredArea",
            self._render_segments_bounded_colored_area,
        )

    def _register_shape(self, module_path: str, class_name: str, handler: Callable[[Any, Any], None]) -> None:
        try:
            module = __import__(module_path, fromlist=[class_name])
            drawable_cls = getattr(module, class_name)
            self.register(drawable_cls, handler)
        except Exception:
            pass

    def register(self, cls: type, handler: Callable[[Any, Any], None]) -> None:
        """Register a handler for a given drawable class."""
        self._handlers_by_type[cls] = handler

    def clear(self) -> None:
        self._plan_cache.clear()
        self._cartesian_cache = None
        try:
            self._shared_primitives.clear_surface()
        except Exception:
            pass
        try:
            document[self._primary_surface_id].clear()
        except Exception:
            pass

    def set_render_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized == "optimized":
            self._render_mode = "optimized"
        else:
            self._render_mode = "legacy"
        self._telemetry.set_mode(self._render_mode)

    def get_render_mode(self) -> str:
        return self._render_mode

    def begin_frame(self) -> None:
        self._telemetry.begin_frame()
        if self._use_offscreen_surface:
            self._pull_main_to_offscreen()
        self._shared_primitives.begin_frame()
        self._frame_seen_plan_keys.clear()
        self._cartesian_rendered_this_frame = False

    def end_frame(self) -> None:
        self._shared_primitives.end_frame()
        if self._use_offscreen_surface:
            self._push_offscreen_to_main()
        if self._render_mode == "optimized":
            self._prune_unused_plan_entries()
        self._telemetry.end_frame()

    def render(self, drawable: Any, coordinate_mapper: Any) -> bool:
        handler: Optional[Callable[[Any, Any], None]] = self._handlers_by_type.get(type(drawable))
        if handler is None:
            return False
        handler(drawable, coordinate_mapper)
        return True

    # ----------------------- Point -----------------------
    def register_point(self, point_cls: type) -> None:
        self.register(point_cls, self._render_point)

    def _render_point(self, point: Any, coordinate_mapper: Any) -> None:
        self._render_with_mode(point, coordinate_mapper, render_point_helper)

    # ----------------------- Segment -----------------------
    def register_segment(self, segment_cls: type) -> None:
        self.register(segment_cls, self._render_segment)

    def _render_segment(self, segment: Any, coordinate_mapper: Any) -> None:
        self._render_with_mode(segment, coordinate_mapper, render_segment_helper)

    # ----------------------- Circle -----------------------
    def register_circle(self, circle_cls: type) -> None:
        self.register(circle_cls, self._render_circle)

    def _render_circle(self, circle: Any, coordinate_mapper: Any) -> None:
        self._render_with_mode(circle, coordinate_mapper, render_circle_helper)

    # ----------------------- Ellipse -----------------------
    def register_ellipse(self, ellipse_cls: type) -> None:
        self.register(ellipse_cls, self._render_ellipse)

    def _render_ellipse(self, ellipse: Any, coordinate_mapper: Any) -> None:
        self._render_with_mode(ellipse, coordinate_mapper, render_ellipse_helper)

    # ----------------------- Vector -----------------------
    def register_vector(self, vector_cls: type) -> None:
        self.register(vector_cls, self._render_vector)

    def _render_vector(self, vector: Any, coordinate_mapper: Any) -> None:
        self._render_with_mode(vector, coordinate_mapper, render_vector_helper)

    # ----------------------- Angle -----------------------
    def register_angle(self, angle_cls: type) -> None:
        self.register(angle_cls, self._render_angle)

    def _render_angle(self, angle: Any, coordinate_mapper: Any) -> None:
        self._render_with_mode(angle, coordinate_mapper, render_angle_helper)

    # ----------------------- Triangle -----------------------
    def register_triangle(self, triangle_cls: type) -> None:
        self.register(triangle_cls, self._render_triangle)

    def _render_triangle(self, triangle: Any, coordinate_mapper: Any) -> None:
        self._render_with_mode(triangle, coordinate_mapper, render_triangle_helper)

    # ----------------------- Rectangle -----------------------
    def register_rectangle(self, rectangle_cls: type) -> None:
        self.register(rectangle_cls, self._render_rectangle)

    def _render_rectangle(self, rectangle: Any, coordinate_mapper: Any) -> None:
        self._render_with_mode(rectangle, coordinate_mapper, render_rectangle_helper)

    # ----------------------- Function -----------------------
    def register_function(self, function_cls: type) -> None:
        self.register(function_cls, self._render_function)

    def _render_function(self, func: Any, coordinate_mapper: Any) -> None:
        self._render_with_mode(func, coordinate_mapper, render_function_helper)

    # ----------------------- Cartesian Grid -----------------------
    def render_cartesian(self, cartesian: Any, coordinate_mapper: Any) -> None:
        width, height = self._get_surface_dimensions()
        setattr(cartesian, "width", width)
        setattr(cartesian, "height", height)
        drawable_name = "Cartesian2Axis"
        map_state = self._capture_map_state(coordinate_mapper)
        signature = self._compute_drawable_signature(cartesian)
        if self._render_mode == "optimized":
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
                plan = build_plan_for_cartesian(cartesian, coordinate_mapper, self.style)
                build_elapsed = self._telemetry.elapsed_since(build_start)
                self._telemetry.record_plan_build(drawable_name, build_elapsed, cartesian=True)
                plan.update_map_state(map_state)
                self._cartesian_cache = {"plan": plan, "signature": signature}
            self._cartesian_rendered_this_frame = True
            if not plan.is_visible(width, height):
                self._telemetry.record_plan_skip(drawable_name)
                return
            apply_start = self._telemetry.mark_time()
            plan.apply(self._shared_primitives)
            apply_elapsed = self._telemetry.elapsed_since(apply_start)
            self._telemetry.record_plan_apply(drawable_name, apply_elapsed, cartesian=True)
            return
        legacy_start = self._telemetry.mark_time()
        try:
            render_cartesian_helper(self._shared_primitives, cartesian, coordinate_mapper, self.style)
            self._cartesian_rendered_this_frame = True
        finally:
            legacy_elapsed = self._telemetry.elapsed_since(legacy_start)
            self._telemetry.record_legacy_render(drawable_name, legacy_elapsed, cartesian=True)

    # ----------------------- Colored Areas: FunctionsBoundedColoredArea -----------------------
    def register_functions_bounded_colored_area(self, cls: type) -> None:
        self.register(cls, self._render_functions_bounded_colored_area)

    def _render_functions_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        self._render_with_mode(area, coordinate_mapper, render_functions_bounded_area_helper)

    # ----------------------- Colored Areas: FunctionSegmentBoundedColoredArea -----------------
    def register_function_segment_bounded_colored_area(self, cls: type) -> None:
        self.register(cls, self._render_function_segment_bounded_colored_area)

    def _render_function_segment_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        self._render_with_mode(area, coordinate_mapper, render_function_segment_area_helper)

    # ----------------------- Colored Areas: SegmentsBoundedColoredArea -----------------------
    def register_segments_bounded_colored_area(self, cls: type) -> None:
        self.register(cls, self._render_segments_bounded_colored_area)

    def _render_segments_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        self._render_with_mode(area, coordinate_mapper, render_segments_bounded_area_helper)

    def _render_with_mode(self, drawable: Any, coordinate_mapper: Any, legacy_callable: Callable[[Any, Any, Any], None]) -> None:
        drawable_name = self._resolve_drawable_name(drawable)
        map_state = self._capture_map_state(coordinate_mapper)
        signature = self._compute_drawable_signature(drawable)
        cache_key = self._plan_cache_key(drawable, drawable_name)
        if self._render_mode == "optimized":
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
                plan = build_plan_for_drawable(drawable, coordinate_mapper, self.style)
                build_elapsed = self._telemetry.elapsed_since(build_start)
                if plan is not None:
                    self._telemetry.record_plan_build(drawable_name, build_elapsed)
                    plan.update_map_state(map_state)
                    if signature is not None:
                        self._plan_cache[cache_key] = {"plan": plan, "signature": signature}
                else:
                    self._telemetry.record_plan_miss(drawable_name)
                    self._plan_cache.pop(cache_key, None)
            if plan is not None:
                self._frame_seen_plan_keys.add(cache_key)
                width, height = self._get_surface_dimensions()
                if not plan.is_visible(width, height):
                    self._telemetry.record_plan_skip(drawable_name)
                    return
                apply_start = self._telemetry.mark_time()
                plan.apply(self._shared_primitives)
                apply_elapsed = self._telemetry.elapsed_since(apply_start)
                self._telemetry.record_plan_apply(drawable_name, apply_elapsed)
                return
        legacy_start = self._telemetry.mark_time()
        try:
            legacy_callable(self._shared_primitives, drawable, coordinate_mapper, self.style)
        finally:
            legacy_elapsed = self._telemetry.elapsed_since(legacy_start)
            self._telemetry.record_legacy_render(drawable_name, legacy_elapsed)

    def drain_telemetry(self) -> Dict[str, Any]:
        return self._telemetry.drain()

    def peek_telemetry(self) -> Dict[str, Any]:
        return self._telemetry.snapshot()

    def invalidate_drawable_cache(self, drawable: Any) -> None:
        cache_key = self._plan_cache_key(drawable, self._resolve_drawable_name(drawable))
        self._plan_cache.pop(cache_key, None)

    def invalidate_all_drawable_caches(self) -> None:
        self._plan_cache.clear()

    def invalidate_cartesian_cache(self) -> None:
        self._cartesian_cache = None

    def _prune_unused_plan_entries(self) -> None:
        if not self._plan_cache:
            self._frame_seen_plan_keys.clear()
            return
        stale_keys = [key for key in self._plan_cache.keys() if key not in self._frame_seen_plan_keys]
        for key in stale_keys:
            self._plan_cache.pop(key, None)
        self._frame_seen_plan_keys.clear()

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

    def _get_surface_dimensions(self) -> Tuple[float, float]:
        try:
            surface = document[self._primary_surface_id]
            rect = surface.getBoundingClientRect()
            width = float(rect.width)
            height = float(rect.height)
            if width <= 0 or height <= 0:
                width_attr = surface.attrs.get("width")
                height_attr = surface.attrs.get("height")
                width = float(width_attr) if width_attr else width
                height = float(height_attr) if height_attr else height
        except Exception:
            width = float(self.style.get("viewport_width", 0) or 0)
            height = float(self.style.get("viewport_height", 0) or 0)
        return (width, height)

    def _should_use_offscreen_surface(self) -> bool:
        try:
            flag = getattr(window, "MatHudSvgOffscreen", None)
            if isinstance(flag, bool):
                return flag
        except Exception:
            pass
        try:
            stored = window.localStorage.getItem("mathud.svg.offscreen")
            if stored and stored.lower() in {"1", "true", "yes", "on"}:
                return True
        except Exception:
            pass
        return False

    def _ensure_offscreen_surface(self) -> Optional[Any]:
        try:
            offscreen = document[self._offscreen_surface_id]
        except Exception:
            offscreen = None
        if offscreen is None:
            try:
                main_surface = document[self._primary_surface_id]
            except Exception:
                return None
            offscreen = svg.svg(id=self._offscreen_surface_id)
            width_attr = main_surface.attrs.get("width")
            height_attr = main_surface.attrs.get("height")
            if width_attr:
                offscreen.attrs["width"] = width_attr
            if height_attr:
                offscreen.attrs["height"] = height_attr
            offscreen.style.display = "none"
            parent = getattr(main_surface, "parentElement", None)
            if parent is None:
                document <= offscreen
            else:
                parent <= offscreen
        self._offscreen_surface = offscreen
        self._sync_offscreen_size()
        return offscreen

    def _sync_offscreen_size(self) -> None:
        if not self._use_offscreen_surface:
            return
        try:
            main_surface = document[self._primary_surface_id]
            offscreen = self._offscreen_surface or document[self._offscreen_surface_id]
        except Exception:
            return
        if not offscreen or not main_surface:
            return
        for attr in ("width", "height"):
            value = main_surface.attrs.get(attr)
            if value:
                offscreen.attrs[attr] = value

    def _get_main_surface(self) -> Optional[Any]:
        try:
            return document[self._primary_surface_id]
        except Exception:
            return None

    def _pull_main_to_offscreen(self) -> None:
        if not self._use_offscreen_surface:
            return
        offscreen = self._ensure_offscreen_surface()
        main_surface = self._get_main_surface()
        if offscreen is None or main_surface is None:
            return
        self._sync_offscreen_size()
        try:
            main_surface.clear()
        except Exception:
            pass

    def _push_offscreen_to_main(self) -> None:
        if not self._use_offscreen_surface:
            return
        main_surface = self._get_main_surface()
        offscreen = self._offscreen_surface
        if main_surface is None or offscreen is None:
            return
        try:
            main_surface.clear()
        except Exception:
            pass
        try:
            children = list(offscreen.children)
        except Exception:
            children = []
        for child in children:
            try:
                clone = child.cloneNode(True)
            except Exception:
                pass
            else:
                try:
                    main_surface <= clone
                    self._telemetry.record_adapter_event("svg_clone_copy")
                except Exception:
                    pass


