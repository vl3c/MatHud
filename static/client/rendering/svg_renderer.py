from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Set, Tuple

from browser import document, svg, window

from rendering.cached_render_plan import (
    OptimizedPrimitivePlan,
    build_plan_for_cartesian,
    build_plan_for_polar,
    build_plan_for_drawable,
)
from rendering.interfaces import RendererProtocol
from rendering.style_manager import get_renderer_style
from rendering.svg_primitive_adapter import SvgPrimitiveAdapter

class SvgTelemetry:
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
                "plan_build_count": 0,
                "plan_apply_count": 0,
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


class SvgRenderer(RendererProtocol):
    """SVG renderer with caching, culling, and optional offscreen staging."""

    SKIP_AUTO_CLEAR = True

    def __init__(self, style_config: Optional[Dict[str, Any]] = None, surface_id: str = "math-svg") -> None:
        self.style: Dict[str, Any] = get_renderer_style(style_config)
        self._handlers_by_type: Dict[type, Callable[[Any, Any], None]] = {}
        self._primary_surface_id: str = surface_id
        self._offscreen_surface_id: str = f"{surface_id}-offscreen"
        self._telemetry = SvgTelemetry()
        self._initialize_plan_state()
        adapter_surface_id = self._configure_surfaces()
        self._shared_primitives: SvgPrimitiveAdapter = SvgPrimitiveAdapter(
            adapter_surface_id, telemetry=self._telemetry
        )
        self._frame_seen_plan_keys: Set[str] = set()
        self._cartesian_rendered_this_frame: bool = False

    def _record_plan_usage(self, name: str, usage_counts: Dict[str, int], *, cartesian: bool = False) -> None:
        if not usage_counts:
            return
        total = 0
        for count in usage_counts.values():
            try:
                total += int(count)
            except Exception:
                continue
        if total <= 0:
            return
        if cartesian:
            for _ in range(total):
                self._telemetry.record_plan_apply(name, 0.0, cartesian=True)
        else:
            for _ in range(total):
                self._telemetry.record_plan_apply(name, 0.0)

    def register_default_drawables(self) -> None:
        self._register_shape("drawables.point", "Point", self._render_point)
        self._register_shape("drawables.segment", "Segment", self._render_segment)
        self._register_shape("drawables.circle", "Circle", self._render_circle)
        self._register_shape("drawables.circle_arc", "CircleArc", self._render_circle_arc)
        self._register_shape("drawables.ellipse", "Ellipse", self._render_ellipse)
        self._register_shape("drawables.vector", "Vector", self._render_vector)
        self._register_shape("drawables.angle", "Angle", self._render_angle)
        self._register_shape("drawables.function", "Function", self._render_function)
        self._register_shape("drawables.piecewise_function", "PiecewiseFunction", self._render_function)
        self._register_shape("drawables.triangle", "Triangle", self._render_triangle)
        self._register_shape("drawables.rectangle", "Rectangle", self._render_rectangle)
        self._register_shape("drawables.bar", "Bar", self._render_drawable)
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
        self._register_shape(
            "drawables.closed_shape_colored_area",
            "ClosedShapeColoredArea",
            self._render_closed_shape_colored_area,
        )
        self._register_shape("drawables.label", "Label", self._render_label)

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
            self._prune_unused_plan_entries()
        # Clear grid SVG elements if grid was not rendered this frame (visibility off)
        if not self._cartesian_rendered_this_frame and self._cartesian_cache:
            self.invalidate_cartesian_cache()
        self._telemetry.end_frame()

    def render(self, drawable: Any, coordinate_mapper: Any) -> bool:
        # Handlers perform the actual drawing; this method only dispatches.
        handler: Optional[Callable[[Any, Any], None]] = self._handlers_by_type.get(type(drawable))
        if handler is None:
            return False
        handler(drawable, coordinate_mapper)
        return True

    # ----------------------- Point -----------------------
    def register_point(self, point_cls: type) -> None:
        self.register(point_cls, self._render_point)

    def _render_point(self, point: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(point, coordinate_mapper)

    # ----------------------- Segment -----------------------
    def register_segment(self, segment_cls: type) -> None:
        self.register(segment_cls, self._render_segment)

    def _render_segment(self, segment: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(segment, coordinate_mapper)

    # ----------------------- Circle -----------------------
    def register_circle(self, circle_cls: type) -> None:
        self.register(circle_cls, self._render_circle)

    def _render_circle(self, circle: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(circle, coordinate_mapper)

    def register_circle_arc(self, arc_cls: type) -> None:
        self.register(arc_cls, self._render_circle_arc)

    def _render_circle_arc(self, arc: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(arc, coordinate_mapper)

    # ----------------------- Ellipse -----------------------
    def register_ellipse(self, ellipse_cls: type) -> None:
        self.register(ellipse_cls, self._render_ellipse)

    def _render_ellipse(self, ellipse: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(ellipse, coordinate_mapper)

    # ----------------------- Vector -----------------------
    def register_vector(self, vector_cls: type) -> None:
        self.register(vector_cls, self._render_vector)

    def _render_vector(self, vector: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(vector, coordinate_mapper)

    # ----------------------- Angle -----------------------
    def register_angle(self, angle_cls: type) -> None:
        self.register(angle_cls, self._render_angle)

    def _render_angle(self, angle: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(angle, coordinate_mapper)

    # ----------------------- Triangle -----------------------
    def register_triangle(self, triangle_cls: type) -> None:
        self.register(triangle_cls, self._render_triangle)

    def _render_triangle(self, triangle: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(triangle, coordinate_mapper)

    # ----------------------- Rectangle -----------------------
    def register_rectangle(self, rectangle_cls: type) -> None:
        self.register(rectangle_cls, self._render_rectangle)

    def _render_rectangle(self, rectangle: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(rectangle, coordinate_mapper)

    # ----------------------- Function -----------------------
    def register_function(self, function_cls: type) -> None:
        self.register(function_cls, self._render_function)

    def _render_function(self, func: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(func, coordinate_mapper)

    # ----------------------- Cartesian Grid -----------------------
    def render_cartesian(self, cartesian: Any, coordinate_mapper: Any) -> None:
        width, height = self._get_surface_dimensions()
        self._assign_cartesian_dimensions(cartesian, width, height)
        drawable_name = "Cartesian2Axis"
        map_state = self._capture_map_state(coordinate_mapper)
        signature = self._compute_drawable_signature(cartesian, coordinate_mapper)
        plan_context = self._resolve_cartesian_plan(
            cartesian, coordinate_mapper, map_state, signature, drawable_name
        )
        if plan_context is None:
            return
        self._cartesian_rendered_this_frame = True
        if self._skip_invisible_cartesian(plan_context, drawable_name, width, height):
            return
        self._apply_cartesian_plan(plan_context, drawable_name)

    # ----------------------- Polar Grid -----------------------
    def render_polar(self, polar_grid: Any, coordinate_mapper: Any) -> None:
        width, height = self._get_surface_dimensions()
        self._assign_polar_dimensions(polar_grid, width, height)
        drawable_name = "PolarGrid"
        map_state = self._capture_map_state(coordinate_mapper)
        signature = self._compute_drawable_signature(polar_grid, coordinate_mapper)
        plan_context = self._resolve_polar_plan(
            polar_grid, coordinate_mapper, map_state, signature, drawable_name
        )
        if plan_context is None:
            return
        self._cartesian_rendered_this_frame = True
        if self._skip_invisible_cartesian(plan_context, drawable_name, width, height):
            return
        self._apply_cartesian_plan(plan_context, drawable_name)

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
    ) -> Optional[Dict[str, Any]]:
        plan_entry = self._cartesian_cache
        if self._is_cached_plan_valid(plan_entry, signature):
            plan = plan_entry["plan"]
            plan.update_map_state(map_state)
            self._mark_screen_space_plan_dirty(plan)
        else:
            if plan_entry is not None:
                self._drop_plan_group(plan_entry.get("plan"))
            plan = self._build_polar_plan_with_metrics(
                polar_grid, coordinate_mapper, map_state, drawable_name
            )
            if plan is None:
                self._cartesian_cache = None
                return None
            self._cartesian_cache = {"plan": plan, "signature": signature}
        return self._create_cartesian_plan_context(plan)

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

    # ----------------------- Colored Areas: FunctionsBoundedColoredArea -----------------------
    def register_functions_bounded_colored_area(self, cls: type) -> None:
        self.register(cls, self._render_functions_bounded_colored_area)

    def _render_functions_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(area, coordinate_mapper)

    # ----------------------- Colored Areas: FunctionSegmentBoundedColoredArea -----------------
    def register_function_segment_bounded_colored_area(self, cls: type) -> None:
        self.register(cls, self._render_function_segment_bounded_colored_area)

    def _render_function_segment_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(area, coordinate_mapper)

    # ----------------------- Colored Areas: SegmentsBoundedColoredArea -----------------------
    def register_segments_bounded_colored_area(self, cls: type) -> None:
        self.register(cls, self._render_segments_bounded_colored_area)

    def _render_segments_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(area, coordinate_mapper)

    # ----------------------- Colored Areas: ClosedShapeColoredArea -----------------------
    def register_closed_shape_colored_area(self, cls: type) -> None:
        self.register(cls, self._render_closed_shape_colored_area)

    def _render_closed_shape_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(area, coordinate_mapper)

    # ----------------------- Label -----------------------
    def register_label(self, label_cls: type) -> None:
        self.register(label_cls, self._render_label)

    def _render_label(self, label: Any, coordinate_mapper: Any) -> None:
        self._render_drawable(label, coordinate_mapper)

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
        plan_context = self._resolve_drawable_plan_context(
            drawable, coordinate_mapper, map_state, signature, drawable_name, cache_key
        )
        if plan_context is None:
            return
        self._frame_seen_plan_keys.add(cache_key)
        self._apply_drawable_plan(plan_context, drawable_name)

    def drain_telemetry(self) -> Dict[str, Any]:
        return self._telemetry.drain()

    def peek_telemetry(self) -> Dict[str, Any]:
        return self._telemetry.snapshot()

    def invalidate_drawable_cache(self, drawable: Any) -> None:
        cache_key = self._plan_cache_key(drawable, self._resolve_drawable_name(drawable))
        entry = self._plan_cache.pop(cache_key, None)
        if entry:
            plan_obj = entry.get("plan")
            drop = getattr(self._shared_primitives, "drop_group", None)
            if callable(drop) and getattr(plan_obj, "plan_key", None):
                drop(plan_obj.plan_key)

    def invalidate_all_drawable_caches(self) -> None:
        if self._plan_cache:
            drop = getattr(self._shared_primitives, "drop_group", None)
            if callable(drop):
                for entry in self._plan_cache.values():
                    plan_obj = entry.get("plan") if isinstance(entry, dict) else None
                    if getattr(plan_obj, "plan_key", None):
                        drop(plan_obj.plan_key)
        self._plan_cache.clear()

    def invalidate_cartesian_cache(self) -> None:
        if self._cartesian_cache:
            plan_obj = self._cartesian_cache.get("plan")
            drop = getattr(self._shared_primitives, "drop_group", None)
            if callable(drop) and getattr(plan_obj, "plan_key", None):
                drop(plan_obj.plan_key)
        self._cartesian_cache = None

    def _prune_unused_plan_entries(self) -> None:
        if not self._plan_cache:
            self._frame_seen_plan_keys.clear()
            return
        stale_keys = [key for key in self._plan_cache.keys() if key not in self._frame_seen_plan_keys]
        for key in stale_keys:
            entry = self._plan_cache.pop(key, None)
            if entry:
                plan_obj = entry.get("plan")
                drop = getattr(self._shared_primitives, "drop_group", None)
                if callable(drop) and getattr(plan_obj, "plan_key", None):
                    drop(plan_obj.plan_key)
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
        if self._needs_scale_in_signature(drawable) and coordinate_mapper is not None:
            scale = getattr(coordinate_mapper, "scale_factor", None)
            if scale is not None:
                snapshot["_view_scale"] = round(float(scale), 4)
            offset = getattr(coordinate_mapper, "offset", None)
            if offset is not None:
                snapshot["_view_offset"] = (round(float(offset.x), 2), round(float(offset.y), 2))
        return self._freeze_signature(snapshot)

    def _needs_scale_in_signature(self, drawable: Any) -> bool:
        class_name = getattr(drawable, "get_class_name", lambda: "")()
        return class_name in ("Function", "PiecewiseFunction", "FunctionsBoundedColoredArea", "FunctionSegmentBoundedColoredArea")

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

    def _initialize_plan_state(self) -> None:
        self._plan_cache = {}
        self._cartesian_cache = None

    def _configure_surfaces(self) -> str:
        self._use_offscreen_surface = self._should_use_offscreen_surface()
        self._offscreen_surface = None
        if self._use_offscreen_surface:
            self._offscreen_surface = self._ensure_offscreen_surface()
            adapter_surface_id = self._offscreen_surface_id
        else:
            adapter_surface_id = self._primary_surface_id
        self._clear_primary_surface()
        return adapter_surface_id

    def _clear_primary_surface(self) -> None:
        try:
            document[self._primary_surface_id].clear()
        except Exception:
            pass

    def _assign_cartesian_dimensions(self, cartesian: Any, width: int, height: int) -> None:
        setattr(cartesian, "width", width)
        setattr(cartesian, "height", height)

    def _resolve_cartesian_plan(
        self,
        cartesian: Any,
        coordinate_mapper: Any,
        map_state: Dict[str, float],
        signature: Optional[Any],
        drawable_name: str,
    ) -> Optional[Dict[str, Any]]:
        plan_entry = self._cartesian_cache
        if self._is_cached_plan_valid(plan_entry, signature):
            plan = plan_entry["plan"]
            plan.update_map_state(map_state)
            self._mark_screen_space_plan_dirty(plan)
        else:
            if plan_entry is not None:
                self._drop_plan_group(plan_entry.get("plan"))
            plan = self._build_cartesian_plan_with_metrics(
                cartesian, coordinate_mapper, map_state, drawable_name
            )
            if plan is None:
                self._cartesian_cache = None
                return None
            self._cartesian_cache = {"plan": plan, "signature": signature}
        return self._create_cartesian_plan_context(plan)

    def _skip_invisible_cartesian(
        self, context: Dict[str, Any], drawable_name: str, width: int, height: int
    ) -> bool:
        plan = context["plan"]
        plan_key = context["plan_key"]
        if plan.is_visible(width, height):
            return False
        if plan_key:
            self._clear_group(plan_key)
        self._telemetry.record_plan_skip(drawable_name)
        return True

    def _apply_cartesian_plan(self, context: Dict[str, Any], drawable_name: str) -> None:
        plan = context["plan"]
        plan_key = context["plan_key"]
        supports_transform = context["supports_transform"]
        needs_apply = context["needs_apply"]
        if needs_apply and plan_key:
            self._clear_group(plan_key)
        push_to_back = getattr(self._shared_primitives, "push_group_to_back", None)
        if callable(push_to_back) and plan_key:
            push_to_back(plan_key)
        apply_start = self._telemetry.mark_time()
        plan.apply(self._shared_primitives)
        if callable(push_to_back) and plan_key:
            push_to_back(plan_key)
        apply_elapsed = self._telemetry.elapsed_since(apply_start)
        self._telemetry.record_plan_apply(drawable_name, apply_elapsed, cartesian=True)
        if supports_transform:
            self._apply_plan_transform(plan_key, plan)
        usage_counts = context["usage_counts"]
        if usage_counts:
            self._record_plan_usage(drawable_name, usage_counts, cartesian=True)
        reserve = context["reserve"]
        if callable(reserve) and usage_counts:
            reserve(usage_counts, trim_excess=True)

    def _resolve_drawable_plan_context(
        self,
        drawable: Any,
        coordinate_mapper: Any,
        map_state: Dict[str, float],
        signature: Optional[Any],
        drawable_name: str,
        cache_key: str,
    ) -> Optional[Dict[str, Any]]:
        cached_entry = self._plan_cache.get(cache_key)
        if self._is_cached_plan_valid(cached_entry, signature):
            plan = cached_entry["plan"]
            plan.update_map_state(map_state)
            self._mark_screen_space_plan_dirty(plan)
        else:
            if cached_entry is not None:
                self._drop_plan_group(cached_entry.get("plan"))
            plan = self._build_drawable_plan_with_metrics(
                drawable, coordinate_mapper, map_state, drawable_name
            )
            if plan is None:
                self._plan_cache.pop(cache_key, None)
                return None
            if signature is not None:
                self._plan_cache[cache_key] = {"plan": plan, "signature": signature}
        return self._create_drawable_plan_context(plan)

    def _apply_drawable_plan(self, context: Dict[str, Any], drawable_name: str) -> None:
        plan = context["plan"]
        plan_key = context["plan_key"]
        supports_transform = context["supports_transform"]
        if supports_transform:
            self._apply_plan_transform(plan_key, plan)
        width, height = self._get_surface_dimensions()
        if not plan.is_visible(width, height, margin=0.0):
            if plan_key:
                self._clear_group(plan_key)
            self._telemetry.record_plan_skip(drawable_name)
            return
        if context["needs_apply"] and plan_key:
            self._clear_group(plan_key)
        apply_start = self._telemetry.mark_time()
        plan.apply(self._shared_primitives)
        apply_elapsed = self._telemetry.elapsed_since(apply_start)
        self._telemetry.record_plan_apply(drawable_name, apply_elapsed)
        usage_counts = context["usage_counts"]
        if usage_counts:
            self._record_plan_usage(drawable_name, usage_counts)
        reserve = context["reserve"]
        if callable(reserve) and usage_counts:
            reserve(usage_counts, trim_excess=True)

    def _is_cached_plan_valid(self, cache_entry: Optional[Dict[str, Any]], signature: Optional[Any]) -> bool:
        return bool(
            cache_entry
            and cache_entry.get("signature") == signature
            and isinstance(cache_entry.get("plan"), OptimizedPrimitivePlan)
        )

    def _build_cartesian_plan_with_metrics(
        self,
        cartesian: Any,
        coordinate_mapper: Any,
        map_state: Dict[str, float],
        drawable_name: str,
    ) -> Optional[OptimizedPrimitivePlan]:
        build_start = self._telemetry.mark_time()
        try:
            plan = build_plan_for_cartesian(cartesian, coordinate_mapper, self.style, supports_transform=False)
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

    def _build_drawable_plan_with_metrics(
        self,
        drawable: Any,
        coordinate_mapper: Any,
        map_state: Dict[str, float],
        drawable_name: str,
    ) -> Optional[OptimizedPrimitivePlan]:
        build_start = self._telemetry.mark_time()
        try:
            plan = build_plan_for_drawable(drawable, coordinate_mapper, self.style, supports_transform=False)
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

    def _create_cartesian_plan_context(self, plan: OptimizedPrimitivePlan) -> Dict[str, Any]:
        usage_counts = getattr(plan, "get_usage_counts", lambda: {})()
        reserve = getattr(self._shared_primitives, "reserve_usage_counts", None)
        plan_key = getattr(plan, "plan_key", None)
        supports_transform = getattr(plan, "supports_transform", lambda: False)()
        needs_apply = getattr(plan, "needs_apply", lambda: True)()
        if supports_transform:
            self._apply_plan_transform(plan_key, plan)
        return {
            "plan": plan,
            "usage_counts": usage_counts,
            "reserve": reserve,
            "plan_key": plan_key,
            "supports_transform": supports_transform,
            "needs_apply": needs_apply,
        }

    def _create_drawable_plan_context(self, plan: OptimizedPrimitivePlan) -> Dict[str, Any]:
        usage_counts = getattr(plan, "get_usage_counts", lambda: {})()
        reserve = getattr(self._shared_primitives, "reserve_usage_counts", None)
        plan_key = getattr(plan, "plan_key", None)
        supports_transform = getattr(plan, "supports_transform", lambda: False)()
        needs_apply = getattr(plan, "needs_apply", lambda: True)()
        return {
            "plan": plan,
            "usage_counts": usage_counts,
            "reserve": reserve,
            "plan_key": plan_key,
            "supports_transform": supports_transform,
            "needs_apply": needs_apply,
        }

    def _mark_screen_space_plan_dirty(self, plan: OptimizedPrimitivePlan) -> None:
        if getattr(plan, "uses_screen_space", lambda: False)():
            plan.mark_dirty()

    def _drop_plan_group(self, plan_obj: Optional[OptimizedPrimitivePlan]) -> None:
        if plan_obj is None:
            return
        drop = getattr(self._shared_primitives, "drop_group", None)
        if callable(drop) and getattr(plan_obj, "plan_key", None):
            drop(plan_obj.plan_key)

    def _clear_group(self, plan_key: Optional[str]) -> None:
        if not plan_key:
            return
        clear_group = getattr(self._shared_primitives, "clear_group", None)
        if callable(clear_group):
            clear_group(plan_key)

    def _apply_plan_transform(self, plan_key: Optional[str], plan: OptimizedPrimitivePlan) -> None:
        if not plan_key:
            return
        set_transform = getattr(self._shared_primitives, "set_group_transform", None)
        transform = getattr(plan, "get_transform", lambda: None)()
        if callable(set_transform):
            set_transform(plan_key, transform)


