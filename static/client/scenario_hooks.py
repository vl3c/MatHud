"""
Browser hooks for the agentic scenario-testing harness.

The harness (``cli/scenarios/``) drives the real app in headless Chrome and talks
to it only through these ``window`` functions. Like ``getMatHudTestResults`` they
take and return JSON strings:

    getMatHudCanvasState(optionsJson?)   canvas state, optionally with an inspection view
    runMatHudToolCalls(callsJson)        run one tool batch exactly as a model batch runs
    resetMatHudSession(optionsJson?)     clear canvas, undo history, traces, metrics and chat
    getMatHudTurnStatus()                whether a chat turn is running, and its progress
    sendMatHudMessage(text, modelId?, optionsJson?)  send a chat message as the user (live mode)
    stopMatHudTurn()                     stop the running chat turn
    fitMatHudView(optionsJson?)          zoom the view to fit every drawable (automation display only)

``fitMatHudView`` is for the automation paths only (the CLI's attach mode,
``desktop prompt`` and ``desktop fit``): in regular use the app never pans or
zooms on its own, so no UI or AI-turn code path calls it.

See documentation/development/agentic_scenario_testing.md (section 4.3).

The helpers below the hook class are plain Python over the canvas objects, so the
Brython test runner can exercise them without a browser session.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from browser import ajax, document, window

if TYPE_CHECKING:
    from ai_interface import AIInterface
    from canvas import Canvas

# Step used to probe listed vertical asymptotes: f(a - h) and f(a + h), h = 1e-7 * max(1, |a|).
ASYMPTOTE_PROBE_STEP = 1e-7
_FUNCTION_CLASSES = ("Function", "PiecewiseFunction")
_POLYGON_CLASSES = (
    "Triangle",
    "Rectangle",
    "Quadrilateral",
    "Pentagon",
    "Hexagon",
    "Heptagon",
    "Octagon",
    "Nonagon",
    "Decagon",
    "GenericPolygon",
)
# fitMatHudView: padding on each side as a share of the content's span, the smallest
# half-span shown (a single point gets a window of about +-1 around it), the function
# samples per curve, the x range for an unbounded function when nothing else gives one,
# and Tukey's fence factor that drops asymptote spikes from function samples.
FIT_PADDING = 0.12
FIT_MIN_HALF_SPAN = 1.0
FIT_SAMPLES = 64
FIT_DEFAULT_X_RANGE = (-10.0, 10.0)
FIT_FENCE = 3.0
FIT_VALUE_LIMIT = 1e9


class ScenarioHooks:
    """Registers the scenario-testing hooks on ``window`` for one AIInterface."""

    def __init__(self, ai_interface: "AIInterface") -> None:
        self.ai = ai_interface

    @property
    def canvas(self) -> "Canvas":
        return self.ai.canvas

    def register(self) -> None:
        """Expose the hooks on ``window``."""
        window.getMatHudCanvasState = self.get_canvas_state
        window.runMatHudToolCalls = self.run_tool_calls
        window.resetMatHudSession = self.reset_session
        window.getMatHudTurnStatus = self.get_turn_status
        window.sendMatHudMessage = self.send_message
        window.stopMatHudTurn = self.stop_turn
        window.fitMatHudView = self.fit_view

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def get_canvas_state(self, options_json: Any = None) -> str:
        """``{"state": ..., "inspection": ...?}``.

        Options: ``inspect`` (bool) adds the inspection view; ``samples``
        (``{function_name or "*": [x, ...]}``) and ``t_samples`` (the same for
        parametric curves) add function values to it.
        """
        try:
            options = parse_options(options_json)
            payload: Dict[str, Any] = {"state": self.canvas.get_canvas_state()}
            if options.get("inspect"):
                payload["inspection"] = build_inspection(self.canvas, options.get("samples"), options.get("t_samples"))
            return to_json(payload)
        except Exception as exc:
            return to_json({"status": "error", "error": str(exc)})

    def run_tool_calls(self, calls_json: Any) -> str:
        """Run one batch through ``AIInterface.execute_tool_batch``, the model's batch path.

        Returns the traced calls, the state after the batch, the undo and redo
        depths before and after, and which calls are undoable. Refused with
        ``{"status": "busy"}`` while a chat turn runs, so scripted calls never
        mix into a user turn's canvas or metrics.
        """
        try:
            calls = normalize_tool_calls(json.loads(str(calls_json)))
        except Exception as exc:
            return to_json({"status": "error", "error": f"Invalid tool calls: {exc}"})
        if self.ai.is_processing or self.ai._turn_metrics.is_active:
            return to_json({"status": "busy", "error": "a chat turn is running"})
        undo_before, redo_before = undo_depths(self.canvas)
        try:
            batch = self.ai.execute_tool_batch(calls, None)
        except Exception as exc:
            undo_after, redo_after = undo_depths(self.canvas)
            return to_json(
                {
                    "status": "error",
                    "error": str(exc),
                    "undo_depth_before": undo_before,
                    "undo_depth_after": undo_after,
                    "redo_depth_before": redo_before,
                    "redo_depth_after": redo_after,
                }
            )
        undo_after, redo_after = undo_depths(self.canvas)
        undoable = set(self.ai.undoable_functions)
        return to_json(
            {
                "status": "ok",
                "traced": batch["traced_calls"],
                "undoable": [call["function_name"] in undoable for call in calls],
                "state": batch["state_after"],
                "trace_id": (batch["trace"] or {}).get("trace_id"),
                "undo_depth_before": undo_before,
                "undo_depth_after": undo_after,
                "redo_depth_before": redo_before,
                "redo_depth_after": redo_after,
            }
        )

    def reset_session(self, options_json: Any = None) -> str:
        """Clear everything a scenario could have left behind, without saving a workspace.

        Options: ``fixture`` (a workspace-shaped canvas state, or ``{"state": ...}``)
        is restored after the reset; ``conversation`` (bool, default false) also
        POSTs ``/new_conversation`` to reset the server's chat history (each POST
        starts a new server session log); ``chat`` (bool, default true) clears the
        chat panel.
        """
        try:
            options = parse_options(options_json)
            if self.ai.is_processing:
                self.ai.stop_ai_processing()
            reset_canvas_session(self.canvas)
            fixture = options.get("fixture")
            if isinstance(fixture, dict):
                state = fixture.get("state", fixture)
                self.ai.workspace_manager._restore_workspace_state(state)
                clear_undo_history(self.canvas)
            self.ai._trace_collector.clear()
            self.ai._turn_metrics.clear()
            if options.get("chat", True):
                # clear_chat also forgets the saved-chat transcript and the typeset math.
                self.ai._chat_ui.clear_chat()
            if options.get("conversation"):
                req = ajax.ajax()
                req.open("POST", "/new_conversation", True)
                req.set_header("content-type", "application/json")
                req.send()
            return to_json({"status": "ok", "fixture": isinstance(fixture, dict)})
        except Exception as exc:
            return to_json({"status": "error", "error": str(exc)})

    def get_turn_status(self) -> str:
        """``{"processing", "completed_turns", "last_outcome", "tool_batches", "requests", "tool_calls"}``."""
        try:
            return to_json(turn_status(bool(self.ai.is_processing), self.ai._turn_metrics))
        except Exception as exc:
            return to_json({"status": "error", "error": str(exc)})

    def send_message(self, text: Any, model_id: Any = None, options_json: Any = None) -> str:
        """Send ``text`` as the user; ``modelId`` selects an existing model option first.

        Vision is switched off for this request, so it does not depend on a canvas
        capture, and the user's toggle is restored once the request is built.
        Options: ``max_requests`` caps the turn's model requests (the turn ends
        instead of sending more); ``response_timeout_ms`` replaces the client's
        response timeouts for this turn (a local model may think longer than 60 s).
        """
        try:
            if self.ai.is_processing:
                return to_json({"status": "busy"})
            options = parse_options(options_json)
            request_limit = _positive_int(options.get("max_requests"))
            timeout_ms = _positive_int(options.get("response_timeout_ms"))
            if model_id:
                selector = document["ai-model-selector"]
                values = [str(option.value) for option in selector.options]
                if str(model_id) not in values:
                    return to_json({"status": "error", "error": f"Model option not found: {model_id}"})
                selector.value = str(model_id)
            toggle = document["vision-toggle"] if "vision-toggle" in document else None
            vision_was_on = bool(toggle.checked) if toggle is not None else False
            if toggle is not None:
                toggle.checked = False
            try:
                # The prompt (including use_vision) is built synchronously inside this call.
                self.ai.send_user_message(str(text), request_limit, timeout_ms)
            finally:
                if toggle is not None:
                    toggle.checked = vision_was_on
            return to_json({"status": "started"})
        except Exception as exc:
            return to_json({"status": "error", "error": str(exc)})

    def stop_turn(self) -> str:
        """Stop the running chat turn, if any."""
        try:
            if not self.ai.is_processing:
                return to_json({"status": "idle"})
            self.ai.stop_ai_processing()
            return to_json({"status": "stopped"})
        except Exception as exc:
            return to_json({"status": "error", "error": str(exc)})

    def fit_view(self, options_json: Any = None) -> str:
        """Zoom the view so every drawable fills it, for someone watching an automated run.

        Presentation only, like a mouse zoom: no undo entry, the redo stack and the
        drawables stay as they are, and the coordinate mode is kept. With nothing
        to fit the view is left alone. Refused (``busy``) while a chat turn runs.
        Options: ``padding`` (share of the span on each side, default 0.12) and
        ``min_half_span`` (default 1). Only the automation paths call this hook.
        """
        try:
            if self.ai.is_processing:
                return to_json({"status": "busy", "error": "a chat turn is running"})
            options = parse_options(options_json)
            padding = float(options.get("padding", FIT_PADDING))
            min_half_span = float(options.get("min_half_span", FIT_MIN_HALF_SPAN))
            extent = content_extent(self.canvas)
            if extent is None:
                return to_json({"status": "ok", "fitted": False, "reason": "nothing to fit"})
            center_x, center_y, range_val, axis = fit_window(
                extent, float(self.canvas.width), float(self.canvas.height), padding, min_half_span
            )
            self.canvas.zoom(center_x, center_y, range_val, axis)
            return to_json(
                {
                    "status": "ok",
                    "fitted": True,
                    "extent": extent,
                    "view": self.canvas.get_canvas_state().get("Cartesian_System_Visibility"),
                }
            )
        except Exception as exc:
            return to_json({"status": "error", "error": str(exc)})


# ----------------------------------------------------------------------
# Plain helpers (tested by client_tests/test_scenario_hooks.py)
# ----------------------------------------------------------------------


def to_json(data: Any) -> str:
    """Serialize hook output; values JSON cannot express become strings."""
    return json.dumps(data, default=str)


def parse_options(raw: Any) -> Dict[str, Any]:
    """Parse an optional JSON options argument; missing or empty means ``{}``."""
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text or text in ("undefined", "null"):
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("options must be a JSON object")
    return parsed


def normalize_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
    """Accept ``[{"function_name", "arguments"}]`` or the scenario form ``[{"tool", "args"}]``."""
    if not isinstance(raw_calls, list):
        raise ValueError("calls must be a JSON list")
    calls: List[Dict[str, Any]] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            raise ValueError("each call must be a JSON object")
        name = raw.get("function_name", raw.get("tool"))
        arguments = raw.get("arguments", raw.get("args", {}))
        if not isinstance(name, str) or not name:
            raise ValueError("each call needs a function_name")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError(f"arguments of {name} must be a JSON object")
        call: Dict[str, Any] = {"function_name": name, "arguments": arguments}
        if raw.get("id") is not None:
            call["id"] = raw["id"]
        calls.append(call)
    return calls


def _positive_int(value: Any) -> Optional[int]:
    """``value`` as a positive int, or None (absent, not a number or not positive)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return int(value)


def undo_depths(canvas: "Canvas") -> tuple[int, int]:
    """Current undo and redo stack sizes."""
    manager = canvas.undo_redo_manager
    return len(manager.undo_stack), len(manager.redo_stack)


def clear_undo_history(canvas: "Canvas") -> None:
    """Empty the undo and redo stacks."""
    manager = canvas.undo_redo_manager
    manager.undo_stack = []
    manager.redo_stack = []


def reset_canvas_session(canvas: "Canvas") -> None:
    """Return the canvas to its start-up state without archiving anything.

    Drawables, computations, name-generator state, view, coordinate system
    (cartesian), grid visibility and the undo and redo stacks are all reset.
    """
    canvas.drawable_manager.drawables.clear()
    canvas._reset_name_generator_state()
    canvas.computations = []
    manager = canvas.coordinate_system_manager
    manager.set_mode("cartesian", redraw=False)
    manager.cartesian_grid.visible = True
    manager.polar_grid.visible = True
    clear_undo_history(canvas)
    canvas.reset()


def turn_status(is_processing: bool, collector: Any) -> Dict[str, Any]:
    """Summarize the chat turn state from the turn-metrics collector."""
    last = collector.last_turn()
    progress = collector.progress()
    return {
        "processing": is_processing,
        "completed_turns": collector.completed_turns,
        "last_outcome": last.get("outcome") if isinstance(last, dict) else None,
        "requests": progress["requests"],
        "tool_batches": progress["tool_batches"],
        "tool_calls": progress["tool_calls"],
    }


def build_inspection(
    canvas: "Canvas",
    samples: Optional[Dict[str, Any]] = None,
    t_samples: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attributes the canvas state omits, for checks and invariants.

    Per drawable: class, name, colour, attached label, and derived values
    (angle degrees, polygon vertices in order). Globally: undo and redo
    depths, grid visibility, the coordinate-system mode and the point-name
    generator's hints. Functions get the requested samples and a probe of
    every listed vertical asymptote.
    """
    undo_depth, redo_depth = undo_depths(canvas)
    manager = canvas.coordinate_system_manager
    drawables = [describe_drawable(drawable, samples or {}, t_samples or {}) for drawable in canvas.get_drawables()]
    return {
        "drawables": drawables,
        "undo_depth": undo_depth,
        "redo_depth": redo_depth,
        "coordinate_mode": str(manager.mode),
        "grid_visible": {
            "cartesian": bool(getattr(manager.cartesian_grid, "visible", False)),
            "polar": bool(getattr(manager.polar_grid, "visible", False)),
            "active": bool(manager.is_grid_visible()),
        },
        "polar_radial_spacing": _plain(getattr(manager.polar_grid, "_current_radial_spacing", None)),
        "name_hints": _name_hints(canvas),
    }


def describe_drawable(drawable: Any, samples: Dict[str, Any], t_samples: Dict[str, Any]) -> Dict[str, Any]:
    """Inspection record for one drawable."""
    class_name = str(drawable.get_class_name())
    info: Dict[str, Any] = {
        "class": class_name,
        "name": str(getattr(drawable, "name", "")),
        "color": _plain(getattr(drawable, "color", None)),
    }
    label = _attached_label(drawable)
    if label is not None:
        info["label"] = {
            "text": str(getattr(label, "text", "") or ""),
            "visible": bool(getattr(label, "visible", False)),
        }
    if class_name == "Angle":
        info.update(_angle_details(drawable))
    elif class_name in _POLYGON_CLASSES:
        info["vertices"] = _polygon_vertices(drawable)
    elif class_name in _FUNCTION_CLASSES:
        xs = _requested_values(samples, info["name"])
        if xs:
            info["samples"] = [[x, _safe_eval(drawable.function, x)] for x in xs]
        probes = _asymptote_probes(drawable)
        if probes:
            info["asymptote_probes"] = probes
    elif class_name == "ParametricFunction":
        ts = _requested_values(t_samples, info["name"])
        if ts:
            info["t_samples"] = [[t] + _safe_eval_pair(drawable.evaluate, t) for t in ts]
    return info


def _attached_label(drawable: Any) -> Any:
    label = getattr(drawable, "label", None)
    if label is None:
        segment = getattr(drawable, "segment", None)
        label = getattr(segment, "label", None) if segment is not None else None
    if label is None or not hasattr(label, "text"):
        return None
    return label


def _angle_details(angle: Any) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "angle_degrees": _plain(getattr(angle, "angle_degrees", None)),
        "raw_angle_degrees": _plain(getattr(angle, "raw_angle_degrees", None)),
        "is_reflex": bool(getattr(angle, "is_reflex", False)),
    }
    for key in ("vertex_point", "arm1_point", "arm2_point"):
        point = getattr(angle, key, None)
        details[key.replace("_point", "")] = [point.x, point.y] if point is not None else None
    return details


def _polygon_vertices(polygon: Any) -> Optional[List[List[float]]]:
    points = getattr(polygon, "_points", None)
    if not points:
        try:
            points = list(polygon.get_vertices())
        except Exception:
            return None
    return [[point.x, point.y] for point in points]


def _requested_values(requests: Dict[str, Any], name: str) -> List[float]:
    values: List[float] = []
    for key in (name, "*"):
        entry = requests.get(key)
        if isinstance(entry, list):
            values.extend(float(value) for value in entry if isinstance(value, (int, float)))
    return values


def _asymptote_probes(function: Any) -> List[List[Any]]:
    probes: List[List[Any]] = []
    for asymptote in getattr(function, "vertical_asymptotes", None) or []:
        try:
            a = float(asymptote)
        except (TypeError, ValueError):
            continue
        h = ASYMPTOTE_PROBE_STEP * max(1.0, abs(a))
        probes.append([a, _safe_eval(function.function, a - h), _safe_eval(function.function, a + h)])
    return probes


def _safe_eval(evaluate: Any, x: float) -> Optional[float]:
    try:
        value = float(evaluate(x))
    except Exception:
        return None
    return value


def _safe_eval_pair(evaluate: Any, t: float) -> List[Optional[float]]:
    try:
        x, y = evaluate(t)
        return [float(x), float(y)]
    except Exception:
        return [None, None]


def content_extent(canvas: "Canvas", samples: int = FIT_SAMPLES) -> Optional[List[float]]:
    """``[x_min, y_min, x_max, y_max]`` of every drawable, or None when there is nothing to fit.

    Points, segment, vector and polygon vertices, circles, arcs and ellipses by
    their extent, labels and bars; functions over their bounds (or the other
    content's x range, else -10..10) with Tukey-fenced samples, so an asymptote
    does not stretch the view; parametric curves by samples over their t range.
    """
    xs: List[float] = []
    ys: List[float] = []
    functions: List[Any] = []
    for drawable in canvas.get_drawables():
        name = str(drawable.get_class_name())
        if name in _FUNCTION_CLASSES:
            functions.append(drawable)
            continue
        for x, y in _drawable_extent_points(drawable, name, samples):
            if _finite(x) and _finite(y):
                xs.append(x)
                ys.append(y)
    for function in functions:
        x_range = _function_x_range(function, xs)
        if x_range is None:
            continue
        left, right = x_range
        values: List[float] = []
        for i in range(samples + 1):
            value = _safe_eval(function.function, left + (right - left) * i / samples)
            if value is not None and _finite(value):
                values.append(value)
        fenced = _fenced(values)
        xs.extend([left, right])
        if fenced:
            ys.extend(fenced)
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def fit_window(
    extent: List[float],
    width: float,
    height: float,
    padding: float = FIT_PADDING,
    min_half_span: float = FIT_MIN_HALF_SPAN,
) -> Tuple[float, float, float, str]:
    """``(center_x, center_y, range_val, range_axis)`` for ``Canvas.zoom`` showing ``extent`` padded.

    The binding axis is the one whose padded half-span needs more room at the
    canvas's aspect ratio, so the whole extent stays visible.
    """
    x_min, y_min, x_max, y_max = extent
    half_x = max((x_max - x_min) / 2.0 * (1.0 + 2.0 * padding), min_half_span)
    half_y = max((y_max - y_min) / 2.0 * (1.0 + 2.0 * padding), min_half_span)
    center_x, center_y = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    aspect = width / height if height > 0 else 1.0
    if half_x >= half_y * aspect:
        return center_x, center_y, half_x, "x"
    return center_x, center_y, half_y, "y"


def _drawable_extent_points(drawable: Any, name: str, samples: int) -> List[Tuple[float, float]]:
    """Points that bound one drawable (empty for types without an extent of their own)."""
    if name == "Point":
        return [(float(drawable.x), float(drawable.y))]
    if name in ("Segment", "Vector"):
        segment = drawable.segment if name == "Vector" else drawable
        return [(float(p.x), float(p.y)) for p in (segment.point1, segment.point2)]
    if name in _POLYGON_CLASSES:
        return [(x, y) for x, y in (_polygon_vertices(drawable) or [])]
    if name in ("Circle", "CircleArc"):
        if name == "Circle":
            cx, cy = float(drawable.center.x), float(drawable.center.y)
        else:
            cx, cy = float(drawable.center_x), float(drawable.center_y)
        r = abs(float(drawable.radius))
        return [(cx - r, cy - r), (cx + r, cy + r)]
    if name == "Ellipse":
        cx, cy = float(drawable.center.x), float(drawable.center.y)
        angle = math.radians(float(getattr(drawable, "rotation_angle", 0.0) or 0.0))
        rx, ry = abs(float(drawable.radius_x)), abs(float(drawable.radius_y))
        half_x = math.hypot(rx * math.cos(angle), ry * math.sin(angle))
        half_y = math.hypot(rx * math.sin(angle), ry * math.cos(angle))
        return [(cx - half_x, cy - half_y), (cx + half_x, cy + half_y)]
    if name == "Label":
        position = drawable.position
        return [(float(position.x), float(position.y))]
    if name == "Bar":
        return [
            (float(drawable.x_left), float(drawable.y_bottom)),
            (float(drawable.x_right), float(drawable.y_top)),
        ]
    if name == "ParametricFunction":
        t_min, t_max = float(drawable.t_min), float(drawable.t_max)
        pairs = [_safe_eval_pair(drawable.evaluate, t_min + (t_max - t_min) * i / samples) for i in range(samples + 1)]
        return [
            (float(x), float(y))
            for x, y in pairs
            if x is not None and y is not None and abs(x) < FIT_VALUE_LIMIT and abs(y) < FIT_VALUE_LIMIT
        ]
    return []


def _function_x_range(function: Any, other_xs: List[float]) -> Optional[Tuple[float, float]]:
    """The x range to sample a function over: its bounds, filled from the other content or -10..10."""
    left = getattr(function, "left_bound", None)
    right = getattr(function, "right_bound", None)
    default_left, default_right = (min(other_xs), max(other_xs)) if other_xs else FIT_DEFAULT_X_RANGE
    if default_left == default_right:
        default_left, default_right = default_left - FIT_DEFAULT_X_RANGE[1], default_right + FIT_DEFAULT_X_RANGE[1]
    lo = float(left) if left is not None and _finite(float(left)) else default_left
    hi = float(right) if right is not None and _finite(float(right)) else default_right
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        return None
    return lo, hi


def _fenced(values: List[float]) -> List[float]:
    """``values`` within Tukey's fences (quartiles -/+ FIT_FENCE x IQR) and FIT_VALUE_LIMIT."""
    kept = sorted(v for v in values if abs(v) < FIT_VALUE_LIMIT)
    if len(kept) < 4:
        return kept
    q1 = kept[len(kept) // 4]
    q3 = kept[(3 * len(kept)) // 4]
    fence = FIT_FENCE * (q3 - q1)
    return [v for v in kept if q1 - fence <= v <= q3 + fence]


def _finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def _name_hints(canvas: "Canvas") -> Dict[str, Any]:
    try:
        generator = canvas.drawable_manager.name_generator.point_generator
        return {str(key): dict(value) for key, value in generator.used_letters_from_names.items()}
    except Exception:
        return {}


def _plain(value: Any) -> Any:
    """Keep JSON scalars (NaN and infinity stay floats), stringify anything else."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
