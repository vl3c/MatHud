"""Parameters (sliders): create, set, animate and delete them, and keep their users in step.

A parameter (drawables/parameter.py) is a named real number with a range and a step. The
expressions of functions, curves, point coordinates and circle radii read it through the
page's parameter namespace (compiled_expression.set_parameter_values), which this manager
alone writes, from the canvas's Parameters bucket.

When a value changes, the objects that read it are refreshed: defined points move, defined
circle radii change, and functions recompute their analysis (asymptotes, discontinuities).
Their render caches key on the values (rendering/signature_state.py), so the next draw
samples them again.

Undo: each tool call is one entry (set_parameter, delete_parameter, animate_parameter);
slider drags archive once when they start; animation frames never archive.

Pure Python apart from the animation scheduler, which ParameterAnimator (Brython) provides.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import compiled_expression
from drawables.parameter import ANIMATION_REPEATS, Parameter, check_parameter_name, resolve_range

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer

# During a live change (slider drag, animation frame) a function's analysis runs at most
# this often, over the visible x range; the full analysis runs when the change settles.
LIVE_ANALYSIS_INTERVAL_MS = 100.0
DEFAULT_ANIMATION_SECONDS = 4.0
MAX_ANIMATION_SECONDS = 600.0


def _now_ms() -> float:
    return time.time() * 1000.0


# The manager whose parameters the expression namespace holds: the page canvas's (main.py
# claims it). Other canvases (client tests) never write the shared namespace unless they
# claim it, so building a test canvas cannot clobber the page's parameters.
_namespace_owner: Optional["ParameterManager"] = None


class ParameterManager:
    """Owns the canvas's parameters and the namespace their users read."""

    def __init__(
        self,
        canvas: "Canvas",
        drawables: "DrawablesContainer",
        now_ms: Callable[[], float] = _now_ms,
    ) -> None:
        self.canvas = canvas
        self.drawables = drawables
        self._now_ms = now_ms
        # Set by the Brython layer (ParameterAnimator) and the slider panel
        self.animator: Optional[Any] = None
        self._change_listeners: List[Callable[[], None]] = []
        self._last_live_analysis: Dict[int, float] = {}
        self._stale_analysis: Dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> List[Parameter]:
        return list(self.drawables.get_by_class_name("Parameter"))  # type: ignore[arg-type]

    def get_parameter(self, name: str) -> Optional[Parameter]:
        normalised = str(name or "").strip()
        for parameter in self.parameters:
            if parameter.name == normalised:
                return parameter
        return None

    def _require(self, name: str) -> Parameter:
        parameter = self.get_parameter(name)
        if parameter is None:
            existing = ", ".join(p.name for p in self.parameters) or "none"
            raise ValueError(f"No parameter named '{name}' (parameters: {existing}). Create it with set_parameter.")
        return parameter

    def users_of(self, names: Sequence[str]) -> List[Any]:
        """The drawables whose expressions read any of the parameters, in canvas order."""
        wanted = set(names)
        users = []
        for drawable in self.drawables.get_all():
            names_of = getattr(drawable, "parameter_names", None)
            if not callable(names_of) or drawable.get_class_name() == "Parameter":
                continue
            try:
                if wanted & set(names_of()):
                    users.append(drawable)
            except Exception:
                continue
        return users

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def set_parameter(
        self,
        name: str,
        value: Optional[float] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        step: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create the parameter or change its value, range or step (one undo entry).

        Nulls keep the current value, range and step (a new parameter takes value 1 in
        -10..10, step 0.1). A value outside a min or max given with it is refused; outside
        the stored range it widens it. Setting an animating parameter stops its animation.

        Raises:
            ValueError: A bad name or number (nothing changes).
        """
        current = self.get_parameter(name)
        if current is None:
            name = check_parameter_name(name, self.drawables.names_in_use())
        new_value, low, high, new_step, notes = resolve_range(value, min, max, step, current)
        if current is not None and self.stop_animation(current.name):
            notes.append("its animation was stopped")
            new_value, low, high, new_step, _ = resolve_range(value, min, max, step, current)
        self._archive()
        created = current is None
        if current is None:
            current = Parameter(name, new_value, low, high, new_step)
            self.drawables.add(current)
        else:
            current.redefine(new_value, low, high, new_step)
        self.after_change([current.name], live=False)
        self._redraw()
        report = self._report(current)
        report["created"] = created
        notes.extend(self._undefined_notes(current.name))
        if notes:
            report["notes"] = notes
        return report

    def delete_parameter(self, name: str) -> Dict[str, Any]:
        """Delete an unused parameter (one undo entry).

        Raises:
            ValueError: No such parameter, or objects still read it (named in the message).
        """
        parameter = self._require(name)
        users = self.users_of([parameter.name])
        if users:
            listed = ", ".join(str(user.name) for user in users)
            raise ValueError(f"Parameter '{parameter.name}' is used by {listed}; delete or redefine them first.")
        self._archive()
        self.stop_animation(parameter.name)
        self.drawables.remove(parameter)
        self.sync_namespace()
        self._notify_change()
        self._redraw()
        return {"deleted": parameter.name}

    def animate_parameter(
        self,
        name: Optional[str],
        start: Optional[float] = None,
        end: Optional[float] = None,
        duration_seconds: Optional[float] = None,
        repeat: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Animate a parameter from start to end (one undo entry), or stop animations.

        repeat: "once" (default, stops at end), "loop" (starts again at start), "bounce"
        (back and forth) or "stop" (stops the animation of name, or of every parameter when
        name is null, keeping the current values). start and end default to the range ends;
        outside the range they widen it.

        Raises:
            ValueError: No such parameter, a bad number, or a duration outside (0, 600] s.
        """
        mode = str(repeat or "once").strip().lower()
        if mode == "stop":
            stopped = self.stop_animation(name)
            return {"stopped": stopped}
        if mode not in ANIMATION_REPEATS:
            raise ValueError(f"repeat must be one of once, loop, bounce or stop, got '{repeat}'.")
        if name is None or not str(name).strip():
            raise ValueError("Give the name of the parameter to animate.")
        parameter = self._require(name)
        low = parameter.min if start is None else self._number(start, "start")
        high = parameter.max if end is None else self._number(end, "end")
        if low == high:
            raise ValueError(f"start and end are both {low}; give two different values.")
        seconds = (
            DEFAULT_ANIMATION_SECONDS
            if duration_seconds is None
            else self._number(duration_seconds, "duration_seconds")
        )
        if not 0 < seconds <= MAX_ANIMATION_SECONDS:
            raise ValueError(
                f"duration_seconds must be more than 0 and at most {MAX_ANIMATION_SECONDS:g}, got {seconds:g}."
            )
        if self.animator is None:
            raise ValueError("Animation is not available on this canvas.")
        self._archive()
        notes: List[str] = []
        new_min, new_max = min(parameter.min, low, high), max(parameter.max, low, high)
        if (new_min, new_max) != (parameter.min, parameter.max):
            notes.append(f"range widened to [{new_min:g}, {new_max:g}]")
            parameter.redefine(parameter.value, new_min, new_max, parameter.step)
        parameter.animation = {"start": low, "end": high, "duration": seconds, "repeat": mode}
        parameter.set_value(low)
        self.after_change([parameter.name], live=False)
        self.animator.start(parameter.name, low, high, seconds, mode)
        self._redraw()
        report: Dict[str, Any] = {
            "name": parameter.name,
            "animating": mode,
            "from": low,
            "to": high,
            "duration_seconds": seconds,
        }
        users = [str(user.name) for user in self.users_of([parameter.name])]
        if users:
            report["used_by"] = users
        if notes:
            report["notes"] = notes
        return report

    # ------------------------------------------------------------------
    # Live changes (slider, animation) and animation bookkeeping
    # ------------------------------------------------------------------

    def begin_live_change(self, name: str) -> None:
        """A slider drag starts: stop the parameter's animation and archive once for the drag."""
        if self.get_parameter(name) is None:
            return
        self.stop_animation(name)
        self._archive()

    def apply_live_value(self, name: str, value: float) -> bool:
        """Set a value during a drag or an animation frame: no undo entry, light analysis.

        Returns False when the parameter no longer exists (the caller stops).
        """
        parameter = self.get_parameter(name)
        if parameter is None or not math.isfinite(float(value)):
            return False
        parameter.set_value(value)
        self.after_change([parameter.name], live=True)
        return True

    def settle(self, names: Optional[Sequence[str]] = None) -> None:
        """A live change ended: give the users of the parameters their full analysis."""
        wanted = list(names) if names is not None else [p.name for p in self.parameters]
        self.sync_namespace()
        for user in self.users_of(wanted):
            if id(user) in self._stale_analysis:
                self._reanalyse(user, None)
        self._redraw()

    def animation_finished(self, name: str) -> None:
        """The animator ended an animation (a "once" run reached its end, or it was stopped)."""
        parameter = self.get_parameter(name)
        if parameter is not None:
            parameter.animation = None
        self.settle([name])
        self._notify_change()

    def stop_animation(self, name: Optional[str] = None) -> List[str]:
        """Stop the animation of one parameter, or all; returns the names stopped."""
        names = [name] if name else [p.name for p in self.parameters if p.animation]
        stopped = []
        for parameter_name in names:
            parameter = self.get_parameter(parameter_name)
            running = parameter is not None and parameter.animation is not None
            if self.animator is not None:
                running = bool(self.animator.stop(parameter_name)) or running
            if parameter is not None:
                parameter.animation = None
            if running:
                stopped.append(parameter_name)
        if stopped:
            self.settle(stopped)
            self._notify_change()
        return stopped

    @property
    def animating(self) -> bool:
        """Whether any parameter is animating now."""
        if self.animator is not None and self.animator.running:
            return True
        return any(parameter.animation for parameter in self.parameters)

    def stop_all_animations(self) -> None:
        """Stop every animation without archiving (undo, redo, clear, workspace load)."""
        if self.animator is not None:
            self.animator.stop_all()
        for parameter in self.parameters:
            parameter.animation = None

    # ------------------------------------------------------------------
    # Namespace and users
    # ------------------------------------------------------------------

    @property
    def owns_namespace(self) -> bool:
        return _namespace_owner is self

    def claim_namespace(self) -> Optional["ParameterManager"]:
        """Make this manager the namespace owner and sync; returns the previous owner."""
        global _namespace_owner
        previous = _namespace_owner
        _namespace_owner = self
        self.sync_namespace()
        return previous

    def release_namespace(self, previous: Optional["ParameterManager"]) -> None:
        """Give the namespace back to the previous owner (or empty it) and resync."""
        global _namespace_owner
        if _namespace_owner is not self:
            return
        _namespace_owner = previous
        if previous is not None:
            previous.sync_namespace()
        else:
            compiled_expression.set_parameter_values({})

    def sync_namespace(self) -> None:
        """Make the expression namespace hold exactly the canvas's parameters (cheap when equal).

        Only the namespace owner writes it.
        """
        if self.owns_namespace:
            values = {parameter.name: parameter.value for parameter in self.parameters}
            compiled_expression.set_parameter_values(values)

    def sync_namespace_from(self, parameters: Sequence[Any]) -> None:
        """Before an undo or redo copies its state: hold that state's parameters already.

        The copies then never evaluate an expression against the old values.
        """
        if self.owns_namespace:
            values = {}
            for parameter in parameters:
                try:
                    values[str(parameter.name)] = float(parameter.value)
                except Exception:
                    continue
            compiled_expression.set_parameter_values(values)

    def on_state_restored(self) -> None:
        """Undo, redo or a workspace load replaced the drawables: stop animations, resync."""
        self.stop_all_animations()
        self.sync_namespace()
        self._last_live_analysis.clear()
        self._stale_analysis.clear()
        self._notify_change()

    def after_change(self, names: Sequence[str], live: bool) -> None:
        """Sync the namespace and refresh the objects that read the parameters."""
        self.sync_namespace()
        users = self.users_of(names)
        # Points first (circles centred on them, and functions, may read them next)
        moved = {id(user) for user in users if user.get_class_name() == "Point" and user.refresh_definition()}
        if moved:
            self.refresh_segments_on(moved)
        for user in users:
            if user.get_class_name() == "Circle":
                user.refresh_definition()
        for user in users:
            if hasattr(user, "reanalyze"):
                self._refresh_analysis(user, live)
        if not live:
            self._notify_change()

    def refresh_segments_on(self, point_ids: Set[int]) -> None:
        """Recompute the line formulas of segments (and vectors) whose endpoints moved."""
        segments = list(getattr(self.drawables, "Segments", []))
        segments += [vector.segment for vector in getattr(self.drawables, "Vectors", []) if hasattr(vector, "segment")]
        for segment in segments:
            if id(segment.point1) in point_ids or id(segment.point2) in point_ids:
                try:
                    segment.line_formula = segment._calculate_line_algebraic_formula()
                except Exception as exc:
                    print(f"Warning: could not refresh {getattr(segment, 'name', '?')}: {exc}")

    def _refresh_analysis(self, user: Any, live: bool) -> None:
        if not live:
            self._reanalyse(user, None)
            return
        now = self._now_ms()
        last = self._last_live_analysis.get(id(user))
        self._stale_analysis[id(user)] = user
        if last is not None and now - last < LIVE_ANALYSIS_INTERVAL_MS:
            return
        self._last_live_analysis[id(user)] = now
        self._reanalyse(user, self._view_bounds())

    def _reanalyse(self, user: Any, view_bounds: Optional[Tuple[float, float]]) -> None:
        try:
            user.reanalyze(view_bounds) if view_bounds is not None else user.reanalyze()
        except Exception as exc:
            print(f"Warning: could not re-analyse {getattr(user, 'name', '?')}: {exc}")
        if view_bounds is None:
            self._stale_analysis.pop(id(user), None)

    def _view_bounds(self) -> Optional[Tuple[float, float]]:
        mapper = getattr(self.canvas, "coordinate_mapper", None)
        try:
            left = float(mapper.get_visible_left_bound())  # type: ignore[union-attr]
            right = float(mapper.get_visible_right_bound())  # type: ignore[union-attr]
        except Exception:
            return None
        return (left, right) if left < right else None

    # ------------------------------------------------------------------
    # Change listeners (the slider panel)
    # ------------------------------------------------------------------

    def add_change_listener(self, listener: Callable[[], None]) -> None:
        """Call listener after parameters are created, deleted, redefined or restored."""
        self._change_listeners.append(listener)

    def _notify_change(self) -> None:
        for listener in list(self._change_listeners):
            try:
                listener()
            except Exception as exc:
                print(f"Warning: parameter change listener failed: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _report(self, parameter: Parameter) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "name": parameter.name,
            "value": parameter.value,
            "min": parameter.min,
            "max": parameter.max,
            "step": parameter.step,
        }
        users = [str(user.name) for user in self.users_of([parameter.name])]
        if users:
            report["used_by"] = users
        return report

    def _undefined_notes(self, name: str) -> List[str]:
        undefined = [str(user.name) for user in self.users_of([name]) if getattr(user, "undefined", False)]
        if not undefined:
            return []
        return [
            f"{', '.join(undefined)} undefined at the new value (not a finite real number); "
            f"shown at the last defined position"
        ]

    @staticmethod
    def _number(value: Any, label: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{label} must be a number, got {value!r}.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a number, got {value!r}.")
        if not math.isfinite(number):
            raise ValueError(f"{label} must be a finite number, got {value!r}.")
        return number

    def _archive(self) -> None:
        undo_redo = getattr(self.canvas, "undo_redo_manager", None)
        if undo_redo is not None:
            undo_redo.archive()

    def _redraw(self) -> None:
        if getattr(self.canvas, "draw_enabled", False):
            self.canvas.draw()
