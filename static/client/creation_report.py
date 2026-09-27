"""Tool results for create calls that name the objects they produced.

A create tool returns the drawable it made (or a dict of drawables for a composite
construction). Instead of the bare success message, the model is told the class and
the real name of each object, whether the object is new, an existing one that was
reused unchanged, or an existing one the call redefined (``draw_function`` on a name in
use), and when a requested name was not used (point names, for example, are a single
capital letter with optional primes, and a name already in use is never reused).
"""

from __future__ import annotations

import json
from typing import Any, Dict, FrozenSet, List, Optional, Set

# Tool name prefixes whose returned drawable is the object the call produced.
CREATING_PREFIXES = ("create_", "construct_", "draw_", "generate_", "plot_", "fit_")


class CreationSnapshot:
    """What existed before a create call: every drawable's id, and the state of drawables named as requested.

    The states let the report tell a redefined object (``draw_function`` on an existing
    name updates that function in place) from one that was reused unchanged.
    """

    def __init__(self, ids: FrozenSet[int], named_states: Dict[int, str]) -> None:
        self.ids: FrozenSet[int] = ids
        self.named_states: Dict[int, str] = named_states


class CreationOutcome:
    """The result message of a create call; ``reused_unchanged`` when it only returned an existing object."""

    def __init__(self, message: str, reused_unchanged: bool = False) -> None:
        self.message: str = message
        self.reused_unchanged: bool = reused_unchanged


class CreationReport:
    """Builds the result message of a create call from the drawables it returned."""

    @staticmethod
    def applies_to(function_name: str) -> bool:
        """True for tools that create objects, whose returned drawables are worth naming."""
        return function_name.startswith(CREATING_PREFIXES)

    @staticmethod
    def take_snapshot(canvas: Any, function_name: str, args: Dict[str, Any]) -> Optional[CreationSnapshot]:
        """Record what exists before the call, or None when the canvas cannot list its drawables."""
        try:
            drawables = list(canvas.drawable_manager.drawables.get_all())
        except Exception:
            return None
        requested = CreationReport._requested_name(function_name, args)
        named_states: Dict[int, str] = {}
        if requested:
            for drawable in drawables:
                if getattr(drawable, "name", None) == requested:
                    state = CreationReport._state_text(drawable)
                    if state is not None:
                        named_states[id(drawable)] = state
        return CreationSnapshot(frozenset(id(drawable) for drawable in drawables), named_states)

    @staticmethod
    def describe(
        function_name: str,
        args: Dict[str, Any],
        result: Any,
        snapshot: Optional[CreationSnapshot] = None,
    ) -> Optional[CreationOutcome]:
        """What a create call produced, or None when the result holds no drawable.

        Args:
            function_name: The tool that was called.
            args: The call's arguments; the requested name is ``name`` (``angle_name`` for create_angle).
            result: The tool's return value: a drawable, or a dict of drawables.
            snapshot: What existed before the call, used to tell a reused or redefined object
                from a new one. None treats every returned drawable as new.
        """
        requested = CreationReport._requested_name(function_name, args)
        if CreationReport._is_drawable(result):
            return CreationReport._describe_single(result, requested, args, snapshot)
        if isinstance(result, dict) and result:
            message = CreationReport._describe_composite(result, requested, snapshot)
            return CreationOutcome(message) if message is not None else None
        return None

    @staticmethod
    def _requested_name(function_name: str, args: Dict[str, Any]) -> str:
        key = "angle_name" if function_name == "create_angle" else "name"
        value = args.get(key) if isinstance(args, dict) else None
        return value if isinstance(value, str) else ""

    @staticmethod
    def _is_drawable(value: Any) -> bool:
        return isinstance(getattr(value, "name", None), str) and callable(getattr(value, "get_class_name", None))

    @staticmethod
    def _class_name(drawable: Any) -> str:
        try:
            return str(drawable.get_class_name())
        except Exception:
            return "object"

    @staticmethod
    def _state_text(drawable: Any) -> Optional[str]:
        try:
            return json.dumps(drawable.get_state(), sort_keys=True, default=str)
        except Exception:
            return None

    @staticmethod
    def _existed(drawable: Any, snapshot: Optional[CreationSnapshot]) -> bool:
        return snapshot is not None and id(drawable) in snapshot.ids

    @staticmethod
    def _was_redefined(drawable: Any, snapshot: Optional[CreationSnapshot]) -> bool:
        """True when an existing object named as requested has a different state after the call."""
        if snapshot is None or id(drawable) not in snapshot.named_states:
            return False
        return CreationReport._state_text(drawable) != snapshot.named_states[id(drawable)]

    @staticmethod
    def _center_name(drawable: Any) -> str:
        """The name of a circle's or ellipse's centre point, whose name the requested name sets; else ""."""
        center = getattr(drawable, "center", None)
        name = getattr(center, "name", None)
        return name if isinstance(name, str) and CreationReport._is_drawable(center) else ""

    @staticmethod
    def _names_of(drawable: Any) -> Set[str]:
        """Names a requested name may have gone to: the object's own, and its centre's."""
        names = {drawable.name}
        center_name = CreationReport._center_name(drawable)
        if center_name:
            names.add(center_name)
        return names

    @staticmethod
    def _label(drawable: Any) -> str:
        """``Circle 'O(2)' centred on 'O'`` or ``Point 'B'``."""
        label = f"{CreationReport._class_name(drawable)} '{drawable.name}'"
        center_name = CreationReport._center_name(drawable)
        return label + (f" centred on '{center_name}'" if center_name else "")

    @staticmethod
    def _describe_single(
        drawable: Any, requested: str, args: Dict[str, Any], snapshot: Optional[CreationSnapshot]
    ) -> CreationOutcome:
        class_name = CreationReport._class_name(drawable)
        label = CreationReport._label(drawable)
        name_differs = bool(requested) and requested not in CreationReport._names_of(drawable)
        if CreationReport._was_redefined(drawable, snapshot):
            expression = getattr(drawable, "function_string", None)
            to_text = f" to {expression}" if isinstance(expression, str) and expression else ""
            return CreationOutcome(f"Updated the existing {label}{to_text}; no new {class_name.lower()} was created.")
        if CreationReport._existed(drawable, snapshot):
            message = f"Used the existing {label}; no new {class_name.lower()} was created."
            if name_differs:
                message += f" The requested name '{requested}' was not applied."
            requested_color = args.get("color") if isinstance(args, dict) else None
            if isinstance(requested_color, str) and requested_color and requested_color != drawable.color:
                message += " The requested color was not applied."
            return CreationOutcome(message, reused_unchanged=True)
        if name_differs:
            return CreationOutcome(f"Created {label} instead of the requested name '{requested}'.")
        return CreationOutcome(f"Created {label}.")

    @staticmethod
    def _describe_composite(
        result: Dict[str, Any], requested: str, snapshot: Optional[CreationSnapshot]
    ) -> Optional[str]:
        """``"foot: created Point 'E'; segment: created Segment 'CE'."`` for a dict of drawables.

        Returns None unless every value is a drawable (or None), so other dicts keep the
        default handling.
        """
        parts: List[str] = []
        names: Set[str] = set()
        for key, value in result.items():
            if value is None:
                continue
            if not CreationReport._is_drawable(value):
                return None
            verb = "existing" if CreationReport._existed(value, snapshot) else "created"
            parts.append(f"{key}: {verb} {CreationReport._label(value)}")
            names |= CreationReport._names_of(value)
        if not parts:
            return None
        message = "; ".join(parts) + "."
        if requested and requested not in names:
            message += f" The requested name '{requested}' was not used."
        return message
