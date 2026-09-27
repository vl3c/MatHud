"""Tool results for create calls that name the objects they produced.

A create tool returns the drawable it made (or a dict of drawables for a composite
construction). Instead of the bare success message, the model is told the class and
the real name of each object, whether the object is new or an existing one that was
reused, and when a requested name was not used (point names, for example, are a single
capital letter with optional primes, and a name already in use is never reused).
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Set

# Tool name prefixes whose returned drawable is the object the call produced.
CREATING_PREFIXES = ("create_", "construct_", "draw_", "generate_", "plot_", "fit_")


class CreationReport:
    """Builds the result message of a create call from the drawables it returned."""

    @staticmethod
    def applies_to(function_name: str) -> bool:
        """True for tools that create objects, whose returned drawables are worth naming."""
        return function_name.startswith(CREATING_PREFIXES)

    @staticmethod
    def snapshot_ids(canvas: Any) -> Optional[FrozenSet[int]]:
        """Ids of every drawable on the canvas, or None when the canvas cannot list them."""
        try:
            return frozenset(id(drawable) for drawable in canvas.drawable_manager.drawables.get_all())
        except Exception:
            return None

    @staticmethod
    def describe(
        function_name: str,
        args: Dict[str, Any],
        result: Any,
        existing_ids: Optional[FrozenSet[int]] = None,
    ) -> Optional[str]:
        """Message naming what a create call produced, or None when the result holds no drawable.

        Args:
            function_name: The tool that was called.
            args: The call's arguments; the requested name is ``name`` (``angle_name`` for create_angle).
            result: The tool's return value: a drawable, or a dict of drawables.
            existing_ids: Ids of the drawables that existed before the call, used to tell a
                reused object from a new one. None treats every returned drawable as new.
        """
        requested = CreationReport._requested_name(function_name, args)
        if CreationReport._is_drawable(result):
            return CreationReport._describe_single(result, requested, existing_ids)
        if isinstance(result, dict) and result:
            return CreationReport._describe_composite(result, requested, existing_ids)
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
    def _existed(drawable: Any, existing_ids: Optional[FrozenSet[int]]) -> bool:
        return existing_ids is not None and id(drawable) in existing_ids

    @staticmethod
    def _describe_single(drawable: Any, requested: str, existing_ids: Optional[FrozenSet[int]]) -> str:
        class_name = CreationReport._class_name(drawable)
        name_differs = bool(requested) and requested != drawable.name
        if CreationReport._existed(drawable, existing_ids):
            message = f"Used the existing {class_name} '{drawable.name}'; no new {class_name.lower()} was created."
            if name_differs:
                message += f" The requested name '{requested}' was not applied."
            return message
        if name_differs:
            return f"Created {class_name} '{drawable.name}' instead of the requested name '{requested}'."
        return f"Created {class_name} '{drawable.name}'."

    @staticmethod
    def _describe_composite(
        result: Dict[str, Any], requested: str, existing_ids: Optional[FrozenSet[int]]
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
            verb = "existing" if CreationReport._existed(value, existing_ids) else "created"
            parts.append(f"{key}: {verb} {CreationReport._class_name(value)} '{value.name}'")
            names.add(value.name)
        if not parts:
            return None
        message = "; ".join(parts) + "."
        if requested and requested not in names:
            message += f" The requested name '{requested}' was not used."
        return message
