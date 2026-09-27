"""Model tools that report a change already in effect instead of repeating it.

An update to the values an object already has, a translation by (0, 0), a zoom to the view
already shown, and a coordinate-system or grid setting already active change nothing. The
wrappers here return a NoChangeResult for them (which ResultProcessor reports as is, adding
no undo entry) and call the real tool otherwise.

The update check is conservative: it answers "nothing changed" only when every requested
field is one it knows how to compare and each already has the requested value. Any other
field, a missing object or an incomplete coordinate pair goes to the real tool, which
applies or rejects it as before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from no_change_result import NoChangeResult

if TYPE_CHECKING:
    from canvas import Canvas

# Two coordinates closer than this are the same (the canvas stores floats).
COORDINATE_TOLERANCE = 1e-9

Finder = Callable[["Canvas", str, Dict[str, Any]], Any]


@dataclass(frozen=True)
class _Field:
    """Tool arguments that together set one property, how to compare them, and how to name them."""

    args: Tuple[str, ...]
    matches: Callable[[Any, Sequence[Any]], bool]
    describe: Callable[[Sequence[Any]], str]


@dataclass(frozen=True)
class _UpdateSpec:
    """What an update tool edits: the object's kind, its name argument, its lookup and its comparable fields.

    ``lookup_args`` only help find the object (``update_polygon``'s ``polygon_type``); they edit nothing.
    """

    kind: str
    name_arg: str
    find: Finder
    fields: Tuple[_Field, ...]
    lookup_args: Tuple[str, ...] = ()


def _format_number(value: Any) -> str:
    """``1.0 -> '1'``, ``2.5 -> '2.5'``, ``-2.0000000000000004 -> '-2'``."""
    return format(float(value), ".10g")


def _same_number(a: Any, b: Any) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=COORDINATE_TOLERANCE)
    except (TypeError, ValueError):
        return False


def _normalize_color(value: Any) -> str:
    return str(value).strip().lower()


def _own_parts(drawable: Any) -> List[Any]:
    """Objects whose update_color recolors only themselves."""
    return [drawable]


def _with_label(drawable: Any) -> List[Any]:
    """Points and segments recolor their attached label too."""
    label = getattr(drawable, "label", None)
    return [drawable] if label is None else [drawable, label]


def _vector_parts(vector: Any) -> List[Any]:
    """A vector recolors its segment, and the segment its label."""
    segment = getattr(vector, "segment", None)
    return [vector] + ([] if segment is None else _with_label(segment))


def _polygon_parts(polygon: Any) -> List[Any]:
    """A polygon recolors its edges, and each edge its label."""
    get_segments = getattr(polygon, "get_segments", None)
    if callable(get_segments):
        edges = list(get_segments())
    else:
        edges = [getattr(polygon, attr, None) for attr in ("segment1", "segment2", "segment3")]
    parts = [polygon]
    for edge in edges:
        if edge is not None:
            parts.extend(_with_label(edge))
    return parts


def _color_field(parts: Callable[[Any], List[Any]]) -> _Field:
    """``new_color``: a no-op when the object and every part its update_color recolors have that color."""

    def matches(drawable: Any, values: Sequence[Any]) -> bool:
        wanted = _normalize_color(values[0])
        return bool(wanted) and all(_normalize_color(getattr(part, "color", "")) == wanted for part in parts(drawable))

    return _Field(("new_color",), matches, lambda values: f"color {str(values[0]).strip()}")


def _position_matches(point: Any, values: Sequence[Any]) -> bool:
    return _same_number(getattr(point, "x", None), values[0]) and _same_number(getattr(point, "y", None), values[1])


def _center_matches(drawable: Any, values: Sequence[Any]) -> bool:
    return _position_matches(getattr(drawable, "center", None), values)


def _name_matches(drawable: Any, values: Sequence[Any]) -> bool:
    return str(getattr(drawable, "name", "")) == str(values[0])


COLOR_FIELD = _color_field(_own_parts)
POINT_NAME_FIELD = _Field(("new_name",), _name_matches, lambda values: f"name {values[0]}")
POINT_POSITION_FIELD = _Field(
    ("new_x", "new_y"),
    _position_matches,
    lambda values: f"position ({_format_number(values[0])}, {_format_number(values[1])})",
)
CENTER_FIELD = _Field(
    ("new_center_x", "new_center_y"),
    _center_matches,
    lambda values: f"center ({_format_number(values[0])}, {_format_number(values[1])})",
)

ARC_SWEEP_FIELD = _Field(
    ("use_major_arc",),
    lambda arc, values: bool(getattr(arc, "use_major_arc", False)) == bool(values[0]),
    lambda values: "the major arc selected" if values[0] else "the minor arc selected",
)


def _text_matches(label: Any, value: Any) -> bool:
    """The label already shows ``value`` once normalised as Label.set_text would store it."""
    if label is None:
        return False
    from drawables.label import Label

    try:
        return str(Label.validate_text(str(value))) == str(getattr(label, "text", ""))
    except ValueError:
        return False


LABEL_TEXT_FIELD = _Field(
    ("new_text",), lambda label, values: _text_matches(label, values[0]), lambda values: f"text '{values[0]}'"
)
SEGMENT_LABEL_TEXT_FIELD = _Field(
    ("new_label_text",),
    lambda segment, values: _text_matches(getattr(segment, "label", None), values[0]),
    lambda values: f"label text '{values[0]}'",
)
SEGMENT_LABEL_VISIBLE_FIELD = _Field(
    ("new_label_visible",),
    lambda segment, values: (
        getattr(segment, "label", None) is not None
        and bool(getattr(segment.label, "visible", False)) == bool(values[0])
    ),
    lambda values: "a visible label" if values[0] else "a hidden label",
)


def _by_name(lookup: Callable[["Canvas"], Callable[[str], Any]]) -> Finder:
    """A finder that calls ``lookup(canvas)(name)``."""
    return lambda canvas, name, _args: lookup(canvas)(name)


UPDATE_SPECS: Dict[str, _UpdateSpec] = {
    "update_point": _UpdateSpec(
        "Point",
        "point_name",
        _by_name(lambda canvas: canvas.get_point_by_name),
        (POINT_NAME_FIELD, POINT_POSITION_FIELD, _color_field(_with_label)),
    ),
    "update_segment": _UpdateSpec(
        "Segment",
        "name",
        _by_name(lambda canvas: canvas.get_segment_by_name),
        (_color_field(_with_label), SEGMENT_LABEL_TEXT_FIELD, SEGMENT_LABEL_VISIBLE_FIELD),
    ),
    "update_vector": _UpdateSpec(
        "Vector",
        "name",
        _by_name(lambda canvas: canvas.drawable_manager.vector_manager.get_vector_by_name),
        (_color_field(_vector_parts),),
    ),
    "update_polygon": _UpdateSpec(
        "Polygon",
        "polygon_name",
        lambda canvas, name, args: canvas.get_polygon_by_name(name, args.get("polygon_type")),
        (_color_field(_polygon_parts),),
        lookup_args=("polygon_type",),
    ),
    "update_circle": _UpdateSpec(
        "Circle", "name", _by_name(lambda canvas: canvas.get_circle_by_name), (COLOR_FIELD, CENTER_FIELD)
    ),
    "update_circle_arc": _UpdateSpec(
        "Circle arc",
        "name",
        _by_name(lambda canvas: canvas.drawable_manager.arc_manager.get_circle_arc_by_name),
        (COLOR_FIELD, ARC_SWEEP_FIELD),
    ),
    "update_ellipse": _UpdateSpec(
        "Ellipse", "name", _by_name(lambda canvas: canvas.get_ellipse_by_name), (COLOR_FIELD, CENTER_FIELD)
    ),
    "update_label": _UpdateSpec(
        "Label", "name", _by_name(lambda canvas: canvas.get_label_by_name), (COLOR_FIELD, LABEL_TEXT_FIELD)
    ),
    "update_function": _UpdateSpec("Function", "name", _by_name(lambda canvas: canvas.get_function), (COLOR_FIELD,)),
    "update_piecewise_function": _UpdateSpec(
        "Piecewise function",
        "name",
        _by_name(lambda canvas: canvas.drawable_manager.piecewise_function_manager.get_piecewise_function),
        (COLOR_FIELD,),
    ),
    "update_parametric_function": _UpdateSpec(
        "Parametric function", "name", _by_name(lambda canvas: canvas.get_parametric_function), (COLOR_FIELD,)
    ),
    "update_angle": _UpdateSpec(
        "Angle",
        "name",
        _by_name(lambda canvas: canvas.drawable_manager.angle_manager.get_angle_by_name),
        (COLOR_FIELD,),
    ),
}


def _join_phrases(phrases: List[str]) -> str:
    """``['a'] -> 'a'``, ``['a', 'b', 'c'] -> 'a, b and c'``."""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def unchanged_update_message(canvas: "Canvas", spec: _UpdateSpec, args: Dict[str, Any]) -> Optional[str]:
    """The no-op message when the update would change nothing, else None (run the real update)."""
    ignored = (spec.name_arg,) + spec.lookup_args
    requested = {key: value for key, value in args.items() if key not in ignored and value is not None}
    name = args.get(spec.name_arg)
    if not requested or not isinstance(name, str):
        return None
    try:
        drawable = spec.find(canvas, name, args)
    except Exception:
        return None
    if drawable is None:
        return None
    phrases: List[str] = []
    covered: set[str] = set()
    for field in spec.fields:
        present = [arg for arg in field.args if arg in requested]
        if not present:
            continue
        if len(present) != len(field.args):
            return None
        values = [requested[arg] for arg in field.args]
        if not field.matches(drawable, values):
            return None
        covered.update(field.args)
        phrases.append(field.describe(values))
    if set(requested) - covered:
        return None
    return f"{spec.kind} '{drawable.name}' already has {_join_phrases(phrases)}; nothing changed."


def update_tool(canvas: "Canvas", tool_name: str, update: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an update tool so that an update to the values already in effect is reported as a no-op."""
    spec = UPDATE_SPECS[tool_name]

    def checked_update(**kwargs: Any) -> Any:
        message = unchanged_update_message(canvas, spec, kwargs)
        if message is not None:
            return NoChangeResult(message)
        return update(**kwargs)

    return checked_update


def translate_tool(canvas: "Canvas") -> Callable[..., Any]:
    """Wrap canvas.translate_object so that a translation by (0, 0) of an existing object is a no-op."""

    def translate_object(name: str, x_offset: float, y_offset: float) -> Any:
        if _same_number(x_offset, 0) and _same_number(y_offset, 0):
            if any(getattr(drawable, "name", None) == name for drawable in canvas.drawable_manager.get_drawables()):
                return NoChangeResult(f"Translating '{name}' by (0, 0) moves nothing; nothing changed.")
        return canvas.translate_object(name, x_offset, y_offset)

    return translate_object


def zoom_tool(canvas: "Canvas") -> Callable[..., Any]:
    """Wrap canvas.zoom so that a zoom to the view already shown is reported as a no-op."""

    def zoom(center_x: float, center_y: float, range_val: float, range_axis: str) -> Any:
        bounds = canvas.get_zoom_bounds_if_shown(center_x, center_y, range_val, range_axis)
        if bounds is not None:
            left, right, top, bottom = (_format_number(value) for value in bounds)
            return NoChangeResult(
                f"The view already shows x from {left} to {right} and y from {bottom} to {top}; nothing changed."
            )
        return canvas.zoom(center_x, center_y, range_val, range_axis)

    return zoom


def set_coordinate_system_tool(canvas: "Canvas") -> Callable[..., Any]:
    """Wrap canvas.set_coordinate_system: the active mode is a no-op, an unknown mode an error."""

    def set_coordinate_system(mode: str) -> Any:
        if mode == canvas.get_coordinate_system():
            return NoChangeResult(f"The coordinate system is already {mode}; nothing changed.")
        if not canvas.set_coordinate_system(mode):
            return f"Error: unknown coordinate system '{mode}'; use 'cartesian' or 'polar'."
        return True

    return set_coordinate_system


def set_grid_visible_tool(canvas: "Canvas") -> Callable[..., Any]:
    """Wrap canvas.set_grid_visible so that the active grid's current visibility is a no-op."""

    def set_grid_visible(visible: bool) -> Any:
        if bool(visible) == canvas.is_grid_visible():
            state = "visible" if visible else "hidden"
            return NoChangeResult(f"The {canvas.get_coordinate_system()} grid is already {state}; nothing changed.")
        return canvas.set_grid_visible(visible)

    return set_grid_visible
