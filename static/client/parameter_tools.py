"""The AI tools that create objects reading parameters, as FunctionRegistry binds them.

- ``checked_plot_tool``: draw_function, draw_piecewise_function and
  draw_parametric_function refuse an expression that reads a name that is neither its
  variable nor a parameter, with a hint (create the parameter, or write a number). The
  drawables themselves accept any name, so undo copies and older workspaces still load.
- ``create_point_tool`` / ``update_point_tool`` / ``create_circle_tool``: point
  coordinates and circle radii may be expressions in parameters ("a", "a^2"); the
  object then follows them.
- ``evaluate_expression_tool``: the current parameter values are variables of the
  expression unless the call gives them.

No browser imports, so server-side pytest can import it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import compiled_expression
from no_change_result import NoChangeResult
from utils.parameter_expressions import Definition, called_name_hint, read_definition, undefined_names_hint

if TYPE_CHECKING:
    from canvas import Canvas


def check_expression_names(
    text: Any, variables: Sequence[str], functions: str = compiled_expression.FUNCTIONS_ALL
) -> None:
    """Refuse an expression that reads names other than its variables and the parameters.

    An expression that does not compile is left to the tool, whose own message explains it,
    unless it calls a parameter or an unknown name like a function (a(x+1)).

    Raises:
        ValueError: With undefined_names_hint's or called_name_hint's message.
    """
    if not isinstance(text, str) or not text.strip():
        return
    try:
        compiled = compiled_expression.compile_expression(text, tuple(variables), functions)
    except Exception:
        hint = called_name_hint(text)
        if hint is not None:
            raise ValueError(hint)
        return
    undefined = compiled.undefined_names
    if undefined:
        raise ValueError(undefined_names_hint(text, undefined, variables))


def _piece_expressions(pieces: Any) -> Iterable[Any]:
    for piece in pieces if isinstance(pieces, list) else []:
        if isinstance(piece, dict):
            yield piece.get("expression")


def checked_plot_tool(tool_name: str, tool: Callable[..., Any]) -> Callable[..., Any]:
    """The plot tool, refusing expressions that read undefined names before it runs."""

    def run(*args: Any, **kwargs: Any) -> Any:
        if tool_name == "draw_function":
            check_expression_names(kwargs.get("function_string", args[0] if args else None), ("x",))
        elif tool_name == "draw_piecewise_function":
            for expression in _piece_expressions(kwargs.get("pieces", args[0] if args else None)):
                check_expression_names(expression, ("x",))
        elif tool_name == "draw_parametric_function":
            for key, index in (("x_expression", 0), ("y_expression", 1)):
                value = kwargs.get(key, args[index] if len(args) > index else None)
                check_expression_names(value, ("t",), compiled_expression.FUNCTIONS_CURVE)
        return tool(*args, **kwargs)

    return run


def _definitions(values: Dict[str, Any]) -> Dict[str, Optional[Definition]]:
    """read_definition for each given (non-None) value."""
    return {label: (read_definition(value, label) if value is not None else None) for label, value in values.items()}


def _plain(given: Any, definition: Optional[Definition]) -> Any:
    """What the plain tool gets: the number as given (3 stays 3), a text's value ("3" -> 3.0)."""
    if definition is None:
        return None
    return definition.value if isinstance(given, str) else given


def _described(definition: Optional[Definition]) -> Optional[str]:
    return definition.expression if definition is not None else None


def create_point_tool(canvas: "Canvas", create_point: Callable[..., Any]) -> Callable[..., Any]:
    """create_point taking numbers or expressions in parameters for x and y.

    With an expression, a new point is made even where another point is now (the defined
    point only passes through that spot at the current values), and it follows the
    parameters. Plain numbers (or texts without parameters, "3", "2*pi") behave as before.
    """

    def run(x: Any, y: Any, name: str = "", color: Optional[str] = None, extra_graphics: bool = True) -> Any:
        given = _definitions({"x": x, "y": y})
        x_definition, y_definition = given["x"], given["y"]
        assert x_definition is not None and y_definition is not None
        if x_definition.expression is None and y_definition.expression is None:
            return create_point(_plain(x, x_definition), _plain(y, y_definition), name, color, extra_graphics)
        point = canvas.drawable_manager.point_manager.create_point(
            x_definition.value, y_definition.value, name or "", color=color, extra_graphics=False, distinct=True
        )
        point.set_definition(x_definition.expression, y_definition.expression)
        if getattr(canvas, "draw_enabled", False):
            canvas.draw()
        return point

    return run


def update_point_tool(canvas: "Canvas", update_point: Callable[..., Any]) -> Callable[..., Any]:
    """update_point taking numbers or expressions in parameters for new_x and new_y.

    Numbers move a free point as before (the solitary-point rule applies). Numbers for a
    point defined by parameters make it free at them, even where it stands and even when
    other objects use it, so update_point always frees a point translate_object refuses.
    An expression defines that coordinate on any point, referenced or not (segments on it
    follow); a coordinate not given keeps its number or its expression.
    """

    def run(
        point_name: str,
        new_name: Optional[str] = None,
        new_x: Any = None,
        new_y: Any = None,
        new_color: Optional[str] = None,
    ) -> Any:
        given = _definitions({"new_x": new_x, "new_y": new_y})
        point = canvas.get_point_by_name(point_name)
        defines = any(definition is not None and definition.expression for definition in given.values())
        if not defines and point is not None and point.has_definition and None not in given.values():
            return _free_defined_point(point, given["new_x"], given["new_y"], new_name, new_color)
        if not defines:
            return update_point(
                point_name=point_name,
                new_name=new_name,
                new_x=_plain(new_x, given["new_x"]),
                new_y=_plain(new_y, given["new_y"]),
                new_color=new_color,
            )
        if point is None:
            raise ValueError(f"Point '{point_name}' was not found.")
        x_definition, y_definition = given["new_x"], given["new_y"]
        x_expression = _described(x_definition) if x_definition is not None else point.x_expression
        y_expression = _described(y_definition) if y_definition is not None else point.y_expression
        _rename_or_archive(point_name, new_name, new_color)
        x_now = x_definition.value if x_definition is not None else point.x
        y_now = y_definition.value if y_definition is not None else point.y
        point.update_position(x_now, y_now)
        point.set_definition(x_expression, y_expression)
        canvas.drawable_manager.parameter_manager.refresh_segments_on({id(point)})
        if getattr(canvas, "draw_enabled", False):
            canvas.draw()
        return True

    def _rename_or_archive(point_name: str, new_name: Any, new_color: Any) -> None:
        """Apply a new name or color through update_point (which archives), else archive here."""
        if new_name is None and new_color is None:
            canvas.undo_redo_manager.archive()
            return
        result = update_point(point_name=point_name, new_name=new_name, new_x=None, new_y=None, new_color=new_color)
        if result is not True:  # the name and color were already those: nothing archived
            canvas.undo_redo_manager.archive()

    def _free_defined_point(point: Any, x_definition: Any, y_definition: Any, new_name: Any, new_color: Any) -> Any:
        """Numbers for a point defined by parameters: it becomes free at them, even where it
        stands and even when other objects use it (they follow, as with a parameter change)."""
        _rename_or_archive(point.name, new_name, new_color)
        point.clear_definition()
        point.update_position(x_definition.value, y_definition.value)
        canvas.drawable_manager.parameter_manager.refresh_segments_on({id(point)})
        if getattr(canvas, "draw_enabled", False):
            canvas.draw()
        return True

    return run


def create_circle_tool(canvas: "Canvas", create_circle: Callable[..., Any]) -> Callable[..., Any]:
    """create_circle taking a number or an expression in parameters for the radius.

    A radius expression defines the radius (its absolute value), the circle is named after
    it, such as A(b), and it never reuses an existing circle. The radius must not be 0 when
    the circle is created. A centre follows parameters when it is a point defined by them:
    create that point first, then the circle at its coordinates.
    """

    def run(
        center_x: float,
        center_y: float,
        radius: Any,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> Any:
        r = read_definition(radius, "radius")
        if r.expression is None:
            return create_circle(center_x, center_y, _plain(radius, r), name, color, extra_graphics)
        if r.value == 0:
            raise ValueError(f"radius = '{radius}' is 0 at the current parameter values; a circle needs a radius.")
        circle = canvas.drawable_manager.circle_manager.create_circle(
            center_x,
            center_y,
            abs(r.value),
            name or "",
            color=color,
            extra_graphics=False,
            reuse_existing=False,
        )
        circle.set_radius_definition(r.expression)
        if not getattr(circle, "has_custom_name", False):
            circle.regenerate_name(canvas.drawable_manager.drawables.names_in_use(exclude=circle))
        if getattr(canvas, "draw_enabled", False):
            canvas.draw()
        return circle

    return run


def evaluate_expression_tool(evaluate: Callable[..., Any]) -> Callable[..., Any]:
    """evaluate_expression with the current parameter values as default variables."""

    def run(expression: str, variables: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        values: Dict[str, Any] = dict(compiled_expression.parameter_values())
        if values and isinstance(expression, str):
            try:
                read = set(compiled_expression.compile_expression(expression, ()).free_names)
            except Exception:
                read = set(values)
            values = {name: value for name, value in values.items() if name in read}
        if isinstance(variables, dict):
            values.update(variables)
        return evaluate(expression, values or variables, **kwargs)

    return run


def parameter_report_names(objects: Iterable[Any]) -> List[str]:
    """Names of objects, for tool results."""
    return [str(getattr(obj, "name", "")) for obj in objects]


__all__: Tuple[str, ...] = (
    "check_expression_names",
    "checked_plot_tool",
    "create_point_tool",
    "update_point_tool",
    "create_circle_tool",
    "evaluate_expression_tool",
)
