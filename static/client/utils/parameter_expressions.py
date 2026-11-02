"""Expressions in parameters: point coordinates and circle radii such as "a" or "2*a + 1".

``read_definition(value, label)`` takes what a tool was given for one number: a number
(returned as is, no definition), a text without parameters ("3", "2*pi"; evaluated to a
number, no definition) or a text that reads parameters ("a^2"; kept as a definition and
evaluated at the current values). ``evaluate_definition(text)`` gives a definition's value
at the current values, or None where it is not a finite real number.

``undefined_names_hint(text, names, variables)`` is the refusal every tool gives for an
expression that reads names that are neither its variables nor parameters.

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional, Sequence, Tuple

import compiled_expression
import expression_functions as ef

# Variables a glued product most often hides (ax -> a*x, at -> a*t)
_GLUED_VARIABLES = ("x", "t", "θ")
# A name followed by "(": a call, or a product written without "*" (a(x+1))
_CALLED_NAME = re.compile(r"(\w+)\s*\(")
_QUOTES = ('"', "'")
# Names that are always variables, never parameters (drawables/parameter.py CURVE_VARIABLES)
_VARIABLE_NAMES = frozenset({"x", "y", "θ", "theta"})
# Where the model finds set_parameter when tools are searched
_CREATE_HINT = "set_parameter (search_tools 'slider' finds it if it is not loaded)"


class Definition:
    """A number given to a tool: its value now and, when it reads parameters, its text."""

    __slots__ = ("value", "expression")

    def __init__(self, value: float, expression: Optional[str]) -> None:
        self.value = value
        self.expression = expression


def undefined_names_hint(text: str, names: Sequence[str], variables: Sequence[str] = ()) -> str:
    """The refusal for an expression reading names that are neither variables nor parameters.

    A name glued from a name and a variable (ax, bx) gets "for a times x write a*x"; the
    input is never rewritten. Otherwise the hint says to create the parameter first.
    """
    quoted = ", ".join(f"'{name}'" for name in names)
    verb = "is" if len(names) == 1 else "are"
    message = f"'{text}' reads {quoted}, which {verb} not defined."
    for name in names:
        glued = _glued_product(name, variables)
        if glued is not None:
            head, variable = glued
            return (
                f"{message} For {head} times {variable} write {head}*{variable} (products need '*'); "
                f"create {head} with {_CREATE_HINT} if it is not a parameter yet."
            )
    first = names[0] if names else "a"
    return (
        f"{message} Create the parameter first with {_CREATE_HINT}: set_parameter(name='{first}', value=...), "
        f"or write a number in its place."
    )


def _glued_product(name: str, variables: Sequence[str]) -> Optional[Tuple[str, str]]:
    """(head, variable) when name is a name glued to a variable (ax -> a, x), else None."""
    for variable in tuple(variables) + _GLUED_VARIABLES:
        if name.endswith(variable) and len(name) > len(variable):
            head = name[: -len(variable)]
            if head.isidentifier() and head not in ef.RESERVED_NAMES:
                return head, variable
    return None


def called_name_hint(text: Any) -> Optional[str]:
    """The refusal for a parameter or an unknown name written like a function: a(x+1).

    None when every called name is a function expressions may call, or for an equation
    (f(x) = x^2), whose tool explains it.
    """
    if not isinstance(text, str) or "=" in text:
        return None
    parameters = compiled_expression.parameter_values()
    for match in _CALLED_NAME.finditer(text):
        name = match.group(1)
        before = text[match.start() - 1] if match.start() > 0 else ""
        if name[0].isdigit() or before == "." or before.isalnum() or before == "_":
            continue
        if name in ef.ALLOWED_FUNCTIONS or name in ef.CALL_ALIASES or name in ef.WORD_ALIASES:
            continue
        rest = text[match.end() - 1 :]
        argument = _parenthesised(rest)
        product = f"{name}*{argument}"
        if name in parameters:
            return f"'{text}': {name} is a parameter, not a function. For {name} times {argument} write {product}."
        if name in _VARIABLE_NAMES:
            return f"'{text}': {name} is a variable, not a function. For {name} times {argument} write {product}."
        return (
            f"'{text}': {name} is not a function expressions may call. For a product write {product} "
            f"(and create {name} with {_CREATE_HINT}), or use a known function such as sin(x)."
        )
    return None


def _parenthesised(text: str) -> str:
    """The balanced parenthesised group text starts with ('(x+1)' from '(x+1)^2')."""
    depth = 0
    for index, char in enumerate(text):
        depth += char == "("
        depth -= char == ")"
        if depth == 0:
            return text[: index + 1]
    return text


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unquoted(text: str) -> str:
    """The text without one pair of matching quotes around all of it ('"b"' is b).

    Models sometimes quote an expression inside a JSON string; the quotes are never part
    of an expression, so one surrounding pair is dropped and any other quote is refused.
    """
    if len(text) >= 2 and text[0] == text[-1] and text[0] in _QUOTES:
        return text[1:-1].strip()
    return text


def compile_definition(text: str) -> Tuple[str, Tuple[str, ...]]:
    """The stored form of a definition text and the parameters it reads.

    Raises:
        ValueError: The text does not parse, or reads names that are not parameters.
    """
    source = _unquoted(str(text).strip())
    if not source:
        raise ValueError("The expression is empty; give a number or an expression in parameters, such as a^2.")
    if any(quote in source for quote in _QUOTES):
        raise ValueError(f"Could not read {source}: write the expression without quotes, such as a^2 or 2*a.")
    try:
        compiled = compiled_expression.compile_expression(source, ())
    except Exception as exc:
        hint = called_name_hint(source)
        if hint is not None:
            raise ValueError(hint) from exc
        raise ValueError(
            f"Could not read '{source}': {exc}. Write every function with parentheses and products "
            f"explicitly: 2*a, sqrt(a), a^2."
        ) from exc
    undefined = compiled.undefined_names
    if undefined:
        raise ValueError(undefined_names_hint(source, undefined))
    from expression_validator import ExpressionValidator

    return ExpressionValidator.fix_math_expression(source), compiled.parameter_names


def evaluate_definition(text: str) -> Optional[float]:
    """The definition's value at the current parameter values, or None if not a finite real."""
    try:
        value = compiled_expression.compile_expression(text, ()).bind(parameters=True)()
    except Exception:
        return None
    return _finite(value)


def definition_parameters(text: Optional[str]) -> Tuple[str, ...]:
    """The current parameters a stored definition reads (none for None or an unreadable text)."""
    if not text:
        return ()
    try:
        return compiled_expression.compile_expression(text, ()).parameter_names
    except Exception:
        return ()


def read_definition(value: Any, label: str) -> Definition:
    """A tool's number or expression for one coordinate or radius.

    Raises:
        ValueError: Not a finite number, an unreadable text, a text reading names that are
            not parameters, or an expression that is not a finite real number now.
    """
    if not isinstance(value, str):
        number = _finite(value)
        if number is None:
            raise ValueError(f"{label} must be a finite number or an expression in parameters, got {value!r}.")
        return Definition(number, None)
    stored, parameters = compile_definition(value)
    number = evaluate_definition(stored)
    if number is None:
        raise ValueError(
            f"{label} = '{value}' is not a finite real number at the current parameter values; "
            f"change the parameters or the expression."
        )
    return Definition(number, stored if parameters else None)
