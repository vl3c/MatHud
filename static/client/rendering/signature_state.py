"""Render state of drawables that read parameters.

``signature_state``: the state a render cache is keyed on, ``get_state()`` plus the
parameter values read. ``is_undefined``: whether a drawable is not drawn because a
parameter value leaves it undefined.

A function such as ``a*sin(x)`` keeps the same ``get_state()`` when the parameter ``a``
changes, yet its samples change. Render caches (plan signatures, area dependencies, the
function renderable's model signature) key on ``signature_state(drawable)`` instead, which
adds the current values of the parameters the drawable reads (``parameter_names()``).
"""

from __future__ import annotations

from typing import Any, Optional

import compiled_expression


def signature_state(drawable: Any) -> Optional[Any]:
    """``drawable.get_state()`` with ``"_parameters"`` added when it reads parameters; None on failure."""
    get_state = getattr(drawable, "get_state", None)
    if not callable(get_state):
        return None
    try:
        state = get_state()
    except Exception:
        return None
    if not compiled_expression.has_parameters() or not isinstance(state, dict):
        return state
    names_of = getattr(drawable, "parameter_names", None)
    if not callable(names_of):
        return state
    try:
        names = names_of()
    except Exception:
        return state
    if not names:
        return state
    keyed = dict(state)
    keyed["_parameters"] = tuple((name, compiled_expression.parameter_value(name)) for name in names)
    return keyed


def is_undefined(drawable: Any) -> bool:
    """Whether the renderers skip drawable: it, or a point it is built on, is undefined.

    A point defined by parameters (``sqrt(a)`` with a < 0) or a circle whose radius is not
    a finite real number is marked ``undefined``; the point, its label, segments and vectors
    ending on it and a circle centred on it are then not drawn. Only parameters make an
    object undefined, so with none (most frames) nothing is looked up.
    """
    if not compiled_expression.has_parameters():
        return False
    if getattr(drawable, "undefined", False) is True:
        return True
    for attribute in ("point1", "point2", "center"):
        if getattr(getattr(drawable, attribute, None), "undefined", False) is True:
            return True
    segment = getattr(drawable, "segment", None)
    return segment is not None and segment is not drawable and is_undefined(segment)
