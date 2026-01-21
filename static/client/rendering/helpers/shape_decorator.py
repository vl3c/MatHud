"""Shape lifecycle decorator for primitive rendering.

This module provides the _manages_shape decorator that wraps render functions
with begin_shape/end_shape calls for proper batching and grouping.

Key Features:
    - Automatic shape lifecycle management
    - Safe handling when primitives lack lifecycle methods
    - Exception-safe with finally cleanup
"""

from __future__ import annotations


def _manages_shape(render_fn):
    """Decorator that wraps render logic with begin_shape/end_shape lifecycle calls.

    Ensures that render functions properly notify the primitives interface
    about shape boundaries, enabling batching and grouping optimizations.

    Args:
        render_fn: The render function to wrap. Must take primitives as first arg.

    Returns:
        Wrapped function that calls begin_shape before and end_shape after.
    """
    def wrapper(primitives, *args, **kwargs):
        begin_shape = getattr(primitives, "begin_shape", None)
        end_shape = getattr(primitives, "end_shape", None)
        managing = callable(begin_shape) and callable(end_shape)
        if managing:
            begin_shape()
        try:
            return render_fn(primitives, *args, **kwargs)
        finally:
            if managing:
                end_shape()
    return wrapper

