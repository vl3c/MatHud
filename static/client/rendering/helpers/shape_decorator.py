from __future__ import annotations


def _manages_shape(render_fn):
    """Decorator that wraps render logic with begin_shape/end_shape lifecycle calls."""
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

