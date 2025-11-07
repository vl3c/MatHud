from __future__ import annotations

from typing import Optional

from browser import window

from rendering.interfaces import RendererProtocol

try:
    from rendering.svg_renderer import SvgRenderer
except Exception:
    SvgRenderer = None  # type: ignore

try:
    from rendering.canvas2d_renderer import Canvas2DRenderer
except Exception:
    Canvas2DRenderer = None  # type: ignore

try:
    from rendering.webgl_renderer import WebGLRenderer
except Exception:
    WebGLRenderer = None  # type: ignore


def create_renderer(preferred: Optional[str] = None) -> Optional[RendererProtocol]:
    """Instantiate a renderer based on preference and environment support."""

    preference_chain: list[str] = []

    if preferred:
        preference_chain.append(preferred)

    # Allow setting via window.CONFIG or localStorage for quick experimentation
    try:
        runtime_pref = getattr(window, "MatHudRenderer", None)
        if runtime_pref and runtime_pref not in preference_chain:
            preference_chain.append(runtime_pref)
    except Exception:
        pass

    try:
        stored_pref = window.localStorage.getItem("mathud.renderer")
        if stored_pref and stored_pref not in preference_chain:
            preference_chain.append(stored_pref)
    except Exception:
        pass

    # Only fall back to SVG if no explicit preference is supplied
    if not preference_chain:
        preference_chain.append("svg")

    for mode in preference_chain:
        if mode == "canvas2d" and Canvas2DRenderer is not None:
            try:
                renderer = Canvas2DRenderer()
                if renderer is None:
                    raise RuntimeError("Canvas2DRenderer returned None")
                return renderer
            except Exception:
                continue
        if mode == "webgl" and WebGLRenderer is not None:
            try:
                renderer = WebGLRenderer()
                return renderer
            except Exception:
                continue
        if mode == "svg" and SvgRenderer is not None:
            try:
                renderer = SvgRenderer()
                return renderer
            except Exception:
                continue

    return None

