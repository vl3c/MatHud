from __future__ import annotations

from typing import Optional, TypeVar

RendererType = TypeVar("RendererType")


def _load_renderer(module_path: str, attr: str) -> Optional[RendererType]:
    try:
        module = __import__(module_path, fromlist=[attr])
        return getattr(module, attr)
    except Exception:
        return None

from rendering.interfaces import RendererProtocol

SvgRenderer = _load_renderer("rendering.svg_renderer", "SvgRenderer")
Canvas2DRenderer = _load_renderer("rendering.canvas2d_renderer", "Canvas2DRenderer")
WebGLRenderer = _load_renderer("rendering.webgl_renderer", "WebGLRenderer")


def create_renderer(preferred: Optional[str] = None) -> Optional[RendererProtocol]:
    """Instantiate a renderer based on preference and environment support."""

    preference_chain: list[str] = []

    if preferred:
        preference_chain.append(preferred)

    default_order = ["canvas2d", "svg", "webgl"]
    for fallback in default_order:
        if fallback not in preference_chain:
            preference_chain.append(fallback)

    for mode in preference_chain:
        if mode == "canvas2d" and Canvas2DRenderer is not None:
            try:
                renderer = Canvas2DRenderer()
                if renderer is None:
                    raise RuntimeError("Canvas2DRenderer returned None")
                return renderer
            except Exception:
                continue
        if mode == "svg" and SvgRenderer is not None:
            try:
                renderer = SvgRenderer()
                if renderer is None:
                    raise RuntimeError("SvgRenderer returned None")
                return renderer
            except Exception:
                continue
        if mode == "webgl" and WebGLRenderer is not None:
            try:
                renderer = WebGLRenderer()
                if renderer is None:
                    raise RuntimeError("WebGLRenderer returned None")
                return renderer
            except Exception:
                continue

    return None
