from __future__ import annotations

from typing import Callable, Optional, TypeVar

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


def _build_preference_chain(preferred: Optional[str]) -> list[str]:
    chain: list[str] = []
    if preferred:
        chain.append(preferred)
    default_order = ["canvas2d", "svg", "webgl"]
    for fallback in default_order:
        if fallback not in chain:
            chain.append(fallback)
    return chain


def _safe_instantiate(
    factory: Callable[[], Optional[RendererProtocol]], *, error_message: str
) -> RendererProtocol:
    renderer = factory()
    if renderer is None:
        raise RuntimeError(error_message)
    return renderer


def _attempt_renderer(mode: str) -> Optional[RendererProtocol]:
    if mode == "canvas2d" and Canvas2DRenderer is not None:
        try:
            return _safe_instantiate(Canvas2DRenderer, error_message="Canvas2DRenderer returned None")
        except Exception:
            return None
    if mode == "svg" and SvgRenderer is not None:
        try:
            return _safe_instantiate(SvgRenderer, error_message="SvgRenderer returned None")
        except Exception:
            return None
    if mode == "webgl" and WebGLRenderer is not None:
        try:
            return _safe_instantiate(WebGLRenderer, error_message="WebGLRenderer returned None")
        except Exception:
            return None
    return None


def create_renderer(preferred: Optional[str] = None) -> Optional[RendererProtocol]:
    """Instantiate a renderer based on preference and environment support."""

    preference_chain = _build_preference_chain(preferred)

    for mode in preference_chain:
        renderer = _attempt_renderer(mode)
        if renderer is not None:
            return renderer
    return None
