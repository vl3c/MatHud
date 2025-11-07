from __future__ import annotations

from typing import Optional, TypeVar

RendererType = TypeVar("RendererType")


def _load_renderer(module_path: str, attr: str) -> Optional[RendererType]:
    try:
        module = __import__(module_path, fromlist=[attr])
        return getattr(module, attr)
    except Exception:
        return None

from browser import window

from rendering.interfaces import RendererProtocol

SvgRenderer = _load_renderer("rendering.svg_renderer", "SvgRenderer")
Canvas2DRenderer = _load_renderer("rendering.canvas2d_renderer", "Canvas2DRenderer")
WebGLRenderer = _load_renderer("rendering.webgl_renderer", "WebGLRenderer")


def create_renderer(preferred: Optional[str] = None) -> Optional[RendererProtocol]:
    """Instantiate a renderer based on preference and environment support."""

    preference_chain: list[str] = []
    strategy_preference: list[str] = []

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

    try:
        runtime_strategy = getattr(window, "MatHudRendererStrategy", None)
        if runtime_strategy:
            strategy_preference.append(runtime_strategy)
    except Exception:
        pass
    try:
        stored_strategy = window.localStorage.getItem("mathud.renderer.strategy")
        if stored_strategy:
            strategy_preference.append(stored_strategy)
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
                _apply_strategy(renderer, strategy_preference)
                return renderer
            except Exception:
                continue
        if mode == "webgl" and WebGLRenderer is not None:
            try:
                renderer = WebGLRenderer()
                _apply_strategy(renderer, strategy_preference)
                return renderer
            except Exception:
                continue
        if mode == "svg" and SvgRenderer is not None:
            try:
                renderer = SvgRenderer()
                _apply_strategy(renderer, strategy_preference)
                return renderer
            except Exception:
                continue

    return None


def _apply_strategy(renderer: RendererProtocol, strategies: list[str]) -> None:
    if not hasattr(renderer, "set_render_mode"):
        return
    for strategy in strategies:
        if isinstance(strategy, str) and strategy:
            try:
                renderer.set_render_mode(strategy)
                return
            except Exception:
                continue
