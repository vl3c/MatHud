"""Renderer factory for creating the appropriate rendering backend.

This module provides a factory function that instantiates renderers based on
preference and environment support. It tries renderers in a preference chain
until one successfully initializes.

Key Features:
    - Preference-based renderer selection (canvas2d, svg)
    - Automatic fallback through available renderers
    - Lazy loading: a renderer module is imported only when it is attempted
    - Error handling for failed instantiation

Default Preference Order:
    1. canvas2d - HTML5 Canvas 2D API
    2. svg - SVG DOM elements
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, cast

from rendering.interfaces import RendererProtocol

# Renderer mode -> (module path, class name), in default preference order.
_RENDERER_MODULES: Dict[str, Tuple[str, str]] = {
    "canvas2d": ("rendering.canvas2d_renderer", "Canvas2DRenderer"),
    "svg": ("rendering.svg_renderer", "SvgRenderer"),
}


def _load_renderer(module_path: str, attr: str) -> Optional[Callable[[], Optional[RendererProtocol]]]:
    """Lazily load a renderer class from a module.

    Args:
        module_path: Dotted module path to import.
        attr: Attribute name to retrieve from the module.

    Returns:
        The renderer class, or None if loading failed.
    """
    try:
        module = __import__(module_path, fromlist=[attr])
        return cast(Optional[Callable[[], Optional[RendererProtocol]]], getattr(module, attr))
    except Exception:
        return None


def _build_preference_chain(preferred: Optional[str]) -> list[str]:
    """Build an ordered list of renderer modes to try.

    Args:
        preferred: Optional preferred renderer mode.

    Returns:
        List of modes starting with preferred, then defaults.
    """
    chain: list[str] = []
    if preferred:
        chain.append(preferred)
    for fallback in _RENDERER_MODULES:
        if fallback not in chain:
            chain.append(fallback)
    return chain


def _safe_instantiate(factory: Callable[[], Optional[RendererProtocol]], *, error_message: str) -> RendererProtocol:
    """Instantiate a renderer and raise if it returns None."""
    renderer = factory()
    if renderer is None:
        raise RuntimeError(error_message)
    return renderer


def _attempt_renderer(mode: str) -> Optional[RendererProtocol]:
    """Try to import and instantiate the renderer for the given mode."""
    spec = _RENDERER_MODULES.get(mode)
    if spec is None:
        return None
    module_path, class_name = spec
    renderer_cls = _load_renderer(module_path, class_name)
    if renderer_cls is None:
        return None
    try:
        return _safe_instantiate(renderer_cls, error_message=f"{class_name} returned None")
    except Exception:
        return None


def create_renderer(preferred: Optional[str] = None) -> Optional[RendererProtocol]:
    """Instantiate a renderer based on preference and environment support.

    Tries renderers in preference order until one successfully initializes.
    Falls back through the default chain if the preferred renderer fails.

    Args:
        preferred: Optional renderer mode ("canvas2d", "svg").

    Returns:
        An instantiated renderer, or None if all renderers failed.
    """
    preference_chain = _build_preference_chain(preferred)

    for mode in preference_chain:
        renderer = _attempt_renderer(mode)
        if renderer is not None:
            return renderer
    return None
