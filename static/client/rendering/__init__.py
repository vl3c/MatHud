"""Rendering package (local).

This file ensures Brython treats `rendering` as a local package so that
imports like `from rendering.svg_renderer import SvgRenderer` resolve to
`static/client/rendering/svg_renderer.py` instead of attempting to fetch
`rendering.py` from the CDN.
"""


