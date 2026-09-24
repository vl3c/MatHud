"""
MatHud Canvas Snapshot (vision)

Captures what the user sees on the math canvas as a PNG data URL, in the
browser, for vision-enabled AI requests. Replaces the former server-side
Selenium/headless-Firefox capture.

The canvas is layered inside ``#math-container``: the ``#math-svg`` surface is
the base layer (it holds the whole scene with the SVG renderer) and the
Canvas2D ``<canvas>`` sits above it (it holds the whole scene, labels
included, with the Canvas2D renderer). The snapshot composites both in that
order onto an offscreen canvas at CSS-pixel size, so HiDPI bitmaps are scaled
down, and caps the longest side at ``VISION_SNAPSHOT_MAX_SIDE``.

A non-empty SVG layer has to be decoded as an image first, which is
asynchronous, so ``capture`` reports its result through a callback. It is
called exactly once, synchronously when there is no SVG content to draw.
If the SVG layer cannot be drawn, the snapshot falls back to the Canvas2D
layer alone; if nothing can be captured, the callback receives None.

Dependencies:
    - browser: DOM access (document, html, window) for canvases, SVG serialization and images
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from browser import document, html, window

# Longest side of the snapshot in pixels; larger canvases are scaled down.
VISION_SNAPSHOT_MAX_SIDE: int = 1280
# Filled behind all layers so the PNG has no transparent areas.
VISION_SNAPSHOT_BACKGROUND: str = "#ffffff"
# Mirrors the server's per-image cap (static/config.py MAX_IMAGE_BASE64_BYTES).
VISION_SNAPSHOT_MAX_DATA_URL_CHARS: int = 20 * 1024 * 1024
# How long to wait for the SVG layer to decode before sending the Canvas2D layer alone.
SVG_DECODE_TIMEOUT_MS: int = 2000

PNG_DATA_URL_PREFIX: str = "data:image/png"

SnapshotCallback = Callable[[Optional[str]], None]


def snapshot_size(css_width: float, css_height: float, max_side: int = VISION_SNAPSHOT_MAX_SIDE) -> Tuple[int, int]:
    """Return the snapshot size for a canvas of the given CSS size.

    Keeps the aspect ratio, never scales up, and caps the longest side at ``max_side``.
    """
    width = max(float(css_width), 1.0)
    height = max(float(css_height), 1.0)
    scale = min(1.0, float(max_side) / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


class CanvasSnapshotter:
    """Composites the visible canvas layers into a PNG data URL."""

    def __init__(
        self,
        container_id: str = "math-container",
        canvas_id: str = "math-canvas-2d",
        svg_id: str = "math-svg",
        max_side: int = VISION_SNAPSHOT_MAX_SIDE,
        background: str = VISION_SNAPSHOT_BACKGROUND,
    ) -> None:
        self._container_id = container_id
        self._canvas_id = canvas_id
        self._svg_id = svg_id
        self._max_side = max_side
        self._background = background

    def capture(self, on_done: SnapshotCallback) -> None:
        """Capture the canvas and call ``on_done`` once with a PNG data URL (or None)."""
        try:
            css_size = self._css_size()
            if css_size is None:
                on_done(None)
                return
            css_width, css_height = css_size
            width, height = snapshot_size(css_width, css_height, self._max_side)
            output, ctx = self._create_output(width, height)
            svg_markup = self._svg_markup(css_width, css_height)
        except Exception as exc:
            print(f"Canvas snapshot failed: {exc}")
            on_done(None)
            return

        finish = self._once(lambda: on_done(self._finish(output, ctx, width, height)))
        if svg_markup is None:
            finish()
            return
        self._draw_svg_then(svg_markup, ctx, width, height, finish)

    # ----- sizing and output -----

    def _css_size(self) -> Optional[Tuple[float, float]]:
        container = document.getElementById(self._container_id)
        if container is None:
            return None
        width = float(getattr(container, "clientWidth", 0) or 0)
        height = float(getattr(container, "clientHeight", 0) or 0)
        if width <= 0 or height <= 0:
            return None
        return width, height

    def _create_output(self, width: int, height: int) -> Tuple[Any, Any]:
        output = html.CANVAS()
        # Set the bitmap size through the attributes: Brython maps ``.width`` on
        # DOM nodes to the element's CSS box, not the canvas bitmap.
        output.setAttribute("width", str(width))
        output.setAttribute("height", str(height))
        ctx = output.getContext("2d")
        ctx.fillStyle = self._background
        ctx.fillRect(0, 0, width, height)
        return output, ctx

    def _finish(self, output: Any, ctx: Any, width: int, height: int) -> Optional[str]:
        """Draw the Canvas2D layer on top and encode the result."""
        self._draw_canvas_layer(ctx, width, height)
        return self._encode(output)

    def _encode(self, output: Any) -> Optional[str]:
        try:
            data_url = output.toDataURL("image/png")
        except Exception as exc:
            # A tainted canvas (e.g. an SVG the browser refuses to export) cannot be read back.
            print(f"Canvas snapshot could not be encoded: {exc}")
            return None
        if not isinstance(data_url, str) or not data_url.startswith(PNG_DATA_URL_PREFIX):
            return None
        if len(data_url) > VISION_SNAPSHOT_MAX_DATA_URL_CHARS:
            print("Canvas snapshot is too large to send; sending the request without it.")
            return None
        return data_url

    # ----- layers -----

    def _draw_canvas_layer(self, ctx: Any, width: int, height: int) -> None:
        canvas_el = document.getElementById(self._canvas_id)
        if canvas_el is None or not self._is_visible(canvas_el):
            return
        try:
            # Scales the (devicePixelRatio-sized) bitmap to the snapshot size.
            ctx.drawImage(canvas_el, 0, 0, width, height)
        except Exception as exc:
            print(f"Canvas snapshot could not draw the Canvas2D layer: {exc}")

    def _svg_markup(self, css_width: float, css_height: float) -> Optional[str]:
        """Serialize the SVG layer as a standalone document, or None when it has no content."""
        svg_el = document.getElementById(self._svg_id)
        if svg_el is None or not self._is_visible(svg_el):
            return None
        if int(getattr(svg_el, "childElementCount", 0) or 0) == 0:
            return None
        clone = svg_el.cloneNode(True)
        clone.setAttribute("width", str(int(round(css_width))))
        clone.setAttribute("height", str(int(round(css_height))))
        # The live element is sized "100%" by CSS, which means nothing in a standalone image.
        clone.style.width = f"{int(round(css_width))}px"
        clone.style.height = f"{int(round(css_height))}px"
        # XMLSerializer adds the SVG namespace declaration a standalone image needs.
        markup = window.XMLSerializer.new().serializeToString(clone)
        return markup if isinstance(markup, str) and markup else None

    def _draw_svg_then(self, svg_markup: str, ctx: Any, width: int, height: int, done: Callable[[], None]) -> None:
        """Decode the SVG markup as an image, draw it, then call ``done`` (also on failure or timeout)."""
        image = self._new_image()

        def on_load(_event: Any = None) -> None:
            try:
                ctx.drawImage(image, 0, 0, width, height)
            except Exception as exc:
                print(f"Canvas snapshot could not draw the SVG layer: {exc}")
            done()

        def on_error(_event: Any = None) -> None:
            print("Canvas snapshot could not decode the SVG layer; using the Canvas2D layer only.")
            done()

        image.onload = on_load
        image.onerror = on_error
        self._schedule(done, SVG_DECODE_TIMEOUT_MS)
        image.src = "data:image/svg+xml;charset=utf-8," + window.encodeURIComponent(svg_markup)

    # ----- browser seams (replaced in tests) -----

    def _new_image(self) -> Any:
        return window.Image.new()

    def _schedule(self, callback: Callable[[], None], delay_ms: int) -> None:
        window.setTimeout(callback, delay_ms)

    # ----- helpers -----

    @staticmethod
    def _is_visible(element: Any) -> bool:
        style = getattr(element, "style", None)
        return getattr(style, "display", "") != "none" and getattr(style, "visibility", "") != "hidden"

    @staticmethod
    def _once(action: Callable[[], None]) -> Callable[[], None]:
        """Wrap ``action`` so only the first call runs it (load, error and timeout race)."""
        called: List[bool] = []

        def run(*_args: Any) -> None:
            if called:
                return
            called.append(True)
            action()

        return run
