"""
Tests for the browser-side vision snapshot (canvas_snapshot.py).

Uses real DOM elements: an off-screen container holding a Canvas2D bitmap
(sized like a HiDPI canvas) and an SVG layer, mirroring #math-container.
"""

from __future__ import annotations

import unittest
from typing import Any, List, Optional, Tuple

from browser import document, html, svg, window

from canvas_snapshot import CanvasSnapshotter, snapshot_size

CONTAINER_ID = "snapshot-test-container"
CANVAS_ID = "snapshot-test-canvas"
SVG_ID = "snapshot-test-svg"
CSS_WIDTH = 400
CSS_HEIGHT = 200
PNG_PREFIX = "data:image/png;base64,"


def _png_size(data_url: str) -> Tuple[int, int]:
    """Read width and height from the PNG IHDR chunk of a data URL."""
    raw = window.atob(data_url[len(PNG_PREFIX) :])
    width = (ord(raw[16]) << 24) | (ord(raw[17]) << 16) | (ord(raw[18]) << 8) | ord(raw[19])
    height = (ord(raw[20]) << 24) | (ord(raw[21]) << 16) | (ord(raw[22]) << 8) | ord(raw[23])
    return width, height


def _pixel(canvas_el: Any, x: int, y: int) -> List[int]:
    data = canvas_el.getContext("2d").getImageData(x, y, 1, 1).data
    return [int(data[i]) for i in range(4)]


class TestSnapshotSize(unittest.TestCase):
    def test_small_canvas_keeps_its_css_size(self) -> None:
        self.assertEqual(snapshot_size(800, 600), (800, 600))

    def test_wide_canvas_is_capped_at_the_longest_side(self) -> None:
        self.assertEqual(snapshot_size(2560, 1440), (1280, 720))

    def test_tall_canvas_is_capped_at_the_longest_side(self) -> None:
        self.assertEqual(snapshot_size(1000, 3000), (427, 1280))

    def test_custom_cap(self) -> None:
        self.assertEqual(snapshot_size(400, 200, max_side=100), (100, 50))

    def test_degenerate_size_is_at_least_one_pixel(self) -> None:
        self.assertEqual(snapshot_size(0, 0), (1, 1))


class TestCanvasSnapshotter(unittest.TestCase):
    def setUp(self) -> None:
        self.container = html.DIV(id=CONTAINER_ID)
        style = self.container.style
        style.position = "absolute"
        style.left = "-10000px"
        style.top = "0px"
        style.width = f"{CSS_WIDTH}px"
        style.height = f"{CSS_HEIGHT}px"
        document <= self.container

    def tearDown(self) -> None:
        self.container.remove()

    # ----- fixtures -----

    def _add_canvas(self, ratio: int = 2) -> Any:
        """A Canvas2D layer with a devicePixelRatio-sized bitmap: left half red, right half blue."""
        canvas_el = html.CANVAS(id=CANVAS_ID)
        canvas_el.setAttribute("width", str(CSS_WIDTH * ratio))
        canvas_el.setAttribute("height", str(CSS_HEIGHT * ratio))
        canvas_el.style.width = f"{CSS_WIDTH}px"
        canvas_el.style.height = f"{CSS_HEIGHT}px"
        ctx = canvas_el.getContext("2d")
        ctx.fillStyle = "#ff0000"
        ctx.fillRect(0, 0, CSS_WIDTH * ratio / 2, CSS_HEIGHT * ratio)
        ctx.fillStyle = "#0000ff"
        ctx.fillRect(CSS_WIDTH * ratio / 2, 0, CSS_WIDTH * ratio / 2, CSS_HEIGHT * ratio)
        self.container <= canvas_el
        return canvas_el

    def _add_svg(self, with_content: bool) -> Any:
        svg_el = svg.svg(id=SVG_ID)
        svg_el.style.width = "100%"
        svg_el.style.height = "100%"
        if with_content:
            svg_el <= svg.rect(x=10, y=10, width=50, height=20, fill="#00ff00")
            svg_el <= svg.text("A", x=20, y=60)
        self.container <= svg_el
        return svg_el

    def _snapshotter(self, **kwargs: Any) -> CanvasSnapshotter:
        return CanvasSnapshotter(container_id=CONTAINER_ID, canvas_id=CANVAS_ID, svg_id=SVG_ID, **kwargs)

    def _capture_sync(self, snapshotter: CanvasSnapshotter) -> List[Optional[str]]:
        results: List[Optional[str]] = []
        snapshotter.capture(results.append)
        return results

    def _capture_output_canvas(self, snapshotter: CanvasSnapshotter) -> Any:
        """Run a capture and return the composited output canvas instead of its encoding."""
        outputs: List[Any] = []

        def keep(output: Any) -> Optional[str]:
            outputs.append(output)
            return None

        setattr(snapshotter, "_encode", keep)
        snapshotter.capture(lambda _snapshot: None)
        self.assertEqual(len(outputs), 1)
        return outputs[0]

    # ----- tests -----

    def test_canvas_layer_snapshot_is_png_at_css_size(self) -> None:
        self._add_svg(with_content=False)
        self._add_canvas(ratio=2)
        results = self._capture_sync(self._snapshotter())
        self.assertEqual(len(results), 1, "callback must run once, synchronously, without SVG content")
        data_url = results[0]
        self.assertIsNotNone(data_url)
        assert data_url is not None
        self.assertTrue(data_url.startswith(PNG_PREFIX))
        self.assertEqual(_png_size(data_url), (CSS_WIDTH, CSS_HEIGHT))

    def test_longest_side_is_capped(self) -> None:
        self._add_canvas(ratio=2)
        results = self._capture_sync(self._snapshotter(max_side=100))
        data_url = results[0]
        assert data_url is not None
        self.assertEqual(_png_size(data_url), (100, 50))

    def test_hidpi_bitmap_is_scaled_not_cropped(self) -> None:
        self._add_canvas(ratio=2)
        output = self._capture_output_canvas(self._snapshotter())
        left = _pixel(output, CSS_WIDTH // 4, CSS_HEIGHT // 2)
        right = _pixel(output, 3 * CSS_WIDTH // 4, CSS_HEIGHT // 2)
        self.assertGreater(left[0], 200)
        self.assertLess(left[2], 50)
        self.assertGreater(right[2], 200)
        self.assertLess(right[0], 50)

    def test_background_is_filled_without_layers(self) -> None:
        output = self._capture_output_canvas(self._snapshotter())
        self.assertEqual(_pixel(output, 5, 5), [255, 255, 255, 255])

    def test_hidden_canvas_layer_is_skipped(self) -> None:
        canvas_el = self._add_canvas(ratio=1)
        canvas_el.style.display = "none"
        output = self._capture_output_canvas(self._snapshotter())
        self.assertEqual(_pixel(output, CSS_WIDTH // 4, CSS_HEIGHT // 2), [255, 255, 255, 255])

    def test_missing_container_reports_none(self) -> None:
        results: List[Optional[str]] = []
        CanvasSnapshotter(container_id="no-such-container").capture(results.append)
        self.assertEqual(results, [None])

    def test_empty_svg_layer_is_not_serialized(self) -> None:
        self._add_svg(with_content=False)
        self.assertIsNone(self._snapshotter()._svg_markup(CSS_WIDTH, CSS_HEIGHT))

    def test_svg_layer_is_serialized_as_standalone_image(self) -> None:
        self._add_svg(with_content=True)
        markup = self._snapshotter()._svg_markup(CSS_WIDTH, CSS_HEIGHT)
        self.assertIsNotNone(markup)
        assert markup is not None
        self.assertIn("http://www.w3.org/2000/svg", markup)
        self.assertIn(f'width="{CSS_WIDTH}"', markup)
        self.assertIn(f'height="{CSS_HEIGHT}"', markup)
        self.assertIn(f"width: {CSS_WIDTH}px", markup)
        self.assertNotIn("100%", markup)
        self.assertIn("<rect", markup)
        self.assertEqual(markup.count("xmlns="), 1)

    def test_live_svg_is_not_modified(self) -> None:
        svg_el = self._add_svg(with_content=True)
        self._snapshotter()._svg_markup(CSS_WIDTH, CSS_HEIGHT)
        self.assertEqual(svg_el.style.width, "100%")
        self.assertFalse(svg_el.hasAttribute("width"))

    def test_callback_runs_once(self) -> None:
        calls: List[int] = []
        run = CanvasSnapshotter._once(lambda: calls.append(1))
        run()
        run()
        run()
        self.assertEqual(calls, [1])
