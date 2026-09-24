"""Zoom step constants must be exact inverses so zoom in/out cycles do not drift."""

from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import unittest

from constants import zoom_in_scale_factor, zoom_out_scale_factor


class TestZoomConstants(unittest.TestCase):
    def test_zoom_out_undoes_zoom_in(self) -> None:
        self.assertAlmostEqual(zoom_in_scale_factor * zoom_out_scale_factor, 1.0, places=12)

    def test_repeated_zoom_cycles_do_not_drift(self) -> None:
        scale = 1.0
        for _ in range(50):
            scale *= zoom_in_scale_factor
            scale *= zoom_out_scale_factor
        self.assertAlmostEqual(scale, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
