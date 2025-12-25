from __future__ import annotations

import unittest

from rendering.helpers.label_overlap_resolver import ScreenOffsetLabelOverlapResolver


class TestLabelOverlapResolver(unittest.TestCase):
    def test_second_label_shifts_when_overlapping(self) -> None:
        resolver = ScreenOffsetLabelOverlapResolver(max_steps=3, padding_px=0.0)
        dy1 = resolver.get_or_place_dy("a", (0.0, 10.0, 0.0, 10.0), step=10.0)
        dy2 = resolver.get_or_place_dy("b", (0.0, 10.0, 0.0, 10.0), step=10.0)

        self.assertEqual(dy1, 0.0)
        self.assertEqual(dy2, 10.0)

    def test_block_height_affects_collision(self) -> None:
        resolver = ScreenOffsetLabelOverlapResolver(max_steps=3, padding_px=0.0)
        resolver.get_or_place_dy("block", (0.0, 10.0, 0.0, 20.0), step=10.0)

        dy = resolver.get_or_place_dy("line", (0.0, 10.0, 0.0, 10.0), step=10.0)
        self.assertEqual(dy, -10.0)

        dy_again = resolver.get_or_place_dy("line", (0.0, 10.0, 0.0, 10.0), step=10.0)
        self.assertEqual(dy_again, dy)


