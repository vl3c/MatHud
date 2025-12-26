from __future__ import annotations

import unittest

from rendering.helpers.screen_offset_label_layout import LabelBlock, make_label_text_call, solve_dy, solve_dy_with_hide_for_text_calls


class TestScreenOffsetLabelLayout(unittest.TestCase):
    def test_prefers_smallest_abs_dy(self) -> None:
        a = LabelBlock(group="A", order=0, base_rect=(0.0, 10.0, 0.0, 12.0), step=12.0)
        b = LabelBlock(group="B", order=1, base_rect=(0.0, 10.0, 0.0, 12.0), step=12.0)
        dy = solve_dy([a, b], max_steps=3, iteration_cap=200, cell_size=32.0)

        self.assertEqual(dy.get("A"), 0.0)
        self.assertEqual(dy.get("B"), 12.0)

    def test_can_return_to_zero_when_free(self) -> None:
        # Chain overlaps: X overlaps Y, Y overlaps Z, X does not overlap Z.
        x = LabelBlock(group="X", order=2, base_rect=(0.0, 10.0, 0.0, 12.0), step=12.0)
        y = LabelBlock(group="Y", order=1, base_rect=(8.0, 18.0, 0.0, 12.0), step=12.0)
        z = LabelBlock(group="Z", order=0, base_rect=(16.0, 26.0, 0.0, 12.0), step=12.0)
        dy = solve_dy([x, y, z], max_steps=3, iteration_cap=500, cell_size=32.0)

        self.assertEqual(dy.get("Z"), 0.0)
        self.assertEqual(dy.get("X"), 0.0)
        self.assertEqual(dy.get("Y"), 12.0)

    def test_multiline_block_moves_as_one(self) -> None:
        # A is taller (eg 2-line block). B starts overlapping A's lower half.
        a = LabelBlock(group="A", order=0, base_rect=(0.0, 10.0, 0.0, 24.0), step=12.0)
        b = LabelBlock(group="B", order=1, base_rect=(0.0, 10.0, 12.0, 24.0), step=12.0)
        dy = solve_dy([a, b], max_steps=3, iteration_cap=200, cell_size=32.0)

        self.assertEqual(dy.get("A"), 0.0)
        self.assertEqual(dy.get("B"), 12.0)

    def test_lookahead_picks_smaller_best_dy_mover(self) -> None:
        # A and B collide at dy=0. B is the later label (order=10) but is constrained so its
        # first collision-free placement is 2*step. A can resolve at 1*step, so lookahead
        # must move A instead of B.
        step = 12.0
        a = LabelBlock(group="A", order=0, base_rect=(0.0, 10.0, 0.0, step), step=step)
        b = LabelBlock(group="B", order=10, base_rect=(8.0, 18.0, 0.0, step), step=step)
        # Blockers that prevent B from using +/-1*step, but do not affect A at +1*step.
        up_block = LabelBlock(group="U", order=1, base_rect=(10.0, 18.0, -step, 0.0), step=step)
        down_block = LabelBlock(group="D", order=2, base_rect=(10.0, 18.0, step, 2 * step), step=step)

        dy = solve_dy([a, b, up_block, down_block], max_steps=3, iteration_cap=500, cell_size=32.0)

        self.assertEqual(dy.get("B"), 0.0)
        self.assertEqual(dy.get("A"), step)

    def test_max_steps_zero_cannot_resolve_overlap(self) -> None:
        # With max_steps=0, no label can move (only dy=0 is considered).
        a = LabelBlock(group="A", order=0, base_rect=(0.0, 10.0, 0.0, 12.0), step=12.0)
        b = LabelBlock(group="B", order=1, base_rect=(0.0, 10.0, 0.0, 12.0), step=12.0)
        dy = solve_dy([a, b], max_steps=0, iteration_cap=10, cell_size=32.0)
        self.assertEqual(dy.get("A"), 0.0)
        self.assertEqual(dy.get("B"), 0.0)

    def test_iteration_cap_coerces_and_returns_all_groups(self) -> None:
        # iteration_cap<=0 is coerced to a minimal positive cap; should still return a dy entry for each group.
        a = LabelBlock(group="A", order=0, base_rect=(0.0, 10.0, 0.0, 12.0), step=12.0)
        b = LabelBlock(group="B", order=1, base_rect=(0.0, 10.0, 0.0, 12.0), step=12.0)
        c = LabelBlock(group="C", order=2, base_rect=(0.0, 10.0, 0.0, 12.0), step=12.0)
        dy = solve_dy([a, b, c], max_steps=1, iteration_cap=0, cell_size=32.0)
        self.assertEqual(set(dy.keys()), {"A", "B", "C"})

    def test_lookahead_prefers_smaller_step_when_it_yields_smaller_abs_dy(self) -> None:
        # A and B collide at dy=0. B is constrained to require 2*B_step, while A can resolve at 1*A_step.
        # The solver should move A because abs(1*A_step) < abs(2*B_step).
        a_step = 20.0
        b_step = 12.0
        a = LabelBlock(group="A", order=0, base_rect=(0.0, 10.0, 0.0, a_step), step=a_step)
        b = LabelBlock(group="B", order=10, base_rect=(8.0, 18.0, 0.0, b_step), step=b_step)

        # Block B at +/-1*b_step using x-range that touches A at x=10 (edge-touch is non-overlap).
        up_block = LabelBlock(group="U", order=1, base_rect=(10.0, 18.0, -b_step, 0.0), step=b_step)
        down_block = LabelBlock(group="D", order=2, base_rect=(10.0, 18.0, b_step, 2 * b_step), step=b_step)

        dy = solve_dy([a, b, up_block, down_block], max_steps=3, iteration_cap=500, cell_size=32.0)
        self.assertEqual(dy.get("B"), 0.0)
        self.assertEqual(dy.get("A"), a_step)

    def test_make_label_text_call_falls_back_when_layout_group_is_unhashable(self) -> None:
        class _Font:
            def __init__(self, size: float) -> None:
                self.size = size

        metadata = {
            "point_label": {
                "layout_group": [],  # unhashable
                "layout_line_index": 0,
                "layout_line_count": 1,
                "layout_max_line_len": 2,
            }
        }
        call = make_label_text_call(
            order=0,
            text="hi",
            position=(1.0, 2.0),
            font=_Font(12.0),
            color="#000",
            alignment=None,
            style_overrides=None,
            metadata=metadata,
        )
        self.assertIsNotNone(call)
        self.assertEqual(call.group, ("hi", 1.0, 2.0))

    def test_hide_groups_when_abs_dy_exceeds_factor(self) -> None:
        class _Font:
            def __init__(self, size: float) -> None:
                self.size = size

        font_size = 10.0
        calls = []
        for idx in range(9):
            group = f"G{idx}"
            metadata = {
                "point_label": {
                    "layout_group": group,
                    "layout_line_index": 0,
                    "layout_line_count": 1,
                    "layout_max_line_len": 1,
                    "screen_offset": (0.0, 0.0),
                }
            }
            call = make_label_text_call(
                order=idx,
                text="x",
                position=(0.0, 0.0),
                font=_Font(font_size),
                color="#000",
                alignment=None,
                style_overrides=None,
                metadata=metadata,
            )
            self.assertIsNotNone(call)
            if call is not None:
                calls.append(call)

        dy, hidden = solve_dy_with_hide_for_text_calls(calls, max_abs_dy_factor=3.0, max_passes=2)
        # Hard bound: visible labels must have abs(dy) <= 3*font_size.
        for idx in range(9):
            group = f"G{idx}"
            if group in hidden:
                continue
            self.assertTrue(abs(float(dy.get(group, 0.0))) <= 3.0 * font_size)

    def test_proximity_hide_hides_later_label_when_anchors_are_too_close(self) -> None:
        class _Font:
            def __init__(self, size: float) -> None:
                self.size = size

        font_size = 10.0
        # Width ~= 0.6 * font_size * max_line_len = 60.0 when max_line_len=10
        # Threshold = (wA+wB)/4 = 30.0. Set anchor distance < 30.0 to trigger hide.
        common_meta = {
            "layout_line_index": 0,
            "layout_line_count": 1,
            "layout_max_line_len": 10,
        }
        call_a = make_label_text_call(
            order=0,
            text="abcdefghij",
            position=(0.0, 0.0),
            font=_Font(font_size),
            color="#000",
            alignment=None,
            style_overrides=None,
            metadata={"point_label": {**common_meta, "layout_group": "A", "screen_offset": (0.0, 0.0)}},
        )
        call_b = make_label_text_call(
            order=10,
            text="abcdefghij",
            position=(0.0, 0.0),
            font=_Font(font_size),
            color="#000",
            alignment=None,
            style_overrides=None,
            metadata={"point_label": {**common_meta, "layout_group": "B", "screen_offset": (-10.0, 0.0)}},
        )
        self.assertIsNotNone(call_a)
        self.assertIsNotNone(call_b)
        dy, hidden = solve_dy_with_hide_for_text_calls([call_a, call_b], max_abs_dy_factor=3.0)
        self.assertIn("B", hidden)
        self.assertNotIn("A", hidden)
        self.assertIn("A", dy)

    def test_proximity_hide_does_not_trigger_when_anchors_are_far(self) -> None:
        class _Font:
            def __init__(self, size: float) -> None:
                self.size = size

        font_size = 10.0
        common_meta = {
            "layout_line_index": 0,
            "layout_line_count": 1,
            "layout_max_line_len": 10,
        }
        call_a = make_label_text_call(
            order=0,
            text="abcdefghij",
            position=(0.0, 0.0),
            font=_Font(font_size),
            color="#000",
            alignment=None,
            style_overrides=None,
            metadata={"point_label": {**common_meta, "layout_group": "A", "screen_offset": (0.0, 0.0)}},
        )
        call_b = make_label_text_call(
            order=10,
            text="abcdefghij",
            position=(0.0, 0.0),
            font=_Font(font_size),
            color="#000",
            alignment=None,
            style_overrides=None,
            metadata={"point_label": {**common_meta, "layout_group": "B", "screen_offset": (-100.0, 0.0)}},
        )
        self.assertIsNotNone(call_a)
        self.assertIsNotNone(call_b)
        dy, hidden = solve_dy_with_hide_for_text_calls([call_a, call_b], max_abs_dy_factor=3.0)
        # They may still overlap in rect-space, but proximity rule should not hide.
        self.assertNotIn("B", hidden)


