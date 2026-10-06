"""Tests for prompt_canvas_state: the canvas size in pixels added to a prompt's canvas state."""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from prompt_canvas_state import CANVAS_SIZE_KEY, canvas_size_px, with_canvas_size

from .test_turn_metrics import TestTurnBookkeeping


class _Sized:
    def __init__(self, width: Any, height: Any) -> None:
        self.width = width
        self.height = height


class TestPromptCanvasSize(unittest.TestCase):
    def test_size_of_a_canvas(self) -> None:
        self.assertEqual(canvas_size_px(_Sized(800, 600)), {"width": 800.0, "height": 600.0})

    def test_fractional_css_pixels_are_rounded(self) -> None:
        self.assertEqual(canvas_size_px(_Sized(812.3456, 600.004)), {"width": 812.35, "height": 600.0})

    def test_unknown_or_invalid_sizes_give_none(self) -> None:
        self.assertIsNone(canvas_size_px(None))
        self.assertIsNone(canvas_size_px(_Sized(0, 600)))
        self.assertIsNone(canvas_size_px(_Sized(800, -1)))
        self.assertIsNone(canvas_size_px(_Sized("wide", 600)))
        self.assertIsNone(canvas_size_px(_Sized(float("nan"), 600)))

    def test_size_is_added_to_a_copy(self) -> None:
        state: Dict[str, Any] = {"Points": [], "Cartesian_System_Visibility": {"left_bound": -400}}
        result = with_canvas_size(state, _Sized(800, 600))
        self.assertEqual(result[CANVAS_SIZE_KEY], {"width": 800.0, "height": 600.0})
        self.assertEqual(result["Points"], [])
        self.assertNotIn(CANVAS_SIZE_KEY, state)

    def test_state_is_returned_unchanged_without_a_size(self) -> None:
        state: Dict[str, Any] = {"Points": []}
        self.assertIs(with_canvas_size(state, None), state)
        self.assertIs(with_canvas_size(None, _Sized(800, 600)), None)

    def test_key_matches_the_live_canvas(self) -> None:
        """The real Canvas exposes its CSS pixel size as width and height."""
        from canvas import Canvas

        canvas = Canvas(640, 480, draw_enabled=False)
        self.assertEqual(canvas_size_px(canvas), {"width": 640.0, "height": 480.0})


class TestPromptCarriesCanvasSize(unittest.TestCase):
    """AIInterface sends the size with every prompt, without touching the state it was given."""

    def _ai(self) -> Any:
        ai = TestTurnBookkeeping()._ai()
        ai._turn_request_limit = None
        ai._turn_requests_sent = 0
        self.sent: List[Any] = []
        ai._send_prompt_json = lambda *args: self.sent.append(args)
        return ai

    def test_tool_batch_prompt_carries_the_size(self) -> None:
        ai = self._ai()
        ai.canvas = _Sized(900, 700)
        state_after: Dict[str, Any] = {"Points": []}
        ai._send_prompt_to_ai(None, "[]", canvas_state=state_after)
        prompt_json = self.sent[-1][0]
        self.assertEqual(prompt_json["canvas_state"][CANVAS_SIZE_KEY], {"width": 900.0, "height": 700.0})
        self.assertNotIn(CANVAS_SIZE_KEY, state_after)

    def test_prompt_without_a_canvas_is_sent_as_is(self) -> None:
        ai = self._ai()
        ai._send_prompt_to_ai(None, "[]", canvas_state={})
        self.assertEqual(self.sent[-1][0]["canvas_state"], {})


__all__ = ["TestPromptCanvasSize", "TestPromptCarriesCanvasSize"]
