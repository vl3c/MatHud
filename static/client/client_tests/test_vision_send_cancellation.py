"""
Tests that a vision request stopped by the user is never sent.

With vision on, the request leaves from the snapshot callback, which can
arrive up to the SVG decode timeout later. A stopped (or superseded) turn's
late callback must not send, since sending also aborts the current stream.

AIInterface is built without __init__ and its UI, network and timer
collaborators are replaced by recorders, so only the send path is exercised.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Callable, Dict, List, Optional

from browser import document

SNAPSHOT = "data:image/png;base64,iVBORw0KGgo="


class _FakeSnapshotter:
    """Holds capture callbacks so the test decides when (and whether) each snapshot arrives."""

    def __init__(self) -> None:
        self.pending: List[Callable[[Optional[str]], None]] = []

    def capture(self, on_done: Callable[[Optional[str]], None]) -> None:
        self.pending.append(on_done)


class _Stub:
    """Accepts any method call and does nothing."""

    def __init__(self, **attrs: Any) -> None:
        for name, value in attrs.items():
            setattr(self, name, value)

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


class TestVisionSendCancellation(unittest.TestCase):
    def setUp(self) -> None:
        if "vision-toggle" not in document or "ai-model-selector" not in document:
            self.skipTest("vision toggle or model selector not in DOM")
        self._vision_was_checked = document["vision-toggle"].checked
        document["vision-toggle"].checked = True
        self.sent: List[Dict[str, Any]] = []
        self.snapshotter = _FakeSnapshotter()
        self.ai = self._create_ai_interface()

    def tearDown(self) -> None:
        document["vision-toggle"].checked = self._vision_was_checked

    def _create_ai_interface(self) -> Any:
        from ai_interface import AIInterface

        ai = AIInterface.__new__(AIInterface)
        ai.is_processing = False
        ai._stop_requested = False
        ai._send_token = 0
        ai.canvas = _Stub(get_canvas_state=lambda: {})
        ai._canvas_snapshotter = self.snapshotter
        ai._chat_ui = _Stub(stream_buffer="", request_start_time=0)
        ai._image_attachment = _Stub(images=[])
        ai.slash_command_handler = _Stub(is_slash_command=lambda message: False)

        def disable_send_controls() -> None:
            ai.is_processing = True
            ai._stop_requested = False

        def enable_send_controls() -> None:
            ai.is_processing = False
            ai._stop_requested = False

        def send_request(prompt: Optional[str], action_trace: Any = None) -> None:
            self.sent.append(json.loads(prompt or "{}"))

        for name in (
            "_abort_current_stream",
            "_cancel_response_timeout",
            "_save_partial_response",
            "_finalize_stream_message",
            "_print_system_message_in_chat",
            "_print_user_message_in_chat",
        ):
            setattr(ai, name, lambda *args, **kwargs: None)
        setattr(ai, "_disable_send_controls", disable_send_controls)
        setattr(ai, "_enable_send_controls", enable_send_controls)
        setattr(ai, "_send_request", send_request)
        return ai

    def _deliver_snapshot(self, index: int) -> None:
        self.snapshotter.pending[index](SNAPSHOT)

    def test_snapshot_is_sent_while_the_turn_is_active(self) -> None:
        self.ai.send_user_message("first")
        self.assertEqual(self.sent, [], "a vision request waits for its snapshot")
        self._deliver_snapshot(0)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0]["user_message"], "first")
        self.assertEqual(self.sent[0]["canvas_snapshot"], SNAPSHOT)

    def test_snapshot_arriving_after_stop_is_not_sent(self) -> None:
        self.ai.send_user_message("first")
        self.ai.stop_ai_processing()
        self._deliver_snapshot(0)
        self.assertEqual(self.sent, [])

    def test_stopped_snapshot_does_not_send_over_the_next_message(self) -> None:
        self.ai.send_user_message("first")
        self.ai.stop_ai_processing()
        self.ai.send_user_message("second")
        self.assertEqual(len(self.snapshotter.pending), 2)
        self._deliver_snapshot(0)
        self.assertEqual(self.sent, [], "the stopped turn must not send (it would abort the new stream)")
        self._deliver_snapshot(1)
        self.assertEqual([prompt["user_message"] for prompt in self.sent], ["second"])

    def test_snapshot_arriving_after_the_turn_ended_is_not_sent(self) -> None:
        self.ai.send_user_message("first")
        self.ai._enable_send_controls()  # e.g. the response timeout fired
        self._deliver_snapshot(0)
        self.assertEqual(self.sent, [])
