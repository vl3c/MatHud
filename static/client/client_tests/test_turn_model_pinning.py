"""
Tests that a turn's requests go to the model it started with.

Turn tokens belong to a provider instance, so a model picked while a reply runs
must not receive the turn's tool-result follow-ups (its instance would drop them
as abandoned). The selector stays usable: the new model answers from the next
message, and the chat says so.

AIInterface is built without __init__ and its UI, network, timer and send-control
collaborators are replaced by stubs. The page's model selector gets three
temporary options (CI has no API keys, so its selector lists no models), which
are removed, with the selector value and vision toggle restored, after each test.
"""

from __future__ import annotations

import unittest
from typing import Any, Callable, Dict, List, Optional

from browser import document, html

TOOL_CALLS = [{"function_name": "create_point", "arguments": {"x": 1, "y": 2}}]
# Temporary selector options: (value, label)
TEST_MODELS = [
    ("pin-test-a", "Pin Test A"),
    ("pin-test-b", "Pin Test B"),
    ("pin-test-c", "Pin Test C (text only)"),
]


class _Stub:
    """Accepts any method call and does nothing."""

    def __init__(self, **attrs: Any) -> None:
        for name, value in attrs.items():
            setattr(self, name, value)

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


class TestTurnModelPinning(unittest.TestCase):
    def setUp(self) -> None:
        if "ai-model-selector" not in document or "vision-toggle" not in document:
            self.skipTest("model selector or vision toggle not in DOM")
        self.selector = document["ai-model-selector"]
        saved_model = str(self.selector.value)
        saved_vision = document["vision-toggle"].checked
        added = [html.OPTION(label, value=value) for value, label in TEST_MODELS]
        for option in added:
            self.selector <= option

        def restore() -> None:
            for option in added:
                option.remove()
            self.selector.value = saved_model
            document["vision-toggle"].checked = saved_vision

        self.addCleanup(restore)
        self.first_model, self.second_model, self.third_model = (value for value, _ in TEST_MODELS)
        document["vision-toggle"].checked = False
        self.selector.value = self.first_model
        self.sent: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.ai = self._create_ai_interface()

    def _create_ai_interface(self) -> Any:
        from ai_interface import AIInterface

        ai = AIInterface.__new__(AIInterface)
        ai.is_processing = False
        ai._stop_requested = False
        ai._send_token = 0
        ai._server_turn = None
        ai._turn_model = None
        ai._turn_request_limit = None
        ai._turn_requests_sent = 0
        ai._turn_timeout_ms = None
        ai._response_timeout_id = None
        ai._last_user_message = ""
        ai.canvas = _Stub(get_canvas_state=lambda: {})
        ai._chat_ui = _Stub(stream_buffer="", stream_container=object(), stream_content=None)
        ai._image_attachment = _Stub(images=[])
        ai._tool_call_log = _Stub()
        ai._turn_metrics_collector = _Stub()  # backs the _turn_metrics property
        ai.slash_command_handler = _Stub(is_slash_command=lambda message: False)

        def execute_tool_batch(tool_calls: Any, turn_token: Optional[int] = None) -> Dict[str, Any]:
            return {"call_results": {}, "traced_calls": [], "trace": None, "state_after": {}}

        def send_prompt_json(prompt_json: Dict[str, Any], *args: Any) -> None:
            self.sent.append(dict(prompt_json))

        def disable_send_controls() -> None:
            ai.is_processing = True
            ai._stop_requested = False

        def enable_send_controls() -> None:
            ai.is_processing = False
            ai._stop_requested = False

        for name in (
            "_start_response_timeout",
            "_cancel_response_timeout",
            "_abort_current_stream",
            "_save_partial_response",
            "_finalize_stream_message",
            "_print_user_message_in_chat",
            "_debug_log_ai_response",
        ):
            setattr(ai, name, lambda *args, **kwargs: None)
        setattr(ai, "_print_system_message_in_chat", lambda message: self.notes.append(message))
        setattr(ai, "_trace_summary", lambda trace: None)
        setattr(ai, "execute_tool_batch", execute_tool_batch)
        setattr(ai, "_send_prompt_json", send_prompt_json)
        setattr(ai, "_disable_send_controls", disable_send_controls)
        setattr(ai, "_enable_send_controls", enable_send_controls)
        return ai

    def _select(self, model_id: str) -> None:
        """Pick a model the way the user does: set the value, then fire the change handler."""
        self.selector.value = model_id
        self.ai.on_model_selected(None)

    def _run_tool_calls(self) -> None:
        self.ai._on_stream_final({"finish_reason": "tool_calls", "ai_message": "", "ai_tool_calls": TOOL_CALLS})

    def _finish_turn(self) -> None:
        self.ai._on_stream_final({"finish_reason": "stop", "ai_message": "Done.", "ai_tool_calls": []})

    def test_first_request_uses_the_selected_model(self) -> None:
        self.ai.send_user_message("draw a point")
        self.assertEqual(self.sent[0]["ai_model"], self.first_model)

    def test_follow_up_keeps_the_turns_model_after_a_mid_turn_change(self) -> None:
        self.ai.send_user_message("draw a point")
        self._select(self.second_model)
        self._run_tool_calls()
        self.assertEqual(len(self.sent), 2)
        self.assertIsNotNone(self.sent[1]["tool_call_results"])
        self.assertEqual(self.sent[1]["ai_model"], self.first_model, "the turn's follow-up stays on its model")

    def test_next_message_uses_the_model_picked_mid_turn(self) -> None:
        self.ai.send_user_message("draw a point")
        self._select(self.second_model)
        self._finish_turn()
        self.ai.send_user_message("draw another")
        self.assertEqual(self.sent[-1]["ai_model"], self.second_model)
        self.assertIsNone(self.sent[-1]["tool_call_results"])

    def test_mid_turn_change_explains_when_the_new_model_answers(self) -> None:
        from ai_interface import AIInterface

        self.ai.send_user_message("draw a point")
        self._select(self.second_model)
        self.assertEqual(len(self.notes), 1)
        self.assertIn(AIInterface._model_label(self.second_model), self.notes[0])
        self.assertIn(AIInterface._model_label(self.first_model), self.notes[0])
        self.assertIn("next message", self.notes[0])

    def test_change_while_idle_adds_no_note(self) -> None:
        self._select(self.second_model)
        self.assertEqual(self.notes, [])

    def test_switching_back_says_the_turns_model_keeps_answering(self) -> None:
        from ai_interface import AIInterface

        self.ai.send_user_message("draw a point")
        self._select(self.second_model)
        self._select(self.first_model)
        self.assertEqual(len(self.notes), 2, "the earlier note must not be left promising the other model")
        self.assertIn(AIInterface._model_label(self.first_model), self.notes[1])
        self.assertIn("keep answering", self.notes[1])
        self.assertEqual(self.ai.turn_model_id(), self.first_model)

    def test_reselecting_the_turns_model_without_a_switch_adds_no_note(self) -> None:
        self.ai.send_user_message("draw a point")
        self._select(self.first_model)
        self.assertEqual(self.notes, [])

    def test_reselecting_the_announced_model_adds_no_note(self) -> None:
        self.ai.send_user_message("draw a point")
        self._select(self.second_model)
        self._select(self.second_model)
        self.assertEqual(len(self.notes), 1)

    def test_latest_note_names_the_model_that_answers_next(self) -> None:
        from ai_interface import AIInterface

        self.ai.send_user_message("draw a point")
        self._select(self.second_model)
        self._select(self.third_model)
        self.assertEqual(len(self.notes), 2)
        self.assertIn(AIInterface._model_label(self.third_model), self.notes[-1])
        self._finish_turn()
        self.ai.send_user_message("draw another")
        self.assertEqual(self.sent[-1]["ai_model"], self.third_model)

    def test_each_turn_starts_without_an_announced_model(self) -> None:
        self.ai.send_user_message("draw a point")
        self._select(self.second_model)
        self._finish_turn()
        self.ai.send_user_message("draw another")  # this turn runs on the second model
        self._select(self.first_model)
        self.assertEqual(len(self.notes), 2)
        self.assertIn("next message", self.notes[1])

    def test_turn_model_id_follows_the_selector_between_turns(self) -> None:
        self.ai.send_user_message("draw a point")
        self._select(self.second_model)
        self.assertEqual(self.ai.turn_model_id(), self.first_model)
        self._finish_turn()
        self.assertEqual(self.ai.turn_model_id(), self.second_model)

    def test_stop_releases_the_turns_model(self) -> None:
        self.ai.send_user_message("draw a point")
        self._select(self.second_model)
        self.ai.stop_ai_processing()
        self.assertEqual(self.ai.turn_model_id(), self.second_model)

    def test_model_label_uses_the_option_text_without_the_text_only_suffix(self) -> None:
        from ai_interface import AIInterface

        self.assertEqual(AIInterface._model_label("pin-test-b"), "Pin Test B")
        self.assertEqual(AIInterface._model_label("pin-test-c"), "Pin Test C")
        self.assertEqual(AIInterface._model_label("vendor/unlisted-model"), "unlisted-model")


class TestSearchToolsModel(unittest.TestCase):
    def test_search_uses_the_turns_model(self) -> None:
        from function_registry import FunctionRegistry

        self.assertEqual(FunctionRegistry._search_model_id(lambda: "turn-model"), "turn-model")

    def test_empty_turn_model_sends_no_model(self) -> None:
        from function_registry import FunctionRegistry

        self.assertIsNone(FunctionRegistry._search_model_id(lambda: ""))

    def test_without_a_getter_the_selector_is_used(self) -> None:
        from function_registry import FunctionRegistry

        if "ai-model-selector" not in document:
            self.skipTest("model selector not in DOM")
        expected = str(document["ai-model-selector"].value) or None
        self.assertEqual(FunctionRegistry._search_model_id(None), expected)

    def test_registry_gives_search_tools_the_turns_model(self) -> None:
        from canvas import Canvas
        from function_registry import FunctionRegistry
        from workspace_manager import WorkspaceManager

        captured: List[Any] = []
        original = FunctionRegistry._create_search_tools_handler

        def capture(get_model_id: Any = None) -> Any:
            captured.append(get_model_id)
            return original(get_model_id)

        canvas = Canvas(500, 500, draw_enabled=False)
        ai = _Stub(turn_model_id=lambda: "turn-model", run_tests=lambda: {})
        FunctionRegistry._create_search_tools_handler = staticmethod(capture)  # type: ignore[method-assign]
        try:
            FunctionRegistry.get_available_functions(canvas, WorkspaceManager(canvas), ai)  # type: ignore[arg-type]
        finally:
            FunctionRegistry._create_search_tools_handler = staticmethod(original)  # type: ignore[method-assign]
        self.assertEqual(len(captured), 1)
        self.assertEqual(FunctionRegistry._search_model_id(captured[0]), "turn-model")
