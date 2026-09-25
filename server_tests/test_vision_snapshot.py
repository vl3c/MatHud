"""
How the browser-captured canvas snapshot reaches each provider when vision is on.

The client puts the snapshot (a PNG data URL) into the prompt JSON as
``canvas_snapshot`` next to ``use_vision``; providers must send it as an image
part together with the <canvas> text block and the user's text.
"""

from __future__ import annotations

import base64
import json
import os
import unittest
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

from server_tests.test_canvas_state_prompts import CANVAS_BLOCK, CanvasFormatEnv, user_prompt
from static.app_manager import AppManager, MatHudFlask
from static.config import MAX_IMAGE_BASE64_BYTES
from static.routes import validate_canvas_snapshot

PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
SNAPSHOT = f"data:image/png;base64,{PNG_BASE64}"
ATTACHED = "data:image/jpeg;base64,/9j/AAAA"
USER_TEXT = f"{CANVAS_BLOCK}\n\nHow long is AB?"


def vision_prompt(snapshot: Optional[str] = SNAPSHOT, **extra: Any) -> str:
    fields: Dict[str, Any] = {"use_vision": True}
    if snapshot is not None:
        fields["canvas_snapshot"] = snapshot
    fields.update(extra)
    return user_prompt(**fields)


class TestChatCompletionsVision(CanvasFormatEnv):
    def test_snapshot_is_sent_with_the_canvas_block(self) -> None:
        api = self.chat_api()
        api._prepare_messages_for_request(vision_prompt())
        self.assertEqual(
            api.messages[-1],
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_TEXT},
                    {"type": "image_url", "image_url": {"url": SNAPSHOT}},
                ],
            },
        )

    def test_snapshot_comes_before_attached_images(self) -> None:
        api = self.chat_api()
        content = api._prepare_message_content(vision_prompt(attached_images=[ATTACHED]))
        assert isinstance(content, list)
        self.assertEqual([part.get("image_url", {}).get("url") for part in content[1:]], [SNAPSHOT, ATTACHED])

    def test_vision_without_snapshot_sends_text_only(self) -> None:
        api = self.chat_api()
        self.assertEqual(api._prepare_message_content(vision_prompt(snapshot=None)), USER_TEXT)

    def test_snapshot_is_ignored_when_vision_is_off(self) -> None:
        api = self.chat_api()
        content = api._prepare_message_content(user_prompt(canvas_snapshot=SNAPSHOT))
        self.assertEqual(content, USER_TEXT)

    def test_non_image_snapshot_is_ignored(self) -> None:
        api = self.chat_api()
        self.assertEqual(api._prepare_message_content(vision_prompt(snapshot="javascript:alert(1)")), USER_TEXT)

    def test_only_base64_png_snapshots_are_sent(self) -> None:
        api = self.chat_api()
        for snapshot in (
            "data:image/svg+xml;charset=utf-8,%3Csvg%3E%3C%2Fsvg%3E",
            "data:image/png,rawbytes",
            "data:image/jpeg;base64,/9j/AAAA",
        ):
            with self.subTest(snapshot=snapshot):
                self.assertEqual(api._prepare_message_content(vision_prompt(snapshot=snapshot)), USER_TEXT)

    def test_snapshot_text_never_leaks_into_the_text_part(self) -> None:
        api = self.chat_api()
        content = api._prepare_message_content(vision_prompt())
        assert isinstance(content, list)
        self.assertNotIn(PNG_BASE64, content[0]["text"])

    def test_images_are_removed_from_history(self) -> None:
        api = self.chat_api()
        api._prepare_messages_for_request(vision_prompt())
        api._clean_conversation_history()
        self.assertIsInstance(api.messages[-1]["content"], str)


class TestJsonFormatVision(CanvasFormatEnv):
    canvas_format = "json"

    def test_legacy_json_path_sends_the_snapshot(self) -> None:
        api = self.chat_api()
        content = api._prepare_message_content(vision_prompt())
        assert isinstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "How long is AB?"})
        self.assertEqual(content[1], {"type": "image_url", "image_url": {"url": SNAPSHOT}})


class TestResponsesVision(CanvasFormatEnv):
    def test_snapshot_becomes_an_input_image(self) -> None:
        api = self.responses_api()
        api._prepare_messages_for_stream(vision_prompt())
        self.assertEqual(
            api._get_latest_user_message_for_input(),
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": USER_TEXT},
                        {"type": "input_image", "image_url": SNAPSHOT},
                    ],
                }
            ],
        )


class TestAnthropicVision(CanvasFormatEnv):
    def test_snapshot_becomes_a_base64_image_block(self) -> None:
        api = self.anthropic_api()
        message = api._parse_and_prepare_message(vision_prompt())
        assert message is not None
        api.messages.append(message)
        converted = api._convert_messages_to_anthropic()
        self.assertEqual(
            converted[-1],
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_TEXT},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64},
                    },
                ],
            },
        )


class TestLocalAgentVision(CanvasFormatEnv):
    def test_local_models_get_text_only(self) -> None:
        api = self.local_api()
        message = api._parse_and_prepare_message(vision_prompt(attached_images=[ATTACHED]))
        self.assertEqual(message, {"role": "user", "content": USER_TEXT})


class TestVisionRoutes(unittest.TestCase):
    """The routes accept the client snapshot inside the prompt JSON; no server-side capture."""

    def setUp(self) -> None:
        self._env = patch.dict(os.environ, {"REQUIRE_AUTH": "false"})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.app: MatHudFlask = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def _payload(self, snapshot: object) -> Dict[str, Any]:
        prompt = {
            "user_message": "what is drawn?",
            "use_vision": True,
            "canvas_snapshot": snapshot,
            # Not in MODEL_CONFIGS: OpenAI Chat Completions, so the mock below intercepts it
            "ai_model": "chat-completions-test-model",
        }
        return {"message": json.dumps(prompt)}

    def test_init_webdriver_route_is_gone(self) -> None:
        self.assertEqual(self.client.get("/init_webdriver").status_code, 404)

    @patch("static.openai_completions_api.OpenAIChatCompletionsAPI.create_chat_completion")
    def test_send_message_passes_the_snapshot_to_the_provider(self, mock_chat: Mock) -> None:
        mock_chat.return_value = Mock(message=Mock(content="ok", tool_calls=None), finish_reason="stop")
        response = self.client.post("/send_message", json=self._payload(SNAPSHOT))
        self.assertEqual(response.status_code, 200)
        sent_prompt = json.loads(mock_chat.call_args.args[0])
        self.assertEqual(sent_prompt["canvas_snapshot"], SNAPSHOT)
        self.assertTrue(sent_prompt["use_vision"])

    def test_oversized_snapshot_is_rejected(self) -> None:
        oversized = "data:image/png;base64," + "A" * MAX_IMAGE_BASE64_BYTES
        for route in ("/send_message", "/send_message_stream"):
            with self.subTest(route=route):
                response = self.client.post(route, json=self._payload(oversized))
                self.assertEqual(response.status_code, 400)
                self.assertIn("Canvas snapshot exceeds", json.loads(response.data)["message"])

    @patch("static.openai_completions_api.OpenAIChatCompletionsAPI.create_chat_completion_stream")
    @patch("static.openai_completions_api.OpenAIChatCompletionsAPI.create_chat_completion")
    def test_non_png_snapshot_is_rejected(self, mock_chat: Mock, mock_stream: Mock) -> None:
        for snapshot in (
            "data:image/svg+xml;charset=utf-8,%3Csvg%3E%3C%2Fsvg%3E",
            "data:image/png,rawbytes",
            "data:image/jpeg;base64,/9j/AAAA",
            "javascript:alert(1)",
        ):
            for route in ("/send_message", "/send_message_stream"):
                with self.subTest(snapshot=snapshot, route=route):
                    response = self.client.post(route, json=self._payload(snapshot))
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("base64 PNG", json.loads(response.data)["message"])
        mock_chat.assert_not_called()
        mock_stream.assert_not_called()

    @patch("static.openai_completions_api.OpenAIChatCompletionsAPI.create_chat_completion")
    def test_non_string_snapshot_is_rejected(self, mock_chat: Mock) -> None:
        response = self.client.post("/send_message", json=self._payload(42))
        self.assertEqual(response.status_code, 400)
        mock_chat.assert_not_called()

    @patch("static.openai_completions_api.OpenAIChatCompletionsAPI.create_chat_completion")
    def test_png_snapshot_is_accepted_by_both_routes(self, mock_chat: Mock) -> None:
        mock_chat.return_value = Mock(message=Mock(content="ok", tool_calls=None), finish_reason="stop")
        self.assertEqual(self.client.post("/send_message", json=self._payload(SNAPSHOT)).status_code, 200)
        self.assertIsNone(validate_canvas_snapshot(SNAPSHOT))
        self.assertIsNone(validate_canvas_snapshot(None))

    def test_snapshot_payload_is_a_real_png(self) -> None:
        # Guard the fixture itself: the data URL decodes to PNG bytes.
        self.assertTrue(base64.b64decode(PNG_BASE64).startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
