"""
Tests for the image attachment feature.

Tests server-side handling of attached images in:
- Route payload extraction
- OpenAI API base content preparation
- OpenAI Responses API content conversion
"""

from __future__ import annotations

import base64
import json
import os
import unittest
from unittest.mock import Mock, patch

from static.openai_api_base import OpenAIAPIBase
from static.openai_responses_api import OpenAIResponsesAPI
from static.routes import extract_vision_payload


# Create a small test image as base64 (1x1 red pixel PNG)
TEST_IMAGE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)
TEST_IMAGE_DATA_URL = f"data:image/png;base64,{base64.b64encode(TEST_IMAGE_PNG_BYTES).decode()}"


class TestExtractVisionPayload(unittest.TestCase):
    """Tests for extract_vision_payload function in routes.py."""

    def test_extract_empty_payload(self) -> None:
        """Test extracting from empty payload returns all None."""
        svg_state, canvas_image, renderer_mode, attached_images = extract_vision_payload({})
        self.assertIsNone(svg_state)
        self.assertIsNone(canvas_image)
        self.assertIsNone(renderer_mode)
        self.assertIsNone(attached_images)

    def test_extract_svg_state(self) -> None:
        """Test extracting svg_state from payload."""
        payload = {"svg_state": {"elements": []}}
        svg_state, _, _, _ = extract_vision_payload(payload)
        self.assertEqual(svg_state, {"elements": []})

    def test_extract_renderer_mode(self) -> None:
        """Test extracting renderer_mode from payload."""
        payload = {"renderer_mode": "svg"}
        _, _, renderer_mode, _ = extract_vision_payload(payload)
        self.assertEqual(renderer_mode, "svg")

    def test_extract_canvas_image_from_vision_snapshot(self) -> None:
        """Test extracting canvas_image from nested vision_snapshot."""
        payload = {"vision_snapshot": {"canvas_image": TEST_IMAGE_DATA_URL}}
        _, canvas_image, _, _ = extract_vision_payload(payload)
        self.assertEqual(canvas_image, TEST_IMAGE_DATA_URL)

    def test_extract_svg_state_from_vision_snapshot(self) -> None:
        """Test extracting svg_state from nested vision_snapshot."""
        payload = {"vision_snapshot": {"svg_state": {"elements": ["rect", "circle"]}}}
        svg_state, _, _, _ = extract_vision_payload(payload)
        self.assertEqual(svg_state, {"elements": ["rect", "circle"]})

    def test_vision_snapshot_overrides_top_level(self) -> None:
        """Test that vision_snapshot values override top-level values."""
        payload = {
            "svg_state": {"elements": ["old"]},
            "renderer_mode": "canvas2d",
            "vision_snapshot": {"svg_state": {"elements": ["new"]}, "renderer_mode": "webgl"},
        }
        svg_state, _, renderer_mode, _ = extract_vision_payload(payload)
        self.assertEqual(svg_state, {"elements": ["new"]})
        self.assertEqual(renderer_mode, "webgl")

    def test_invalid_types_ignored(self) -> None:
        """Test that invalid types for fields are ignored."""
        payload = {
            "svg_state": "not a dict",
            "renderer_mode": 123,
        }
        svg_state, _, renderer_mode, _ = extract_vision_payload(payload)
        self.assertIsNone(svg_state)
        self.assertIsNone(renderer_mode)

    def test_extract_attached_images(self) -> None:
        """Test extracting attached_images from payload."""
        payload = {"attached_images": ["data:image/png;base64,img1", "data:image/jpeg;base64,img2"]}
        _, _, _, attached_images = extract_vision_payload(payload)
        self.assertEqual(len(attached_images), 2)
        self.assertIn("data:image/png;base64,img1", attached_images)
        self.assertIn("data:image/jpeg;base64,img2", attached_images)

    def test_extract_attached_images_filters_non_strings(self) -> None:
        """Test that non-string items are filtered from attached_images."""
        payload = {"attached_images": ["data:image/png;base64,valid", 123, None, {"invalid": "object"}]}
        _, _, _, attached_images = extract_vision_payload(payload)
        self.assertEqual(len(attached_images), 1)
        self.assertEqual(attached_images[0], "data:image/png;base64,valid")

    def test_extract_attached_images_empty_list(self) -> None:
        """Test extracting empty attached_images list."""
        payload = {"attached_images": []}
        _, _, _, attached_images = extract_vision_payload(payload)
        self.assertEqual(attached_images, [])

    def test_extract_attached_images_invalid_type(self) -> None:
        """Test that non-list attached_images is ignored."""
        payload = {"attached_images": "not a list"}
        _, _, _, attached_images = extract_vision_payload(payload)
        self.assertIsNone(attached_images)


class TestPrepareMessageContent(unittest.TestCase):
    """Tests for _prepare_message_content in OpenAIAPIBase."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.original_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-api-key"

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.original_api_key:
            os.environ["OPENAI_API_KEY"] = self.original_api_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    @patch("static.openai_api_base.OpenAI")
    def test_plain_text_prompt_returns_original(self, mock_openai: Mock) -> None:
        """Test plain text prompt returns unchanged."""
        api = OpenAIAPIBase()
        result = api._prepare_message_content("plain text message")
        self.assertEqual(result, "plain text message")

    @patch("static.openai_api_base.OpenAI")
    def test_no_vision_no_images_returns_original(self, mock_openai: Mock) -> None:
        """Test JSON prompt with no vision and no images returns original."""
        api = OpenAIAPIBase()
        prompt = json.dumps({"user_message": "test", "use_vision": False})
        result = api._prepare_message_content(prompt)
        self.assertEqual(result, prompt)

    @patch("static.openai_api_base.OpenAI")
    def test_attached_images_without_vision(self, mock_openai: Mock) -> None:
        """Test that attached images are processed even without vision toggle."""
        api = OpenAIAPIBase()
        prompt = json.dumps(
            {"user_message": "What is this image?", "use_vision": False, "attached_images": [TEST_IMAGE_DATA_URL]}
        )
        result = api._prepare_message_content(prompt)

        # Should return a list (enhanced prompt) even without vision
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # text + 1 image
        self.assertEqual(result[0]["type"], "text")
        self.assertEqual(result[0]["text"], "What is this image?")
        self.assertEqual(result[1]["type"], "image_url")
        self.assertEqual(result[1]["image_url"]["url"], TEST_IMAGE_DATA_URL)

    @patch("static.openai_api_base.OpenAI")
    def test_multiple_attached_images(self, mock_openai: Mock) -> None:
        """Test multiple attached images are all included."""
        api = OpenAIAPIBase()
        images = ["data:image/png;base64,img1", "data:image/jpeg;base64,img2", "data:image/png;base64,img3"]
        prompt = json.dumps({"user_message": "Compare these images", "use_vision": False, "attached_images": images})
        result = api._prepare_message_content(prompt)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)  # text + 3 images

        # Check all images are present
        image_urls = [part["image_url"]["url"] for part in result if part.get("type") == "image_url"]
        self.assertEqual(len(image_urls), 3)
        for img in images:
            self.assertIn(img, image_urls)

    @patch("static.openai_api_base.OpenAI")
    def test_invalid_attached_images_filtered(self, mock_openai: Mock) -> None:
        """Test that invalid image data URLs are filtered out."""
        api = OpenAIAPIBase()
        prompt = json.dumps(
            {
                "user_message": "test",
                "use_vision": False,
                "attached_images": [
                    "data:image/png;base64,valid",
                    "not-a-data-url",
                    123,  # Not a string
                    "data:text/plain;base64,notimage",  # Wrong MIME type
                ],
            }
        )
        result = api._prepare_message_content(prompt)

        self.assertIsInstance(result, list)
        # Only valid image should be included
        image_parts = [part for part in result if part.get("type") == "image_url"]
        self.assertEqual(len(image_parts), 1)
        self.assertEqual(image_parts[0]["image_url"]["url"], "data:image/png;base64,valid")

    @patch("static.openai_api_base.OpenAI")
    def test_empty_attached_images_returns_original(self, mock_openai: Mock) -> None:
        """Test empty attached_images array without vision returns original."""
        api = OpenAIAPIBase()
        prompt = json.dumps({"user_message": "test", "use_vision": False, "attached_images": []})
        result = api._prepare_message_content(prompt)
        # Empty images array with no vision should return original
        self.assertEqual(result, prompt)


class TestCreateEnhancedPromptWithImage(unittest.TestCase):
    """Tests for _create_enhanced_prompt_with_image in OpenAIAPIBase."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.original_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-api-key"

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.original_api_key:
            os.environ["OPENAI_API_KEY"] = self.original_api_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    @patch("static.openai_api_base.OpenAI")
    def test_attached_images_only(self, mock_openai: Mock) -> None:
        """Test creating enhanced prompt with only attached images."""
        api = OpenAIAPIBase()
        images = [TEST_IMAGE_DATA_URL]
        result = api._create_enhanced_prompt_with_image(
            user_message="Describe this", attached_images=images, include_canvas_snapshot=False
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # text + 1 image
        self.assertEqual(result[0]["type"], "text")
        self.assertEqual(result[0]["text"], "Describe this")
        self.assertEqual(result[1]["type"], "image_url")

    @patch("static.openai_api_base.OpenAI")
    def test_no_images_returns_none(self, mock_openai: Mock) -> None:
        """Test returns None when no images available."""
        api = OpenAIAPIBase()
        result = api._create_enhanced_prompt_with_image(
            user_message="Hello", attached_images=None, include_canvas_snapshot=False
        )
        self.assertIsNone(result)

    @patch("static.openai_api_base.OpenAI")
    def test_empty_images_returns_none(self, mock_openai: Mock) -> None:
        """Test returns None with empty images list."""
        api = OpenAIAPIBase()
        result = api._create_enhanced_prompt_with_image(
            user_message="Hello", attached_images=[], include_canvas_snapshot=False
        )
        self.assertIsNone(result)


class TestConvertContentForResponsesAPI(unittest.TestCase):
    """Tests for _convert_content_for_responses_api in OpenAIResponsesAPI."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.original_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-api-key"

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.original_api_key:
            os.environ["OPENAI_API_KEY"] = self.original_api_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    @patch("static.openai_api_base.OpenAI")
    def test_string_content_unchanged(self, mock_openai: Mock) -> None:
        """Test string content passes through unchanged."""
        api = OpenAIResponsesAPI()
        result = api._convert_content_for_responses_api("plain text")
        self.assertEqual(result, "plain text")

    @patch("static.openai_api_base.OpenAI")
    def test_convert_text_type(self, mock_openai: Mock) -> None:
        """Test 'text' type converted to 'input_text'."""
        api = OpenAIResponsesAPI()
        content = [{"type": "text", "text": "Hello"}]
        result = api._convert_content_for_responses_api(content)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "input_text")
        self.assertEqual(result[0]["text"], "Hello")

    @patch("static.openai_api_base.OpenAI")
    def test_convert_image_url_type(self, mock_openai: Mock) -> None:
        """Test 'image_url' type converted to 'input_image'."""
        api = OpenAIResponsesAPI()
        content = [{"type": "image_url", "image_url": {"url": TEST_IMAGE_DATA_URL}}]
        result = api._convert_content_for_responses_api(content)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "input_image")
        self.assertEqual(result[0]["image_url"], TEST_IMAGE_DATA_URL)

    @patch("static.openai_api_base.OpenAI")
    def test_convert_mixed_content(self, mock_openai: Mock) -> None:
        """Test converting mixed text and image content."""
        api = OpenAIResponsesAPI()
        content = [
            {"type": "text", "text": "Look at this:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "What do you see?"},
        ]
        result = api._convert_content_for_responses_api(content)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], {"type": "input_text", "text": "Look at this:"})
        self.assertEqual(result[1], {"type": "input_image", "image_url": "data:image/png;base64,abc"})
        self.assertEqual(result[2], {"type": "input_text", "text": "What do you see?"})

    @patch("static.openai_api_base.OpenAI")
    def test_unknown_type_preserved(self, mock_openai: Mock) -> None:
        """Test unknown content types are preserved."""
        api = OpenAIResponsesAPI()
        content = [{"type": "custom", "data": "value"}]
        result = api._convert_content_for_responses_api(content)

        self.assertEqual(result, content)

    @patch("static.openai_api_base.OpenAI")
    def test_non_dict_items_preserved(self, mock_openai: Mock) -> None:
        """Test non-dict items in list are preserved."""
        api = OpenAIResponsesAPI()
        content = ["string item", 123]
        result = api._convert_content_for_responses_api(content)

        self.assertEqual(result, content)


class TestRemoveImagesFromUserMessages(unittest.TestCase):
    """Tests for _remove_images_from_user_messages in OpenAIAPIBase."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.original_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-api-key"

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.original_api_key:
            os.environ["OPENAI_API_KEY"] = self.original_api_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    @patch("static.openai_api_base.OpenAI")
    def test_removes_image_content(self, mock_openai: Mock) -> None:
        """Test images are removed from user messages."""
        api = OpenAIAPIBase()
        api.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "test message"},
                    {"type": "image_url", "image_url": {"url": TEST_IMAGE_DATA_URL}},
                ],
            }
        )

        api._remove_images_from_user_messages()

        # Content should now be just the text string
        self.assertEqual(api.messages[-1]["content"], "test message")

    @patch("static.openai_api_base.OpenAI")
    def test_preserves_string_content(self, mock_openai: Mock) -> None:
        """Test string content is preserved unchanged."""
        api = OpenAIAPIBase()
        api.messages.append({"role": "user", "content": "simple text"})

        api._remove_images_from_user_messages()

        self.assertEqual(api.messages[-1]["content"], "simple text")


class TestPreviousResponseId(unittest.TestCase):
    """Tests for previous_response_id functionality in OpenAIResponsesAPI."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.original_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-api-key"

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.original_api_key:
            os.environ["OPENAI_API_KEY"] = self.original_api_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    @patch("static.openai_api_base.OpenAI")
    def test_initial_previous_response_id_is_none(self, mock_openai: Mock) -> None:
        """Test that _previous_response_id starts as None."""
        api = OpenAIResponsesAPI()
        self.assertIsNone(api._previous_response_id)

    @patch("static.openai_api_base.OpenAI")
    def test_reset_conversation_clears_previous_response_id(self, mock_openai: Mock) -> None:
        """Test that reset_conversation clears the previous_response_id."""
        api = OpenAIResponsesAPI()
        api._previous_response_id = "resp_test123"

        api.reset_conversation()

        self.assertIsNone(api._previous_response_id)

    @patch("static.openai_api_base.OpenAI")
    def test_is_regular_message_turn_with_user_message(self, mock_openai: Mock) -> None:
        """Test _is_regular_message_turn returns True for user messages."""
        api = OpenAIResponsesAPI()
        api.messages.append({"role": "user", "content": "Hello"})

        self.assertTrue(api._is_regular_message_turn())

    @patch("static.openai_api_base.OpenAI")
    def test_is_regular_message_turn_with_tool_result(self, mock_openai: Mock) -> None:
        """Test _is_regular_message_turn returns False for tool results."""
        api = OpenAIResponsesAPI()
        api.messages.append({"role": "tool", "tool_call_id": "call_123", "content": "result"})

        self.assertFalse(api._is_regular_message_turn())

    @patch("static.openai_api_base.OpenAI")
    def test_is_regular_message_turn_empty_messages(self, mock_openai: Mock) -> None:
        """Test _is_regular_message_turn returns False for empty messages."""
        api = OpenAIResponsesAPI()
        api.messages = []

        self.assertFalse(api._is_regular_message_turn())

    @patch("static.openai_api_base.OpenAI")
    def test_get_latest_user_message_for_input(self, mock_openai: Mock) -> None:
        """Test _get_latest_user_message_for_input extracts last user message."""
        api = OpenAIResponsesAPI()
        api.messages.append({"role": "developer", "content": "system msg"})
        api.messages.append({"role": "user", "content": "first question"})
        api.messages.append({"role": "assistant", "content": "answer"})
        api.messages.append({"role": "user", "content": "second question"})

        result = api._get_latest_user_message_for_input()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[0]["content"], "second question")

    @patch("static.openai_api_base.OpenAI")
    def test_get_latest_user_message_converts_content(self, mock_openai: Mock) -> None:
        """Test _get_latest_user_message_for_input converts content format."""
        api = OpenAIResponsesAPI()
        api.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        )

        result = api._get_latest_user_message_for_input()

        self.assertEqual(len(result), 1)
        content = result[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[1]["type"], "input_image")

    @patch("static.openai_api_base.OpenAI")
    def test_get_latest_user_message_empty_returns_empty(self, mock_openai: Mock) -> None:
        """Test _get_latest_user_message_for_input returns empty for no messages."""
        api = OpenAIResponsesAPI()
        api.messages = []

        result = api._get_latest_user_message_for_input()

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
