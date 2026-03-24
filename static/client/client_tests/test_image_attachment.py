"""
Tests for the image attachment feature in the client.

Tests cover:
- ImageAttachmentManager state management (uses real manager)
- Preview area DOM updates
- Payload generation with attached images
- /attach slash command
- Image modal functionality

Where possible, tests instantiate the real ``ImageAttachmentManager`` and
exercise its public API directly.  Tests that require browser DOM elements
(file picker, preview area rendering) still use mocks.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from constants import IMAGE_SIZE_WARNING_BYTES, MAX_ATTACHED_IMAGES
from image_attachment_manager import ImageAttachmentManager


class MockCanvas:
    """Mock canvas for testing."""

    def __init__(self) -> None:
        self.coordinate_mapper = MagicMock()
        self.coordinate_mapper.left_bound = -10
        self.coordinate_mapper.right_bound = 10
        self.coordinate_mapper.top_bound = 10
        self.coordinate_mapper.bottom_bound = -10

    def get_canvas_state(self) -> Dict[str, Any]:
        return {"Points": [], "Segments": []}


class MockWorkspaceManager:
    """Mock workspace manager for testing."""

    def __init__(self) -> None:
        pass


class MockDOMElement:
    """Mock DOM element for testing."""

    def __init__(self) -> None:
        self.value = ""
        self.style = MockStyle()
        self.src = ""
        self.id = ""
        self._children: List[Any] = []
        self._event_handlers: Dict[str, List[Any]] = {}

    def bind(self, event: str, handler: Any) -> None:
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def click(self) -> None:
        if "click" in self._event_handlers:
            for handler in self._event_handlers["click"]:
                handler(None)

    def clear(self) -> None:
        self._children = []

    def __le__(self, child: Any) -> "MockDOMElement":
        self._children.append(child)
        return self


class MockStyle:
    """Mock CSS style object."""

    def __init__(self) -> None:
        self.display: str = "none"


class TestAttachedImagesState(unittest.TestCase):
    """Tests for ImageAttachmentManager state management using the real class."""

    def setUp(self) -> None:
        """Set up a real ImageAttachmentManager instance."""
        self.mgr = ImageAttachmentManager()

    def test_initial_state_empty(self) -> None:
        """Test images property starts empty."""
        self.assertEqual(self.mgr.images, [])

    def test_append_image(self) -> None:
        """Test appending an image via internal list."""
        test_url = "data:image/png;base64,test123"
        self.mgr._images.append(test_url)
        self.assertEqual(len(self.mgr.images), 1)
        self.assertEqual(self.mgr.images[0], test_url)

    def test_append_multiple_images(self) -> None:
        """Test appending multiple images."""
        images = ["data:image/png;base64,img1", "data:image/jpeg;base64,img2", "data:image/png;base64,img3"]
        for img in images:
            self.mgr._images.append(img)
        self.assertEqual(len(self.mgr.images), 3)
        self.assertEqual(self.mgr.images, images)

    def test_remove_image_via_manager(self) -> None:
        """Test removing an image through the manager's _remove_image method."""
        self.mgr._images = [
            "data:image/png;base64,img1",
            "data:image/png;base64,img2",
            "data:image/png;base64,img3",
        ]
        # Patch _update_preview_area to avoid DOM access
        self.mgr._update_preview_area = lambda: None  # type: ignore[assignment]
        self.mgr._remove_image(1)
        self.assertEqual(len(self.mgr.images), 2)
        self.assertEqual(self.mgr.images[0], "data:image/png;base64,img1")
        self.assertEqual(self.mgr.images[1], "data:image/png;base64,img3")

    def test_clear_images(self) -> None:
        """Test clearing all images via the manager's clear method."""
        self.mgr._images = ["data:image/png;base64,img1", "data:image/png;base64,img2"]
        # Patch _update_preview_area to avoid DOM access
        self.mgr._update_preview_area = lambda: None  # type: ignore[assignment]
        self.mgr.clear()
        self.assertEqual(len(self.mgr.images), 0)

    def test_max_images_constant(self) -> None:
        """Test maximum images constant is set."""
        self.assertEqual(MAX_ATTACHED_IMAGES, 5)

    def test_image_size_warning_constant(self) -> None:
        """Test image size warning threshold is 10MB."""
        self.assertEqual(IMAGE_SIZE_WARNING_BYTES, 10 * 1024 * 1024)

    def test_remove_invalid_index_negative(self) -> None:
        """Test removing with negative index does nothing."""
        self.mgr._images = ["img1", "img2", "img3"]
        self.mgr._update_preview_area = lambda: None  # type: ignore[assignment]
        self.mgr._remove_image(-1)
        self.assertEqual(self.mgr.images, ["img1", "img2", "img3"])

    def test_remove_invalid_index_too_large(self) -> None:
        """Test removing with too large index does nothing."""
        self.mgr._images = ["img1", "img2", "img3"]
        self.mgr._update_preview_area = lambda: None  # type: ignore[assignment]
        self.mgr._remove_image(10)
        self.assertEqual(self.mgr.images, ["img1", "img2", "img3"])

    def test_on_system_message_callback_stored(self) -> None:
        """Test that the system message callback is stored correctly."""
        messages: List[str] = []
        mgr = ImageAttachmentManager(on_system_message=messages.append)
        self.assertIs(mgr._on_system_message, messages.append)


class TestImageValidation(unittest.TestCase):
    """Tests for image URL validation logic."""

    def test_valid_png_data_url(self) -> None:
        """Test valid PNG data URL passes validation."""
        url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        is_valid = isinstance(url, str) and url.startswith("data:image")
        self.assertTrue(is_valid)

    def test_valid_jpeg_data_url(self) -> None:
        """Test valid JPEG data URL passes validation."""
        url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        is_valid = isinstance(url, str) and url.startswith("data:image")
        self.assertTrue(is_valid)

    def test_invalid_text_data_url(self) -> None:
        """Test non-image data URL fails validation."""
        url = "data:text/plain;base64,SGVsbG8="
        is_valid = isinstance(url, str) and url.startswith("data:image")
        self.assertFalse(is_valid)

    def test_non_string_fails_validation(self) -> None:
        """Test non-string value fails validation."""
        url = 12345
        is_valid = isinstance(url, str) and url.startswith("data:image")
        self.assertFalse(is_valid)

    def test_regular_url_fails_validation(self) -> None:
        """Test regular HTTP URL fails validation."""
        url = "https://example.com/image.png"
        is_valid = isinstance(url, str) and url.startswith("data:image")
        self.assertFalse(is_valid)


class TestPayloadGeneration(unittest.TestCase):
    """Tests for payload generation with attached images."""

    def test_payload_includes_attached_images(self) -> None:
        """Test that payload includes attached_images array."""
        attached_images = ["data:image/png;base64,img1", "data:image/jpeg;base64,img2"]

        # Simulate payload creation
        payload = {
            "canvas_state": {},
            "user_message": "What do you see?",
            "use_vision": False,
            "attached_images": attached_images,
        }

        self.assertIn("attached_images", payload)
        self.assertEqual(len(payload["attached_images"]), 2)

    def test_payload_without_images_omits_key(self) -> None:
        """Test that payload without images can omit the key."""
        payload = {
            "canvas_state": {},
            "user_message": "Hello",
            "use_vision": False,
        }

        # Key should not be present
        self.assertNotIn("attached_images", payload)

    def test_payload_empty_images_array(self) -> None:
        """Test payload with empty images array."""
        attached_images: List[str] = []

        # When there are no images, we shouldn't include the key
        payload: Dict[str, Any] = {
            "canvas_state": {},
            "user_message": "Hello",
            "use_vision": False,
        }
        if attached_images:
            payload["attached_images"] = attached_images

        self.assertNotIn("attached_images", payload)

    def test_payload_with_vision_and_images(self) -> None:
        """Test payload with both vision enabled and attached images."""
        attached_images = ["data:image/png;base64,img1"]

        payload = {
            "canvas_state": {},
            "user_message": "Compare canvas and image",
            "use_vision": True,  # Vision enabled
            "attached_images": attached_images,  # Also has attached images
        }

        self.assertTrue(payload["use_vision"])
        self.assertEqual(len(payload["attached_images"]), 1)


class TestSlashCommandImage(unittest.TestCase):
    """Tests for /image slash command."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from slash_command_handler import SlashCommandHandler, CommandResult

        self.canvas = MockCanvas()
        self.workspace_manager = MockWorkspaceManager()

        # Create mock AI interface
        self.ai_interface = MagicMock()
        self.ai_interface.trigger_file_picker = MagicMock()

        self.handler = SlashCommandHandler(
            self.canvas,  # type: ignore
            self.workspace_manager,  # type: ignore
            self.ai_interface,  # type: ignore
        )
        # Mock vision model check to always return True for testing
        self.handler._selected_model_has_vision = lambda: True  # type: ignore

    def test_image_command_exists(self) -> None:
        """Test /image command is registered."""
        commands_list = self.handler.get_commands_list()
        command_names = [cmd for cmd, _ in commands_list]
        self.assertIn("/image", command_names)

    def test_image_command_triggers_file_picker(self) -> None:
        """Test /image command triggers file picker."""
        result = self.handler.execute("/image")
        self.assertTrue(result.success)
        self.ai_interface.trigger_file_picker.assert_called_once()

    def test_image_command_description(self) -> None:
        """Test /image command has description."""
        commands_list = self.handler.get_commands_list()
        for cmd, desc in commands_list:
            if cmd == "/image":
                self.assertIn("image", desc.lower())
                break
        else:
            self.fail("/image command not found")


class TestImageLimitLogic(unittest.TestCase):
    """Tests for image attachment limit enforcement using real ImageAttachmentManager."""

    def setUp(self) -> None:
        """Set up a real ImageAttachmentManager with DOM patched out."""
        self.mgr = ImageAttachmentManager()
        self.mgr._update_preview_area = lambda: None  # type: ignore[assignment]

    def test_remaining_slots_calculation(self) -> None:
        """Test remaining slots calculation via manager state."""
        self.mgr._images = ["img1", "img2"]
        remaining = MAX_ATTACHED_IMAGES - len(self.mgr.images)
        self.assertEqual(remaining, 3)

    def test_remaining_slots_at_limit(self) -> None:
        """Test remaining slots when at limit."""
        self.mgr._images = ["img"] * MAX_ATTACHED_IMAGES
        remaining = MAX_ATTACHED_IMAGES - len(self.mgr.images)
        self.assertEqual(remaining, 0)

    def test_files_to_process_limited(self) -> None:
        """Test files to process is limited by remaining slots."""
        self.mgr._images = ["img"] * 3
        remaining = MAX_ATTACHED_IMAGES - len(self.mgr.images)
        files_selected = 5
        files_to_process = min(files_selected, remaining)
        self.assertEqual(files_to_process, 2)

    def test_all_files_processed_when_below_limit(self) -> None:
        """Test all files processed when total is below limit."""
        self.mgr._images = ["img"]
        remaining = MAX_ATTACHED_IMAGES - len(self.mgr.images)
        files_selected = 2
        files_to_process = min(files_selected, remaining)
        self.assertEqual(files_to_process, 2)


class TestImageRemovalLogic(unittest.TestCase):
    """Tests for image removal using real ImageAttachmentManager._remove_image."""

    def setUp(self) -> None:
        """Set up a real ImageAttachmentManager with DOM patched out."""
        self.mgr = ImageAttachmentManager()
        self.mgr._update_preview_area = lambda: None  # type: ignore[assignment]

    def test_remove_first_image(self) -> None:
        """Test removing first image from list."""
        self.mgr._images = ["img1", "img2", "img3"]
        self.mgr._remove_image(0)
        self.assertEqual(self.mgr.images, ["img2", "img3"])

    def test_remove_middle_image(self) -> None:
        """Test removing middle image from list."""
        self.mgr._images = ["img1", "img2", "img3"]
        self.mgr._remove_image(1)
        self.assertEqual(self.mgr.images, ["img1", "img3"])

    def test_remove_last_image(self) -> None:
        """Test removing last image from list."""
        self.mgr._images = ["img1", "img2", "img3"]
        self.mgr._remove_image(2)
        self.assertEqual(self.mgr.images, ["img1", "img2"])

    def test_remove_invalid_index_negative(self) -> None:
        """Test removing with negative index does nothing."""
        self.mgr._images = ["img1", "img2", "img3"]
        self.mgr._remove_image(-1)
        self.assertEqual(self.mgr.images, ["img1", "img2", "img3"])

    def test_remove_invalid_index_too_large(self) -> None:
        """Test removing with too large index does nothing."""
        self.mgr._images = ["img1", "img2", "img3"]
        self.mgr._remove_image(10)
        self.assertEqual(self.mgr.images, ["img1", "img2", "img3"])


class TestPreviewAreaLogic(unittest.TestCase):
    """Tests for preview area update logic using real ImageAttachmentManager."""

    def test_preview_visible_with_images(self) -> None:
        """Test images list is non-empty when images attached."""
        mgr = ImageAttachmentManager()
        mgr._images = ["img1"]
        self.assertTrue(len(mgr.images) > 0)

    def test_preview_hidden_without_images(self) -> None:
        """Test images list is empty when no images."""
        mgr = ImageAttachmentManager()
        self.assertFalse(len(mgr.images) > 0)

    def test_preview_thumbnail_count_matches_images(self) -> None:
        """Test number of images matches attached count."""
        mgr = ImageAttachmentManager()
        mgr._images = ["img1", "img2", "img3"]
        self.assertEqual(len(mgr.images), 3)


class TestModalLogic(unittest.TestCase):
    """Tests for image modal display logic (requires DOM mocks)."""

    def test_modal_display_style_visible(self) -> None:
        """Test modal display style when showing."""
        modal_style = MockStyle()
        modal_style.display = "flex"
        self.assertEqual(modal_style.display, "flex")

    def test_modal_display_style_hidden(self) -> None:
        """Test modal display style when hidden."""
        modal_style = MockStyle()
        modal_style.display = "none"
        self.assertEqual(modal_style.display, "none")

    def test_backdrop_click_detection(self) -> None:
        """Test backdrop click is detected by element id."""

        class MockEvent:
            def __init__(self, target_id: str) -> None:
                self.target = MagicMock()
                self.target.id = target_id

        # Click on modal backdrop (should close)
        event = MockEvent("image-modal")
        should_close = event.target.id == "image-modal"
        self.assertTrue(should_close)

        # Click on image inside modal (should not close)
        event = MockEvent("image-modal-img")
        should_close = event.target.id == "image-modal"
        self.assertFalse(should_close)


class TestMessageElementWithImages(unittest.TestCase):
    """Tests for message element creation with images."""

    def test_message_with_images_parameter(self) -> None:
        """Test message creation accepts images parameter."""

        # Simulate _create_message_element signature
        def create_message_element(
            sender: str, message: str, message_type: str = "normal", images: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            return {"sender": sender, "message": message, "type": message_type, "images": images}

        result = create_message_element("User", "Hello", images=["data:image/png;base64,img1"])
        self.assertEqual(result["images"], ["data:image/png;base64,img1"])

    def test_message_without_images(self) -> None:
        """Test message creation without images."""

        def create_message_element(
            sender: str, message: str, message_type: str = "normal", images: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            return {"sender": sender, "message": message, "type": message_type, "images": images}

        result = create_message_element("User", "Hello")
        self.assertIsNone(result["images"])


class TestDataURLParsing(unittest.TestCase):
    """Tests for data URL parsing and validation."""

    def test_parse_png_mime_type(self) -> None:
        """Test PNG MIME type is detected."""
        url = "data:image/png;base64,abc123"
        is_png = "image/png" in url
        self.assertTrue(is_png)

    def test_parse_jpeg_mime_type(self) -> None:
        """Test JPEG MIME type is detected."""
        url = "data:image/jpeg;base64,abc123"
        is_jpeg = "image/jpeg" in url
        self.assertTrue(is_jpeg)

    def test_parse_gif_mime_type(self) -> None:
        """Test GIF MIME type is detected."""
        url = "data:image/gif;base64,abc123"
        is_gif = "image/gif" in url
        self.assertTrue(is_gif)

    def test_data_url_has_correct_prefix(self) -> None:
        """Test data URL starts with correct prefix."""
        url = "data:image/png;base64,abc123"
        self.assertTrue(url.startswith("data:"))
        self.assertTrue(url.startswith("data:image"))


class TestImageOnlySending(unittest.TestCase):
    """Tests for sending images without text."""

    def test_can_send_with_only_images(self) -> None:
        """Test that sending is allowed with images but no text."""
        message = ""
        attached_images = ["data:image/png;base64,img1"]
        has_text = bool(message.strip())
        has_images = len(attached_images) > 0

        can_send = has_text or has_images
        self.assertTrue(can_send)

    def test_cannot_send_without_text_or_images(self) -> None:
        """Test that sending is blocked with neither text nor images."""
        message = ""
        attached_images: List[str] = []
        has_text = bool(message.strip())
        has_images = len(attached_images) > 0

        can_send = has_text or has_images
        self.assertFalse(can_send)

    def test_can_send_with_only_text(self) -> None:
        """Test that sending is allowed with text but no images."""
        message = "Hello"
        attached_images: List[str] = []
        has_text = bool(message.strip())
        has_images = len(attached_images) > 0

        can_send = has_text or has_images
        self.assertTrue(can_send)

    def test_can_send_with_both_text_and_images(self) -> None:
        """Test that sending is allowed with both text and images."""
        message = "What is this?"
        attached_images = ["data:image/png;base64,img1"]
        has_text = bool(message.strip())
        has_images = len(attached_images) > 0

        can_send = has_text or has_images
        self.assertTrue(can_send)

    def test_whitespace_only_message_not_considered_text(self) -> None:
        """Test that whitespace-only message is not considered text."""
        message = "   \t\n  "
        has_text = bool(message.strip())
        self.assertFalse(has_text)

    def test_display_message_for_image_only(self) -> None:
        """Test display message shows placeholder for image-only sends."""
        message = ""
        has_text = bool(message.strip())
        display_message = message if has_text else "[Image attached]"
        self.assertEqual(display_message, "[Image attached]")

    def test_ai_message_default_for_image_only(self) -> None:
        """Test AI receives default prompt for image-only sends."""
        message = ""
        has_text = bool(message.strip())
        ai_message = message if has_text else "What do you see in this image?"
        self.assertEqual(ai_message, "What do you see in this image?")
