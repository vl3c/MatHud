"""Tests for login rate limiting, request-size limits, and image validation.

Covers the server-side security hardening added to the authentication flow and
the AI message endpoints:

    - Correct/incorrect PIN handling with constant-time comparison.
    - Per-IP failed-attempt cooldown keyed on the real socket peer (X-Forwarded-For
      spoofing must not reset or bypass the limit).
    - A global failed-attempt ceiling that locks out even fresh IPs.
    - Server-side attached-image validation (count and per-image size caps).
    - MAX_CONTENT_LENGTH configuration and the JSON 413 error handler.
"""

from __future__ import annotations

import json
import os
import unittest
from typing import List, Optional
from unittest.mock import Mock, patch

from werkzeug.test import TestResponse

from static.app_manager import AppManager, MatHudFlask
from static.config import MAX_ATTACHED_IMAGES, MAX_CONTENT_LENGTH_BYTES, MAX_IMAGE_BASE64_BYTES
from static.routes import GLOBAL_FAILED_ATTEMPTS_LIMIT, reset_login_rate_limit_state


TEST_PIN = "123456"


class TestLoginRateLimiting(unittest.TestCase):
    """Exercise the /login handler's authentication and rate-limiting logic."""

    def setUp(self) -> None:
        self._original_require_auth: Optional[str] = os.environ.get("REQUIRE_AUTH")
        self._original_auth_pin: Optional[str] = os.environ.get("AUTH_PIN")
        os.environ["REQUIRE_AUTH"] = "true"
        os.environ["AUTH_PIN"] = TEST_PIN

        self.app: MatHudFlask = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Ensure no leaked rate-limit state from other tests.
        reset_login_rate_limit_state()

    def tearDown(self) -> None:
        reset_login_rate_limit_state()
        self._restore_env("REQUIRE_AUTH", self._original_require_auth)
        self._restore_env("AUTH_PIN", self._original_auth_pin)

    @staticmethod
    def _restore_env(name: str, value: Optional[str]) -> None:
        if value is not None:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)

    def _post_login(self, pin: str, remote_addr: str = "127.0.0.1", xff: Optional[str] = None) -> TestResponse:
        headers = {"X-Forwarded-For": xff} if xff is not None else None
        return self.client.post(
            "/login",
            data={"pin": pin},
            headers=headers,
            environ_base={"REMOTE_ADDR": remote_addr},
        )

    def test_correct_pin_succeeds(self) -> None:
        response = self._post_login(TEST_PIN)
        self.assertEqual(response.status_code, 302)  # redirect to index
        with self.client.session_transaction() as sess:
            self.assertTrue(sess.get("authenticated"))

    def test_wrong_pin_fails(self) -> None:
        response = self._post_login("000000")
        self.assertEqual(response.status_code, 200)  # re-render login, not a redirect
        with self.client.session_transaction() as sess:
            self.assertFalse(sess.get("authenticated"))

    def test_per_ip_cooldown_triggers_after_failure(self) -> None:
        # First failure records a timestamp and re-renders the login page.
        first = self._post_login("000000")
        self.assertEqual(first.status_code, 200)

        # Second failure within the cooldown window is rejected with 429.
        second = self._post_login("000000")
        self.assertEqual(second.status_code, 429)

    def test_x_forwarded_for_does_not_bypass_per_ip_limit(self) -> None:
        # Same socket peer (REMOTE_ADDR), different spoofed XFF values on each request.
        self.assertEqual(self._post_login("000000", xff="10.0.0.1").status_code, 200)

        # Varying the client-controlled header must NOT reset the per-IP cooldown,
        # because limiting keys on REMOTE_ADDR, not X-Forwarded-For.
        self.assertEqual(self._post_login("000000", xff="10.0.0.2").status_code, 429)
        self.assertEqual(self._post_login("000000", xff="203.0.113.9").status_code, 429)

    def test_global_ceiling_locks_out_fresh_ip(self) -> None:
        # Drive the global counter to its ceiling using a distinct IP per request so
        # the per-IP cooldown never engages.
        for i in range(GLOBAL_FAILED_ATTEMPTS_LIMIT):
            response = self._post_login("000000", remote_addr=f"198.51.100.{i}")
            self.assertEqual(response.status_code, 200)

        # A brand-new, never-seen IP is now locked out purely by the global ceiling.
        locked = self._post_login("000000", remote_addr="192.0.2.55")
        self.assertEqual(locked.status_code, 429)

        # Even a correct PIN is rejected while the global lockout is active.
        locked_correct = self._post_login(TEST_PIN, remote_addr="192.0.2.77")
        self.assertEqual(locked_correct.status_code, 429)


class TestImageAndSizeHardening(unittest.TestCase):
    """Exercise server-side image validation and request-size configuration."""

    SAMPLE_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAwMB/aqVw0sAAAAASUVORK5CYII="

    def setUp(self) -> None:
        self._original_require_auth: Optional[str] = os.environ.get("REQUIRE_AUTH")
        os.environ["REQUIRE_AUTH"] = "false"

        self.app = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self._original_require_auth is not None:
            os.environ["REQUIRE_AUTH"] = self._original_require_auth
        else:
            os.environ.pop("REQUIRE_AUTH", None)

    def _small_image(self) -> str:
        return f"data:image/png;base64,{self.SAMPLE_PNG_BASE64}"

    def _post_message(self, images: List[str]) -> TestResponse:
        payload = {"message": json.dumps({"user_message": "hi", "attached_images": images})}
        return self.client.post("/send_message", json=payload)

    def test_too_many_images_rejected(self) -> None:
        images = [self._small_image()] * (MAX_ATTACHED_IMAGES + 1)
        response = self._post_message(images)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "error")
        self.assertIn("Too many attached images", data["message"])

    def test_oversized_single_image_rejected(self) -> None:
        oversized = "data:image/png;base64," + ("A" * (MAX_IMAGE_BASE64_BYTES + 1))
        response = self._post_message([oversized])
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "error")
        self.assertIn("maximum size", data["message"])

    @patch("static.openai_completions_api.OpenAIChatCompletionsAPI.create_chat_completion")
    def test_valid_image_payload_passes_validation(self, mock_chat: Mock) -> None:
        class MockMessage:
            content = "ok"
            tool_calls = None

        class MockResponse:
            message = MockMessage()
            finish_reason = "stop"

        mock_chat.return_value = MockResponse()

        response = self._post_message([self._small_image()] * MAX_ATTACHED_IMAGES)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")

    def test_max_content_length_configured(self) -> None:
        self.assertEqual(self.app.config["MAX_CONTENT_LENGTH"], MAX_CONTENT_LENGTH_BYTES)

    def test_413_handler_returns_json(self) -> None:
        # Shrink the limit for this request so we don't have to send 80 MB.
        self.app.config["MAX_CONTENT_LENGTH"] = 64
        response = self.client.post(
            "/send_message",
            data="x" * 512,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("application/json", response.content_type)
        data = json.loads(response.data)
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
