"""Tests for OAuth-related routes (OAuth initiation, callback, API key entry, auth status)."""

from __future__ import annotations

import os
import unittest
from typing import Optional
from unittest.mock import patch, Mock

from static.app_manager import AppManager, MatHudFlask
from static.openai_oauth import OAuthError, OpenAIOAuth


class TestOAuthRoutes(unittest.TestCase):
    """Test the OAuth authentication routes."""

    def setUp(self) -> None:
        """Set up test client with authentication required."""
        self.original_require_auth: Optional[str] = os.environ.get("REQUIRE_AUTH")
        os.environ["REQUIRE_AUTH"] = "true"

        self.original_auth_pin: Optional[str] = os.environ.get("AUTH_PIN")
        os.environ["AUTH_PIN"] = "123456"

        self.app: MatHudFlask = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self.original_require_auth is not None:
            os.environ["REQUIRE_AUTH"] = self.original_require_auth
        else:
            os.environ.pop("REQUIRE_AUTH", None)

        if self.original_auth_pin is not None:
            os.environ["AUTH_PIN"] = self.original_auth_pin
        else:
            os.environ.pop("AUTH_PIN", None)

    def test_auth_openai_redirects_to_openai(self) -> None:
        """GET /auth/openai should redirect to OpenAI authorization endpoint."""
        resp = self.client.get("/auth/openai")
        self.assertEqual(resp.status_code, 302)
        location = resp.headers["Location"]
        self.assertIn("auth.openai.com", location)
        self.assertIn("oauth/authorize", location)
        self.assertIn(OpenAIOAuth.CLIENT_ID, location)

    def test_auth_openai_stores_pkce_in_session(self) -> None:
        """GET /auth/openai should store code_verifier and state in session."""
        with self.client.session_transaction() as sess:
            self.assertNotIn("oauth_code_verifier", sess)
            self.assertNotIn("oauth_state", sess)

        self.client.get("/auth/openai")

        with self.client.session_transaction() as sess:
            self.assertIn("oauth_code_verifier", sess)
            self.assertIn("oauth_state", sess)

    def test_callback_error_parameter(self) -> None:
        """Callback with error param should redirect to login with flash."""
        resp = self.client.get("/auth/openai/callback?error=access_denied&error_description=User+denied")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_callback_invalid_state(self) -> None:
        """Callback with wrong state should redirect to login."""
        # First initiate the flow to get a session
        self.client.get("/auth/openai")

        resp = self.client.get("/auth/openai/callback?state=wrong-state&code=test-code")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_callback_missing_code(self) -> None:
        """Callback without code should redirect to login."""
        self.client.get("/auth/openai")

        with self.client.session_transaction() as sess:
            state = sess.get("oauth_state")

        resp = self.client.get(f"/auth/openai/callback?state={state}")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    @patch("static.routes.OpenAIOAuth.exchange_code_for_tokens")
    def test_callback_token_exchange_failure(self, mock_exchange: Mock) -> None:
        """Token exchange failure should redirect to login."""
        mock_exchange.side_effect = OAuthError("invalid_grant", "Bad code")

        self.client.get("/auth/openai")
        with self.client.session_transaction() as sess:
            state = sess.get("oauth_state")

        resp = self.client.get(f"/auth/openai/callback?state={state}&code=test-code")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    @patch("static.routes.OpenAIOAuth.exchange_token_for_api_key")
    @patch("static.routes.OpenAIOAuth.decode_and_verify_id_token")
    @patch("static.routes.OpenAIOAuth.exchange_code_for_tokens")
    def test_callback_success(
        self,
        mock_exchange: Mock,
        mock_decode: Mock,
        mock_api_key: Mock,
    ) -> None:
        """Successful callback should set session and redirect to index."""
        mock_exchange.return_value = {
            "access_token": "at-123",
            "refresh_token": "rt-456",
            "id_token": "idt-789",
            "expires_in": 3600,
        }
        mock_decode.return_value = {
            "iss": "https://auth.openai.com",
            "aud": OpenAIOAuth.CLIENT_ID,
            "email": "user@test.com",
            "name": "Test User",
        }
        mock_api_key.return_value = "sk-test-oauth-key"

        self.client.get("/auth/openai")
        with self.client.session_transaction() as sess:
            state = sess.get("oauth_state")

        resp = self.client.get(f"/auth/openai/callback?state={state}&code=test-code")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/", resp.headers["Location"])

        with self.client.session_transaction() as sess:
            self.assertTrue(sess.get("authenticated"))
            self.assertEqual(sess.get("openai_api_key"), "sk-test-oauth-key")
            self.assertEqual(sess.get("openai_auth_method"), "oauth")
            self.assertEqual(sess.get("openai_user_email"), "user@test.com")
            self.assertIn("openai_refresh_token", sess)
            self.assertIn("openai_token_expiry", sess)
            # PKCE state should be cleaned up
            self.assertNotIn("oauth_code_verifier", sess)
            self.assertNotIn("oauth_state", sess)


class TestApiKeyRoute(unittest.TestCase):
    """Test the manual API key entry route."""

    def setUp(self) -> None:
        self.original_require_auth: Optional[str] = os.environ.get("REQUIRE_AUTH")
        os.environ["REQUIRE_AUTH"] = "true"

        self.app: MatHudFlask = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self.original_require_auth is not None:
            os.environ["REQUIRE_AUTH"] = self.original_require_auth
        else:
            os.environ.pop("REQUIRE_AUTH", None)

    def test_valid_api_key(self) -> None:
        """POST /auth/apikey with valid key should authenticate."""
        resp = self.client.post("/auth/apikey", data={"api_key": "sk-test123"})
        self.assertEqual(resp.status_code, 302)

        with self.client.session_transaction() as sess:
            self.assertTrue(sess.get("authenticated"))
            self.assertEqual(sess.get("openai_api_key"), "sk-test123")
            self.assertEqual(sess.get("openai_auth_method"), "api_key")

    def test_invalid_api_key_format(self) -> None:
        """POST /auth/apikey with invalid key format should redirect to login."""
        resp = self.client.post("/auth/apikey", data={"api_key": "not-a-key"})
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_empty_api_key(self) -> None:
        """POST /auth/apikey with empty key should redirect to login."""
        resp = self.client.post("/auth/apikey", data={"api_key": ""})
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])


class TestLogout(unittest.TestCase):
    """Test that logout clears OAuth session keys."""

    def setUp(self) -> None:
        self.original_require_auth: Optional[str] = os.environ.get("REQUIRE_AUTH")
        os.environ["REQUIRE_AUTH"] = "true"

        self.app: MatHudFlask = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self.original_require_auth is not None:
            os.environ["REQUIRE_AUTH"] = self.original_require_auth
        else:
            os.environ.pop("REQUIRE_AUTH", None)

    def test_logout_clears_oauth_keys(self) -> None:
        """Logout should clear all OAuth session keys."""
        # Set up session with OAuth data
        with self.client.session_transaction() as sess:
            sess["authenticated"] = True
            sess["openai_api_key"] = "sk-test"
            sess["openai_refresh_token"] = "rt-test"
            sess["openai_id_token"] = "idt-test"
            sess["openai_token_expiry"] = 9999999999
            sess["openai_user_email"] = "test@test.com"
            sess["openai_auth_method"] = "oauth"

        self.client.get("/logout")

        with self.client.session_transaction() as sess:
            self.assertNotIn("authenticated", sess)
            self.assertNotIn("openai_api_key", sess)
            self.assertNotIn("openai_refresh_token", sess)
            self.assertNotIn("openai_id_token", sess)
            self.assertNotIn("openai_token_expiry", sess)
            self.assertNotIn("openai_user_email", sess)
            self.assertNotIn("openai_auth_method", sess)


class TestAuthStatus(unittest.TestCase):
    """Test the /auth_status endpoint returns OAuth info."""

    def setUp(self) -> None:
        self.original_require_auth: Optional[str] = os.environ.get("REQUIRE_AUTH")
        os.environ["REQUIRE_AUTH"] = "true"

        self.app: MatHudFlask = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self.original_require_auth is not None:
            os.environ["REQUIRE_AUTH"] = self.original_require_auth
        else:
            os.environ.pop("REQUIRE_AUTH", None)

    def test_unauthenticated_status(self) -> None:
        resp = self.client.get("/auth_status")
        data = resp.get_json()
        self.assertTrue(data["data"]["auth_required"])
        self.assertFalse(data["data"]["authenticated"])
        self.assertIsNone(data["data"]["openai_auth_method"])

    def test_oauth_authenticated_status(self) -> None:
        with self.client.session_transaction() as sess:
            sess["authenticated"] = True
            sess["openai_auth_method"] = "oauth"
            sess["openai_user_email"] = "user@test.com"

        resp = self.client.get("/auth_status")
        data = resp.get_json()
        self.assertTrue(data["data"]["authenticated"])
        self.assertEqual(data["data"]["openai_auth_method"], "oauth")
        self.assertEqual(data["data"]["openai_user_email"], "user@test.com")

    def test_pin_authenticated_status(self) -> None:
        """PIN auth should show server_key as method."""
        with self.client.session_transaction() as sess:
            sess["authenticated"] = True

        resp = self.client.get("/auth_status")
        data = resp.get_json()
        self.assertTrue(data["data"]["authenticated"])
        self.assertEqual(data["data"]["openai_auth_method"], "server_key")

    def test_apikey_authenticated_status(self) -> None:
        with self.client.session_transaction() as sess:
            sess["authenticated"] = True
            sess["openai_auth_method"] = "api_key"

        resp = self.client.get("/auth_status")
        data = resp.get_json()
        self.assertEqual(data["data"]["openai_auth_method"], "api_key")


class TestLoginPageRendering(unittest.TestCase):
    """Test that the login page renders correctly with auth tiers."""

    def setUp(self) -> None:
        self.original_require_auth: Optional[str] = os.environ.get("REQUIRE_AUTH")
        os.environ["REQUIRE_AUTH"] = "true"

        self.original_auth_pin: Optional[str] = os.environ.get("AUTH_PIN")

        self.app: MatHudFlask = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self.original_require_auth is not None:
            os.environ["REQUIRE_AUTH"] = self.original_require_auth
        else:
            os.environ.pop("REQUIRE_AUTH", None)

        if self.original_auth_pin is not None:
            os.environ["AUTH_PIN"] = self.original_auth_pin
        else:
            os.environ.pop("AUTH_PIN", None)

    def test_login_page_shows_oauth_button(self) -> None:
        resp = self.client.get("/login")
        html = resp.data.decode()
        self.assertIn("Sign in with OpenAI", html)
        self.assertIn("/auth/openai", html)

    def test_login_page_shows_api_key_section(self) -> None:
        resp = self.client.get("/login")
        html = resp.data.decode()
        self.assertIn("Enter an API key", html)
        self.assertIn("/auth/apikey", html)

    def test_login_page_shows_pin_when_configured(self) -> None:
        os.environ["AUTH_PIN"] = "123456"
        # Re-create app to pick up new env
        self.app = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        resp = self.client.get("/login")
        html = resp.data.decode()
        self.assertIn("Use access code", html)

    @patch("static.routes.AppManager.get_auth_pin", return_value=None)
    def test_login_page_hides_pin_when_not_configured(self, _mock_pin: Mock) -> None:
        os.environ.pop("AUTH_PIN", None)
        self.app = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        resp = self.client.get("/login")
        html = resp.data.decode()
        self.assertNotIn("Use access code", html)


if __name__ == "__main__":
    unittest.main()
