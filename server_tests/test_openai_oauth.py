"""Tests for the OpenAI OAuth module (PKCE, URL building, token parsing, JWT decode)."""

from __future__ import annotations

import base64
import hashlib
import json
import time
import unittest
from typing import Any, Dict
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

from static.openai_oauth import OAuthError, OpenAIOAuth


def _make_jwt(payload: Dict[str, Any], header: Dict[str, Any] | None = None) -> str:
    """Helper to create a minimal JWT (unsigned) for testing."""
    if header is None:
        header = {"alg": "RS256", "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    sig_b64 = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=").decode()
    return f"{header_b64}.{payload_b64}.{sig_b64}"


class TestPKCE(unittest.TestCase):
    """Test PKCE code verifier and challenge generation."""

    def test_generate_pkce_returns_tuple(self) -> None:
        verifier, challenge = OpenAIOAuth.generate_pkce()
        self.assertIsInstance(verifier, str)
        self.assertIsInstance(challenge, str)

    def test_generate_pkce_verifier_length(self) -> None:
        verifier, _ = OpenAIOAuth.generate_pkce()
        # 32 bytes -> 43 base64url chars (no padding)
        self.assertEqual(len(verifier), 43)

    def test_generate_pkce_challenge_matches_verifier(self) -> None:
        verifier, challenge = OpenAIOAuth.generate_pkce()
        # Verify challenge = base64url(sha256(verifier))
        expected_digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected_challenge = base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("ascii")
        self.assertEqual(challenge, expected_challenge)

    def test_generate_pkce_uniqueness(self) -> None:
        pairs = [OpenAIOAuth.generate_pkce() for _ in range(10)]
        verifiers = [v for v, _ in pairs]
        self.assertEqual(len(set(verifiers)), 10, "Each PKCE verifier should be unique")


class TestBuildAuthorizationURL(unittest.TestCase):
    """Test authorization URL construction."""

    def test_url_contains_required_params(self) -> None:
        url = OpenAIOAuth.build_authorization_url(
            redirect_uri="http://localhost:5000/auth/openai/callback",
            state="test-state-123",
            code_challenge="test-challenge-abc",
        )
        parsed = urlparse(url)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.hostname, "auth.openai.com")
        self.assertEqual(parsed.path, "/oauth/authorize")

        params = parse_qs(parsed.query)
        self.assertEqual(params["client_id"], [OpenAIOAuth.CLIENT_ID])
        self.assertEqual(params["redirect_uri"], ["http://localhost:5000/auth/openai/callback"])
        self.assertEqual(params["response_type"], ["code"])
        self.assertEqual(params["code_challenge"], ["test-challenge-abc"])
        self.assertEqual(params["code_challenge_method"], ["S256"])
        self.assertEqual(params["state"], ["test-state-123"])
        self.assertIn("openid", params["scope"][0])

    def test_url_includes_openai_specific_params(self) -> None:
        url = OpenAIOAuth.build_authorization_url("http://x", "s", "c")
        params = parse_qs(urlparse(url).query)
        self.assertEqual(params["id_token_add_organizations"], ["true"])
        self.assertEqual(params["codex_cli_simplified_flow"], ["true"])


class TestDecodeAndVerifyIdToken(unittest.TestCase):
    """Test JWT decoding and claim verification."""

    def _valid_claims(self) -> Dict[str, Any]:
        return {
            "iss": "https://auth.openai.com",
            "aud": OpenAIOAuth.CLIENT_ID,
            "exp": time.time() + 3600,
            "email": "user@example.com",
            "name": "Test User",
        }

    def test_valid_token(self) -> None:
        claims = self._valid_claims()
        token = _make_jwt(claims)
        result = OpenAIOAuth.decode_and_verify_id_token(token)
        self.assertEqual(result["email"], "user@example.com")
        self.assertEqual(result["iss"], "https://auth.openai.com")

    def test_invalid_jwt_format(self) -> None:
        with self.assertRaises(OAuthError) as ctx:
            OpenAIOAuth.decode_and_verify_id_token("not-a-jwt")
        self.assertEqual(ctx.exception.error_code, "invalid_token")

    def test_wrong_issuer(self) -> None:
        claims = self._valid_claims()
        claims["iss"] = "https://evil.com"
        token = _make_jwt(claims)
        with self.assertRaises(OAuthError) as ctx:
            OpenAIOAuth.decode_and_verify_id_token(token)
        self.assertEqual(ctx.exception.error_code, "invalid_issuer")

    def test_wrong_audience(self) -> None:
        claims = self._valid_claims()
        claims["aud"] = "wrong-client-id"
        token = _make_jwt(claims)
        with self.assertRaises(OAuthError) as ctx:
            OpenAIOAuth.decode_and_verify_id_token(token)
        self.assertEqual(ctx.exception.error_code, "invalid_audience")

    def test_expired_token(self) -> None:
        claims = self._valid_claims()
        claims["exp"] = time.time() - 100
        token = _make_jwt(claims)
        with self.assertRaises(OAuthError) as ctx:
            OpenAIOAuth.decode_and_verify_id_token(token)
        self.assertEqual(ctx.exception.error_code, "token_expired")

    def test_missing_exp_is_ok(self) -> None:
        """Tokens without exp should not raise (exp check is only if present)."""
        claims = self._valid_claims()
        del claims["exp"]
        token = _make_jwt(claims)
        result = OpenAIOAuth.decode_and_verify_id_token(token)
        self.assertEqual(result["email"], "user@example.com")


class TestExtractUserInfo(unittest.TestCase):
    """Test user info extraction from JWT claims."""

    def test_extracts_email_and_name(self) -> None:
        claims = {"email": "alice@example.com", "name": "Alice"}
        info = OpenAIOAuth.extract_user_info(claims)
        self.assertEqual(info["email"], "alice@example.com")
        self.assertEqual(info["name"], "Alice")

    def test_missing_fields_return_none(self) -> None:
        info = OpenAIOAuth.extract_user_info({})
        self.assertIsNone(info["email"])
        self.assertIsNone(info["name"])

    def test_non_string_values_return_none(self) -> None:
        info = OpenAIOAuth.extract_user_info({"email": 123, "name": True})
        self.assertIsNone(info["email"])
        self.assertIsNone(info["name"])


class TestPostTokenEndpoint(unittest.TestCase):
    """Test the token endpoint POST helper."""

    @patch("static.openai_oauth.requests.post")
    def test_successful_response(self, mock_post: Mock) -> None:
        mock_resp = Mock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"access_token": "test-token", "expires_in": 3600}
        mock_post.return_value = mock_resp

        result = OpenAIOAuth._post_token_endpoint({"grant_type": "test"})
        self.assertEqual(result["access_token"], "test-token")

    @patch("static.openai_oauth.requests.post")
    def test_error_response(self, mock_post: Mock) -> None:
        mock_resp = Mock()
        mock_resp.ok = False
        mock_resp.json.return_value = {"error": "invalid_grant", "error_description": "Bad code"}
        mock_post.return_value = mock_resp

        with self.assertRaises(OAuthError) as ctx:
            OpenAIOAuth._post_token_endpoint({"grant_type": "test"})
        self.assertEqual(ctx.exception.error_code, "invalid_grant")
        self.assertIn("Bad code", ctx.exception.description)

    @patch("static.openai_oauth.requests.post")
    def test_network_error(self, mock_post: Mock) -> None:
        import requests as req
        mock_post.side_effect = req.ConnectionError("No network")

        with self.assertRaises(OAuthError) as ctx:
            OpenAIOAuth._post_token_endpoint({"grant_type": "test"})
        self.assertEqual(ctx.exception.error_code, "request_failed")

    @patch("static.openai_oauth.requests.post")
    def test_non_json_error_response(self, mock_post: Mock) -> None:
        mock_resp = Mock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.json.side_effect = ValueError("No JSON")
        mock_post.return_value = mock_resp

        with self.assertRaises(OAuthError) as ctx:
            OpenAIOAuth._post_token_endpoint({"grant_type": "test"})
        self.assertEqual(ctx.exception.error_code, "http_error")


class TestExchangeTokenForApiKey(unittest.TestCase):
    """Test the token-to-API-key exchange."""

    @patch("static.openai_oauth.OpenAIOAuth._post_token_endpoint")
    def test_returns_api_key(self, mock_post: Mock) -> None:
        mock_post.return_value = {"access_token": "sk-test-key-123"}
        result = OpenAIOAuth.exchange_token_for_api_key("test-id-token")
        self.assertEqual(result, "sk-test-key-123")

    @patch("static.openai_oauth.OpenAIOAuth._post_token_endpoint")
    def test_missing_api_key_raises(self, mock_post: Mock) -> None:
        mock_post.return_value = {"token_type": "bearer"}
        with self.assertRaises(OAuthError) as ctx:
            OpenAIOAuth.exchange_token_for_api_key("test-id-token")
        self.assertEqual(ctx.exception.error_code, "missing_api_key")

    @patch("static.openai_oauth.OpenAIOAuth._post_token_endpoint")
    def test_sends_correct_params(self, mock_post: Mock) -> None:
        mock_post.return_value = {"access_token": "sk-key"}
        OpenAIOAuth.exchange_token_for_api_key("my-id-token")

        call_args = mock_post.call_args[0][0]
        self.assertEqual(call_args["grant_type"], "urn:ietf:params:oauth:grant-type:token-exchange")
        self.assertEqual(call_args["subject_token"], "my-id-token")
        self.assertEqual(call_args["requested_token_type"], "openai-api-key")


class TestOAuthError(unittest.TestCase):
    """Test the OAuthError exception."""

    def test_str_without_description(self) -> None:
        err = OAuthError("invalid_grant")
        self.assertIn("invalid_grant", str(err))

    def test_str_with_description(self) -> None:
        err = OAuthError("invalid_grant", "The code has expired")
        self.assertIn("invalid_grant", str(err))
        self.assertIn("The code has expired", str(err))

    def test_attributes(self) -> None:
        err = OAuthError("test_error", "test desc")
        self.assertEqual(err.error_code, "test_error")
        self.assertEqual(err.description, "test desc")


if __name__ == "__main__":
    unittest.main()
