"""
MatHud OpenAI OAuth Module

Implements the OAuth 2.0 PKCE flow for OpenAI authentication, allowing users
to sign in with their OpenAI account and use their subscription for API calls.

Follows the same flow used by the Codex CLI:
1. Generate PKCE verifier/challenge pair
2. Redirect user to OpenAI authorization endpoint
3. Exchange authorization code for tokens (access_token, refresh_token, id_token)
4. Exchange id_token for an OpenAI API key via token-exchange grant
5. Refresh tokens when they expire

Dependencies:
    - requests (already in requirements.txt)
    - Standard library: hashlib, secrets, base64, json, time, urllib.parse
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import requests

_logger = logging.getLogger("mathud")


class OAuthError(Exception):
    """Raised when an OAuth token endpoint request fails.

    Attributes:
        error_code: The OAuth error code from the response (e.g. 'invalid_grant').
        description: Human-readable error description from the response.
    """

    def __init__(self, error_code: str, description: str = "") -> None:
        self.error_code = error_code
        self.description = description
        message = f"OAuth error: {error_code}"
        if description:
            message += f" - {description}"
        super().__init__(message)


class OpenAIOAuth:
    """Handles the OpenAI OAuth 2.0 PKCE flow for per-user authentication."""

    CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
    AUTH_ENDPOINT = "https://auth.openai.com/oauth/authorize"
    TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"
    SCOPES = "openid profile email offline_access"

    @staticmethod
    def generate_pkce() -> Tuple[str, str]:
        """Generate a PKCE code verifier and challenge pair.

        Returns:
            Tuple of (code_verifier, code_challenge) as base64url-encoded strings.
        """
        verifier_bytes = secrets.token_bytes(32)
        code_verifier = base64.urlsafe_b64encode(verifier_bytes).rstrip(b"=").decode("ascii")

        challenge_digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        code_challenge = base64.urlsafe_b64encode(challenge_digest).rstrip(b"=").decode("ascii")

        return code_verifier, code_challenge

    @staticmethod
    def build_authorization_url(
        redirect_uri: str,
        state: str,
        code_challenge: str,
    ) -> str:
        """Construct the full OpenAI authorization URL.

        Args:
            redirect_uri: The callback URL for the OAuth flow.
            state: CSRF protection state parameter.
            code_challenge: The PKCE code challenge (S256).

        Returns:
            The full authorization URL to redirect the user to.
        """
        params = {
            "client_id": OpenAIOAuth.CLIENT_ID,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": OpenAIOAuth.SCOPES,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
        }
        return f"{OpenAIOAuth.AUTH_ENDPOINT}?{urlencode(params)}"

    @staticmethod
    def exchange_code_for_tokens(
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> Dict[str, Any]:
        """Exchange an authorization code for tokens.

        Args:
            code: The authorization code from the callback.
            redirect_uri: The same redirect URI used in the authorization request.
            code_verifier: The original PKCE code verifier.

        Returns:
            Dict with keys: access_token, refresh_token, id_token, expires_in.

        Raises:
            OAuthError: If the token exchange fails.
        """
        params = {
            "grant_type": "authorization_code",
            "client_id": OpenAIOAuth.CLIENT_ID,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }
        return OpenAIOAuth._post_token_endpoint(params)

    @staticmethod
    def exchange_token_for_api_key(id_token: str) -> str:
        """Exchange an id_token for an OpenAI API key via token-exchange grant.

        Args:
            id_token: The id_token from the initial token exchange.

        Returns:
            The OpenAI API key string.

        Raises:
            OAuthError: If the token exchange fails.
        """
        params = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "client_id": OpenAIOAuth.CLIENT_ID,
            "requested_token_type": "openai-api-key",
            "subject_token": id_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
        }
        result = OpenAIOAuth._post_token_endpoint(params)
        api_key = result.get("access_token", "")
        if not api_key:
            raise OAuthError("missing_api_key", "Token exchange did not return an API key")
        return str(api_key)

    @staticmethod
    def refresh_tokens(refresh_token: str) -> Dict[str, Any]:
        """Refresh OAuth tokens using a refresh_token.

        Args:
            refresh_token: The refresh token from a previous exchange.

        Returns:
            Dict with keys: access_token, refresh_token, id_token, expires_in.

        Raises:
            OAuthError: If the refresh fails.
        """
        params = {
            "grant_type": "refresh_token",
            "client_id": OpenAIOAuth.CLIENT_ID,
            "refresh_token": refresh_token,
        }
        return OpenAIOAuth._post_token_endpoint(params)

    @staticmethod
    def decode_and_verify_id_token(id_token: str) -> Dict[str, Any]:
        """Decode and verify critical claims in an OpenAI id_token (JWT).

        Validates issuer, audience, and expiration. Does not perform full
        signature verification since the token arrives directly from OpenAI
        over HTTPS, but claim validation is mandatory before trusting identity
        or using the token for exchange.

        Args:
            id_token: The JWT id_token string.

        Returns:
            The decoded JWT payload as a dict.

        Raises:
            OAuthError: If the token is malformed or claims are invalid.
        """
        parts = id_token.split(".")
        if len(parts) != 3:
            raise OAuthError("invalid_token", "ID token is not a valid JWT (expected 3 parts)")

        try:
            # Decode the payload (second part), adding padding as needed
            payload_b64 = parts[1]
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            claims = json.loads(payload_bytes)
        except (ValueError, json.JSONDecodeError) as exc:
            raise OAuthError("invalid_token", f"Failed to decode ID token payload: {exc}") from exc

        if not isinstance(claims, dict):
            raise OAuthError("invalid_token", "ID token payload is not a JSON object")

        # Validate issuer
        issuer = claims.get("iss")
        if issuer != "https://auth.openai.com":
            raise OAuthError("invalid_issuer", f"Expected issuer 'https://auth.openai.com', got '{issuer}'")

        # Validate audience
        audience = claims.get("aud")
        if audience != OpenAIOAuth.CLIENT_ID:
            raise OAuthError("invalid_audience", f"Expected audience '{OpenAIOAuth.CLIENT_ID}', got '{audience}'")

        # Validate expiration
        exp = claims.get("exp")
        if isinstance(exp, (int, float)) and time.time() > exp:
            raise OAuthError("token_expired", "ID token has expired")

        return claims

    @staticmethod
    def extract_user_info(claims: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Extract user display info from verified JWT claims.

        Args:
            claims: The decoded JWT payload from decode_and_verify_id_token().

        Returns:
            Dict with keys: email, name.
        """
        email: Optional[str] = None
        name: Optional[str] = None

        raw_email = claims.get("email")
        if isinstance(raw_email, str):
            email = raw_email

        raw_name = claims.get("name")
        if isinstance(raw_name, str):
            name = raw_name

        return {"email": email, "name": name}

    @staticmethod
    def _post_token_endpoint(params: Dict[str, str]) -> Dict[str, Any]:
        """Send a POST request to the OpenAI token endpoint.

        Args:
            params: Form-encoded parameters for the request.

        Returns:
            The parsed JSON response as a dict.

        Raises:
            OAuthError: If the request fails or returns an error response.
        """
        try:
            resp = requests.post(
                OpenAIOAuth.TOKEN_ENDPOINT,
                data=params,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
        except requests.RequestException as exc:
            _logger.error("OAuth token endpoint request failed: %s", exc)
            raise OAuthError("request_failed", f"Failed to contact OpenAI: {exc}") from exc

        try:
            body = resp.json()
        except (ValueError, json.JSONDecodeError):
            if not resp.ok:
                raise OAuthError("http_error", f"Token endpoint returned HTTP {resp.status_code}")
            raise OAuthError("invalid_response", "Token endpoint returned non-JSON response")

        if not isinstance(body, dict):
            raise OAuthError("invalid_response", "Token endpoint returned unexpected response format")

        if "error" in body:
            error_code = str(body.get("error", "unknown_error"))
            error_desc = str(body.get("error_description", ""))
            _logger.warning("OAuth token endpoint error: %s - %s", error_code, error_desc)
            raise OAuthError(error_code, error_desc)

        if not resp.ok:
            raise OAuthError("http_error", f"Token endpoint returned HTTP {resp.status_code}")

        return body
