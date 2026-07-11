"""Tests for the centralized environment-variable loading module.

Covers ``load_env_files()`` discovery logic and ``get_api_key()`` fast-path,
required/optional, and fallback behaviours.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from static.env_config import get_api_key, load_env_files


class TestLoadEnvFiles(unittest.TestCase):
    """Test cases for load_env_files()."""

    @patch("static.env_config.load_dotenv")
    @patch("static.env_config.os.path.exists", return_value=True)
    def test_loads_project_root_and_parent_env(
        self, mock_exists: unittest.mock.MagicMock, mock_load: unittest.mock.MagicMock
    ) -> None:
        """When a parent .env exists, both project-root and parent are loaded."""
        load_env_files()

        # First call: project-root (no explicit path)
        # Second call: parent directory .env
        self.assertEqual(mock_load.call_count, 2)
        # The first call uses no arguments (project root default)
        mock_load.assert_any_call()

    @patch("static.env_config.load_dotenv")
    @patch("static.env_config.os.path.exists", return_value=False)
    def test_skips_parent_env_when_missing(
        self, mock_exists: unittest.mock.MagicMock, mock_load: unittest.mock.MagicMock
    ) -> None:
        """When the parent .env does not exist, only the project-root is loaded."""
        load_env_files()

        # Only the project-root call (no path arg) should occur
        mock_load.assert_called_once_with()

    @patch("static.env_config.load_dotenv")
    @patch("static.env_config.os.path.exists", return_value=False)
    def test_does_not_crash_when_no_env_files(
        self, mock_exists: unittest.mock.MagicMock, mock_load: unittest.mock.MagicMock
    ) -> None:
        """Calling load_env_files when no .env files exist must not raise."""
        # Should complete without error
        load_env_files()

    @patch("static.env_config.load_dotenv")
    @patch("static.env_config.os.path.exists", return_value=True)
    def test_idempotent_multiple_calls(
        self, mock_exists: unittest.mock.MagicMock, mock_load: unittest.mock.MagicMock
    ) -> None:
        """Calling load_env_files twice delegates to load_dotenv each time.

        python-dotenv itself handles the idempotency guarantee (it will not
        overwrite variables already present in os.environ), so repeated calls
        are safe even though load_dotenv is invoked again.
        """
        load_env_files()
        load_env_files()

        # Two invocations x 2 load_dotenv calls each = 4 total
        self.assertEqual(mock_load.call_count, 4)

    @patch("static.env_config.load_dotenv")
    @patch("static.env_config.os.path.exists", return_value=True)
    def test_parent_env_path_constructed_correctly(
        self, mock_exists: unittest.mock.MagicMock, mock_load: unittest.mock.MagicMock
    ) -> None:
        """The parent .env path should be <cwd>/../.env."""
        load_env_files()

        expected_parent = os.path.join(os.path.dirname(os.getcwd()), ".env")
        mock_exists.assert_called_once_with(expected_parent)
        mock_load.assert_any_call(expected_parent)


class TestGetApiKeyFastPath(unittest.TestCase):
    """Test cases for the get_api_key() fast-path (key already in environ)."""

    @patch("static.env_config.load_env_files")
    @patch.dict(os.environ, {"MY_KEY": "fast-value"}, clear=False)
    def test_returns_key_without_loading_env(
        self, mock_load: unittest.mock.MagicMock
    ) -> None:
        """When the key is already in os.environ, skip load_env_files entirely."""
        result = get_api_key("MY_KEY")

        self.assertEqual(result, "fast-value")
        mock_load.assert_not_called()


class TestGetApiKeyRequired(unittest.TestCase):
    """Test cases for get_api_key() with required=True (the default)."""

    @patch("static.env_config.load_env_files")
    @patch.dict(os.environ, {"FOUND_KEY": "secret-123"}, clear=False)
    def test_returns_key_when_present_in_environ(
        self, mock_load: unittest.mock.MagicMock
    ) -> None:
        """Returns the key when it exists in os.environ (fast-path)."""
        result = get_api_key("FOUND_KEY", required=True)
        self.assertEqual(result, "secret-123")

    @patch("static.env_config.load_env_files")
    def test_returns_key_loaded_from_env_file(
        self, mock_load: unittest.mock.MagicMock
    ) -> None:
        """Returns the key when load_env_files populates it on the retry."""
        key_name = "LAZY_KEY"
        # Ensure the key is absent initially
        env_copy = os.environ.copy()
        env_copy.pop(key_name, None)

        def _inject_key() -> None:
            os.environ[key_name] = "loaded-from-dotenv"

        mock_load.side_effect = _inject_key

        with patch.dict(os.environ, env_copy, clear=True):
            result = get_api_key(key_name, required=True)

        self.assertEqual(result, "loaded-from-dotenv")
        mock_load.assert_called_once()
        # Clean up injected key
        os.environ.pop(key_name, None)

    @patch("static.env_config.load_env_files")
    @patch.dict(os.environ, {}, clear=True)
    def test_raises_value_error_when_missing(
        self, mock_load: unittest.mock.MagicMock
    ) -> None:
        """Raises ValueError when required=True and the key cannot be found."""
        with self.assertRaises(ValueError) as ctx:
            get_api_key("MISSING_KEY", required=True)

        self.assertIn("MISSING_KEY", str(ctx.exception))

    @patch("static.env_config.load_env_files")
    @patch.dict(os.environ, {}, clear=True)
    def test_raises_value_error_by_default(
        self, mock_load: unittest.mock.MagicMock
    ) -> None:
        """required defaults to True, so omitting it still raises on missing key."""
        with self.assertRaises(ValueError):
            get_api_key("ABSENT_KEY")


class TestGetApiKeyOptional(unittest.TestCase):
    """Test cases for get_api_key() with required=False."""

    @patch("static.env_config.load_env_files")
    @patch.dict(os.environ, {}, clear=True)
    def test_returns_fallback_when_missing(
        self, mock_load: unittest.mock.MagicMock
    ) -> None:
        """Returns the explicit fallback value when key is absent."""
        result = get_api_key("NOPE", required=False, fallback="default-val")
        self.assertEqual(result, "default-val")

    @patch("static.env_config.load_env_files")
    @patch.dict(os.environ, {}, clear=True)
    def test_returns_empty_string_when_no_fallback(
        self, mock_load: unittest.mock.MagicMock
    ) -> None:
        """Returns empty string when key is absent and no fallback is given."""
        result = get_api_key("NOPE", required=False)
        self.assertEqual(result, "")

    @patch("static.env_config.load_env_files")
    @patch.dict(os.environ, {"OPT_KEY": "found-it"}, clear=False)
    def test_returns_key_when_present_even_if_not_required(
        self, mock_load: unittest.mock.MagicMock
    ) -> None:
        """When the key exists, it is returned regardless of required flag."""
        result = get_api_key("OPT_KEY", required=False, fallback="ignored")
        self.assertEqual(result, "found-it")


if __name__ == "__main__":
    unittest.main()
