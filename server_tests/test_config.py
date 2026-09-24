"""Tests for static.config — server-side configuration constants."""

from __future__ import annotations

import unittest

from static.config import (
    CURRENT_WORKSPACE_SCHEMA_VERSION,
    WORKSPACES_DIR,
)


class TestConfigConstants(unittest.TestCase):
    """Verify types, values, and consistency of configuration constants."""

    # -- Constant types and values --

    def test_workspaces_dir_is_non_empty_string(self) -> None:
        self.assertIsInstance(WORKSPACES_DIR, str)
        self.assertTrue(len(WORKSPACES_DIR) > 0)

    def test_schema_version_is_positive_integer(self) -> None:
        self.assertIsInstance(CURRENT_WORKSPACE_SCHEMA_VERSION, int)
        self.assertGreater(CURRENT_WORKSPACE_SCHEMA_VERSION, 0)

    # -- Consistency with usage --

    def test_workspaces_dir_matches_expected_name(self) -> None:
        self.assertEqual(WORKSPACES_DIR, "workspaces")


if __name__ == "__main__":
    unittest.main()
