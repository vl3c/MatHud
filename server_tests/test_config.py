"""Tests for static.config — server-side configuration constants."""

from __future__ import annotations

import os
import unittest

from static.config import (
    CANVAS_SNAPSHOT_DIR,
    CANVAS_SNAPSHOT_PATH,
    CURRENT_WORKSPACE_SCHEMA_VERSION,
    WORKSPACES_DIR,
)


class TestConfigConstants(unittest.TestCase):
    """Verify types, values, and consistency of configuration constants."""

    # -- Constant types and values --

    def test_workspaces_dir_is_non_empty_string(self) -> None:
        self.assertIsInstance(WORKSPACES_DIR, str)
        self.assertTrue(len(WORKSPACES_DIR) > 0)

    def test_canvas_snapshot_dir_is_non_empty_string(self) -> None:
        self.assertIsInstance(CANVAS_SNAPSHOT_DIR, str)
        self.assertTrue(len(CANVAS_SNAPSHOT_DIR) > 0)

    def test_schema_version_is_positive_integer(self) -> None:
        self.assertIsInstance(CURRENT_WORKSPACE_SCHEMA_VERSION, int)
        self.assertGreater(CURRENT_WORKSPACE_SCHEMA_VERSION, 0)

    def test_canvas_snapshot_path_is_string(self) -> None:
        self.assertIsInstance(CANVAS_SNAPSHOT_PATH, str)

    def test_canvas_snapshot_path_contains_dir_and_filename(self) -> None:
        self.assertIn(CANVAS_SNAPSHOT_DIR, CANVAS_SNAPSHOT_PATH)
        self.assertIn("canvas.png", CANVAS_SNAPSHOT_PATH)

    # -- Path construction --

    def test_canvas_snapshot_path_equals_os_path_join(self) -> None:
        expected = os.path.join(CANVAS_SNAPSHOT_DIR, "canvas.png")
        self.assertEqual(CANVAS_SNAPSHOT_PATH, expected)

    # -- Consistency with usage --

    def test_workspaces_dir_matches_expected_name(self) -> None:
        self.assertEqual(WORKSPACES_DIR, "workspaces")

    def test_schema_version_at_least_one(self) -> None:
        self.assertGreaterEqual(CURRENT_WORKSPACE_SCHEMA_VERSION, 1)


if __name__ == "__main__":
    unittest.main()
