"""Tests for static.config — server-side configuration constants."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from static.config import (
    CURRENT_WORKSPACE_SCHEMA_VERSION,
    WORKSPACES_DIR,
    WORKSPACES_DIR_ENV,
    get_workspaces_dir,
)
from static.workspace_manager import WorkspaceManager


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


class TestWorkspacesDirOverride(unittest.TestCase):
    """MATHUD_WORKSPACES_DIR moves workspace storage (used by the scenario runner)."""

    def test_default_without_override(self) -> None:
        with mock.patch.dict(os.environ, {WORKSPACES_DIR_ENV: ""}):
            self.assertEqual(get_workspaces_dir(), WORKSPACES_DIR)

    def test_override_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {WORKSPACES_DIR_ENV: tmp}):
                self.assertEqual(get_workspaces_dir(), tmp)
                manager = WorkspaceManager()
            self.assertEqual(manager.workspaces_dir, os.path.abspath(tmp))

    def test_explicit_directory_wins_over_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            with mock.patch.dict(os.environ, {WORKSPACES_DIR_ENV: other}):
                manager = WorkspaceManager(tmp)
            self.assertEqual(manager.workspaces_dir, os.path.abspath(tmp))

    def test_saves_go_to_the_override_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {WORKSPACES_DIR_ENV: tmp}):
                manager = WorkspaceManager()
            manager.save_workspace({"Points": []}, "scn_override")
            self.assertTrue(os.path.exists(os.path.join(tmp, "scn_override.json")))


if __name__ == "__main__":
    unittest.main()
