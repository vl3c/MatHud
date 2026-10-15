"""Every drawable class is described to the model, restored from saved workspaces and round-trip tested.

The drawable classes and the browser-side lists are read with ``ast``: those modules import
browser code that CPython cannot load. A drawable class is saved in the bucket named after
it plus "s" (``DrawablesContainer.get_state``).
"""

from __future__ import annotations

import ast
import glob
import json
import os
import unittest
from typing import Any, Set

from static.canvas_state_formatter import _KNOWN_BUCKETS
from static.workspace_manager import WorkspaceManager as ServerWorkspaceManager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_DIR = os.path.join(PROJECT_ROOT, "static", "client")
FIXTURE_DIR = os.path.join(CLIENT_DIR, "client_tests", "data", "saved_workspaces_2026-10-15")

# Base classes that are never put on a canvas themselves.
BASE_CLASSES = {"ColoredArea", "Graph"}
# Buckets the round-trip scenes cannot produce: Bars are rebuilt from their plots and not
# saved, and Plots only come from workspaces saved before the plot classes were split.
NOT_ROUND_TRIPPED = {"Bars", "Plots"}


def _drawable_buckets() -> Set[str]:
    """The bucket of every drawable class that can be on a canvas."""
    names: Set[str] = set()
    for path in glob.glob(os.path.join(CLIENT_DIR, "drawables", "*.py")):
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_class_name":
                for statement in ast.walk(node):
                    if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Constant):
                        names.add(statement.value.value)
    return {name + "s" for name in names - BASE_CLASSES}


def _literal(relative_path: str, name: str) -> Any:
    """The literal value assigned to ``name`` (module or class level) in a client module."""
    tree = ast.parse(open(os.path.join(CLIENT_DIR, relative_path), encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            assert node.value is not None
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {relative_path}")


class TestDrawableWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.buckets = _drawable_buckets()

    def test_drawable_classes_are_found(self) -> None:
        self.assertIn("Points", self.buckets)
        self.assertIn("ParametricFunctions", self.buckets)
        self.assertGreater(len(self.buckets), 30)

    def test_every_drawable_has_a_canvas_formatter_spec(self) -> None:
        self.assertEqual(sorted(self.buckets - _KNOWN_BUCKETS), [])

    def test_every_drawable_is_restored_from_saved_workspaces(self) -> None:
        restored = set(_literal("workspace_manager.py", "RESTORED_BUCKETS"))
        self.assertEqual(sorted(self.buckets - restored), [])

    def test_every_saved_drawable_is_in_the_round_trip_test(self) -> None:
        round_tripped = set(
            _literal(os.path.join("client_tests", "test_workspace_round_trip.py"), "EXPECTED_DRAWABLE_KEYS")
        )
        self.assertEqual(sorted(self.buckets - NOT_ROUND_TRIPPED - round_tripped), [])
        self.assertEqual(sorted(round_tripped - self.buckets), [])


class TestSavedWorkspaceFixtures(unittest.TestCase):
    def test_the_client_test_lists_every_fixture(self) -> None:
        listed = _literal(os.path.join("client_tests", "test_saved_workspace_fixtures.py"), "FIXTURE_NAMES")
        on_disk = sorted(os.path.splitext(name)[0] for name in os.listdir(FIXTURE_DIR) if name.endswith(".json"))
        self.assertEqual(sorted(listed), on_disk)

    def test_fixtures_load_on_the_server_unchanged(self) -> None:
        manager = ServerWorkspaceManager(workspaces_dir=FIXTURE_DIR)
        for name in os.listdir(FIXTURE_DIR):
            with self.subTest(workspace=name):
                record = manager.load_workspace_record(os.path.splitext(name)[0])
                with open(os.path.join(FIXTURE_DIR, name), encoding="utf-8") as handle:
                    self.assertEqual(record["state"], json.load(handle)["state"])


if __name__ == "__main__":
    unittest.main()
