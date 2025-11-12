from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import unittest

from drawables.position import Position


class TestPosition(unittest.TestCase):
    def test_init(self) -> None:
        pos = Position(1, 2)
        self.assertEqual(pos.x, 1)
        self.assertEqual(pos.y, 2)

    def test_str(self) -> None:
        pos = Position(1, 2)
        self.assertEqual(str(pos), 'Position: 1, 2')

    def test_get_state(self) -> None:
        pos = Position(1, 2)
        expected_state = {"Position": {"x": 1, "y": 2}}
        self.assertEqual(pos.get_state(), expected_state)
