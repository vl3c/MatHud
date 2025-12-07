from __future__ import annotations

import unittest

from .simple_mock import SimpleMock
from drawables.point import Point
from drawables.segment import Segment
from utils.math_utils import MathUtils
from workspace_manager import WorkspaceManager


class TestWorkspaceSegmentPersistence(unittest.TestCase):
    def test_segment_state_preserves_endpoint_order(self) -> None:
        point_o = Point(-70.0, -200.0, name="O")
        point_k = Point(-20.0, -180.0, name="K")
        segment = Segment(point_o, point_k)

        state = segment.get_state()

        self.assertEqual(state["args"]["p1"], "O")
        self.assertEqual(state["args"]["p2"], "K")
        self.assertEqual(state["args"]["p1_coords"], [point_o.x, point_o.y])
        self.assertEqual(state["args"]["p2_coords"], [point_k.x, point_k.y])

    def test_legacy_segment_state_restores_original_orientation(self) -> None:
        point_o = Point(-70.0, -200.0, name="O")
        point_k = Point(-20.0, -180.0, name="K")
        points = {"O": point_o, "K": point_k}

        captured_segments = []

        def get_point_by_name(name: str) -> Point | None:
            return points.get(name)

        def get_point(x: float, y: float) -> Point | None:
            for point in points.values():
                if MathUtils.point_matches_coordinates(point, x, y):
                    return point
            return None

        def create_segment(x1: float, y1: float, x2: float, y2: float, name: str = "", **kwargs: object) -> SimpleMock:
            captured_segments.append((x1, y1, x2, y2, name, kwargs))
            return SimpleMock()

        canvas = SimpleMock(
            get_point_by_name=get_point_by_name,
            get_point=get_point,
            create_segment=create_segment,
        )

        manager = WorkspaceManager(canvas)
        legacy_state = {
            "Segments": [
                {
                    "name": "OK",
                    "args": {
                        "p1": "K",
                        "p2": "O",
                        "p1_coords": [-70.0, -200.0],
                        "p2_coords": [-20.0, -180.0],
                    },
                }
            ]
        }

        manager._create_segments(legacy_state)

        self.assertEqual(len(captured_segments), 1)
        x1, y1, x2, y2, name, kwargs = captured_segments[0]
        self.assertEqual((x1, y1), (-70.0, -200.0))
        self.assertEqual((x2, y2), (-20.0, -180.0))
        self.assertEqual(name, "OK")
        self.assertEqual(kwargs.get("label_text"), "")
        self.assertFalse(bool(kwargs.get("label_visible", False)))


if __name__ == "__main__":
    unittest.main()

