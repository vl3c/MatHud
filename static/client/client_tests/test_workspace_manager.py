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


class TestWorkspaceLabelRestore(unittest.TestCase):
    def test_create_labels_restores_visible_scale_and_render_mode(self) -> None:
        created_labels = []

        def create_label(x: float, y: float, text: str, name: str = "", **kwargs: object) -> SimpleMock:
            label = SimpleMock(update_reference_scale=SimpleMock())
            created_labels.append((x, y, text, name, kwargs, label))
            return label

        canvas = SimpleMock(create_label=create_label)
        manager = WorkspaceManager(canvas)

        state = {
            "Labels": [
                {
                    "name": "lbl_A",
                    "args": {
                        "position": {"x": 1.0, "y": 2.0},
                        "text": "hello",
                        "visible": False,
                        "reference_scale_factor": 2.5,
                        "render_mode": {
                            "kind": "screen_offset",
                            "text_format": "text_only",
                            "offset_from_point_radius": True,
                        },
                    },
                }
            ]
        }

        manager._create_labels(state)

        self.assertEqual(len(created_labels), 1)
        x, y, text, name, kwargs, label = created_labels[0]
        self.assertEqual((x, y), (1.0, 2.0))
        self.assertEqual(text, "hello")
        self.assertEqual(name, "lbl_A")

        self.assertFalse(bool(getattr(label, "visible", True)))
        label.update_reference_scale.assert_called_once()
        args, _ = label.update_reference_scale.calls[0]
        self.assertEqual(args[0], 2.5)

        render_mode = getattr(label, "render_mode", None)
        self.assertEqual(getattr(render_mode, "kind", None), "screen_offset")

    def test_create_segments_restores_embedded_label_fields(self) -> None:
        points = {"A": Point(0.0, 0.0, name="A"), "B": Point(10.0, 0.0, name="B")}

        def get_point_by_name(name: str) -> Point | None:
            return points.get(name)

        def get_point(x: float, y: float) -> Point | None:
            for point in points.values():
                if MathUtils.point_matches_coordinates(point, x, y):
                    return point
            return None

        embedded_label = SimpleMock(update_font_size=SimpleMock(), update_rotation=SimpleMock())
        segment_obj = SimpleMock(label=embedded_label)

        def create_segment(x1: float, y1: float, x2: float, y2: float, name: str = "", **kwargs: object) -> SimpleMock:
            return segment_obj

        canvas = SimpleMock(
            get_point_by_name=get_point_by_name,
            get_point=get_point,
            create_segment=create_segment,
        )

        manager = WorkspaceManager(canvas)
        state = {
            "Segments": [
                {
                    "name": "AB",
                    "args": {
                        "p1": "A",
                        "p2": "B",
                        "p1_coords": [0.0, 0.0],
                        "p2_coords": [10.0, 0.0],
                        "label": {
                            "text": "mid",
                            "visible": True,
                            "font_size": 18.0,
                            "rotation_degrees": 30.0,
                            "render_mode": {"kind": "screen_offset", "text_format": "text_only"},
                        },
                    },
                }
            ]
        }

        manager._create_segments(state)

        embedded_label.update_font_size.assert_called_once()
        embedded_label.update_rotation.assert_called_once()
        render_mode = getattr(embedded_label, "render_mode", None)
        self.assertEqual(getattr(render_mode, "kind", None), "screen_offset")


if __name__ == "__main__":
    unittest.main()

