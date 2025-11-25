from __future__ import annotations

import unittest

import copy

from drawables.closed_shape_colored_area import ClosedShapeColoredArea
from constants import default_closed_shape_resolution
from rendering.renderables import ClosedShapeAreaRenderable
from utils.geometry_utils import GeometryUtils
from .simple_mock import SimpleMock


class TestClosedShapeColoredArea(unittest.TestCase):
    def setUp(self) -> None:
        self.point_a = SimpleMock(name="A", x=0.0, y=0.0)
        self.point_b = SimpleMock(name="B", x=2.0, y=0.0)
        self.point_c = SimpleMock(name="C", x=0.0, y=2.0)
        self.segment_ab = SimpleMock(name="AB", point1=self.point_a, point2=self.point_b)
        self.segment_bc = SimpleMock(name="BC", point1=self.point_b, point2=self.point_c)
        self.segment_ca = SimpleMock(name="CA", point1=self.point_c, point2=self.point_a)
        self.circle = SimpleMock(
            name="CircleA",
            center=SimpleMock(x=1.0, y=1.0),
            radius=5.0,
        )
        self.ellipse = SimpleMock(
            name="EllipseA",
            center=SimpleMock(x=3.0, y=4.0),
            radius_x=6.0,
            radius_y=2.0,
            rotation_angle=15.0,
        )

    def test_segments_form_closed_loop(self) -> None:
        segments = [self.segment_ab, self.segment_bc, self.segment_ca]
        self.assertTrue(GeometryUtils.segments_form_closed_loop(segments))
        ordered = GeometryUtils.polygon_math_coordinates_from_segments(segments)
        self.assertIsNotNone(ordered)
        self.assertEqual(len(ordered), 3)
        self.assertEqual(ordered[0], (0.0, 0.0))

    def test_closed_shape_state_serialization(self) -> None:
        area = ClosedShapeColoredArea(
            shape_type="polygon",
            segments=[self.segment_ab, self.segment_bc, self.segment_ca],
            color="salmon",
            opacity=0.4,
        )
        state = area.get_state()
        self.assertEqual(state["args"]["shape_type"], "polygon")
        self.assertEqual(state["args"]["segments"], ["AB", "BC", "CA"])
        self.assertEqual(state["args"]["resolution"], area.resolution)

    def test_circle_segment_renderable(self) -> None:
        circle = SimpleMock(center=SimpleMock(x=0.0, y=0.0), radius=5.0, name="CircleMock")
        chord_segment = SimpleMock(
            name="Chord",
            point1=SimpleMock(x=5.0, y=0.0),
            point2=SimpleMock(x=-5.0, y=0.0),
        )
        area_model = SimpleMock(
            get_geometry_spec=SimpleMock(
                return_value={
                    "shape_type": "circle_segment",
                    "circle": circle,
                    "chord_segment": chord_segment,
                    "arc_clockwise": False,
                    "resolution": 32,
                }
            ),
            color="gold",
            opacity=0.5,
        )
        renderable = ClosedShapeAreaRenderable(area_model, SimpleMock())
        closed_area = renderable.build_screen_area()
        self.assertIsNotNone(closed_area)
        self.assertGreater(len(closed_area.forward_points), 3)
        self.assertEqual(closed_area.forward_points[0], (5.0, 0.0))
        self.assertEqual(closed_area.reverse_points[-1], (5.0, 0.0))

    def test_generate_name_for_circle(self) -> None:
        area = ClosedShapeColoredArea(
            shape_type="circle",
            circle=SimpleMock(name="CircleA"),
            resolution=default_closed_shape_resolution,
        )
        self.assertEqual(area._generate_name(), "closed_CircleA")

    def test_generate_name_for_polygon_without_segments(self) -> None:
        area = ClosedShapeColoredArea(shape_type="polygon")
        self.assertEqual(area._generate_name(), "closed_shape_polygon")

    def test_chord_segment_added_if_missing(self) -> None:
        chord = SimpleMock(name="Chord", point1=self.point_a, point2=self.point_b)
        area = ClosedShapeColoredArea(
            shape_type="circle_segment",
            segments=[self.segment_ab],
            circle=SimpleMock(name="CircleA"),
            chord_segment=chord,
        )
        self.assertIn(chord, area.segments)

    def test_get_state_serializes_geometry(self) -> None:
        area = ClosedShapeColoredArea(
            shape_type="polygon",
            segments=[self.segment_ab, self.segment_bc, self.segment_ca],
            circle=self.circle,
            ellipse=self.ellipse,
            chord_segment=self.segment_ab,
            arc_clockwise=True,
            resolution=123,
            color="salmon",
            opacity=0.4,
        )
        state = area.get_state()
        args = state["args"]
        self.assertEqual(args["shape_type"], "polygon")
        self.assertEqual(args["segments"], ["AB", "BC", "CA"])
        self.assertEqual(args["circle"], "CircleA")
        self.assertEqual(args["ellipse"], "EllipseA")
        self.assertEqual(args["chord_segment"], "AB")
        self.assertTrue(args["arc_clockwise"])
        self.assertEqual(args["resolution"], 123)
        snapshot = args["geometry_snapshot"]
        self.assertIn("polygon_coords", snapshot)
        self.assertEqual(snapshot["polygon_coords"], [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        self.assertEqual(
            snapshot["circle"],
            {"center": [1.0, 1.0], "radius": 5.0},
        )
        self.assertEqual(
            snapshot["ellipse"],
            {
                "center": [3.0, 4.0],
                "radius_x": 6.0,
                "radius_y": 2.0,
                "rotation": 15.0,
            },
        )
        self.assertEqual(
            snapshot["chord_endpoints"],
            [[0.0, 0.0], [2.0, 0.0]],
        )
        self.assertIn("segments", snapshot)

    def test_uses_helpers(self) -> None:
        area = ClosedShapeColoredArea(
            shape_type="polygon",
            segments=[self.segment_ab, self.segment_bc],
            circle=SimpleMock(name="CircleA"),
            ellipse=SimpleMock(name="EllipseA"),
        )
        self.assertTrue(area.uses_segment(self.segment_ab))
        self.assertFalse(area.uses_segment(self.segment_ca))
        self.assertTrue(area.uses_circle(SimpleMock(name="CircleA")))
        self.assertFalse(area.uses_circle(SimpleMock(name="Other")))
        self.assertTrue(area.uses_ellipse(SimpleMock(name="EllipseA")))
        self.assertFalse(area.uses_ellipse(SimpleMock(name="Other")))

    def test_deepcopy_creates_independent_copy(self) -> None:
        area = ClosedShapeColoredArea(
            shape_type="polygon",
            segments=[self.segment_ab],
            circle=SimpleMock(name="CircleA"),
        )
        copied = copy.deepcopy(area)
        self.assertIsNot(area, copied)
        self.assertNotEqual(id(area.segments), id(copied.segments))
        copied.segments.append(SimpleMock(name="New", point1=self.point_a, point2=self.point_b))
        self.assertEqual(len(area.segments), 1)

    def test_invalid_shape_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            ClosedShapeColoredArea(shape_type="unsupported")


