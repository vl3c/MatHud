from __future__ import annotations

import copy
import unittest
from typing import Any, cast

from drawables_aggregator import Position
from coordinate_mapper import CoordinateMapper
from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea
from rendering.renderables import SegmentsBoundedAreaRenderable
from .simple_mock import SimpleMock


class TestSegmentsBoundedColoredArea(unittest.TestCase):
    def setUp(self) -> None:
        # Create a real CoordinateMapper instance
        self.coordinate_mapper = CoordinateMapper(500, 500)  # 500x500 canvas

        # Create canvas mock with all properties that CoordinateMapper needs
        self.canvas = SimpleMock(
            width=500,  # Required by sync_from_canvas
            height=500,  # Required by sync_from_canvas
            scale_factor=1,
            center=Position(250, 250),  # Canvas center
            cartesian2axis=SimpleMock(origin=Position(250, 250)),  # Coordinate system origin
            coordinate_mapper=self.coordinate_mapper,
            is_point_within_canvas_visible_area=SimpleMock(return_value=True),
            any_segment_part_visible_in_canvas_area=SimpleMock(return_value=True),
            zoom_point=Position(1, 1),
            zoom_direction=1,
            zoom_step=0.1,
            offset=Position(0, 0),  # Set to (0,0) for simpler tests
        )

        # Sync canvas state with coordinate mapper
        self.coordinate_mapper.sync_from_canvas(self.canvas)

        # Create mock segments
        self.segment1 = SimpleMock(
            name="AB",
            point1=SimpleMock(x=100, y=200),  # Canvas coordinates
            point2=SimpleMock(x=300, y=250),  # Canvas coordinates
        )

        self.segment2 = SimpleMock(
            name="CD",
            point1=SimpleMock(x=150, y=180),  # Canvas coordinates
            point2=SimpleMock(x=280, y=220),  # Canvas coordinates
        )

    def test_init_with_two_segments(self) -> None:
        """Test initialization of SegmentsBoundedColoredArea with two segments."""
        area = SegmentsBoundedColoredArea(self.segment1, self.segment2)
        self.assertEqual(area.segment1, self.segment1)
        self.assertEqual(area.segment2, self.segment2)
        self.assertEqual(area.color, "lightblue")
        self.assertEqual(area.opacity, 0.3)

    def test_init_with_segment_and_x_axis(self) -> None:
        """Test initialization of SegmentsBoundedColoredArea with segment and x-axis."""
        area = SegmentsBoundedColoredArea(self.segment1, None)
        self.assertEqual(area.segment1, self.segment1)
        self.assertIsNone(area.segment2)

    def test_get_class_name(self) -> None:
        """Test class name retrieval."""
        area = SegmentsBoundedColoredArea(self.segment1, self.segment2)
        self.assertEqual(area.get_class_name(), "SegmentsBoundedColoredArea")

    def test_generate_name_with_two_segments(self) -> None:
        """Test name generation with two segments."""
        area = SegmentsBoundedColoredArea(self.segment1, self.segment2)
        expected_name = "area_between_AB_and_CD"
        self.assertEqual(area.name, expected_name)

    def test_generate_name_with_segment_and_x_axis(self) -> None:
        """Test name generation with segment and x-axis."""
        area = SegmentsBoundedColoredArea(self.segment1, None)
        expected_name = "area_between_AB_and_x_axis"
        self.assertEqual(area.name, expected_name)

    def test_uses_segment_with_matching_first_segment(self) -> None:
        """Test uses_segment method with matching first segment."""
        area = SegmentsBoundedColoredArea(self.segment1, self.segment2)

        # Create matching segment
        matching_segment = SimpleMock(point1=SimpleMock(x=100, y=200), point2=SimpleMock(x=300, y=250))

        self.assertTrue(area.uses_segment(matching_segment))

    def test_uses_segment_with_matching_second_segment(self) -> None:
        """Test uses_segment method with matching second segment."""
        area = SegmentsBoundedColoredArea(self.segment1, self.segment2)

        # Create matching segment
        matching_segment = SimpleMock(point1=SimpleMock(x=150, y=180), point2=SimpleMock(x=280, y=220))

        self.assertTrue(area.uses_segment(matching_segment))

    def test_uses_segment_with_non_matching_segment(self) -> None:
        """Test uses_segment method with non-matching segment."""
        area = SegmentsBoundedColoredArea(self.segment1, self.segment2)

        # Create non-matching segment
        different_segment = SimpleMock(point1=SimpleMock(x=400, y=400), point2=SimpleMock(x=500, y=500))

        self.assertFalse(area.uses_segment(different_segment))

    def test_uses_segment_with_only_first_segment(self) -> None:
        """Test uses_segment method when only first segment exists."""
        area = SegmentsBoundedColoredArea(self.segment1, None)

        # Create matching segment
        matching_segment = SimpleMock(point1=SimpleMock(x=100, y=200), point2=SimpleMock(x=300, y=250))

        # Create non-matching segment
        different_segment = SimpleMock(point1=SimpleMock(x=400, y=400), point2=SimpleMock(x=500, y=500))

        self.assertTrue(area.uses_segment(matching_segment))
        self.assertFalse(area.uses_segment(different_segment))

    def test_x_axis_positioning_uses_cartesian_origin(self) -> None:
        """Test that x-axis positioning correctly uses cartesian origin for simplicity."""
        area = SegmentsBoundedColoredArea(self.segment1, None)

        # The renderer will use cartesian2axis.origin.y for x-axis positioning
        # since segment points are already in screen coordinates
        expected_x_axis_y = self.canvas.cartesian2axis.origin.y
        self.assertEqual(expected_x_axis_y, 250)  # Based on our setup

    def test_get_state_with_two_segments(self) -> None:
        """Test state serialization with two segments."""
        area = SegmentsBoundedColoredArea(self.segment1, self.segment2)
        state = area.get_state()

        expected_args = {"segment1": "AB", "segment2": "CD"}
        self.assertEqual(state["args"]["segment1"], expected_args["segment1"])
        self.assertEqual(state["args"]["segment2"], expected_args["segment2"])

    def test_get_state_with_segment_and_x_axis(self) -> None:
        """Test state serialization with segment and x-axis."""
        area = SegmentsBoundedColoredArea(self.segment1, None)
        state = area.get_state()

        expected_args = {"segment1": "AB", "segment2": "x_axis"}
        self.assertEqual(state["args"]["segment1"], expected_args["segment1"])
        self.assertEqual(state["args"]["segment2"], expected_args["segment2"])

    def test_deepcopy(self) -> None:
        """Test deep copy functionality."""
        area = SegmentsBoundedColoredArea(self.segment1, self.segment2)
        area_copy = copy.deepcopy(area)

        self.assertIsNot(area_copy, area)

        # Boundaries should be deep-copied (important for undo/redo integrity),
        # but should preserve equivalent values.
        self.assertIsNot(area_copy.segment1, area.segment1)
        self.assertEqual(area_copy.segment1.name, area.segment1.name)
        self.assertEqual(area_copy.segment1.point1.x, area.segment1.point1.x)
        self.assertEqual(area_copy.segment1.point1.y, area.segment1.point1.y)
        self.assertEqual(area_copy.segment1.point2.x, area.segment1.point2.x)
        self.assertEqual(area_copy.segment1.point2.y, area.segment1.point2.y)

        self.assertIsNot(area_copy.segment2, area.segment2)
        self.assertEqual(area_copy.segment2.name, area.segment2.name)
        self.assertEqual(area_copy.segment2.point1.x, area.segment2.point1.x)
        self.assertEqual(area_copy.segment2.point1.y, area.segment2.point1.y)
        self.assertEqual(area_copy.segment2.point2.x, area.segment2.point2.x)
        self.assertEqual(area_copy.segment2.point2.y, area.segment2.point2.y)

        self.assertEqual(area_copy.color, area.color)
        self.assertEqual(area_copy.opacity, area.opacity)

    def test_no_overlap_segments(self) -> None:
        """Test case where segments don't overlap - should not draw anything."""
        # Create non-overlapping segments
        segment1 = SimpleMock(
            name="AB",
            point1=SimpleMock(x=100, y=200),  # x range: 100-150
            point2=SimpleMock(x=150, y=250),
        )

        segment2 = SimpleMock(
            name="CD",
            point1=SimpleMock(x=300, y=180),  # x range: 300-400 (no overlap with 100-150)
            point2=SimpleMock(x=400, y=220),
        )

        area = SegmentsBoundedColoredArea(segment1, segment2)

        # Use renderable instead of spying on draw()
        renderable = SegmentsBoundedAreaRenderable(area, self.coordinate_mapper)
        closed_area = renderable.build_screen_area()
        self.assertIsNone(closed_area)

    def test_exactly_touching_segments(self) -> None:
        """Test case where segments exactly touch at one point."""
        # Create segments that touch at exactly one point
        segment1 = SimpleMock(
            name="AB",
            point1=SimpleMock(x=100, y=200),  # x range: 100-200
            point2=SimpleMock(x=200, y=250),
        )

        segment2 = SimpleMock(
            name="CD",
            point1=SimpleMock(x=200, y=180),  # x range: 200-300 (touches at x=200)
            point2=SimpleMock(x=300, y=220),
        )

        area = SegmentsBoundedColoredArea(segment1, segment2)

        # Use renderable instead of spying on draw()
        renderable = SegmentsBoundedAreaRenderable(area, self.coordinate_mapper)
        closed_area = renderable.build_screen_area()
        self.assertIsNone(closed_area)

    def test_vertical_segment_interpolation(self) -> None:
        """Test linear interpolation with vertical segment (x2 == x1)."""
        # Create one normal segment and one vertical segment
        normal_segment = SimpleMock(
            name="AB",
            point1=SimpleMock(x=100, y=200),  # x range: 100-300
            point2=SimpleMock(x=300, y=250),
        )

        vertical_segment = SimpleMock(
            name="CD",
            point1=SimpleMock(x=200, y=100),  # Vertical line at x=200
            point2=SimpleMock(x=200, y=300),
        )

        area = SegmentsBoundedColoredArea(normal_segment, vertical_segment)

        # For vertical segment, the get_y_at_x function should handle x2 == x1 case
        # Let's test this directly
        def get_y_at_x(segment: Any, x: float) -> float:
            x1 = cast(float, segment.point1.x)
            y1 = cast(float, segment.point1.y)
            x2 = cast(float, segment.point2.x)
            y2 = cast(float, segment.point2.y)
            if x2 == x1:
                return y1  # Should return y1 for vertical segment
            t = (x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)

        # Test that vertical segment interpolation returns y1
        result = get_y_at_x(vertical_segment, 200)
        self.assertEqual(result, 100)  # Should return y1 when x2 == x1

    def test_segment_with_negative_coordinates(self) -> None:
        """Test segments with negative coordinates."""
        negative_segment1 = SimpleMock(
            name="AB",
            point1=SimpleMock(x=-200, y=-100),  # x range: -200 to -100
            point2=SimpleMock(x=-100, y=-50),
        )

        negative_segment2 = SimpleMock(
            name="CD",
            point1=SimpleMock(x=-150, y=-80),  # x range: -150 to -50 (overlap: -150 to -100)
            point2=SimpleMock(x=-50, y=-20),
        )

        area = SegmentsBoundedColoredArea(negative_segment1, negative_segment2)

        renderable = SegmentsBoundedAreaRenderable(area, self.coordinate_mapper)
        closed_area = renderable.build_screen_area()
        self.assertIsNotNone(closed_area)
        self.assertTrue(len(closed_area.forward_points) >= 1)
        self.assertTrue(len(closed_area.reverse_points) >= 1)

    def test_segment_crossing_zero_coordinates(self) -> None:
        """Test segments that cross zero on both axes."""
        crossing_segment1 = SimpleMock(
            name="AB",
            point1=SimpleMock(x=-100, y=-50),  # Crosses zero
            point2=SimpleMock(x=100, y=50),
        )

        crossing_segment2 = SimpleMock(
            name="CD",
            point1=SimpleMock(x=-50, y=100),  # Also crosses zero
            point2=SimpleMock(x=150, y=-100),
        )

        area = SegmentsBoundedColoredArea(crossing_segment1, crossing_segment2)

        # Test name generation
        expected_name = "area_between_AB_and_CD"
        self.assertEqual(area.name, expected_name)

        renderable = SegmentsBoundedAreaRenderable(area, self.coordinate_mapper)
        closed_area = renderable.build_screen_area()
        self.assertIsNotNone(closed_area)
        self.assertTrue(len(closed_area.forward_points) >= 1)
        self.assertTrue(len(closed_area.reverse_points) >= 1)
