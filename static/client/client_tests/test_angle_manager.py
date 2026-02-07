import unittest
from unittest.mock import MagicMock, patch
from managers.angle_manager import AngleManager
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper
from drawables_aggregator import Position
from typing import Any

class TestAngleManager(unittest.TestCase):
    def setUp(self) -> None:
        # Create a real CoordinateMapper instance
        self.coordinate_mapper = CoordinateMapper(500, 500)  # 500x500 canvas
        
        self.canvas_mock = SimpleMock(
            name="CanvasMock",
            undo_redo_manager=SimpleMock(
                name="UndoRedoManagerMock",
                archive=MagicMock(),
                capture_state=MagicMock(return_value={"drawables": {}, "computations": []}),
                suspend_archiving=MagicMock(),
                resume_archiving=MagicMock(),
                restore_state=MagicMock(),
                push_undo_state=MagicMock(),
            ),
            draw_enabled=True,
            draw=MagicMock(),
            # Add minimal coordinate_mapper properties
            width=500,
            height=500,
            scale_factor=1,
            center=Position(250, 250),
            cartesian2axis=SimpleMock(origin=Position(250, 250)),
            coordinate_mapper=self.coordinate_mapper,
            zoom_point=Position(1, 1),
            zoom_direction=1,
            zoom_step=0.1,
            offset=Position(0, 0)
        )
        
        # Sync canvas state with coordinate mapper
        self.coordinate_mapper.sync_from_canvas(self.canvas_mock)
        self.drawables_container_mock: Any = SimpleMock(
            name="DrawablesContainerMock",
            Angles=[], # Holds created Angle instances
            add=MagicMock(side_effect=lambda x: self.drawables_container_mock.Angles.append(x))
        )
        self.name_generator_mock = SimpleMock(name="NameGeneratorMock") # Basic mock for now
        
        self.dependency_manager_mock = SimpleMock(
            name="DependencyManagerMock",
            register_dependency=MagicMock(),
            # Add other methods like remove_drawable, unregister_dependency if needed for other tests
        )

        # Mock Point Class Behavior (used by point_manager)
        self.MockPoint = SimpleMock # Use SimpleMock directly
        # We will define the behavior in the create_point mock below

        self.point_manager_mock = SimpleMock(
            name="PointManagerMock",
            create_point=lambda x, y, name=None, extra_graphics=True, label=None, color=None, size=None, display_type=None, is_visible=True, is_fixed=False: 
                SimpleMock(
                    name=name or f"P({x},{y})", 
                    x=x, 
                    y=y,
                    label=label,
                    color=color,
                    size=size,
                    display_type=display_type,
                    is_visible=is_visible,
                    is_fixed=is_fixed
                )
        )

        # Mock Segment Class Behavior (used by segment_manager)
        self.MockSegment = SimpleMock # Use SimpleMock directly
        # We will define the behavior in the create_segment mock below

        self.segment_manager_mock = SimpleMock(
            name="SegmentManagerMock",
            create_segment=lambda x1, y1, x2, y2, name=None, extra_graphics=True, label=None, color=None, thickness=None, is_visible=True, has_direction=False:
                SimpleMock(
                    name=name or f"S_({x1},{y1})-({x2},{y2})", 
                    point1=self.point_manager_mock.create_point(x1,y1), 
                    point2=self.point_manager_mock.create_point(x2,y2),
                    # Ensure all attributes that might be accessed are present
                    label=label,
                    color=color,
                    thickness=thickness,
                    is_visible=is_visible,
                    has_direction=has_direction
                )
        )
        
        self.drawable_manager_proxy_mock = SimpleMock(
            name="DrawableManagerProxyMock",
            create_drawables_from_new_connections=MagicMock()
            # Add get_segment_by_name if AngleManager needs it directly from proxy
        )

        # The actual Angle class will be imported, but we might patch its __init__ for some tests.
        # from drawables.angle import Angle 

        self.angle_manager = AngleManager(
            canvas=self.canvas_mock,
            drawables_container=self.drawables_container_mock,
            name_generator=self.name_generator_mock,
            dependency_manager=self.dependency_manager_mock,
            point_manager=self.point_manager_mock,
            segment_manager=self.segment_manager_mock,
            drawable_manager_proxy=self.drawable_manager_proxy_mock
        )

        # Helper points for tests
        self.A = self.point_manager_mock.create_point(0, 0, name="A")
        self.B = self.point_manager_mock.create_point(10, 0, name="B")
        self.C = self.point_manager_mock.create_point(0, 10, name="C")
        self.D = self.point_manager_mock.create_point(-10, 0, name="D") # For a different angle

        # Helper segments for tests
        self.seg_AB = self.segment_manager_mock.create_segment(self.A.x, self.A.y, self.B.x, self.B.y, name="AB")
        self.seg_AC = self.segment_manager_mock.create_segment(self.A.x, self.A.y, self.C.x, self.C.y, name="AC")
        self.seg_AD = self.segment_manager_mock.create_segment(self.A.x, self.A.y, self.D.x, self.D.y, name="AD")


    def test_initialization(self) -> None:
        self.assertIsNotNone(self.angle_manager.canvas)
        self.assertIsNotNone(self.angle_manager.drawables)
        self.assertIsNotNone(self.angle_manager.point_manager)
        self.assertIsNotNone(self.angle_manager.segment_manager)

    def test_get_angle_by_name_found(self) -> None:
        # Add a mock angle to the container
        mock_angle = SimpleMock(name="TestAngle1", segment1=self.seg_AB, segment2=self.seg_AC)
        self.drawables_container_mock.Angles.append(mock_angle)
        
        found_angle = self.angle_manager.get_angle_by_name("TestAngle1")
        self.assertIs(found_angle, mock_angle)

    def test_get_angle_by_name_not_found(self) -> None:
        found_angle = self.angle_manager.get_angle_by_name("NonExistentAngle")
        self.assertIsNone(found_angle)

    def test_get_angle_by_segments_found(self) -> None:
        mock_angle = SimpleMock(name="AngleBySegs", segment1=self.seg_AB, segment2=self.seg_AC, is_reflex=False)
        self.drawables_container_mock.Angles.append(mock_angle)

        # Test finding in original order
        found_angle = self.angle_manager.get_angle_by_segments(self.seg_AB, self.seg_AC)
        self.assertIs(found_angle, mock_angle)

        # Test finding in reverse order
        found_angle_reverse = self.angle_manager.get_angle_by_segments(self.seg_AC, self.seg_AB)
        self.assertIs(found_angle_reverse, mock_angle)

    def test_get_angle_by_segments_not_found(self) -> None:
        # Segments that don't form a known angle in the container
        # Create a new point E for the other segment to avoid confusion
        E = self.point_manager_mock.create_point(20,20, name="E")
        other_segment = self.segment_manager_mock.create_segment(self.B.x, self.B.y, E.x, E.y, name="BE")
        found_angle = self.angle_manager.get_angle_by_segments(self.seg_AB, other_segment)
        self.assertIsNone(found_angle)

    def test_get_angle_by_points_found(self) -> None:     
        # Setup points that an Angle would have derived
        A_obj = SimpleMock(name="A_obj", x=0,y=0)
        B_obj = SimpleMock(name="B_obj", x=10,y=0)
        C_obj = SimpleMock(name="C_obj", x=0,y=10)

        mock_angle = SimpleMock(
            name="AngleByPoints", 
            vertex_point=A_obj, 
            arm1_point=B_obj, 
            arm2_point=C_obj,
            # segment1 and segment2 would be composed of these points
            segment1=SimpleMock(point1=A_obj, point2=B_obj),
            segment2=SimpleMock(point1=A_obj, point2=C_obj),
            is_reflex=False
        )
        self.drawables_container_mock.Angles.append(mock_angle)

        # Search using the same point objects
        found_angle = self.angle_manager.get_angle_by_points(A_obj, B_obj, C_obj)
        self.assertIs(found_angle, mock_angle)

        # Search with arm points reversed
        found_angle_reverse = self.angle_manager.get_angle_by_points(A_obj, C_obj, B_obj)
        self.assertIs(found_angle_reverse, mock_angle)

    def test_get_angle_by_points_not_found_wrong_vertex(self) -> None:
        A_obj = SimpleMock(name="A_obj", x=0,y=0)
        B_obj = SimpleMock(name="B_obj", x=10,y=0)
        C_obj = SimpleMock(name="C_obj", x=0,y=10)
        wrong_V_obj = SimpleMock(name="WrongV_obj", x=1,y=1) # Changed from wrong_v_obj

        mock_angle = SimpleMock(name="AngleByPoints", vertex_point=A_obj, arm1_point=B_obj, arm2_point=C_obj)
        self.drawables_container_mock.Angles.append(mock_angle)
        
        found_angle = self.angle_manager.get_angle_by_points(wrong_V_obj, B_obj, C_obj)
        self.assertIsNone(found_angle)

    def test_get_angle_by_points_not_found_wrong_arm(self) -> None:
        A_obj = SimpleMock(name="A_obj", x=0,y=0)
        B_obj = SimpleMock(name="B_obj", x=10,y=0)
        C_obj = SimpleMock(name="C_obj", x=0,y=10)
        wrong_D_obj = SimpleMock(name="WrongD_obj", x=-1,y=-1) # Changed from wrong_a3_obj

        mock_angle = SimpleMock(name="AngleByPoints", vertex_point=A_obj, arm1_point=B_obj, arm2_point=C_obj)
        self.drawables_container_mock.Angles.append(mock_angle)

        found_angle = self.angle_manager.get_angle_by_points(A_obj, B_obj, wrong_D_obj)
        self.assertIsNone(found_angle)

    def test_get_angle_by_points_input_points_none(self) -> None:
        A_obj = SimpleMock(name="A_obj", x=0,y=0) # Changed from v_obj
        B_obj = SimpleMock(name="B_obj", x=10,y=0) # Changed from a1_obj
        self.assertIsNone(self.angle_manager.get_angle_by_points(None, B_obj, B_obj))
        self.assertIsNone(self.angle_manager.get_angle_by_points(A_obj, None, B_obj))
        self.assertIsNone(self.angle_manager.get_angle_by_points(A_obj, B_obj, None))

    def test_update_angle_changes_color_and_draws(self) -> None:
        angle = SimpleMock(
            name="angle_ABC",
            color="#ff0000",
            remove_svg_elements=MagicMock(),
        )
        angle.update_color = MagicMock(side_effect=lambda color: setattr(angle, "color", color))
        self.drawables_container_mock.Angles.append(angle)

        result = self.angle_manager.update_angle("angle_ABC", new_color="#00ff00")

        self.assertTrue(result)
        self.assertEqual(angle.color, "#00ff00")
        angle.remove_svg_elements.assert_called()
        self.canvas_mock.undo_redo_manager.archive.assert_called()
        self.canvas_mock.draw.assert_called()

    def test_update_angle_rejects_empty_color(self) -> None:
        angle = SimpleMock(
            name="angle_DEF",
            color="#ff0000",
            remove_svg_elements=MagicMock(),
        )
        angle.update_color = MagicMock(side_effect=lambda color: setattr(angle, "color", color))
        self.drawables_container_mock.Angles.append(angle)

        with self.assertRaises(ValueError):
            self.angle_manager.update_angle("angle_DEF", new_color="  ")

    def test_update_angle_raises_when_not_found(self) -> None:
        with self.assertRaises(ValueError):
            self.angle_manager.update_angle("missing_angle", new_color="#123456")

    @patch("managers.angle_manager.Angle")
    def test_create_angle_records_single_undo_on_success(self, mock_angle_cls: MagicMock) -> None:
        mock_angle = SimpleMock(name="angle_ABC", segment1=self.seg_AB, segment2=self.seg_AC, is_reflex=False)
        mock_angle_cls.return_value = mock_angle

        result = self.angle_manager.create_angle(0, 0, 10, 0, 0, 10, extra_graphics=False)

        self.assertIs(result, mock_angle)
        self.canvas_mock.undo_redo_manager.capture_state.assert_called_once()
        self.canvas_mock.undo_redo_manager.suspend_archiving.assert_called_once()
        self.canvas_mock.undo_redo_manager.push_undo_state.assert_called_once()
        self.canvas_mock.undo_redo_manager.restore_state.assert_not_called()
        self.canvas_mock.undo_redo_manager.resume_archiving.assert_called_once()

    def test_create_angle_rolls_back_when_segment_creation_fails(self) -> None:
        self.segment_manager_mock.create_segment = MagicMock(side_effect=[None, None])

        result = self.angle_manager.create_angle(0, 0, 10, 0, 0, 10, extra_graphics=False)

        self.assertIsNone(result)
        self.canvas_mock.undo_redo_manager.capture_state.assert_called_once()
        self.canvas_mock.undo_redo_manager.suspend_archiving.assert_called_once()
        self.canvas_mock.undo_redo_manager.restore_state.assert_called_once()
        self.canvas_mock.undo_redo_manager.push_undo_state.assert_not_called()
        self.canvas_mock.undo_redo_manager.resume_archiving.assert_called_once()

    @patch("managers.angle_manager.Angle", side_effect=ValueError("invalid angle"))
    def test_create_angle_rolls_back_when_angle_constructor_fails(self, _: MagicMock) -> None:
        result = self.angle_manager.create_angle(0, 0, 10, 0, 0, 10, extra_graphics=False)

        self.assertIsNone(result)
        self.canvas_mock.undo_redo_manager.capture_state.assert_called_once()
        self.canvas_mock.undo_redo_manager.suspend_archiving.assert_called_once()
        self.canvas_mock.undo_redo_manager.restore_state.assert_called_once()
        self.canvas_mock.undo_redo_manager.push_undo_state.assert_not_called()
        self.canvas_mock.undo_redo_manager.resume_archiving.assert_called_once()
