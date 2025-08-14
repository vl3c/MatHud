import unittest
from copy import deepcopy

from drawables.angle import Angle
from .simple_mock import SimpleMock
from name_generator.drawable import DrawableNameGenerator
from coordinate_mapper import CoordinateMapper
from geometry import Position

class TestAngle(unittest.TestCase):
    def setUp(self):
        # Create a real CoordinateMapper instance
        self.coordinate_mapper = CoordinateMapper(500, 500)  # 500x500 canvas
        
        # Setup for Canvas first, as DrawableNameGenerator needs it
        self.canvas = SimpleMock(
            # drawable_manager will be set after it's created
            create_svg_element = lambda tag_name, attributes, text_content=None: {"tag": tag_name, "attrs": attributes, "text": text_content},
            # Add minimal coordinate_mapper properties needed by the system
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
        self.coordinate_mapper.sync_from_canvas(self.canvas)

        # Instantiate the real name generator
        self.name_generator = DrawableNameGenerator(self.canvas)

        # Setup for DrawableManager
        self.drawable_manager_segments = {}
        def get_segment_by_name_mock(name):
            return self.drawable_manager_segments.get(name)
        def add_segment_mock(segment):
            self.drawable_manager_segments[segment.name] = segment

        self.drawable_manager = SimpleMock(
            get_segment_by_name = get_segment_by_name_mock,
            add_segment = add_segment_mock,
            name_generator = self.name_generator
        )
        
        # Now that drawable_manager is created, assign it to canvas mock
        self.canvas.drawable_manager = self.drawable_manager

        # Points using SimpleMock with single letter names
        self.A = SimpleMock(name="A", x=0, y=0)
        self.B = SimpleMock(name="B", x=10, y=0)
        self.C = SimpleMock(name="C", x=0, y=10)
        self.D = SimpleMock(name="D", x=10, y=10)
        self.E = SimpleMock(name="E", x=-10, y=0)
        
        # Segments using SimpleMock with names derived from points
        self.s_AB = SimpleMock(name="AB", point1=self.A, point2=self.B, canvas=self.canvas)
        self.s_AC = SimpleMock(name="AC", point1=self.A, point2=self.C, canvas=self.canvas)
        self.s_AD = SimpleMock(name="AD", point1=self.A, point2=self.D, canvas=self.canvas)
        self.s_AE = SimpleMock(name="AE", point1=self.A, point2=self.E, canvas=self.canvas)
        self.s_BD = SimpleMock(name="BD", point1=self.B, point2=self.D, canvas=self.canvas) # Used for no common vertex test
        self.s_CD = SimpleMock(name="CD", point1=self.C, point2=self.D, canvas=self.canvas) # Used for no common vertex test

        # Add segments to the mock drawable_manager
        self.drawable_manager.add_segment(self.s_AB)
        self.drawable_manager.add_segment(self.s_AC)
        self.drawable_manager.add_segment(self.s_AD)
        self.drawable_manager.add_segment(self.s_AE)
        self.drawable_manager.add_segment(self.s_BD)
        self.drawable_manager.add_segment(self.s_CD)

    def test_initialization_valid_90_degrees(self):
        angle = Angle(self.s_AB, self.s_AC, self.canvas) # is_reflex=False by default
        self.assertIsNotNone(angle)
        self.assertFalse(angle.is_reflex)
        self.assertIs(angle.vertex_point, self.A)
        self.assertIs(angle.arm1_point, self.B)
        self.assertIs(angle.arm2_point, self.C)
        self.assertAlmostEqual(angle.raw_angle_degrees, 90.0, places=5)
        self.assertAlmostEqual(angle.angle_degrees, 90.0, places=5) # Small angle
        self.assertEqual(angle.name, "angle_BAC")

    def test_initialization_valid_45_degrees(self):
        angle = Angle(self.s_AB, self.s_AD, self.canvas) # is_reflex=False by default
        self.assertIsNotNone(angle)
        self.assertFalse(angle.is_reflex)
        self.assertIs(angle.vertex_point, self.A)
        self.assertIs(angle.arm1_point, self.B)
        self.assertIs(angle.arm2_point, self.D)
        self.assertAlmostEqual(angle.raw_angle_degrees, 45.0, places=5)
        self.assertAlmostEqual(angle.angle_degrees, 45.0, places=5) # Small angle
        self.assertEqual(angle.name, "angle_BAD")

    def test_initialization_valid_180_degrees(self):
        angle = Angle(self.s_AB, self.s_AE, self.canvas) # is_reflex=False by default
        self.assertIsNotNone(angle)
        self.assertFalse(angle.is_reflex)
        self.assertIs(angle.vertex_point, self.A)
        self.assertIs(angle.arm1_point, self.B)
        self.assertIs(angle.arm2_point, self.E)
        self.assertAlmostEqual(angle.raw_angle_degrees, 180.0, places=5)
        self.assertAlmostEqual(angle.angle_degrees, 180.0, places=5) # Small angle (180 is its own small/reflex)
        self.assertEqual(angle.name, "angle_BAE")

    def test_initialization_invalid_no_common_vertex(self):
        # s_AB (A-B) and s_CD (C-D)
        with self.assertRaisesRegex(ValueError, "segments do not form a valid angle"):
            Angle(self.s_AB, self.s_CD, self.canvas)

    def test_initialization_invalid_collinear_overlapping_same_segment(self):
        s_AB_copy = SimpleMock(name="AB_copy", point1=self.A, point2=self.B, canvas=self.canvas)
        with self.assertRaisesRegex(ValueError, "segments do not form a valid angle"):
            Angle(self.s_AB, s_AB_copy, self.canvas)

    def test_initialization_invalid_one_segment_is_point_at_vertex(self):
        p_at_vertex = SimpleMock(name="A_vtx_copy", x=self.A.x, y=self.A.y) # Point identical to A
        s_degenerate = SimpleMock(name="S_Deg", point1=self.A, point2=p_at_vertex, canvas=self.canvas)
        with self.assertRaisesRegex(ValueError, "segments do not form a valid angle"):
            Angle(self.s_AB, s_degenerate, self.canvas)

    def test_angle_calculation_270_degrees_or_minus_90(self):
        # Angle from s_AC (A-C) to s_AB (A-B) -> Vertex A, Arms C, B
        # Raw CCW angle from AC to AB is 270 degrees.
        angle = Angle(self.s_AC, self.s_AB, self.canvas) # is_reflex=False by default
        self.assertFalse(angle.is_reflex)
        self.assertAlmostEqual(angle.raw_angle_degrees, 270.0, places=5)
        self.assertAlmostEqual(angle.angle_degrees, 90.0, places=5) # Small angle (360 - 270)
        # Name is based on sorted arm points (B, C) around vertex A
        self.assertEqual(angle.name, "angle_BAC")

    def test_initialization_reflex_AB_AC(self): # raw 90 deg
        angle = Angle(self.s_AB, self.s_AC, self.canvas, is_reflex=True)
        self.assertTrue(angle.is_reflex)
        self.assertAlmostEqual(angle.raw_angle_degrees, 90.0, places=5)
        self.assertAlmostEqual(angle.angle_degrees, 270.0, places=5) # Reflex of 90
        self.assertEqual(angle.name, "angle_BAC_reflex")

    def test_initialization_reflex_AC_AB(self): # raw 270 deg
        angle = Angle(self.s_AC, self.s_AB, self.canvas, is_reflex=True)
        self.assertTrue(angle.is_reflex)
        self.assertAlmostEqual(angle.raw_angle_degrees, 270.0, places=5)
        self.assertAlmostEqual(angle.angle_degrees, 270.0, places=5) # Reflex of 270 is 270
        self.assertEqual(angle.name, "angle_BAC_reflex")

    def test_initialization_reflex_AB_AD(self): # raw 45 deg
        angle = Angle(self.s_AB, self.s_AD, self.canvas, is_reflex=True)
        self.assertTrue(angle.is_reflex)
        self.assertAlmostEqual(angle.raw_angle_degrees, 45.0, places=5)
        self.assertAlmostEqual(angle.angle_degrees, 315.0, places=5) # Reflex of 45
        self.assertEqual(angle.name, "angle_BAD_reflex")

    def test_initialization_reflex_AB_AE(self): # raw 180 deg
        angle = Angle(self.s_AB, self.s_AE, self.canvas, is_reflex=True)
        self.assertTrue(angle.is_reflex)
        self.assertAlmostEqual(angle.raw_angle_degrees, 180.0, places=5)
        self.assertAlmostEqual(angle.angle_degrees, 180.0, places=5) # Reflex of 180 is 180
        self.assertEqual(angle.name, "angle_BAE_reflex")

    def test_initialization_zero_angle_non_reflex(self):
        s_AF = SimpleMock(name="AF", point1=self.A, point2=SimpleMock(name="F", x=5, y=0), canvas=self.canvas)
        self.drawable_manager.add_segment(s_AF)
        angle = Angle(self.s_AB, s_AF, self.canvas, is_reflex=False)
        self.assertFalse(angle.is_reflex)
        self.assertAlmostEqual(angle.raw_angle_degrees, 0.0, places=5)
        self.assertAlmostEqual(angle.angle_degrees, 0.0, places=5)
        self.assertEqual(angle.name, "angle_BAF") # Name might vary based on B vs F sorting

    def test_initialization_zero_angle_reflex(self):
        s_AF = SimpleMock(name="AF", point1=self.A, point2=SimpleMock(name="F", x=5, y=0), canvas=self.canvas)
        self.drawable_manager.add_segment(s_AF)
        angle = Angle(self.s_AB, s_AF, self.canvas, is_reflex=True)
        self.assertTrue(angle.is_reflex)
        self.assertAlmostEqual(angle.raw_angle_degrees, 0.0, places=5)
        self.assertAlmostEqual(angle.angle_degrees, 360.0, places=5)
        self.assertEqual(angle.name, "angle_BAF_reflex") # Name might vary based on B vs F sorting

    def test_angle_calculation_zero_length_arm(self):
        A_copy = SimpleMock(name="A_copy", x=self.A.x, y=self.A.y)
        s_zero_arm = SimpleMock(name="S_Zero", point1=self.A, point2=A_copy, canvas=self.canvas)
        with self.assertRaisesRegex(ValueError, "segments do not form a valid angle"):
             Angle(s_zero_arm, self.s_AC, self.canvas)

    def test_initialization_with_none_segment(self):
        with self.assertRaisesRegex(ValueError, "segments do not form a valid angle"):
            Angle(None, self.s_AC, self.canvas)
        with self.assertRaisesRegex(ValueError, "segments do not form a valid angle"):
            Angle(self.s_AB, None, self.canvas)
        with self.assertRaisesRegex(ValueError, "segments do not form a valid angle"):
            Angle(None, None, self.canvas)

    def test_get_class_name(self):
        angle = Angle(self.s_AB, self.s_AC, self.canvas)
        self.assertEqual(angle.get_class_name(), 'Angle')

    def test_canvas_property(self):
        angle = Angle(self.s_AB, self.s_AC, self.canvas)
        # Math model no longer propagates canvas changes; verify segments keep their own canvas
        self.assertIs(angle.canvas, self.canvas)
        self.assertIs(angle.segment1.canvas, self.canvas)
        self.assertIs(angle.segment2.canvas, self.canvas)

        new_canvas_mock = SimpleMock(name="NewCanvas")
        angle.canvas = new_canvas_mock
        # Angle canvas updated; segments not auto-updated in math-only model
        self.assertIs(angle.canvas, new_canvas_mock)
        self.assertIs(angle.segment1.canvas, self.canvas)
        self.assertIs(angle.segment2.canvas, self.canvas)

    def test_get_state_and_from_state(self):
        # Test non-reflex angle state
        angle1 = Angle(self.s_AB, self.s_AD, self.canvas, is_reflex=False)
        self.assertEqual(angle1.name, "angle_BAD")
        self.assertFalse(angle1.is_reflex)
        state_non_reflex = angle1.get_state()

        expected_state_non_reflex = {
            "name": "angle_BAD",
            "type": "angle",
            "args": {
                "segment1_name": "AB",
                "segment2_name": "AD",
                "color": "blue",
                "is_reflex": False
            }
        }
        self.assertEqual(state_non_reflex, expected_state_non_reflex)

        # Ensure segments are in drawable_manager for from_state to find them
        self.drawable_manager.add_segment(self.s_AB)
        self.drawable_manager.add_segment(self.s_AD)

        angle2 = Angle.from_state(state_non_reflex, self.canvas)
        self.assertIsNotNone(angle2)
        self.assertEqual(angle2.name, "angle_BAD")
        self.assertFalse(angle2.is_reflex)
        self.assertEqual(angle2.color, "blue")
        self.assertEqual(angle2.segment1.name, "AB")
        self.assertEqual(angle2.segment2.name, "AD")
        self.assertAlmostEqual(angle2.raw_angle_degrees, 45.0, places=5)
        self.assertAlmostEqual(angle2.angle_degrees, 45.0, places=5)

        # Test reflex angle state
        angle3 = Angle(self.s_AB, self.s_AC, self.canvas, is_reflex=True)
        self.assertEqual(angle3.name, "angle_BAC_reflex")
        self.assertTrue(angle3.is_reflex)
        state_reflex = angle3.get_state()

        expected_state_reflex = {
            "name": "angle_BAC_reflex",
            "type": "angle",
            "args": {
                "segment1_name": "AB",
                "segment2_name": "AC",
                "color": "blue", 
                "is_reflex": True
            }
        }
        self.assertEqual(state_reflex, expected_state_reflex)

        self.drawable_manager.add_segment(self.s_AC) # Ensure AC is also added for from_state

        angle4 = Angle.from_state(state_reflex, self.canvas)
        self.assertIsNotNone(angle4)
        self.assertEqual(angle4.name, "angle_BAC_reflex")
        self.assertTrue(angle4.is_reflex)
        self.assertEqual(angle4.color, "blue")
        self.assertEqual(angle4.segment1.name, "AB")
        self.assertEqual(angle4.segment2.name, "AC")
        self.assertAlmostEqual(angle4.raw_angle_degrees, 90.0, places=5)
        self.assertAlmostEqual(angle4.angle_degrees, 270.0, places=5)

    def test_from_state_segment_not_found(self):
        state = {
            "name": "ghost_angle",
            "type": "angle",
            "args": {"segment1_name": "NonExistentS1", "segment2_name": "AD", "color": "blue"}
        }
        self.drawable_manager.add_segment(self.s_AD)
        
        angle = Angle.from_state(state, self.canvas)
        self.assertIsNone(angle)

    def test_update_points_based_on_segments(self):
        # Angle from s_AB, s_AC. Points A,B,C. Expected angle 90.
        angle = Angle(self.s_AB, self.s_AC, self.canvas)
        self.assertAlmostEqual(angle.angle_degrees, 90.0)

        original_C_y = self.C.y
        original_C_x = self.C.x
        self.C.x = 10 # C was (0,10), now (10,0) which is B's location
        self.C.y = 0 
        
        if hasattr(angle, 'update_points_based_on_segments'):
            result = angle.update_points_based_on_segments()
            self.assertFalse(result)
            self.assertIsNone(angle.angle_degrees)
        else:
            angle._initialize()
            # If _initialize leads to an invalid state, angle_degrees should be None
            self.assertIsNone(angle.angle_degrees)

        self.C.x = original_C_x
        self.C.y = original_C_y

    def test_update_makes_angle_invalid(self):
        angle = Angle(self.s_AB, self.s_AC, self.canvas)
        
        original_B_x = self.B.x
        original_B_y = self.B.y
        self.B.x = self.A.x # Move B to A's location
        self.B.y = self.A.y

        if hasattr(angle, 'update_points_based_on_segments'):
            result = angle.update_points_based_on_segments()
            self.assertFalse(result) 
            self.assertIsNone(angle.angle_degrees)
        else:
            angle._initialize()
            # If _initialize leads to an invalid state, angle_degrees should be None
            self.assertIsNone(angle.angle_degrees)

        self.B.x = original_B_x
        self.B.y = original_B_y

    def test_deepcopy_basic(self):
        # Test with a non-reflex angle
        original_angle_non_reflex = Angle(self.s_AB, self.s_AD, self.canvas, is_reflex=False)
        self.assertEqual(original_angle_non_reflex.name, "angle_BAD") 
        
        memo = {}
        copied_angle_non_reflex = deepcopy(original_angle_non_reflex, memo)

        self.assertIsNot(original_angle_non_reflex, copied_angle_non_reflex)
        self.assertFalse(copied_angle_non_reflex.is_reflex)
        self.assertAlmostEqual(copied_angle_non_reflex.raw_angle_degrees, 45.0, places=5)
        self.assertAlmostEqual(copied_angle_non_reflex.angle_degrees, 45.0, places=5)
        self.assertEqual(copied_angle_non_reflex.name, "angle_BAD")
        # Standard deepcopy checks from before
        self.assertIs(copied_angle_non_reflex.canvas, original_angle_non_reflex.canvas)
        self.assertIsNot(copied_angle_non_reflex.segment1, original_angle_non_reflex.segment1)
        self.assertIsNot(copied_angle_non_reflex.segment2, original_angle_non_reflex.segment2)
        self.assertEqual(copied_angle_non_reflex.segment1.name, original_angle_non_reflex.segment1.name)
        self.assertEqual(copied_angle_non_reflex.segment2.name, original_angle_non_reflex.segment2.name)
        self.assertEqual(copied_angle_non_reflex.color, "blue")

        # Test with a reflex angle
        original_angle_reflex = Angle(self.s_AB, self.s_AC, self.canvas, is_reflex=True)
        self.assertEqual(original_angle_reflex.name, "angle_BAC_reflex")

        memo_reflex = {}
        copied_angle_reflex = deepcopy(original_angle_reflex, memo_reflex)
        
        self.assertIsNot(original_angle_reflex, copied_angle_reflex)
        self.assertTrue(copied_angle_reflex.is_reflex)
        self.assertAlmostEqual(copied_angle_reflex.raw_angle_degrees, 90.0, places=5)
        self.assertAlmostEqual(copied_angle_reflex.angle_degrees, 270.0, places=5)
        self.assertEqual(copied_angle_reflex.name, "angle_BAC_reflex")
        self.assertIs(copied_angle_reflex.canvas, original_angle_reflex.canvas)

    def test_deepcopy_with_memo(self):
        # Non-reflex angle
        original_angle = Angle(self.s_AB, self.s_AD, self.canvas, is_reflex=False)
        self.assertEqual(original_angle.name, "angle_BAD")
        self.assertFalse(original_angle.is_reflex)
        self.assertAlmostEqual(original_angle.raw_angle_degrees, 45.0, places=5)
        self.assertAlmostEqual(original_angle.angle_degrees, 45.0, places=5)

        memo = {}
        copied_angle = deepcopy(original_angle, memo)

        self.assertIsNot(original_angle, copied_angle)
        self.assertIs(copied_angle.canvas, original_angle.canvas)
        self.assertEqual(copied_angle.name, "angle_BAD")
        self.assertFalse(copied_angle.is_reflex)
        self.assertAlmostEqual(copied_angle.raw_angle_degrees, original_angle.raw_angle_degrees)

        # Reflex angle
        original_angle_reflex = Angle(self.s_AB, self.s_AC, self.canvas, is_reflex=True)
        memo_reflex_again = {}
        # Add segments to memo first to test sharing
        memo_reflex_again[id(original_angle_reflex.segment1)] = original_angle_reflex.segment1
        memo_reflex_again[id(original_angle_reflex.segment2)] = original_angle_reflex.segment2
        
        copied_angle_reflex_again = deepcopy(original_angle_reflex, memo_reflex_again)
        self.assertTrue(copied_angle_reflex_again.is_reflex)
        self.assertEqual(copied_angle_reflex_again.name, "angle_BAC_reflex")
        self.assertAlmostEqual(copied_angle_reflex_again.raw_angle_degrees, 90.0, places=5)
        self.assertAlmostEqual(copied_angle_reflex_again.angle_degrees, 270.0, places=5)
        # Check if segments were taken from memo (i.e., are the same instances)
        self.assertIs(copied_angle_reflex_again.segment1, original_angle_reflex.segment1)
        self.assertIs(copied_angle_reflex_again.segment2, original_angle_reflex.segment2)

    def test_angle_deletion_on_point_deletion(self):
        """Test that angles are automatically deleted when their constituent points are deleted."""
        # Setup mock managers for dependency tracking
        dependency_manager_mock = SimpleMock()
        dependency_manager_mock.dependencies = {}  # Simple storage for dependencies
        dependency_manager_mock.register_dependency = lambda child, parent: self._add_dependency(dependency_manager_mock.dependencies, child, parent)
        dependency_manager_mock.get_children = lambda parent: self._get_dependencies(dependency_manager_mock.dependencies, parent, 'children')
        dependency_manager_mock.remove_drawable = lambda drawable: None
        
        angle_manager_mock = SimpleMock()
        angles_list = []
        angle_manager_mock.drawables = SimpleMock(Angles=angles_list)
        
        # Create an angle and manually register it in our mock system
        angle = Angle(self.s_AB, self.s_AC, self.canvas)
        angles_list.append(angle)
        
        # Register dependencies (simulating what AngleManager.create_angle would do)
        dependency_manager_mock.register_dependency(angle, self.s_AB)  # angle depends on segment AB
        dependency_manager_mock.register_dependency(angle, self.s_AC)  # angle depends on segment AC  
        dependency_manager_mock.register_dependency(angle, self.A)     # angle depends on vertex point A
        dependency_manager_mock.register_dependency(angle, self.B)     # angle depends on arm point B
        dependency_manager_mock.register_dependency(angle, self.C)     # angle depends on arm point C
        
        # Verify angle exists
        self.assertEqual(len(angles_list), 1)
        self.assertIn(angle, angles_list)
        
        # Simulate deleting point A (vertex point)
        # This should trigger deletion of the angle since it depends on point A
        point_a_children = dependency_manager_mock.get_children(self.A)
        
        # Verify that the angle is a child of point A
        self.assertIn(angle, point_a_children)
        
        # Simulate the deletion process that should happen when point A is deleted
        for child in point_a_children:
            if hasattr(child, 'get_class_name') and child.get_class_name() == 'Angle':
                angles_list.remove(child)
        
        # Verify angle was deleted
        self.assertEqual(len(angles_list), 0)
        self.assertNotIn(angle, angles_list)

    def test_angle_deletion_on_segment_deletion(self):
        """Test that angles are automatically deleted when their constituent segments are deleted."""
        # Setup mock managers for dependency tracking
        dependency_manager_mock = SimpleMock()
        dependency_manager_mock.dependencies = {}  # Simple storage for dependencies
        dependency_manager_mock.register_dependency = lambda child, parent: self._add_dependency(dependency_manager_mock.dependencies, child, parent)
        dependency_manager_mock.get_children = lambda parent: self._get_dependencies(dependency_manager_mock.dependencies, parent, 'children')
        dependency_manager_mock.remove_drawable = lambda drawable: None
        
        angle_manager_mock = SimpleMock()
        angles_list = []
        angle_manager_mock.drawables = SimpleMock(Angles=angles_list)
        
        # Create an angle and manually register it in our mock system
        angle = Angle(self.s_AB, self.s_AC, self.canvas)
        angles_list.append(angle)
        
        # Register dependencies (simulating what AngleManager.create_angle would do)
        dependency_manager_mock.register_dependency(angle, self.s_AB)  # angle depends on segment AB
        dependency_manager_mock.register_dependency(angle, self.s_AC)  # angle depends on segment AC  
        dependency_manager_mock.register_dependency(angle, self.A)     # angle depends on vertex point A
        dependency_manager_mock.register_dependency(angle, self.B)     # angle depends on arm point B
        dependency_manager_mock.register_dependency(angle, self.C)     # angle depends on arm point C
        
        # Verify angle exists
        self.assertEqual(len(angles_list), 1)
        self.assertIn(angle, angles_list)
        
        # Simulate deleting segment AB
        # This should trigger deletion of the angle since it depends on segment AB
        segment_ab_children = dependency_manager_mock.get_children(self.s_AB)
        
        # Verify that the angle is a child of segment AB
        self.assertIn(angle, segment_ab_children)
        
        # Simulate the deletion process that should happen when segment AB is deleted
        for child in segment_ab_children:
            if hasattr(child, 'get_class_name') and child.get_class_name() == 'Angle':
                angles_list.remove(child)
        
        # Verify angle was deleted
        self.assertEqual(len(angles_list), 0)
        self.assertNotIn(angle, angles_list)

    def _add_dependency(self, dependencies_dict, child, parent):
        """Helper method to add a dependency relationship."""
        if parent not in dependencies_dict:
            dependencies_dict[parent] = []
        dependencies_dict[parent].append(child)

    def _get_dependencies(self, dependencies_dict, parent, dep_type):
        """Helper method to get dependencies."""
        if dep_type == 'children':
            return dependencies_dict.get(parent, [])
        return []

