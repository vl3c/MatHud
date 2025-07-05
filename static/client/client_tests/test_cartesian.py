import unittest
from geometry import Point, Position
from cartesian_system_2axis import Cartesian2Axis
from .simple_mock import SimpleMock


class TestCartesian2Axis(unittest.TestCase):
    def setUp(self):
        self.canvas = SimpleMock(width=800, height=600, center=Position(400, 300), scale_factor=1, cartesian2axis=None, \
                                 zoom_direction=0, offset=Position(10, 10), zoom_point=Position(0, 0), zoom_step=0.1)
        self.cartesian_system = Cartesian2Axis(canvas=self.canvas)
        self.canvas.cartesian2axis = self.cartesian_system
        self.cartesian_system.origin = Point(x=0, y=0, canvas=self.canvas, name="o")

    def test_init(self):
        self.assertEqual(self.cartesian_system.name, "cartesian-2axis-system")
        self.assertEqual(self.cartesian_system.width, self.canvas.width)
        self.assertEqual(self.cartesian_system.height, self.canvas.height)
        self.assertEqual(self.cartesian_system.origin.x, self.canvas.center.x)
        self.assertEqual(self.cartesian_system.origin.y, self.canvas.center.y)
        self.assertEqual(self.cartesian_system.current_tick_spacing, self.cartesian_system.default_tick_spacing)

    def test_get_visible_bounds(self):
        # Test initial bounds
        left_bound = self.cartesian_system.get_visible_left_bound()
        right_bound = self.cartesian_system.get_visible_right_bound()
        top_bound = self.cartesian_system.get_visible_top_bound()
        bottom_bound = self.cartesian_system.get_visible_bottom_bound()
        
        self.assertEqual(left_bound, -400.0)  # -self.origin.x / self.canvas.scale_factor
        self.assertEqual(right_bound, 400.0)  # (self.width - self.origin.x) / self.canvas.scale_factor
        self.assertEqual(top_bound, 300.0)    # self.origin.y / self.canvas.scale_factor
        self.assertEqual(bottom_bound, -300.0)  # (self.origin.y - self.height) / self.canvas.scale_factor

        # Test bounds after zooming in
        self.canvas.scale_factor = 2
        self.cartesian_system.zoom()
        
        left_bound = self.cartesian_system.get_visible_left_bound()
        right_bound = self.cartesian_system.get_visible_right_bound()
        top_bound = self.cartesian_system.get_visible_top_bound()
        bottom_bound = self.cartesian_system.get_visible_bottom_bound()
        
        self.assertEqual(left_bound, -200.0)  # Bounds should be halved when scale_factor doubles
        self.assertEqual(right_bound, 200.0)
        self.assertEqual(top_bound, 150.0)
        self.assertEqual(bottom_bound, -150.0)

    def test_reset(self):
        # Changing the state of the Cartesian2Axis instance
        self.cartesian_system.current_tick_spacing = 50
        self.cartesian_system.reset()
        # Testing reset functionality
        self.assertEqual(self.cartesian_system.current_tick_spacing, self.cartesian_system.default_tick_spacing)
        self.assertEqual(self.cartesian_system.origin.x, self.canvas.center.x)
        self.assertEqual(self.cartesian_system.origin.y, self.canvas.center.y)

    def test_calculate_tick_spacing(self):
        # Set up a Cartesian2Axis object with a specific width, max_ticks, and canvas scale factor
        self.cartesian_system.width = 1000
        self.cartesian_system.max_ticks = 10
        self.cartesian_system.canvas.scale_factor = 2  # Add this line
        # Calculate the tick spacing
        tick_spacing = self.cartesian_system._calculate_tick_spacing()
        # The relative width is self.width / self.canvas.scale_factor: 1000 / 2 = 500.
        # The ideal tick spacing is relative_width / max_ticks: 500 / 10 = 50.
        # The order of magnitude of the ideal tick spacing is 10 ** math.floor(math.log10(ideal_tick_spacing)): 10^1 = 10.
        # The possible spacings are [10, 25, 50, 100].
        # The closest spacing to the ideal tick spacing that is larger or equal to it is 50.
        self.assertEqual(tick_spacing, 50)

    def test_zoom(self):
        # Set up a Cartesian2Axis object with specific properties
        self.cartesian_system.width = 1000
        self.cartesian_system.max_ticks = 10
        self.cartesian_system.canvas.scale_factor = 2
        self.cartesian_system.current_tick_spacing = 100
        self.cartesian_system.canvas.zoom_direction = -1  # Zoom in
        # Call the zoom method
        self.cartesian_system.zoom()
        # The relative width is 500, so the ideal tick spacing is 50.
        # The proposed tick spacing is also 50, which is less than twice the current tick spacing (200).
        # Therefore, the current tick spacing should be updated to the proposed tick spacing.
        self.assertEqual(self.cartesian_system.current_tick_spacing, 50)
        # Change the zoom direction to zoom out
        self.cartesian_system.canvas.zoom_direction = 1
        self.cartesian_system.canvas.scale_factor = 0.5  # Decrease scale factor to simulate zooming out
        # Call the zoom method again
        self.cartesian_system.zoom()
        # Now, the relative width is 2000, so the ideal tick spacing is 200.
        # The order of magnitude of the ideal tick spacing is 100.
        # The possible spacings are [100, 250, 500, 1000].
        # The closest spacing to the ideal tick spacing that is larger or equal to it is 250.
        # Therefore, the current tick spacing should be updated to 250.
        self.assertEqual(self.cartesian_system.current_tick_spacing, 250)

    def test_pan(self):
        original_origin = Position(self.cartesian_system.origin.x, self.cartesian_system.origin.y)
        self.cartesian_system.pan()
        # Verifying that the origin has been moved according to the canvas offset
        self.assertNotEqual(self.cartesian_system.origin.x, original_origin.x)
        self.assertNotEqual(self.cartesian_system.origin.y, original_origin.y)

    def test_state_retrieval(self):
        state = self.cartesian_system.get_state()
        expected_keys = ["Cartesian_System_Visibility"]
        # Test the state retrieval
        self.assertTrue(all(key in state for key in expected_keys))
        self.assertTrue(isinstance(state["Cartesian_System_Visibility"], dict))

    def test_get_axis_helpers(self):
        # Test the axis helper methods we added during refactoring
        self.assertEqual(self.cartesian_system._get_axis_origin('x'), self.cartesian_system.origin.x)
        self.assertEqual(self.cartesian_system._get_axis_origin('y'), self.cartesian_system.origin.y)
        
        self.assertEqual(self.cartesian_system._get_axis_boundary('x'), self.cartesian_system.width)
        self.assertEqual(self.cartesian_system._get_axis_boundary('y'), self.cartesian_system.height)
    
    def test_should_continue_drawing(self):
        # Test the boundary condition method for drawing
        # Position within boundary, direction positive
        self.assertTrue(self.cartesian_system._should_continue_drawing(50, 100, 1))
        # Position at boundary, direction positive
        self.assertFalse(self.cartesian_system._should_continue_drawing(100, 100, 1))
        # Position beyond boundary, direction positive
        self.assertFalse(self.cartesian_system._should_continue_drawing(150, 100, 1))
        
        # Position within boundary, direction negative
        self.assertTrue(self.cartesian_system._should_continue_drawing(50, 100, -1))
        # Position at zero, direction negative
        self.assertFalse(self.cartesian_system._should_continue_drawing(0, 100, -1))
        # Position below zero, direction negative
        self.assertFalse(self.cartesian_system._should_continue_drawing(-10, 100, -1))
    
    def test_calculate_ideal_tick_spacing(self):
        # Test the refactored tick spacing calculation
        self.cartesian_system.width = 1000
        self.cartesian_system.max_ticks = 10
        self.cartesian_system.canvas.scale_factor = 2
        
        ideal_spacing = self.cartesian_system._calculate_ideal_tick_spacing()
        self.assertEqual(ideal_spacing, 50)  # 1000/2/10 = 50
        
        # Test with different values
        self.cartesian_system.width = 800
        self.cartesian_system.max_ticks = 8
        self.cartesian_system.canvas.scale_factor = 1
        
        ideal_spacing = self.cartesian_system._calculate_ideal_tick_spacing()
        self.assertEqual(ideal_spacing, 100)  # 800/1/8 = 100
    
    def test_find_appropriate_spacing(self):
        # Test the spacing selection logic
        # When ideal spacing is exactly a standard spacing value
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(10), 10)
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(25), 25)
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(50), 50)
        
        # When ideal spacing falls between standard values
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(15), 25)
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(30), 50)
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(6), 10)
        
        # When ideal spacing is very small
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(0.15), 0.25)
        
        # When ideal spacing is very large
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(750), 1000)
    
    def test_relative_dimensions(self):
        # Test the relative width and height calculations
        self.cartesian_system.width = 800
        self.cartesian_system.height = 600
        self.cartesian_system.canvas.scale_factor = 2
        
        self.assertEqual(self.cartesian_system.get_relative_width(), 400)
        self.assertEqual(self.cartesian_system.get_relative_height(), 300)
        
        # Test with different scale factor
        self.cartesian_system.canvas.scale_factor = 0.5
        self.assertEqual(self.cartesian_system.get_relative_width(), 1600)
        self.assertEqual(self.cartesian_system.get_relative_height(), 1200)
