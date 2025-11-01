import unittest
from managers.drawables_container import DrawablesContainer
from .simple_mock import SimpleMock


class TestDrawablesContainer(unittest.TestCase):
    """Test cases for the DrawablesContainer class."""
    
    def setUp(self) -> None:
        """Set up test fixtures before each test."""
        self.container = DrawablesContainer()
        # Create mock drawables of different types
        self.point = SimpleMock(get_class_name=SimpleMock(return_value="Point"), 
                               get_state=SimpleMock(return_value={"name": "A", "coords": [1, 2]}))
        self.segment = SimpleMock(get_class_name=SimpleMock(return_value="Segment"), 
                                 get_state=SimpleMock(return_value={"name": "AB", "points": ["A", "B"]}))
        self.circle = SimpleMock(get_class_name=SimpleMock(return_value="Circle"), 
                                get_state=SimpleMock(return_value={"name": "c1", "center": "A", "radius": 5}))
        
    def test_init(self) -> None:
        """Test initialization of the container."""
        container = DrawablesContainer()
        self.assertEqual(len(container.get_all()), 0, "New container should be empty")
        
    def test_add(self) -> None:
        """Test adding drawables to the container."""
        self.container.add(self.point)
        self.assertEqual(len(self.container.get_all()), 1, "Container should have 1 drawable")
        
        self.container.add(self.segment)
        self.assertEqual(len(self.container.get_all()), 2, "Container should have 2 drawables")
        
        # Test adding multiple drawables of the same type
        point2 = SimpleMock(get_class_name=SimpleMock(return_value="Point"), 
                           get_state=SimpleMock(return_value={"name": "B", "coords": [3, 4]}))
        self.container.add(point2)
        self.assertEqual(len(self.container.get_all()), 3, "Container should have 3 drawables")
        self.assertEqual(len(self.container.get_by_class_name("Point")), 2, "Container should have 2 Points")
        
    def test_remove(self) -> None:
        """Test removing drawables from the container."""
        # Add drawables first
        self.container.add(self.point)
        self.container.add(self.segment)
        self.assertEqual(len(self.container.get_all()), 2, "Container should have 2 drawables")
        
        # Remove a drawable
        result = self.container.remove(self.point)
        self.assertTrue(result, "Remove should return True when successful")
        self.assertEqual(len(self.container.get_all()), 1, "Container should have 1 drawable after removal")
        
        # Try to remove a non-existent drawable
        non_existent = SimpleMock(get_class_name=SimpleMock(return_value="Triangle"))
        result = self.container.remove(non_existent)
        self.assertFalse(result, "Remove should return False when drawable not found")
        
    def test_get_by_class_name(self) -> None:
        """Test getting drawables by class name."""
        # Add drawables of different types
        self.container.add(self.point)
        self.container.add(self.segment)
        self.container.add(self.circle)
        
        points = self.container.get_by_class_name("Point")
        self.assertEqual(len(points), 1, "Should have 1 Point")
        self.assertEqual(points[0], self.point, "Should return the correct Point")
        
        # Test with non-existent class name
        non_existent = self.container.get_by_class_name("NonExistent")
        self.assertEqual(len(non_existent), 0, "Should return empty list for non-existent class")
        
    def test_get_all(self) -> None:
        """Test getting all drawables."""
        # Empty container
        self.assertEqual(len(self.container.get_all()), 0, "Empty container should return empty list")
        
        # Add drawables
        self.container.add(self.point)
        self.container.add(self.segment)
        self.container.add(self.circle)
        
        all_drawables = self.container.get_all()
        self.assertEqual(len(all_drawables), 3, "Should return all 3 drawables")
        self.assertIn(self.point, all_drawables, "Point should be in the result")
        self.assertIn(self.segment, all_drawables, "Segment should be in the result")
        self.assertIn(self.circle, all_drawables, "Circle should be in the result")
        
    def test_clear(self) -> None:
        """Test clearing all drawables."""
        # Add drawables
        self.container.add(self.point)
        self.container.add(self.segment)
        self.container.add(self.circle)
        self.assertEqual(len(self.container.get_all()), 3, "Should have 3 drawables")
        
        # Clear container
        self.container.clear()
        self.assertEqual(len(self.container.get_all()), 0, "Container should be empty after clear")
        
    def test_get_state(self) -> None:
        """Test getting state of all drawables."""
        # Add drawables
        self.container.add(self.point)
        self.container.add(self.segment)
        
        state = self.container.get_state()
        self.assertIn("Points", state, "State should include Points")
        self.assertIn("Segments", state, "State should include Segments")
        self.assertEqual(len(state["Points"]), 1, "Should have 1 Point state")
        self.assertEqual(len(state["Segments"]), 1, "Should have 1 Segment state")
        self.assertEqual(state["Points"][0], {"name": "A", "coords": [1, 2]}, "Point state should match")
        
    def test_property_accessors(self) -> None:
        """Test property-style access for drawable types."""
        # Add drawables
        self.container.add(self.point)
        self.container.add(self.segment)
        self.container.add(self.circle)
        
        # Test property accessors
        points = self.container.Points
        segments = self.container.Segments
        circles = self.container.Circles
        
        self.assertEqual(len(points), 1, "Should have 1 Point")
        self.assertEqual(len(segments), 1, "Should have 1 Segment")
        self.assertEqual(len(circles), 1, "Should have 1 Circle")
        
        # Test property accessor for non-existent type
        triangles = self.container.Triangles
        self.assertEqual(len(triangles), 0, "Should return empty list for non-existent type")
        
    def test_dictionary_access(self) -> None:
        """Test dictionary-like access to drawable types."""
        # Add drawables
        self.container.add(self.point)
        self.container.add(self.segment)
        
        # Test __getitem__
        points = self.container["Point"]
        self.assertEqual(len(points), 1, "Should have 1 Point")
        self.assertEqual(points[0], self.point, "Should return the correct Point")
        
        # Test __contains__
        self.assertTrue("Point" in self.container, "Container should contain Point")
        self.assertTrue("Segment" in self.container, "Container should contain Segment")
        self.assertFalse("Circle" in self.container, "Container should not contain Circle") 