from __future__ import annotations

import unittest
from typing import Any, List

from name_generator.drawable import DrawableNameGenerator
from .simple_mock import SimpleMock


class TestDrawableNameGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas: Any = SimpleMock()
        # Ensure canvas has the get_drawables_by_class_name method from the start
        self.get_drawables_mock = SimpleMock(return_value=[])
        setattr(self.canvas, "get_drawables_by_class_name", self.get_drawables_mock)
        self.generator = DrawableNameGenerator(self.canvas)

    def test_get_drawable_names(self) -> None:
        # Here, get_drawables_by_class_name is expected to be a callable that returns a list of mocks when called
        set_drawables = SimpleMock(
            return_value=[SimpleMock(name='Point1'), SimpleMock(name='Point2')]
        )
        setattr(self.canvas, "get_drawables_by_class_name", set_drawables)
        result = self.generator.get_drawable_names('Point')
        self.assertEqual(result, ['Point1', 'Point2'])

    def test_filter_string(self) -> None:
        # Test with a string that contains letters, digits, apostrophes, and parentheses
        result = self.generator.filter_string("Test(123)'")
        self.assertEqual(result, "Test(123)'")
        # Test with a string that contains special characters
        result = self.generator.filter_string("Hello, World!")
        self.assertEqual(result, "HelloWorld")
        # Test with a string that contains whitespace
        result = self.generator.filter_string("Hello World")
        self.assertEqual(result, "HelloWorld")
        # Test with a string that contains only disallowed characters
        result = self.generator.filter_string("!@#$%^&*")
        self.assertEqual(result, "")
        # Test with an empty string
        result = self.generator.filter_string("")
        self.assertEqual(result, "")

    def test_print_names(self) -> None:
        """Test the print_names method outputs the expected information."""
        # Create a completely fresh generator with a new mock canvas
        mock_canvas: Any = SimpleMock()
        
        # Define predictable returns for each class name
        mock_returns = {
            'Point': [SimpleMock(name='Point1'), SimpleMock(name='Point2')],
            'Segment': [SimpleMock(name='Segment1'), SimpleMock(name='Segment2')],
            'Triangle': [SimpleMock(name='Triangle1'), SimpleMock(name='Triangle2')],
            'Rectangle': [SimpleMock(name='Rectangle1'), SimpleMock(name='Rectangle2')],
            'Circle': [SimpleMock(name='Circle1'), SimpleMock(name='Circle2')],
            'Ellipse': [SimpleMock(name='Ellipse1'), SimpleMock(name='Ellipse2')],
            'Function': [SimpleMock(name='Function1'), SimpleMock(name='Function2')]
        }
        
        # Set up the canvas.get_drawables_by_class_name to return the appropriate mock objects
        def mock_get_drawables(class_name: str) -> List[SimpleMock]:
            return mock_returns.get(class_name, [])

        setattr(mock_canvas, "get_drawables_by_class_name", mock_get_drawables)
        
        # Create a test generator with our fully controlled mock canvas
        test_generator = DrawableNameGenerator(mock_canvas)
        
        # Capture print output
        original_print = __builtins__.print
        printed_lines = []
        
        def mock_print(*args: object, **kwargs: object) -> None:
            line = ' '.join(str(arg) for arg in args)
            printed_lines.append(line)
        
        # Replace print with our mock
        __builtins__.print = mock_print
        
        try:
            # Call the method being tested
            test_generator.print_names()
        finally:
            # Restore original print function
            __builtins__.print = original_print
        
        # Check the output
        expected_lines = [
            "Point names: ['Point1', 'Point2']",
            "Segment names: ['Segment1', 'Segment2']",
            "Triangle names: ['Triangle1', 'Triangle2']",
            "Rectangle names: ['Rectangle1', 'Rectangle2']",
            "Circle names: ['Circle1', 'Circle2']",
            "Ellipse names: ['Ellipse1', 'Ellipse2']",
            "Function names: ['Function1', 'Function2']"
        ]
        
        # Compare line by line for easier debugging
        self.assertEqual(len(printed_lines), len(expected_lines), 
                         f"Expected {len(expected_lines)} lines but got {len(printed_lines)}")
        for i, (actual, expected) in enumerate(zip(printed_lines, expected_lines)):
            self.assertEqual(actual, expected, f"Line {i+1} doesn't match: {actual} != {expected}")

    def test_split_point_names_basic(self) -> None:
        result = self.generator.split_point_names("A'B'CD", 4)
        self.assertEqual(result, ["A'", "B'", "C", "D"])
    
    def test_split_point_names_empty(self) -> None:
        # Test with empty expression
        result = self.generator.split_point_names("", 3)
        self.assertEqual(result, ["", "", ""])
        
        # Test with None expression
        result = self.generator.split_point_names(None, 2)
        self.assertEqual(result, ["", ""])
    
    def test_split_point_names_with_special_chars(self) -> None:
        # Test with expression containing special characters
        result = self.generator.split_point_names("A@B#C", 3)
        self.assertEqual(result, ["A", "B", "C"])
    
    def test_split_point_names_with_repeated_calls(self) -> None:
        # First call should return first n letters
        result1 = self.generator.split_point_names("ABCDE", 2)
        self.assertEqual(result1, ["A", "B"])
        
        # Second call with same expression should return next n letters
        result2 = self.generator.split_point_names("ABCDE", 2)
        self.assertEqual(result2, ["C", "D"])
        
        # Third call with same expression should return remaining letters and empty strings if needed
        result3 = self.generator.split_point_names("ABCDE", 2)
        self.assertEqual(result3, ["E", ""])
    
    def test_generate_unique_point_name(self) -> None:
        self.get_drawables_mock = SimpleMock(return_value=[])
        setattr(self.canvas, "get_drawables_by_class_name", self.get_drawables_mock)
        point_names = []
        for _ in range(52):
            new_name = self.generator._generate_unique_point_name()
            point_names.append(new_name)
            # Update the return_value of get_drawables_by_class_name with each new name
            self.get_drawables_mock.return_value.append(SimpleMock(name=new_name))
        # The next point name should be 'A'' (A with two apostrophes)
        result = self.generator._generate_unique_point_name()
        self.assertEqual(result, "A''")

    def test_generate_point_name(self) -> None:
        setattr(
            self.canvas,
            "get_drawables_by_class_name",
            SimpleMock(return_value=[SimpleMock(name='A')]),
        )
        result = self.generator.generate_point_name(None)
        self.assertEqual(result, 'B')

    def test_generate_point_name_with_preferred_name(self) -> None:
        setattr(
            self.canvas,
            "get_drawables_by_class_name",
            SimpleMock(return_value=[SimpleMock(name='A')]),
        )
        result = self.generator.generate_point_name('B')
        self.assertEqual(result, 'B')

    def test_generate_point_name_with_used_preferred_name(self) -> None:
        setattr(
            self.canvas,
            "get_drawables_by_class_name",
            SimpleMock(return_value=[SimpleMock(name='A'), SimpleMock(name='B')]),
        )
        result = self.generator.generate_point_name('B')
        self.assertNotEqual(result, 'B')

    def test_generate_point_name_with_complex_preferred_name(self) -> None:
        # When we pass "AB'C" as preferred_name, and 'A' is already used,
        # the code should attempt to generate 'A'' (A with an apostrophe)
        # rather than moving to next letter 'B''
        setattr(
            self.canvas,
            "get_drawables_by_class_name",
            SimpleMock(return_value=[SimpleMock(name='A'), SimpleMock(name="B'")]),
        )
        
        # Reset the dictionary for a clean test
        self.generator.used_letters_from_names = {}
        
        result = self.generator.generate_point_name("AB'C")
        # The algorithm tries A first, but it's taken
        # So it adds an apostrophe to get A', which is not taken
        self.assertEqual(result, "A'")  # Should return A', not C

    def test_increment_function_name(self) -> None:
        # Test with a function name that ends with a number
        result = self.generator._increment_function_name('f4')
        self.assertEqual(result, 'f5')
        # Test with a function name that does not end with a number
        result = self.generator._increment_function_name('f')
        self.assertEqual(result, 'f1')
        # Test with a function name that ends with a large number
        result = self.generator._increment_function_name('f99')
        self.assertEqual(result, 'f100')
        # Test with a function name that ends with a number and has other numbers in it
        result = self.generator._increment_function_name('f4f4')
        self.assertEqual(result, 'f4f5')
        # Test with a function name that does not end with a number and has other numbers in it
        result = self.generator._increment_function_name('f4f')
        self.assertEqual(result, 'f4f1')

    def test_generate_unique_function_name(self) -> None:
        self.canvas.get_drawables_by_class_name = SimpleMock(return_value=[])
        function_names = []
        for _ in range(42):
            new_name = self.generator._generate_unique_function_name()
            function_names.append(new_name)
            # Update the return_value of get_drawables_by_class_name with each new name
            self.canvas.get_drawables_by_class_name.return_value.append(SimpleMock(name=new_name))
        # The next function name should be 'r1'
        result = self.generator._generate_unique_function_name()
        self.assertEqual(result, "v1")

    def test_generate_function_name(self) -> None:
        self.canvas.get_drawables_by_class_name = SimpleMock(
            return_value=[SimpleMock(name='f'), SimpleMock(name='f1')]
        )
        result = self.generator.generate_function_name(None)
        self.assertEqual(result, 'g')

    def test_generate_function_name_with_preferred_name(self) -> None:
        self.canvas.get_drawables_by_class_name = SimpleMock(
            return_value=[SimpleMock(name='f1')]
        )
        result = self.generator.generate_function_name('f2')
        self.assertEqual(result, 'f2')

    def test_generate_function_name_with_used_preferred_name(self) -> None:
        self.canvas.get_drawables_by_class_name = SimpleMock(
            return_value=[SimpleMock(name='f1'), SimpleMock(name='f2')]
        )
        result = self.generator.generate_function_name('f2')
        self.assertEqual(result, 'f3')

    def test_generate_function_name_with_preferred_name_and_parentheses(self) -> None:
        self.canvas.get_drawables_by_class_name = SimpleMock(
            return_value=[SimpleMock(name='f1')]
        )
        result = self.generator.generate_function_name('f2(x)')
        self.assertEqual(result, 'f2')

    def test_generate_function_name_with_used_preferred_name_and_parentheses(self) -> None:
        self.canvas.get_drawables_by_class_name = SimpleMock(
            return_value=[SimpleMock(name='f1'), SimpleMock(name='f2')]
        )
        result = self.generator.generate_function_name('f2(x)')
        self.assertEqual(result, 'f3')

    def test_generate_function_name_with_complex_expression(self) -> None:
        self.canvas.get_drawables_by_class_name = SimpleMock(
            return_value=[SimpleMock(name='f1'), SimpleMock(name='f2')]
        )
        result = self.generator.generate_function_name('g(x) = sin(x)')
        self.assertEqual(result, 'g')

    def test_generate_angle_name_from_segments_valid(self) -> None:
        # Standard case: AB, AC -> angle_BAC (assuming B, C sorted)
        result = self.generator.generate_angle_name_from_segments("AB", "AC")
        self.assertEqual(result, "angle_BAC")

        # Test with apostrophes, ensuring sorting of arm points
        # BA', BC' -> common B, arms A', C' -> sorted A', C' -> angle_A'BC'
        result_apostrophe = self.generator.generate_angle_name_from_segments("BA'", "BC'")
        self.assertEqual(result_apostrophe, "angle_A'BC'")

        # Test where arm points might be out of order if not sorted
        # CA, CB -> common C, arms A, B -> sorted A,B -> angle_ACB
        result_order = self.generator.generate_angle_name_from_segments("CA", "CB")
        self.assertEqual(result_order, "angle_ACB")

        # New test cases for multiple apostrophes and specific sorting

        # Case 1: Segments P'Q, P'R -> Vertex P', Arms Q, R (sorted) -> angle_QP'R
        name1 = self.generator.generate_angle_name_from_segments("P'Q", "P'R")
        self.assertEqual(name1, "angle_QP'R")

        # Case 2: Segments QP', RP' -> Vertex P', Arms Q, R (sorted) -> angle_QP'R (order of segs doesn't matter)
        name2 = self.generator.generate_angle_name_from_segments("QP'", "RP'")
        self.assertEqual(name2, "angle_QP'R")

        # Case 3: Segments P'P'', P'P''' -> Vertex P', Arms P'', P''' (sorted) -> angle_P''P'P'''
        # Note: Sorting "P''" and "P'''": "P''" comes before "P'''" alphabetically.
        name3 = self.generator.generate_angle_name_from_segments("P'P''", "P'P'''")
        self.assertEqual(name3, "angle_P''P'P'''")
        
        # Case 4: Segments P'''P', P''P' -> Vertex P', Arms P''', P'' (sorted P'', P''') -> angle_P''P'P'''
        name4 = self.generator.generate_angle_name_from_segments("P'''P'", "P''P'")
        self.assertEqual(name4, "angle_P''P'P'''")

        # Case 5: Segments A'B'', A'C''' -> Vertex A', Arms B'', C''' (sorted) -> angle_B''A'C'''
        name5 = self.generator.generate_angle_name_from_segments("A'B''", "A'C'''")
        self.assertEqual(name5, "angle_B''A'C'''")

    def test_generate_angle_name_invalid_no_common_vertex(self) -> None:
        name = self.generator.generate_angle_name_from_segments("AB", "CD")
        self.assertIsNone(name)

    def test_generate_angle_name_collinear_same_segment_implicitly(self) -> None:
        # e.g. AB, BA - this implies 2 points, not 3 unique points for an angle
        name = self.generator.generate_angle_name_from_segments("AB", "BA")
        self.assertIsNone(name) # Should result in 2 unique points, not 3

    def test_generate_angle_name_identical_segments(self) -> None:
        result = self.generator.generate_angle_name_from_segments("AB", "AB")
        self.assertIsNone(result)

    def test_generate_angle_name_malformed_segment_names(self) -> None:
        # Test that generate_angle_name_from_segments returns None if a segment name
        # is too short (e.g., single letter), causing split_point_names to produce
        # an invalid list like ['A', ''].

        # Case 1: First segment name is a single letter "A".
        # split_point_names("A") -> ["A", ""], which is invalid.
        result = self.generator.generate_angle_name_from_segments("A", "BC") # BC is valid
        self.assertIsNone(result, "Should be None for segment 'A'.")

        # Case 2: Second segment name is a single letter "A".
        result = self.generator.generate_angle_name_from_segments("BC", "A") # BC is valid
        self.assertIsNone(result, "Should be None for segment 'A' (second arg)." )

        # Case 3: Segment name is a single letter with an apostrophe "A'".
        # split_point_names("A'") -> ["A'", ""], which is invalid.
        result = self.generator.generate_angle_name_from_segments("A'", "BC")
        self.assertIsNone(result, "Should be None for segment 'A\''.")

    def test_generate_angle_name_empty_or_none_segment_names(self) -> None:
        name = self.generator.generate_angle_name_from_segments("", "AB")
        self.assertIsNone(name)
        name = self.generator.generate_angle_name_from_segments("AB", "")
        self.assertIsNone(name)
        name = self.generator.generate_angle_name_from_segments(None, "AB")
        self.assertIsNone(name)
        name = self.generator.generate_angle_name_from_segments("AB", None)
        self.assertIsNone(name)
        name = self.generator.generate_angle_name_from_segments(None, None)
        self.assertIsNone(name)

    # -------------------------------------------------------------------------
    # Arc Name Generation Tests
    # -------------------------------------------------------------------------

    def test_extract_point_names_from_arc_name_simple(self) -> None:
        """Test extracting point names from simple arc name suggestions."""
        p1, p2 = self.generator.extract_point_names_from_arc_name("AB")
        self.assertEqual(p1, "A")
        self.assertEqual(p2, "B")

    def test_extract_point_names_from_arc_name_with_primes(self) -> None:
        """Test extracting point names with prime symbols."""
        p1, p2 = self.generator.extract_point_names_from_arc_name("A'B''")
        self.assertEqual(p1, "A'")
        self.assertEqual(p2, "B''")

    def test_extract_point_names_from_arc_name_with_prefix(self) -> None:
        """Test extracting point names when arc prefix is present."""
        p1, p2 = self.generator.extract_point_names_from_arc_name("ArcMin_CD")
        self.assertEqual(p1, "C")
        self.assertEqual(p2, "D")
        
        p1, p2 = self.generator.extract_point_names_from_arc_name("ArcMaj_E'F''")
        self.assertEqual(p1, "E'")
        self.assertEqual(p2, "F''")
        
        p1, p2 = self.generator.extract_point_names_from_arc_name("arc_XY")
        self.assertEqual(p1, "X")
        self.assertEqual(p2, "Y")

    def test_extract_point_names_from_arc_name_prefix_only(self) -> None:
        """Test extracting from names that are just prefixes with nothing after."""
        p1, p2 = self.generator.extract_point_names_from_arc_name("ArcMajor")
        self.assertIsNone(p1)
        self.assertIsNone(p2)
        
        p1, p2 = self.generator.extract_point_names_from_arc_name("ArcMinor")
        self.assertIsNone(p1)
        self.assertIsNone(p2)

    def test_extract_point_names_from_arc_name_empty(self) -> None:
        """Test extracting from empty or None arc names."""
        p1, p2 = self.generator.extract_point_names_from_arc_name("")
        self.assertIsNone(p1)
        self.assertIsNone(p2)
        
        p1, p2 = self.generator.extract_point_names_from_arc_name(None)
        self.assertIsNone(p1)
        self.assertIsNone(p2)

    def test_extract_point_names_from_arc_name_complex(self) -> None:
        """Test extracting point names from complex expressions."""
        p1, p2 = self.generator.extract_point_names_from_arc_name("A''E'")
        self.assertEqual(p1, "A''")
        self.assertEqual(p2, "E'")

    def test_generate_arc_name_minor(self) -> None:
        """Test generating minor arc names."""
        result = self.generator.generate_arc_name(None, "A", "B", False, set())
        self.assertEqual(result, "ArcMin_AB")

    def test_generate_arc_name_major(self) -> None:
        """Test generating major arc names."""
        result = self.generator.generate_arc_name(None, "A", "B", True, set())
        self.assertEqual(result, "ArcMaj_AB")

    def test_generate_arc_name_with_primes(self) -> None:
        """Test generating arc names with prime symbols in point names."""
        result = self.generator.generate_arc_name(None, "A'", "B''", False, set())
        self.assertEqual(result, "ArcMin_A'B''")
        
        result = self.generator.generate_arc_name(None, "C''", "D'''", True, set())
        self.assertEqual(result, "ArcMaj_C''D'''")

    def test_generate_arc_name_avoids_collision(self) -> None:
        """Test that arc name generation avoids existing names."""
        existing = {"ArcMin_AB"}
        result = self.generator.generate_arc_name(None, "A", "B", False, existing)
        self.assertEqual(result, "ArcMin_AB_1")
        
        existing = {"ArcMin_AB", "ArcMin_AB_1"}
        result = self.generator.generate_arc_name(None, "A", "B", False, existing)
        self.assertEqual(result, "ArcMin_AB_2")

    def test_generate_arc_name_uses_proposed_with_prefix(self) -> None:
        """Test that proposed name with proper prefix is used directly."""
        result = self.generator.generate_arc_name("ArcMaj_CustomName", "A", "B", True, set())
        self.assertEqual(result, "ArcMaj_CustomName")
        
        result = self.generator.generate_arc_name("ArcMin_XY", "A", "B", False, set())
        self.assertEqual(result, "ArcMin_XY")

    def test_generate_arc_name_extracts_from_proposed(self) -> None:
        """Test that point names are extracted from proposed name."""
        # "arc_XY" strips prefix, extracts X and Y
        result = self.generator.generate_arc_name("arc_XY", "C", "D", False, set())
        self.assertEqual(result, "ArcMin_XY")
        
        # "ArcMaj_PQ" strips prefix, extracts P and Q
        result = self.generator.generate_arc_name("ArcMaj_PQ", "C", "D", True, set())
        self.assertEqual(result, "ArcMaj_PQ")

    def test_generate_arc_name_fallback_when_prefix_only(self) -> None:
        """Test fallback to point names when proposed name has nothing after prefix."""
        # "ArcMajor" strips to empty, falls back to provided point names
        result = self.generator.generate_arc_name("ArcMajor", "C", "D", True, set())
        self.assertEqual(result, "ArcMaj_CD")
        
        # "ArcMinor" strips to empty, falls back to provided point names
        result = self.generator.generate_arc_name("ArcMinor", "E", "F", False, set())
        self.assertEqual(result, "ArcMin_EF")

    def test_generate_arc_name_proposed_with_collision(self) -> None:
        """Test proposed name with collision gets suffix."""
        existing = {"ArcMaj_AB"}
        result = self.generator.generate_arc_name("ArcMaj_AB", "A", "B", True, existing)
        self.assertEqual(result, "ArcMaj_AB_1")