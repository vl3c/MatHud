from __future__ import annotations

import unittest

from managers.drawable_dependency_manager import DrawableDependencyManager
from client_tests.simple_mock import SimpleMock


class TestDrawableDependencyManager(unittest.TestCase):
    """
    Unit tests for the DrawableDependencyManager class.
    These tests verify dependency tracking, parent/child relationships,
    and canvas reference propagation.
    """

    def _create_mock_point(self, name: str, x: float = 0, y: float = 0) -> SimpleMock:
        """Factory function to create a point mock with SimpleMock"""
        return SimpleMock(
            name=name,
            x=x,
            y=y,
            canvas=None,
            get_class_name=SimpleMock(return_value='Point'),
            __str__=SimpleMock(return_value=f"Point({name})"),
            __repr__=SimpleMock(return_value=f"Point({name})")
        )

    def _create_mock_segment(self, name: str, point1: SimpleMock, point2: SimpleMock) -> SimpleMock:
        """Factory function to create a segment mock with SimpleMock"""
        return SimpleMock(
            name=name,
            point1=point1,
            point2=point2,
            canvas=None,
            get_class_name=SimpleMock(return_value='Segment'),
            __str__=SimpleMock(return_value=f"Segment({name})"),
            __repr__=SimpleMock(return_value=f"Segment({name})")
        )

    def _create_mock_drawable(self, name: str, class_name: str = "MockDrawable") -> SimpleMock:
        """Factory function to create a generic drawable mock with SimpleMock"""
        return SimpleMock(
            name=name,
            _class_name=class_name,
            canvas=None,
            get_class_name=SimpleMock(return_value=class_name),
            __str__=SimpleMock(return_value=f"{class_name}({name})"),
            __repr__=SimpleMock(return_value=f"{class_name}({name})")
        )

    def setUp(self) -> None:
        """Set up test environment before each test"""
        # Create a mock drawable manager
        self.mock_drawable_manager = SimpleMock(
            drawables=SimpleMock(
                Segments=[]
            )
        )
        self.manager = DrawableDependencyManager(drawable_manager_proxy=self.mock_drawable_manager)

        # Create mock drawables using private factory methods
        self.point1 = self._create_mock_point("P1", 100, 100)
        self.point2 = self._create_mock_point("P2", 200, 100)
        self.point3 = self._create_mock_point("P3", 200, 200)
        self.segment1 = self._create_mock_segment("S1", self.point1, self.point2)
        self.segment2 = self._create_mock_segment("S2", self.point2, self.point3)
        self.segment3 = self._create_mock_segment("S3", self.point3, self.point1)
        self.triangle = self._create_mock_drawable("T1", "Triangle")

    def _assert_internal_graph_invariants(self) -> None:
        for child_id, parent_ids in self.manager._parents.items():
            self.assertIn(child_id, self.manager._object_lookup)
            for parent_id in parent_ids:
                self.assertIn(parent_id, self.manager._object_lookup)
                self.assertIn(child_id, self.manager._children.get(parent_id, set()))

        for parent_id, child_ids in self.manager._children.items():
            self.assertIn(parent_id, self.manager._object_lookup)
            for child_id in child_ids:
                self.assertIn(child_id, self.manager._object_lookup)
                self.assertIn(parent_id, self.manager._parents.get(child_id, set()))

        for drawable_id in self.manager._object_lookup:
            self.assertTrue(
                drawable_id in self.manager._parents or drawable_id in self.manager._children
            )

    def test_register_dependency(self) -> None:
        """Test registering dependencies between drawables"""
        # Register segment1 depends on point1 and point2
        self.manager.register_dependency(child=self.segment1, parent=self.point1)
        self.manager.register_dependency(child=self.segment1, parent=self.point2)

        # Check that segment1 has two parents
        parents = self.manager.get_parents(self.segment1)
        self.assertEqual(len(parents), 2, "Segment1 should have 2 parents")
        self.assertIn(self.point1, parents, "Point1 should be a parent of Segment1")
        self.assertIn(self.point2, parents, "Point2 should be a parent of Segment1")

        # Check that point1 and point2 each have segment1 as a child
        children_of_point1 = self.manager.get_children(self.point1)
        self.assertEqual(len(children_of_point1), 1, "Point1 should have 1 child")
        self.assertIn(self.segment1, children_of_point1, "Segment1 should be a child of Point1")

        children_of_point2 = self.manager.get_children(self.point2)
        self.assertEqual(len(children_of_point2), 1, "Point2 should have 1 child")
        self.assertIn(self.segment1, children_of_point2, "Segment1 should be a child of Point2")

    def test_get_all_parents(self) -> None:
        """Test getting all parents recursively"""
        # Set up a hierarchical dependency structure
        # Points are parents of segments
        self.manager.register_dependency(child=self.segment1, parent=self.point1)
        self.manager.register_dependency(child=self.segment1, parent=self.point2)
        self.manager.register_dependency(child=self.segment2, parent=self.point2)
        self.manager.register_dependency(child=self.segment2, parent=self.point3)
        self.manager.register_dependency(child=self.segment3, parent=self.point3)
        self.manager.register_dependency(child=self.segment3, parent=self.point1)

        # Check that segment1 has points as parents
        all_parents = self.manager.get_all_parents(self.segment1)
        self.assertEqual(len(all_parents), 2, "Segment1 should have 2 total parents")
        self.assertIn(self.point1, all_parents, "Point1 should be in Segment1's all parents")
        self.assertIn(self.point2, all_parents, "Point2 should be in Segment1's all parents")

    def test_get_all_children(self) -> None:
        """Test getting all children recursively"""
        # Set up a hierarchical dependency structure
        # Points are parents of segments
        self.manager.register_dependency(child=self.segment1, parent=self.point1)
        self.manager.register_dependency(child=self.segment1, parent=self.point2)
        self.manager.register_dependency(child=self.segment2, parent=self.point2)
        self.manager.register_dependency(child=self.segment2, parent=self.point3)
        self.manager.register_dependency(child=self.segment3, parent=self.point3)
        self.manager.register_dependency(child=self.segment3, parent=self.point1)

        # Check that point1 has segment1 and segment3 as children
        all_children = self.manager.get_all_children(self.point1)
        self.assertEqual(len(all_children), 2, "Point1 should have 2 total children")
        self.assertIn(self.segment1, all_children, "Segment1 should be in Point1's all children")
        self.assertIn(self.segment3, all_children, "Segment3 should be in Point1's all children")

    def test_remove_drawable(self) -> None:
        """Test removing a drawable and its dependencies"""
        # Set up dependencies
        self.manager.register_dependency(child=self.segment1, parent=self.point1)
        self.manager.register_dependency(child=self.segment1, parent=self.point2)

        # Remove segment1
        self.manager.remove_drawable(self.segment1)

        # Check that segment1 is removed from point1's children
        children_of_point1 = self.manager.get_children(self.point1)
        self.assertEqual(len(children_of_point1), 0, "Point1 should have no children after removal")

        # Check that segment1 is removed from point2's children
        children_of_point2 = self.manager.get_children(self.point2)
        self.assertEqual(len(children_of_point2), 0, "Point2 should have no children after removal")

        # Check that segment1 has no parents
        parents = self.manager.get_parents(self.segment1)
        self.assertEqual(len(parents), 0, "Segment1 should have no parents after removal")
        self._assert_internal_graph_invariants()

    def test_resolve_dependency_order(self) -> None:
        """Test resolving dependencies in the correct order"""
        # Set up dependencies: Segments -> Points (points are parents of segments)
        self.manager.register_dependency(child=self.segment1, parent=self.point1)
        self.manager.register_dependency(child=self.segment1, parent=self.point2)
        self.manager.register_dependency(child=self.segment2, parent=self.point2)
        self.manager.register_dependency(child=self.segment2, parent=self.point3)
        self.manager.register_dependency(child=self.segment3, parent=self.point3)
        self.manager.register_dependency(child=self.segment3, parent=self.point1)

        # Resolve the order (points should come before segments)
        drawables = [self.segment1, self.point2, self.point1, self.segment3, self.segment2, self.point3]
        ordered = self.manager.resolve_dependency_order(drawables)

        # Check that points come before their segments
        for point in [self.point1, self.point2, self.point3]:
            point_index = ordered.index(point)
            for segment in [self.segment1, self.segment2, self.segment3]:
                if point in self.manager.get_parents(segment):
                    segment_index = ordered.index(segment)
                    self.assertLess(point_index, segment_index,
                                   f"Point {point.name} should come before its segment {segment.name}")

    def test_circular_dependencies(self) -> None:
        """Test handling of circular dependencies"""
        # Create a circular dependency: A -> B -> C -> A
        a = self._create_mock_drawable("A")
        b = self._create_mock_drawable("B")
        c = self._create_mock_drawable("C")

        self.manager.register_dependency(child=b, parent=a)
        self.manager.register_dependency(child=c, parent=b)
        self.manager.register_dependency(child=a, parent=c)

        # Attempting to get all parents should not cause an infinite loop
        try:
            all_parents_a = self.manager.get_all_parents(a)
            self.assertIn(b, all_parents_a, "B should be in A's all parents")
            self.assertIn(c, all_parents_a, "C should be in A's all parents")
            # A should also be in its own all_parents due to the circular reference
            self.assertIn(a, all_parents_a, "A should be in its own all_parents (circular)")
        except RecursionError:
            self.fail("get_all_parents failed to handle circular dependency")

        # Test dependency order resolution with circular dependencies
        drawables = [a, b, c]
        try:
            ordered = self.manager.resolve_dependency_order(drawables)
            # We just care that it completes without infinite recursion
            # The exact order might vary based on implementation details for cycles
            self.assertEqual(len(ordered), 3, "All three drawables should be in the result")
        except RecursionError:
            self.fail("resolve_dependency_order failed to handle circular dependency")

    def test_analyze_drawable_for_dependencies(self) -> None:
        """Test analyzing drawable for dependencies"""
        # Test Segment
        segment = self._create_mock_segment("TestSegment", self.point1, self.point2)
        dependencies = self.manager.analyze_drawable_for_dependencies(segment)
        self.assertEqual(len(dependencies), 2, "Segment should have 2 dependencies")
        self.assertIn(self.point1, dependencies, "Point1 should be a dependency of segment")
        self.assertIn(self.point2, dependencies, "Point2 should be a dependency of segment")

        # Test Vector
        vector_segment = self._create_mock_segment("VectorSegment", self.point1, self.point2)
        vector = self._create_mock_drawable("TestVector", "Vector")
        vector.segment = vector_segment
        vector_dependencies = self.manager.analyze_drawable_for_dependencies(vector)
        self.assertEqual(len(vector_dependencies), 1, "Vector should have 1 dependency")
        self.assertIn(vector_segment, vector_dependencies, "Segment should be a dependency of vector")

        # Test Triangle
        triangle = self._create_mock_drawable("TestTriangle", "Triangle")
        triangle.segment1 = self.segment1
        triangle.segment2 = self.segment2
        triangle.segment3 = self.segment3
        triangle_dependencies = self.manager.analyze_drawable_for_dependencies(triangle)
        self.assertEqual(len(triangle_dependencies), 3, "Triangle should have 3 segment dependencies")
        self.assertIn(self.segment1, triangle_dependencies, "segment1 should be a dependency of triangle")
        self.assertIn(self.segment2, triangle_dependencies, "segment2 should be a dependency of triangle")
        self.assertIn(self.segment3, triangle_dependencies, "segment3 should be a dependency of triangle")

        # Test Rectangle
        rectangle = self._create_mock_drawable("TestRectangle", "Rectangle")
        rectangle.segment1 = self.segment1
        rectangle.segment2 = self.segment2
        rectangle.segment3 = self.segment3
        rectangle.segment4 = self._create_mock_segment("S4", self.point1, self.point3)
        rectangle_dependencies = self.manager.analyze_drawable_for_dependencies(rectangle)
        self.assertEqual(len(rectangle_dependencies), 4, "Rectangle should have 4 segment dependencies")
        self.assertIn(self.segment1, rectangle_dependencies, "segment1 should be a dependency of rectangle")
        self.assertIn(self.segment2, rectangle_dependencies, "segment2 should be a dependency of rectangle")
        self.assertIn(self.segment3, rectangle_dependencies, "segment3 should be a dependency of rectangle")
        self.assertIn(rectangle.segment4, rectangle_dependencies, "segment4 should be a dependency of rectangle")

        # Test Circle
        circle_center = self._create_mock_point("CenterPoint")
        circle = self._create_mock_drawable("TestCircle", "Circle")
        circle.center = circle_center
        circle_dependencies = self.manager.analyze_drawable_for_dependencies(circle)
        self.assertEqual(len(circle_dependencies), 1, "Circle should have 1 dependency")
        self.assertIn(circle_center, circle_dependencies, "Center point should be a dependency of circle")

        # Test Ellipse
        ellipse_center = self._create_mock_point("EllipseCenter")
        ellipse = self._create_mock_drawable("TestEllipse", "Ellipse")
        ellipse.center = ellipse_center
        ellipse_dependencies = self.manager.analyze_drawable_for_dependencies(ellipse)
        self.assertEqual(len(ellipse_dependencies), 1, "Ellipse should have 1 dependency")
        self.assertIn(ellipse_center, ellipse_dependencies, "Center point should be a dependency of ellipse")

        # Test Function
        function = self._create_mock_drawable("TestFunction", "Function")
        function_dependencies = self.manager.analyze_drawable_for_dependencies(function)
        self.assertEqual(len(function_dependencies), 0, "Function should have no dependencies")

        # Test SegmentsBoundedColoredArea
        segments_area = self._create_mock_drawable("TestSegmentsArea", "SegmentsBoundedColoredArea")
        segments_area.segment1 = self.segment1
        segments_area.segment2 = self.segment2
        segments_area_dependencies = self.manager.analyze_drawable_for_dependencies(segments_area)
        self.assertEqual(len(segments_area_dependencies), 2, "SegmentsBoundedColoredArea should have 2 dependencies")
        self.assertIn(self.segment1, segments_area_dependencies, "segment1 should be a dependency of area")
        self.assertIn(self.segment2, segments_area_dependencies, "segment2 should be a dependency of area")

        # Test FunctionSegmentBoundedColoredArea
        func_seg_area = self._create_mock_drawable("TestFuncSegArea", "FunctionSegmentBoundedColoredArea")
        func_seg_area.func = function
        func_seg_area.segment = self.segment1
        func_seg_area_dependencies = self.manager.analyze_drawable_for_dependencies(func_seg_area)
        self.assertEqual(len(func_seg_area_dependencies), 2, "FunctionSegmentBoundedColoredArea should have 2 dependencies")
        self.assertIn(function, func_seg_area_dependencies, "function should be a dependency of area")
        self.assertIn(self.segment1, func_seg_area_dependencies, "segment should be a dependency of area")

        # Test FunctionsBoundedColoredArea
        funcs_area = self._create_mock_drawable("TestFuncsArea", "FunctionsBoundedColoredArea")
        funcs_area.func1 = function
        function2 = self._create_mock_drawable("TestFunction2", "Function")
        funcs_area.func2 = function2
        funcs_area_dependencies = self.manager.analyze_drawable_for_dependencies(funcs_area)
        self.assertEqual(len(funcs_area_dependencies), 2, "FunctionsBoundedColoredArea should have 2 dependencies")
        self.assertIn(function, funcs_area_dependencies, "func1 should be a dependency of area")
        self.assertIn(function2, funcs_area_dependencies, "func2 should be a dependency of area")

        # Test object with missing get_class_name method
        obj_without_method = SimpleMock(name="NoMethod")
        dependencies = self.manager.analyze_drawable_for_dependencies(obj_without_method)
        self.assertEqual(len(dependencies), 0, "Object without get_class_name should return empty dependencies list")


    def test_drawable_types_completeness(self) -> None:
        """Test that analyze_drawable_for_dependencies has cases for all drawable types"""
        # Use _type_hierarchy as the source of truth for drawable types
        drawable_types = set(self.manager._type_hierarchy.keys())

        # Inspect the analyze_drawable_for_dependencies method to extract the class names it handles
        handled_classes = set()

        # Use direct functional testing since inspect.getsource() in Brython
        # only returns method signatures, not the full method body
        test_cases = [
            ('Point', self._create_mock_drawable("TestPoint", "Point")),
            ('Segment', self._create_mock_drawable("TestSegment", "Segment")),
            ('Vector', self._create_mock_drawable("TestVector", "Vector")),
            ('Triangle', self._create_mock_drawable("TestTriangle", "Triangle")),
            ('Rectangle', self._create_mock_drawable("TestRectangle", "Rectangle")),
            ('Circle', self._create_mock_drawable("TestCircle", "Circle")),
            ('CircleArc', self._create_mock_drawable("TestCircleArc", "CircleArc")),
            ('Ellipse', self._create_mock_drawable("TestEllipse", "Ellipse")),
            ('Function', self._create_mock_drawable("TestFunction", "Function")),
            ('Angle', self._create_mock_drawable("TestAngle", "Angle")),
            ('ColoredArea', self._create_mock_drawable("TestColoredArea", "ColoredArea")),
            ('SegmentsBoundedColoredArea', self._create_mock_drawable("TestSBCA", "SegmentsBoundedColoredArea")),
            ('FunctionSegmentBoundedColoredArea', self._create_mock_drawable("TestFSBCA", "FunctionSegmentBoundedColoredArea")),
            ('FunctionsBoundedColoredArea', self._create_mock_drawable("TestFBCA", "FunctionsBoundedColoredArea")),
            ('Graph', self._create_mock_drawable("TestGraph", "Graph")),
            ('DirectedGraph', self._create_mock_drawable("TestDirectedGraph", "DirectedGraph")),
            ('UndirectedGraph', self._create_mock_drawable("TestUndirectedGraph", "UndirectedGraph")),
            ('Tree', self._create_mock_drawable("TestTree", "Tree")),
        ]

        # Test each drawable type by calling the method and checking it doesn't raise an exception
        for class_name, test_obj in test_cases:
            try:
                # Test if the method handles this type without error
                dependencies = self.manager.analyze_drawable_for_dependencies(test_obj)
                # If we get here without exception, the type is handled
                handled_classes.add(class_name)
                print(f"DEBUG: {class_name} handled successfully, returned {len(dependencies)} dependencies")
            except Exception as test_e:
                print(f"DEBUG: {class_name} failed with error: {test_e}")

        # Also check for any ColoredArea types that might be handled by the endswith logic
        for class_name in drawable_types:
            if class_name.endswith('ColoredArea') and class_name not in handled_classes:
                # Test this ColoredArea type
                try:
                    test_obj = self._create_mock_drawable(f"Test{class_name}", class_name)
                    dependencies = self.manager.analyze_drawable_for_dependencies(test_obj)
                    handled_classes.add(class_name)
                    print(f"DEBUG: {class_name} (ColoredArea type) handled successfully")
                except Exception as test_e:
                    print(f"DEBUG: {class_name} (ColoredArea type) failed: {test_e}")



        # Check for missing implementations
        missing_implementations = drawable_types - handled_classes

        # Print debug info
        print(f"All drawable types to check: {sorted(drawable_types)}")
        print(f"Handled classes: {sorted(handled_classes)}")
        print(f"Missing implementations: {sorted(missing_implementations)}")

        # Assert that all types are handled
        self.assertEqual(len(missing_implementations), 0,
                         f"Missing analyze_drawable_for_dependencies cases for: {', '.join(missing_implementations)}")

    def test_error_handling_none_values(self) -> None:
        """Test handling of None values in various methods"""
        # Test register_dependency with None values
        # Neither should raise exceptions, but should have no effect
        self.manager.register_dependency(None, self.point1)
        self.manager.register_dependency(self.segment1, None)

        # Test get_parents and get_children with None
        self.assertEqual(len(self.manager.get_parents(None)), 0, "get_parents should return empty set for None")
        self.assertEqual(len(self.manager.get_children(None)), 0, "get_children should return empty set for None")

        # Test get_all_parents and get_all_children with None
        self.assertEqual(len(self.manager.get_all_parents(None)), 0, "get_all_parents should return empty set for None")
        self.assertEqual(len(self.manager.get_all_children(None)), 0, "get_all_children should return empty set for None")

        # Test remove_drawable with None
        try:
            self.manager.remove_drawable(None)  # Should not raise exception
        except Exception as e:
            self.fail(f"remove_drawable failed with None: {e}")

        # Test analyze_drawable_for_dependencies with None
        dependencies = self.manager.analyze_drawable_for_dependencies(None)
        self.assertEqual(len(dependencies), 0, "analyze_drawable_for_dependencies should return empty list for None")

        # Test resolve_dependency_order with None in list
        ordered = self.manager.resolve_dependency_order([self.point1, None, self.point2])
        self.assertEqual(len(ordered), 2, "resolve_dependency_order should filter out None values")
        self.assertNotIn(None, ordered, "None should not appear in ordered results")

    def test_edge_cases(self) -> None:
        """Test edge cases like empty inputs and non-existent drawables"""
        # Test empty list for resolve_dependency_order
        ordered = self.manager.resolve_dependency_order([])
        self.assertEqual(len(ordered), 0, "Empty input should produce empty output")

        # Test removing non-existent drawable
        non_existent = self._create_mock_drawable("NonExistent")
        try:
            self.manager.remove_drawable(non_existent)  # Should not raise exception
        except Exception as e:
            self.fail(f"remove_drawable failed with non-existent drawable: {e}")

        # Test get_parents and get_children for non-existent drawable
        self.assertEqual(len(self.manager.get_parents(non_existent)), 0,
                         "get_parents should return empty set for non-existent drawable")
        self.assertEqual(len(self.manager.get_children(non_existent)), 0,
                         "get_children should return empty set for non-existent drawable")

    def test_verify_get_class_name_method(self) -> None:
        """Test verification of get_class_name method"""
        # Test with valid drawable
        valid_drawable = self._create_mock_drawable("Valid", "TestType")
        self.manager._verify_get_class_name_method(valid_drawable, "Test")
        # Should not raise any errors

        # Test with object missing get_class_name
        class NoMethodDrawable:
            def __init__(self) -> None:
                self.name = "NoMethod"
        no_method = NoMethodDrawable()
        self.manager._verify_get_class_name_method(no_method, "Test")
        # Should log warning but not raise error

        # Test with None
        self.manager._verify_get_class_name_method(None, "Test")
        # Should handle None gracefully

        # Test with object that has get_class_name but it's not callable
        class BadMethodDrawable:
            def __init__(self) -> None:
                self.name = "BadMethod"
                self.get_class_name = "not a method"
        bad_method = BadMethodDrawable()
        self.manager._verify_get_class_name_method(bad_method, "Test")
        # Should log warning but not raise error

    def test_unregister_dependency(self) -> None:
        """Test unregistering dependencies between drawables"""
        # Set up initial dependencies
        self.manager.register_dependency(child=self.segment1, parent=self.point1)
        self.manager.register_dependency(child=self.segment1, parent=self.point2)

        # Test unregistering a single dependency
        self.manager.unregister_dependency(child=self.segment1, parent=self.point1)

        # Verify point1 is removed from segment1's parents
        parents = self.manager.get_parents(self.segment1)
        self.assertNotIn(self.point1, parents, "Point1 should be removed from Segment1's parents")
        self.assertIn(self.point2, parents, "Point2 should still be a parent of Segment1")

        # Verify segment1 is removed from point1's children
        children = self.manager.get_children(self.point1)
        self.assertNotIn(self.segment1, children, "Segment1 should be removed from Point1's children")

        # Test unregistering non-existent dependency
        self.manager.unregister_dependency(child=self.segment2, parent=self.point1)
        # Should not raise any errors

        # Test unregistering with None values
        self.manager.unregister_dependency(child=None, parent=self.point1)
        self.manager.unregister_dependency(child=self.segment1, parent=None)
        self.manager.unregister_dependency(child=None, parent=None)
        # Should not raise any errors

    def test_unregister_dependency_prunes_orphaned_lookup_entries(self) -> None:
        """Unregistering the last edge should remove stale internal lookup references."""
        self.manager.register_dependency(child=self.segment1, parent=self.point1)
        self.assertIn(id(self.segment1), self.manager._object_lookup)
        self.assertIn(id(self.point1), self.manager._object_lookup)

        self.manager.unregister_dependency(child=self.segment1, parent=self.point1)

        self.assertNotIn(id(self.segment1), self.manager._object_lookup)
        self.assertNotIn(id(self.point1), self.manager._object_lookup)
        self.assertNotIn(id(self.segment1), self.manager._parents)
        self.assertNotIn(id(self.point1), self.manager._children)
        self._assert_internal_graph_invariants()

    def test_internal_graph_invariants_hold_after_chained_removals(self) -> None:
        self.manager.register_dependency(child=self.segment1, parent=self.point1)
        self.manager.register_dependency(child=self.segment1, parent=self.point2)
        self.manager.register_dependency(child=self.segment2, parent=self.point2)
        self.manager.register_dependency(child=self.segment2, parent=self.point3)
        self.manager.register_dependency(child=self.triangle, parent=self.segment1)
        self.manager.register_dependency(child=self.triangle, parent=self.segment2)

        self._assert_internal_graph_invariants()
        self.manager.remove_drawable(self.segment1)
        self._assert_internal_graph_invariants()
        self.manager.remove_drawable(self.triangle)
        self._assert_internal_graph_invariants()
        self.manager.remove_drawable(self.segment2)
        self._assert_internal_graph_invariants()

    def test_edge_neutral_drawables_register_no_dependencies(self) -> None:
        edge_neutral_classes = [
            "Label",
            "Bar",
            "Plot",
            "BarsPlot",
            "ContinuousPlot",
            "DiscretePlot",
            "ParametricFunction",
            "PiecewiseFunction",
        ]

        for class_name in edge_neutral_classes:
            drawable = self._create_mock_drawable(f"{class_name}1", class_name)
            dependencies = self.manager.analyze_drawable_for_dependencies(drawable)
            self.assertEqual(
                len(dependencies),
                0,
                f"{class_name} should remain edge-neutral in dependency analysis.",
            )

        self.assertEqual(self.manager._parents, {})
        self.assertEqual(self.manager._children, {})
        self.assertEqual(self.manager._object_lookup, {})

    def test_get_parents_and_children(self) -> None:
        """Test getting direct parents and children"""
        # Test empty sets
        self.assertEqual(len(self.manager.get_parents(self.segment1)), 0,
                        "New segment should have no parents")
        self.assertEqual(len(self.manager.get_children(self.point1)), 0,
                        "New point should have no children")

        # Set up multiple dependencies
        self.manager.register_dependency(child=self.segment1, parent=self.point1)
        self.manager.register_dependency(child=self.segment1, parent=self.point2)
        self.manager.register_dependency(child=self.segment2, parent=self.point1)

        # Test get_parents
        parents = self.manager.get_parents(self.segment1)
        self.assertEqual(len(parents), 2, "Segment1 should have 2 parents")
        self.assertIn(self.point1, parents, "Point1 should be a parent of Segment1")
        self.assertIn(self.point2, parents, "Point2 should be a parent of Segment1")

        # Test get_children
        children = self.manager.get_children(self.point1)
        self.assertEqual(len(children), 2, "Point1 should have 2 children")
        self.assertIn(self.segment1, children, "Segment1 should be a child of Point1")
        self.assertIn(self.segment2, children, "Segment2 should be a child of Point1")

        # Test with None values
        self.assertEqual(len(self.manager.get_parents(None)), 0,
                        "None should return empty parents set")
        self.assertEqual(len(self.manager.get_children(None)), 0,
                        "None should return empty children set")

    def test_should_skip_point_point_dependency(self) -> None:
        """Test the point-point dependency skip logic"""
        # Create test points
        point1 = self._create_mock_point("TestPoint1")
        point2 = self._create_mock_point("TestPoint2")
        segment = self._create_mock_segment("TestSegment", point1, point2)

        # Test Point-Point relationship (should skip)
        self.manager.register_dependency(child=point1, parent=point2)
        parents = self.manager.get_parents(point1)
        self.assertEqual(len(parents), 0, "Point-Point dependency should be skipped")

        # Test Point-Segment relationship (should not skip)
        self.manager.register_dependency(child=segment, parent=point1)
        parents = self.manager.get_parents(segment)
        self.assertEqual(len(parents), 1, "Point-Segment dependency should not be skipped")
        self.assertIn(point1, parents, "Point1 should be a parent of Segment")

        # Test with None values
        self.manager.register_dependency(child=None, parent=point1)
        self.manager.register_dependency(child=point1, parent=None)
        # Should not raise any errors

    def test_register_dependency_ignores_none_endpoints(self) -> None:
        """register_dependency should ignore None child/parent without mutating state."""
        self.manager.register_dependency(child=None, parent=self.point1)
        self.manager.register_dependency(child=self.segment1, parent=None)

        self.assertEqual(len(self.manager.get_children(self.point1)), 0)
        self.assertEqual(len(self.manager.get_parents(self.segment1)), 0)

    def test_register_dependency_ignores_non_callable_class_name(self) -> None:
        """register_dependency should ignore objects whose get_class_name is not callable."""
        bad_child = SimpleMock(name="bad_child")
        bad_child.get_class_name = "Point"

        self.manager.register_dependency(child=bad_child, parent=self.point1)
        self.assertEqual(len(self.manager.get_children(self.point1)), 0)

        bad_parent = SimpleMock(name="bad_parent")
        bad_parent.get_class_name = "Point"
        self.manager.register_dependency(child=self.segment1, parent=bad_parent)
        self.assertEqual(len(self.manager.get_parents(self.segment1)), 0)

    def test_append_attr_dependency_if_present_respects_get_class_requirement(self) -> None:
        """Helper should skip attrs lacking get_class_name when requested."""
        drawable = self._create_mock_drawable("Carrier", "FunctionSegmentBoundedColoredArea")
        function_like = self._create_mock_drawable("F1", "Function")
        drawable.func = function_like
        dependencies = []

        self.manager._append_attr_dependency_if_present(
            drawable,
            "func",
            dependencies,
            require_truthy=True,
            require_get_class_name=True,
        )
        self.assertEqual(len(dependencies), 1)
        self.assertIn(function_like, dependencies)

        drawable.func = SimpleMock(name="NoClassName")
        dependencies2 = []
        self.manager._append_attr_dependency_if_present(
            drawable,
            "func",
            dependencies2,
            require_truthy=True,
            require_get_class_name=True,
        )
        self.assertEqual(len(dependencies2), 0)

    def test_append_segment_attrs_collects_expected_count(self) -> None:
        """Helper should collect segment1..segmentN attributes that exist."""
        triangle = self._create_mock_drawable("Tri", "Triangle")
        triangle.segment1 = self.segment1
        triangle.segment2 = self.segment2
        triangle.segment3 = self.segment3
        dependencies = []

        self.manager._append_segment_attrs(triangle, dependencies, count=3)
        self.assertEqual(len(dependencies), 3)
        self.assertIn(self.segment1, dependencies)
        self.assertIn(self.segment2, dependencies)
        self.assertIn(self.segment3, dependencies)

    def test_analyze_drawable_for_dependencies_comprehensive(self) -> None:
        """Test analyzing dependencies for all supported drawable types"""
        # Test Circle
        circle = self._create_mock_drawable("C1", "Circle")
        circle.center = self.point1
        dependencies = self.manager.analyze_drawable_for_dependencies(circle)
        self.assertIn(self.point1, dependencies, "Circle should depend on its center point")

        # Test Ellipse
        ellipse = self._create_mock_drawable("E1", "Ellipse")
        ellipse.center = self.point2
        dependencies = self.manager.analyze_drawable_for_dependencies(ellipse)
        self.assertIn(self.point2, dependencies, "Ellipse should depend on its center point")

        # Test Triangle
        triangle = self._create_mock_drawable("T1", "Triangle")
        triangle.segment1 = self.segment1
        triangle.segment2 = self.segment2
        triangle.segment3 = self.segment3
        dependencies = self.manager.analyze_drawable_for_dependencies(triangle)
        self.assertIn(self.segment1, dependencies, "Triangle should depend on segment1")
        self.assertIn(self.segment2, dependencies, "Triangle should depend on segment2")
        self.assertIn(self.segment3, dependencies, "Triangle should depend on segment3")

        # Test Rectangle
        rectangle = self._create_mock_drawable("R1", "Rectangle")
        rectangle.segment1 = self.segment1
        rectangle.segment2 = self.segment2
        rectangle.segment3 = self.segment3
        rectangle.segment4 = self._create_mock_segment("S4", self.point1, self.point3)
        dependencies = self.manager.analyze_drawable_for_dependencies(rectangle)
        self.assertEqual(len(dependencies), 4, "Rectangle should depend on all four segments")

        # Test Function (should have no dependencies)
        function = self._create_mock_drawable("F1", "Function")
        dependencies = self.manager.analyze_drawable_for_dependencies(function)
        self.assertEqual(len(dependencies), 0, "Function should have no dependencies")

        # Test invalid drawable type
        invalid = self._create_mock_drawable("Invalid", "InvalidType")
        dependencies = self.manager.analyze_drawable_for_dependencies(invalid)
        self.assertEqual(len(dependencies), 0, "Invalid type should have no dependencies")

        # Test with missing get_class_name method
        class BadDrawable:
            def __init__(self) -> None:
                self.name = "Bad"
        bad = BadDrawable()
        dependencies = self.manager.analyze_drawable_for_dependencies(bad)
        self.assertEqual(len(dependencies), 0, "Drawable without get_class_name should have no dependencies")

    def test_find_segment_children(self) -> None:
        """Test finding children segments geometrically"""
        # Create a parent segment with properly initialized points
        parent_point1 = self._create_mock_point("ParentPoint1", 0, 0)
        parent_point2 = self._create_mock_point("ParentPoint2", 100, 0)
        parent_segment = self._create_mock_segment("Parent", parent_point1, parent_point2)

        # Create child segments that should be found
        child_point1 = self._create_mock_point("ChildPoint1", 25, 0)
        child_point2 = self._create_mock_point("ChildPoint2", 75, 0)
        child1 = self._create_mock_segment("Child1", child_point1, child_point2)

        child_point3 = self._create_mock_point("ChildPoint3", 0, 0)
        child_point4 = self._create_mock_point("ChildPoint4", 50, 0)
        child2 = self._create_mock_segment("Child2", child_point3, child_point4)

        # Create a segment that shouldn't be found (outside parent)
        outside_point1 = self._create_mock_point("OutsidePoint1", 150, 0)
        outside_point2 = self._create_mock_point("OutsidePoint2", 200, 0)
        outside = self._create_mock_segment("Outside", outside_point1, outside_point2)

        # Set up the mock drawable manager's segments
        self.mock_drawable_manager.drawables.Segments = [child1, child2, outside]

        # Find children
        try:
            children = self.manager._find_segment_children(parent_segment)
        except AttributeError as e:
            print(f"Error accessing coordinates: {e}")
            print(f"Parent segment: {parent_segment}")
            print(f"Parent segment point1: {parent_segment.point1}")
            print(f"Parent segment point2: {parent_segment.point2}")
            if parent_segment.point1:
                print(f"Parent segment point1 coords: ({parent_segment.point1.x}, {parent_segment.point1.y})")
            if parent_segment.point2:
                print(f"Parent segment point2 coords: ({parent_segment.point2.x}, {parent_segment.point2.y})")
            raise

        # Verify results
        self.assertEqual(len(children), 2, "Should find 2 child segments")
        self.assertIn(child1, children, "Child1 should be found")
        self.assertIn(child2, children, "Child2 should be found")
        self.assertNotIn(outside, children, "Outside segment should not be found")

        # Test with empty segments list
        self.mock_drawable_manager.drawables.Segments = []
        children = self.manager._find_segment_children(parent_segment)
        self.assertEqual(len(children), 0, "Should find no children with empty segments list")

        # Test with invalid segment data
        invalid_segment = self._create_mock_segment("Invalid", None, None)
        children = self.manager._find_segment_children(invalid_segment)
        self.assertEqual(len(children), 0, "Should handle invalid segment data")

        # Test with segment reasonable points
        bad_point1 = self._create_mock_point("BadPoint1", 0, 0)
        bad_point2 = self._create_mock_point("BadPoint2", 100, 0)
        bad_segment = self._create_mock_segment("BadSegment", bad_point1, bad_point2)
        children = self.manager._find_segment_children(bad_segment)
        self.assertEqual(len(children), 0, "Should handle segment without child segments on it")
