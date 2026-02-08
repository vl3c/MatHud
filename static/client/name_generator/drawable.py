"""
MatHud Multi-Type Drawable Name Generator

Coordinated name generation system managing names across all drawable object types.
Provides unified naming interface with type-specific delegation.

Key Features:
    - Multi-type name coordination (points, functions, shapes, areas, arcs)
    - Type-specific naming strategy delegation
    - Global name uniqueness validation
    - Angle name generation from geometric relationships

Dependencies:
    - name_generator.point: Point naming system
    - name_generator.function: Function naming system
    - name_generator.arc: Arc naming system
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, cast

from .point import PointNameGenerator
from .function import FunctionNameGenerator
from .label import LabelNameGenerator
from .arc import ArcNameGenerator


class DrawableNameGenerator:
    """Coordinates name generation across all drawable object types.

    Manages type-specific name generators and provides unified interface
    for consistent naming throughout the mathematical visualization system.

    Attributes:
        canvas (Canvas): Canvas instance for drawable object access
        used_letters_from_names (dict): Backward compatibility tracking
        point_generator (PointNameGenerator): Specialized point name generator
        function_generator (FunctionNameGenerator): Specialized function name generator
        arc_generator (ArcNameGenerator): Specialized arc name generator
    """

    def __init__(self, canvas: Any) -> None:
        """Initialize multi-type name generator with specialized generators.

        Args:
            canvas (Canvas): Canvas instance for drawable object access
        """
        self.canvas: Any = canvas
        self.used_letters_from_names: Dict[str, Any] = {}  # Maintain for backward compatibility

        # Initialize specialized generators
        self.point_generator: PointNameGenerator = PointNameGenerator(canvas)
        self.function_generator: FunctionNameGenerator = FunctionNameGenerator(canvas)
        self.label_generator: LabelNameGenerator = LabelNameGenerator(canvas)
        self.arc_generator: ArcNameGenerator = ArcNameGenerator(canvas, self.point_generator)

    def reset_state(self) -> None:
        """Reset the state of all specialized name generators."""
        self.point_generator.reset_state()
        self.function_generator.reset_state() # Assuming FunctionNameGenerator might also have state
        if hasattr(self.label_generator, "reset_state"):
            self.label_generator.reset_state()

    def print_names(self) -> None:
        """Print all drawable names by category for debugging."""
        print(f"Point names: {self.get_drawable_names('Point')}")
        print(f"Segment names: {self.get_drawable_names('Segment')}")
        print(f"Triangle names: {self.get_drawable_names('Triangle')}")
        print(f"Rectangle names: {self.get_drawable_names('Rectangle')}")
        print(f"Circle names: {self.get_drawable_names('Circle')}")
        print(f"Ellipse names: {self.get_drawable_names('Ellipse')}")
        print(f"Function names: {self.get_drawable_names('Function')}")

    # Delegate methods to specialized generators or base NameGenerator

    def get_drawable_names(self, class_name: str) -> List[str]:
        """Get sorted list of names for drawables of a specific class.

        Args:
            class_name (str): Class name to filter drawables

        Returns:
            list: Sorted list of drawable names for the specified class
        """
        drawables = self.canvas.get_drawables_by_class_name(class_name)
        drawable_names: List[str] = sorted([drawable.name for drawable in drawables])
        return drawable_names

    def filter_string(self, name: str) -> str:
        """Filter a string to keep only letters, apostrophes, digits, and parentheses.

        Args:
            name (str): Input string to filter

        Returns:
            str: Filtered string containing only valid mathematical naming characters
        """
        return self.point_generator.filter_string(name)

    def split_point_names(self, expression: Optional[str], n: int = 2) -> List[str]:
        """Split a point expression into individual point names.

        Args:
            expression (str): Point expression to split
            n (int): Number of point names to extract

        Returns:
            list: List of individual point names
        """
        # Update our internal tracker for backward compatibility
        result: List[str] = self.point_generator.split_point_names(expression, n)
        if expression and expression in self.point_generator.used_letters_from_names:
            self.used_letters_from_names[expression] = self.point_generator.used_letters_from_names[expression]
        return result

    def _generate_unique_point_name(self) -> str:
        """Generate a unique point name using alphabetical sequence with apostrophes.

        Returns:
            str: Unique point name
        """
        return self.point_generator._generate_unique_point_name()

    def generate_point_name(self, preferred_name: Optional[str]) -> str:
        """Generate a unique point name, using preferred_name if possible.

        Args:
            preferred_name (str): Preferred point name

        Returns:
            str: Unique point name
        """
        # Update our internal tracker for backward compatibility
        result: str = self.point_generator.generate_point_name(preferred_name)
        if preferred_name and preferred_name in self.point_generator.used_letters_from_names:
            self.used_letters_from_names[preferred_name] = self.point_generator.used_letters_from_names[preferred_name]
        return result

    def _increment_function_name(self, func_name: str) -> str:
        """Increment a function name by adding or incrementing a number suffix.

        Args:
            func_name (str): Function name to increment

        Returns:
            str: Incremented function name
        """
        return self.function_generator._increment_function_name(func_name)

    def _generate_unique_function_name(self) -> str:
        """Generate a unique function name using alphabetical sequence.

        Returns:
            str: Unique function name
        """
        return self.function_generator._generate_unique_function_name()

    def generate_function_name(self, preferred_name: Optional[str]) -> str:
        """Generate a unique function name, using preferred_name if possible.

        Args:
            preferred_name (str): Preferred function name

        Returns:
            str: Unique function name
        """
        return self.function_generator.generate_function_name(preferred_name)

    def generate_parametric_function_name(self, preferred_name: Optional[str]) -> str:
        """Generate a unique parametric function name.

        Uses the same letter sequence as regular functions (f, g, h, ...)
        but with '_param' suffix to distinguish parametric functions
        (e.g., f_param, g_param, f1_param).

        Args:
            preferred_name: Preferred name (if provided and available, uses as-is)

        Returns:
            Unique parametric function name
        """
        return self.function_generator.generate_parametric_function_name(preferred_name)

    def generate_label_name(self, preferred_name: Optional[str]) -> str:
        """Generate a unique label name, using preferred_name when provided."""
        return str(self.label_generator.generate_label_name(preferred_name))

    def extract_point_names_from_arc_name(
        self, arc_name: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract suggested point names from an arc name suggestion.

        Args:
            arc_name: Arc name or point name suggestion (e.g., "A'B'", "ArcMin_CD")

        Returns:
            Tuple of (point1_name, point2_name)
        """
        return cast(Tuple[Optional[str], Optional[str]], self.arc_generator.extract_point_names_from_arc_name(arc_name))

    def generate_arc_name(
        self,
        proposed_name: Optional[str],
        point1_name: str,
        point2_name: str,
        use_major_arc: bool,
        existing_names: Set[str],
    ) -> str:
        """Generate a unique arc name based on endpoint names.

        Args:
            proposed_name: Optional proposed name
            point1_name: Name of the first endpoint
            point2_name: Name of the second endpoint
            use_major_arc: Whether this is a major arc
            existing_names: Set of existing arc names

        Returns:
            Unique arc name
        """
        return str(self.arc_generator.generate_arc_name(
            proposed_name, point1_name, point2_name, use_major_arc, existing_names
        ))

    def _is_valid_point_list(self, points: List[str]) -> bool:
        """Helper to check if a list of points is valid for angle name generation.

        Args:
            points (list): List of point names to validate

        Returns:
            bool: True if point list is valid for angle generation
        """
        if not points or not isinstance(points, list) or len(points) != 2:
            return False
        # Ensure both point names are non-empty strings
        if not (isinstance(points[0], str) and points[0] and \
                isinstance(points[1], str) and points[1]):
            return False
        return True

    def generate_angle_name_from_segments(self, segment1_name: str, segment2_name: str) -> Optional[str]:
        """Generate canonical angle name from two segment names.

        Segments are assumed to be named like "VP1", "VP2" (Vertex + Point).

        Args:
            segment1_name (str): First segment name
            segment2_name (str): Second segment name

        Returns:
            str or None: Angle name string (e.g., "angle_P1VP2"), or None if invalid
        """
        if not segment1_name or not segment2_name:
            return None

        # Temporarily reset next_index for these specific segment names if they were parsed before
        # This ensures a fresh parse by split_point_names for this method's context
        if hasattr(self.point_generator, 'used_letters_from_names'): # Check if attribute exists
            if segment1_name in self.point_generator.used_letters_from_names:
                self.point_generator.used_letters_from_names[segment1_name]['next_index'] = 0
            if segment2_name in self.point_generator.used_letters_from_names:
                self.point_generator.used_letters_from_names[segment2_name]['next_index'] = 0

        s1_points: List[str] = self.point_generator.split_point_names(segment1_name)
        s2_points: List[str] = self.point_generator.split_point_names(segment2_name)

        valid_s1: bool = self._is_valid_point_list(s1_points)
        valid_s2: bool = self._is_valid_point_list(s2_points)

        if not (valid_s1 and valid_s2):
            return None

        set_s1_points: set[str] = set(s1_points)
        set_s2_points: set[str] = set(s2_points)

        common_points: List[str] = list(set_s1_points.intersection(set_s2_points))
        if len(common_points) != 1:
            return None

        vertex_name: str = common_points[0]

        all_unique_points: List[str] = list(set_s1_points.union(set_s2_points))
        if len(all_unique_points) != 3:
            return None

        # Identify and sort the two arm points (excluding the vertex)
        # Ensure elements are strings before comparison with vertex_name if there's any doubt
        arm_point_candidates: List[str] = [str(p) for p in all_unique_points if p is not None and str(p) != str(vertex_name)]
        arm_point_names: List[str] = sorted(arm_point_candidates)

        if len(arm_point_names) != 2:
            return None

        final_name: str = f"angle_{arm_point_names[0]}{vertex_name}{arm_point_names[1]}"
        return final_name
