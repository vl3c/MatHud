"""
MatHud Segments Bounded Colored Area

Represents a colored area bounded by line segments with overlap detection and linear interpolation.
Provides area visualization between two segments or between a segment and the x-axis.

Key Features:
    - Two-segment or segment-to-axis area visualization
    - Overlap region detection and calculation
    - Linear interpolation for boundary generation
    - Geometric validation and path optimization

Dependencies:
    - drawables.colored_area: Base class for area visualization
    - drawables.segment: Segment objects for boundary definitions
    - copy: Deep copying capabilities for state management
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, cast

from drawables.colored_area import ColoredArea
from drawables.segment import Segment


class SegmentsBoundedColoredArea(ColoredArea):
    """Creates a colored area bounded by line segments with geometric overlap detection.

    This class creates a visual representation of the area between two line segments
    or between a segment and the x-axis, using linear interpolation for smooth boundaries.

    Attributes:
        segment1 (Segment): The first bounding segment
        segment2 (Segment or None): The second bounding segment (None means x-axis)
    """

    def __init__(
        self, segment1: Segment, segment2: Optional[Segment] = None, color: str = "lightblue", opacity: float = 0.3
    ) -> None:
        """Initialize a segments bounded colored area.

        Args:
            segment1 (Segment): The first bounding segment
            segment2 (Segment or None): The second bounding segment (None means x-axis)
            color (str): CSS color value for area fill
            opacity (float): Opacity value between 0.0 and 1.0
        """
        name = self._generate_name(segment1, segment2)
        super().__init__(name=name, color=color, opacity=opacity)
        self.segment1: Segment = segment1
        self.segment2: Optional[Segment] = segment2

    def _generate_name(self, segment1: Segment, segment2: Optional[Segment]) -> str:
        """Generate a descriptive name for the colored area based on segment names."""
        s1_name: str = segment1.name if segment1 else "x_axis"
        s2_name: str = segment2.name if segment2 else "x_axis"
        return f"area_between_{s1_name}_and_{s2_name}"

    def get_class_name(self) -> str:
        """Return the class name 'SegmentsBoundedColoredArea'."""
        return "SegmentsBoundedColoredArea"

    def uses_segment(self, segment: Segment) -> bool:
        """Check if this colored area uses a specific segment for dependency tracking."""

        def segments_match(s1: Segment, s2: Segment) -> bool:
            return bool(
                s1.point1.x == s2.point1.x
                and s1.point1.y == s2.point1.y
                and s1.point2.x == s2.point2.x
                and s1.point2.y == s2.point2.y
            )

        return bool(
            segments_match(self.segment1, segment) or (self.segment2 and segments_match(self.segment2, segment))
        )

    def get_state(self) -> Dict[str, Any]:
        """Serialize segments bounded area state for persistence."""
        state: Dict[str, Any] = super().get_state()
        state["args"].update(
            {"segment1": self.segment1.name, "segment2": self.segment2.name if self.segment2 else "x_axis"}
        )
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        """Create a deep copy for undo/redo functionality."""
        if id(self) in memo:
            return cast(SegmentsBoundedColoredArea, memo[id(self)])

        new_segment1 = copy.deepcopy(self.segment1, memo)
        new_segment2 = copy.deepcopy(self.segment2, memo) if self.segment2 else None

        new_area: SegmentsBoundedColoredArea = SegmentsBoundedColoredArea(
            segment1=new_segment1, segment2=new_segment2, color=self.color, opacity=self.opacity
        )
        new_area.name = self.name
        memo[id(self)] = new_area
        return new_area
