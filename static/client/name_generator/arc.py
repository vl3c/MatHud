"""
MatHud Arc Name Generation System

Naming system for circle arcs using point-based naming conventions.
Provides arc name generation and point name extraction from arc names.

Key Features:
    - Arc naming with ArcMaj/ArcMin prefixes
    - Point name extraction from arc name suggestions
    - Integration with point name generator for proper name parsing
    - Name collision detection and resolution

Dependencies:
    - name_generator.base: Base class functionality
    - name_generator.point: Point name extraction
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Set, Tuple

from .base import NameGenerator

if TYPE_CHECKING:
    from .point import PointNameGenerator


class ArcNameGenerator(NameGenerator):
    """Generates names for circle arcs based on endpoint point names.

    Arc names follow the pattern: ArcMaj_{point1}{point2} or ArcMin_{point1}{point2}
    where point names can include prime symbols (e.g., ArcMin_A'B'').

    Attributes:
        canvas: Canvas instance for accessing drawable objects
        point_generator: Reference to point name generator for name parsing
    """

    # Prefixes to strip before extracting point names (longer prefixes first)
    ARC_PREFIXES = (
        "ArcMajor_",
        "ArcMinor_",
        "ArcMaj_",
        "ArcMin_",
        "ArcMajor",
        "ArcMinor",
        "arc_",
        "Arc_",
        "arc",
        "Arc",
    )

    def __init__(self, canvas: Any, point_generator: "PointNameGenerator") -> None:
        """Initialize arc name generator.

        Args:
            canvas: Canvas instance for drawable object access
            point_generator: Point name generator for parsing point names
        """
        super().__init__(canvas)
        self.point_generator = point_generator

    def extract_point_names_from_arc_name(self, arc_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Extract suggested point names from an arc name suggestion.

        First strips known arc prefixes, then uses the point generator's
        split_point_names for extraction.

        Examples:
            "arc_AB" -> ("A", "B")
            "ArcMaj_C'D''" -> ("C'", "D''")
            "ArcMajor" -> (None, None) - nothing left after stripping

        Args:
            arc_name: Arc name or point name suggestion

        Returns:
            Tuple of (point1_name, point2_name), either or both may be None
        """
        if not arc_name:
            return None, None

        # Strip known arc prefixes first
        name = arc_name
        for prefix in self.ARC_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break

        if not name:
            return None, None

        # Use the point generator to split point names
        point_names = self.point_generator.split_point_names(name, 2)
        p1_name = point_names[0] if point_names and point_names[0] else None
        p2_name = point_names[1] if len(point_names) > 1 and point_names[1] else None
        return p1_name, p2_name

    def generate_arc_name(
        self,
        proposed_name: Optional[str],
        point1_name: str,
        point2_name: str,
        use_major_arc: bool,
        existing_names: Set[str],
    ) -> str:
        """Generate a unique arc name based on endpoint names.

        If proposed_name already has proper format (ArcMaj_... or ArcMin_...),
        uses it directly. Otherwise extracts point names from proposed_name
        or falls back to provided point names.

        Args:
            proposed_name: Optional proposed name
            point1_name: Fallback name of the first endpoint
            point2_name: Fallback name of the second endpoint
            use_major_arc: Whether this is a major arc
            existing_names: Set of existing arc names to avoid collisions

        Returns:
            Unique arc name in format ArcMaj_{p1}{p2} or ArcMin_{p1}{p2}
        """
        # If proposed name already has proper arc format, use it directly
        if proposed_name and (proposed_name.startswith("ArcMaj_") or proposed_name.startswith("ArcMin_")):
            return self._make_unique(proposed_name, existing_names)

        # Try to extract point names from proposed name
        if proposed_name:
            extracted_p1, extracted_p2 = self.extract_point_names_from_arc_name(proposed_name)
            if extracted_p1:
                point1_name = extracted_p1
            if extracted_p2:
                point2_name = extracted_p2

        # Generate arc name from point names
        prefix = "ArcMaj" if use_major_arc else "ArcMin"
        base = f"{prefix}_{point1_name}{point2_name}"

        return self._make_unique(base, existing_names)

    def _make_unique(self, base_name: str, existing_names: Set[str]) -> str:
        """Make a name unique by adding numeric suffix if needed."""
        candidate = base_name
        suffix = 1
        while candidate in existing_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        return candidate
