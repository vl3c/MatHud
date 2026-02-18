"""
MatHud Polar Coordinate Grid System

Implements a polar coordinate grid with concentric circles and radial lines for
visualization. Provides the visual foundation for polar coordinate work.

Key Features:
    - Dynamic radial spacing with zoom-adaptive circle intervals
    - Radial lines at configurable angular intervals (default 12 divisions = 30 degrees)
    - Angle labels at outer edge in degrees
    - Radial distance labels along the positive x-axis
    - Origin marker at coordinate system center

Visual Components:
    - Concentric circles at regular radial intervals
    - Radial lines from origin to viewport edge
    - Angle labels (0, 30, 60, 90, etc.)
    - Radius labels along one axis
    - Origin marker ('O') at center

Dependencies:
    - drawables_aggregator: Drawable base class and Position utilities
    - constants: Default styling and configuration values
    - utils.math_utils: Mathematical calculations and number formatting
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import math

from constants import default_color
from drawables_aggregator import Drawable, Position


class PolarGrid(Drawable):
    """
    Polar coordinate grid with concentric circles and radial lines.

    Inherits from Drawable to provide polar coordinate visualization with
    automatic scaling, radial spacing calculation, and viewport management.

    Attributes:
        width (float): Canvas width for coordinate system bounds
        height (float): Canvas height for coordinate system bounds
        angular_divisions (int): Number of radial lines (360 / angular_step)
        radial_spacing (float): Base spacing between circles in math units
        show_angle_labels (bool): Whether to display angle labels
        show_radius_labels (bool): Whether to display radius labels
    """

    def __init__(
        self,
        coordinate_mapper: Any,
        angular_divisions: int = 24,
        radial_spacing: float = 50.0,
        show_angle_labels: bool = True,
        show_radius_labels: bool = True,
        color: str = default_color,
    ) -> None:
        """Initialize polar grid with a CoordinateMapper and configuration."""
        self.name: str = "polar-grid-system"
        self.coordinate_mapper: Any = coordinate_mapper
        self.width: float = coordinate_mapper.canvas_width
        self.height: float = coordinate_mapper.canvas_height

        # Configuration
        self.angular_divisions: int = angular_divisions
        self.radial_spacing: float = radial_spacing
        self.show_angle_labels: bool = show_angle_labels
        self.show_radius_labels: bool = show_radius_labels

        # Internal zoom-adaptive spacing state
        self._default_radial_spacing: float = radial_spacing
        self._current_radial_spacing: float = radial_spacing
        self._radial_spacing_bias: float = 0.5
        self._max_circles: int = 10
        self._halving_enforcement_threshold: float = 1.0
        self._halving_first_trigger_ratio: float = 0.8
        self._halving_repeat_trigger_ratio: float = 0.4
        self._doubling_first_trigger_ratio: float = 1.25
        self._doubling_repeat_trigger_ratio: float = 2.5
        self._max_progression_steps: int = 8
        self._min_radial_spacing: float = 1e-6

        # Colors
        self.circle_color: str = "lightgrey"
        self.radial_color: str = "lightgrey"
        self.axis_color: str = color
        self.label_color: str = "grey"
        self.label_font_size: int = 8

        # Visibility
        self.visible: bool = True

        super().__init__(name=self.name, color=color)

    def get_class_name(self) -> str:
        """Return the class name 'PolarGrid'."""
        return "PolarGrid"

    @property
    def angle_step(self) -> float:
        """Get the angle step in radians between radial lines."""
        if self.angular_divisions <= 0:
            return 2 * math.pi
        return 2 * math.pi / self.angular_divisions

    @property
    def origin_screen(self) -> Tuple[float, float]:
        """Get the screen coordinates of the mathematical origin (0,0)."""
        ox, oy = self.coordinate_mapper.math_to_screen(0, 0)
        return (ox, oy)

    @property
    def origin(self) -> Position:
        """Get the screen coordinates of the mathematical origin as Position."""
        ox, oy = self.origin_screen
        return Position(ox, oy)

    @property
    def max_radius_screen(self) -> float:
        """Calculate the maximum screen radius needed to cover the viewport."""
        if self.width is None or self.height is None:
            return 0
        ox, oy = self.origin_screen
        corners = [(0, 0), (self.width, 0), (0, self.height), (self.width, self.height)]
        max_dist = 0.0
        for cx, cy in corners:
            dist = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
            if dist > max_dist:
                max_dist = dist
        return max_dist * 1.1  # Add 10% margin

    @property
    def max_radius_math(self) -> float:
        """Get the maximum radius in math units."""
        scale = self.coordinate_mapper.scale_factor
        if scale <= 0:
            scale = 1.0
        return float(self.max_radius_screen / scale)

    @property
    def display_spacing(self) -> float:
        """Get the spacing between circles in screen pixels."""
        return float(abs(self._current_radial_spacing) * self.coordinate_mapper.scale_factor)

    def reset(self) -> None:
        """Reset polar grid to initial state."""
        self._current_radial_spacing = self._default_radial_spacing

    def get_angle_labels(self) -> List[Tuple[float, str]]:
        """Get angle labels for each radial line.

        Returns:
            List of (angle_radians, label_text) tuples
        """
        labels = []
        for i in range(self.angular_divisions):
            angle = i * self.angle_step
            degrees = int(round(math.degrees(angle)))
            labels.append((angle, f"{degrees}°"))
        return labels

    def get_radial_circles(self) -> List[float]:
        """Get the radii for concentric circles in screen pixels.

        Returns:
            List of radii for each concentric circle
        """
        circles: list[float] = []
        spacing = self.display_spacing
        if spacing <= 0:
            return circles
        n = 1
        while n * spacing < self.max_radius_screen:
            circles.append(n * spacing)
            n += 1
        return circles

    def get_visible_radius(self) -> float:
        """Calculate the maximum visible radius from the origin in math units.

        Returns the distance from origin to the farthest visible corner,
        ensuring all concentric circles needed to fill the viewport are drawn.
        """
        left = self.coordinate_mapper.get_visible_left_bound()
        right = self.coordinate_mapper.get_visible_right_bound()
        top = self.coordinate_mapper.get_visible_top_bound()
        bottom = self.coordinate_mapper.get_visible_bottom_bound()

        corners = [
            (left, top),
            (right, top),
            (left, bottom),
            (right, bottom),
        ]
        max_radius = 0.0
        for x, y in corners:
            r = math.sqrt(x * x + y * y)
            if r > max_radius:
                max_radius = r
        return max_radius

    def get_relative_diagonal(self) -> float:
        """Get viewport diagonal adjusted for current scale factor."""
        rel_width = self.width / self.coordinate_mapper.scale_factor
        rel_height = self.height / self.coordinate_mapper.scale_factor
        return math.sqrt(rel_width * rel_width + rel_height * rel_height)

    def draw(self) -> None:
        """No-op: rendering handled via renderer."""
        return

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            numeric = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(numeric):
            return float(default)
        return numeric

    def _calculate_radial_spacing(self) -> float:
        """Calculate ideal radial spacing based on viewport size."""
        ideal_spacing = self._calculate_ideal_radial_spacing()
        return self._find_appropriate_spacing(ideal_spacing)

    def _calculate_ideal_radial_spacing(self) -> float:
        """Calculate ideal spacing to show max_circles circles."""
        visible_radius = self.get_visible_radius()
        if visible_radius <= 0:
            return self._default_radial_spacing
        ideal = visible_radius / self._max_circles
        return ideal

    def _find_appropriate_spacing(self, ideal_spacing: float) -> float:
        """Find a 'nice' spacing value near the ideal."""
        effective_ideal = ideal_spacing * self._radial_spacing_bias
        if effective_ideal <= 0:
            return self._default_radial_spacing
        magnitude = 10 ** math.floor(math.log10(effective_ideal))
        possible_spacings = [magnitude * i for i in [1, 2, 5, 10]]
        for spacing in possible_spacings:
            if spacing >= effective_ideal:
                return float(spacing)
        return float(possible_spacings[0])

    def _invalidate_cache_on_zoom(self) -> None:
        """Update radial spacing for zoom operations."""
        proposed_spacing = self._calculate_radial_spacing()
        if proposed_spacing and proposed_spacing > 0:
            previous_spacing = self._current_radial_spacing
            adjusted_spacing = self._adjust_spacing_for_zoom(previous_spacing, proposed_spacing)
            if adjusted_spacing > 0:
                self._current_radial_spacing = adjusted_spacing

    def _normalize_spacing(self, value: float) -> float:
        try:
            normalized = float(value)
        except Exception:
            return -1.0
        if not math.isfinite(normalized):
            return -1.0
        return normalized

    def _adjust_spacing_for_zoom(self, previous_spacing: float, proposed_spacing: float) -> float:
        previous = self._normalize_spacing(previous_spacing)
        proposed = self._normalize_spacing(proposed_spacing)
        if proposed <= 0:
            return previous if previous > 0 else 1.0
        if previous <= 0:
            return proposed

        threshold = self._halving_enforcement_threshold
        if previous <= threshold:
            return self._progress_spacing(previous, proposed)
        return proposed

    def _progress_spacing(self, current_spacing: float, proposed_spacing: float) -> float:
        spacing = current_spacing
        proposed = proposed_spacing
        if proposed <= 0:
            return max(spacing, self._min_radial_spacing)

        ratio_first = self._halving_first_trigger_ratio
        ratio_repeat = self._halving_repeat_trigger_ratio
        ratio_first = min(max(ratio_first, 0.0), 1.0)
        ratio_repeat = min(max(ratio_repeat, 0.0), 1.0)

        steps = 0
        if proposed < spacing:
            while spacing > self._min_radial_spacing and steps < self._max_progression_steps:
                trigger = ratio_first if steps == 0 else ratio_repeat
                if trigger <= 0:
                    break
                if proposed <= spacing * trigger:
                    spacing *= 0.5
                    steps += 1
                else:
                    break
            return max(spacing, self._min_radial_spacing)

        if proposed > spacing:
            up_first = self._doubling_first_trigger_ratio
            up_repeat = self._doubling_repeat_trigger_ratio
            if up_first <= 1.0:
                up_first = 1.25
            if up_repeat <= 1.0:
                up_repeat = 2.5
            while steps < self._max_progression_steps:
                trigger = up_first if steps == 0 else up_repeat
                if proposed > spacing * trigger:
                    spacing *= 2.0
                    steps += 1
                    if spacing >= self._halving_enforcement_threshold:
                        break
                else:
                    break
            return spacing

        return spacing

    def can_zoom_in_further(self) -> bool:
        """Determine if additional zoom-in should be allowed based on minimum spacing."""
        try:
            spacing = float(self._current_radial_spacing)
            min_spacing = float(self._min_radial_spacing)
        except Exception:
            return True
        if not math.isfinite(spacing) or spacing <= 0:
            return True
        if not math.isfinite(min_spacing) or min_spacing <= 0:
            return True
        tolerance = max(min_spacing * 1e-6, 1e-15)
        limit = min_spacing + tolerance
        return spacing > limit

    def get_state(self) -> Dict[str, Any]:
        """Serialize polar grid state for persistence."""
        return {
            "angular_divisions": self.angular_divisions,
            "radial_spacing": self.radial_spacing,
            "show_angle_labels": self.show_angle_labels,
            "show_radius_labels": self.show_radius_labels,
            "visible": self.visible,
            "_current_radial_spacing": self._current_radial_spacing,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore polar grid state from persistence.

        Args:
            state: Dictionary containing polar grid settings
        """
        if "angular_divisions" in state:
            self.angular_divisions = state["angular_divisions"]
        if "radial_spacing" in state:
            self.radial_spacing = state["radial_spacing"]
            self._default_radial_spacing = state["radial_spacing"]
            self._current_radial_spacing = state["radial_spacing"]
        if "show_angle_labels" in state:
            self.show_angle_labels = state["show_angle_labels"]
        if "show_radius_labels" in state:
            self.show_radius_labels = state["show_radius_labels"]
        if "visible" in state:
            self.visible = bool(state["visible"])
