"""
MatHud Two-Axis Cartesian Coordinate System

Implements a complete Cartesian coordinate system with grid visualization, axis rendering,
and coordinate transformations. Provides the mathematical foundation for geometric object
positioning and user interaction coordinate mapping.

Key Features:
    - Dynamic grid scaling with zoom-adaptive tick spacing
    - Axis rendering with numerical labels and origin marking
    - Coordinate transformation between screen and mathematical space
    - Viewport boundary calculations for efficient rendering
    - Grid line visualization with customizable spacing
    - Origin point management and positioning

Visual Components:
    - X and Y axes with customizable colors and thickness
    - Tick marks with automatic spacing calculation
    - Numerical labels with mathematical formatting
    - Grid lines for visual coordinate reference
    - Origin marker ('O') at coordinate system center

Coordinate Transformations:
    - Screen to mathematical coordinate conversion
    - Mathematical to screen coordinate conversion
    - Zoom and pan transformation support
    - Viewport boundary calculations
    - Visible area determination for rendering optimization

Dependencies:
    - geometry: Drawable base class and Position utilities
    - constants: Default styling and configuration values
    - utils.math_utils: Mathematical calculations and number formatting
"""

from __future__ import annotations

from typing import Any, Dict, cast

import math

from constants import default_color
from drawables_aggregator import Drawable, Position
from utils.math_utils import MathUtils


class Cartesian2Axis(Drawable):
    """
    Two-axis Cartesian coordinate system with dynamic scaling and grid visualization.

    Inherits from Drawable to provide complete coordinate system rendering with
    automatic scaling, tick spacing calculation, and viewport management.

    Attributes:
        width (float): Canvas width for coordinate system bounds
        height (float): Canvas height for coordinate system bounds
        origin (Position): Current origin position in screen coordinates
        default_tick_spacing (float): Base tick spacing for coordinate labels
        current_tick_spacing (float): Current tick spacing adjusted for zoom level
        max_ticks (int): Maximum number of ticks to display
        tick_size (int): Visual size of tick marks in pixels
        tick_color (str): Color for axis lines and tick marks
        tick_label_color (str): Color for numerical coordinate labels
        tick_label_font_size (int): Font size for coordinate labels
        grid_color (str): Color for grid lines
    """

    def __init__(self, coordinate_mapper: Any, color: str = default_color) -> None:
        """Initialize Cartesian coordinate system with a CoordinateMapper and color."""
        self.name: str = "cartesian-2axis-system"
        self.mapper: Any = coordinate_mapper
        self.width: float = coordinate_mapper.canvas_width
        self.height: float = coordinate_mapper.canvas_height
        self.default_tick_spacing: float = 100
        self.current_tick_spacing: float = 100  # Track the previous tick spacing to determine zoom level
        # Bias factor to make intermediate tick spacing appear sooner (tuneable)
        self.tick_spacing_bias: float = 0.5
        self.max_ticks: int = 10
        self.tick_size: int = 3
        self.tick_color: str = color
        self.tick_label_color: str = "grey"
        self.tick_label_font_size: int = 8
        self.grid_color: str = "lightgrey"
        self.halving_enforcement_threshold: float = 1.0
        self.halving_first_trigger_ratio: float = 0.8
        self.halving_repeat_trigger_ratio: float = 0.4
        self.doubling_first_trigger_ratio: float = 1.25
        self.doubling_repeat_trigger_ratio: float = 2.5
        self.max_progression_steps: int = 8
        self.min_tick_spacing: float = 1e-6
        self.visible: bool = True
        super().__init__(name=self.name, color=color)

    # Canvas removed; mapper is the single source of truth

    def reset(self) -> None:
        """Reset coordinate system to initial state with centered origin."""
        self.current_tick_spacing = self.default_tick_spacing

    def get_class_name(self) -> str:
        """Return the class name 'Cartesian2Axis'."""
        return 'Cartesian2Axis'

    @property
    def origin(self) -> Position:
        """Get the screen coordinates of the mathematical origin (0,0) using CoordinateMapper."""
        origin_x: float
        origin_y: float
        origin_x, origin_y = self.mapper.math_to_screen(0, 0)
        return Position(origin_x, origin_y)

    def get_visible_left_bound(self) -> float:
        """Calculate visible left boundary in mathematical coordinates."""
        return cast(float, self.mapper.get_visible_left_bound())

    def get_visible_right_bound(self) -> float:
        """Calculate visible right boundary in mathematical coordinates."""
        return cast(float, self.mapper.get_visible_right_bound())

    def get_visible_top_bound(self) -> float:
        """Calculate visible top boundary in mathematical coordinates."""
        return cast(float, self.mapper.get_visible_top_bound())

    def get_visible_bottom_bound(self) -> float:
        """Calculate visible bottom boundary in mathematical coordinates."""
        return cast(float, self.mapper.get_visible_bottom_bound())

    def get_relative_width(self) -> float:
        """Get canvas width adjusted for current scale factor."""
        return cast(float, self.width / self.mapper.scale_factor)

    def get_relative_height(self) -> float:
        """Get canvas height adjusted for current scale factor."""
        return cast(float, self.height / self.mapper.scale_factor)

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

    def _should_continue_drawing(self, position: float, boundary: float, direction: int) -> bool:
        return (direction == 1 and position < boundary) or (direction == -1 and position > 0)

    def _calculate_tick_spacing(self) -> float:
        ideal_spacing: float = self._calculate_ideal_tick_spacing()
        return self._find_appropriate_spacing(ideal_spacing)

    def _calculate_ideal_tick_spacing(self) -> float:
        relative_width: float = self.get_relative_width()  # Width of the visible cartesian system in units
        ideal: float = relative_width / self.max_ticks  # Ideal width of a tick spacing
        return ideal

    def _find_appropriate_spacing(self, ideal_spacing: float) -> float:
        # Bias to densify sooner and coarsen later
        effective_ideal: float = ideal_spacing * self.tick_spacing_bias
        # Find the order of magnitude of the (biased) ideal spacing
        magnitude: float = 10 ** math.floor(math.log10(effective_ideal))
        # Standard nice steps (denser than 2.5): 1, 2, 5, 10
        possible_spacings: list[float] = [magnitude * i for i in [1, 2, 5, 10]]
        # Pick the smallest spacing not less than effective_ideal,
        # but avoid collapsing to the same spacing if viewport changed significantly.
        for spacing in possible_spacings:
            if spacing >= effective_ideal:
                return spacing

        # Fallback
        return possible_spacings[0]

    def _invalidate_cache_on_zoom(self) -> None:
        """Update tick spacing for zoom operations with dynamic spacing calculation."""
        # Recompute ideal spacing in math units based on current scale and viewport
        proposed_tick_spacing: float = self._calculate_tick_spacing()
        # Apply directly to keep ticks consistent with zoom (no dependency on zoom_direction)
        if proposed_tick_spacing and proposed_tick_spacing > 0:
            previous_spacing = self.current_tick_spacing
            adjusted_spacing = self._adjust_spacing_for_zoom(previous_spacing, proposed_tick_spacing)
            if adjusted_spacing > 0:
                self.current_tick_spacing = adjusted_spacing

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

        threshold = self.halving_enforcement_threshold
        if previous <= threshold:
            return self._progress_spacing(previous, proposed)
        return proposed

    def _progress_spacing(self, current_spacing: float, proposed_spacing: float) -> float:
        spacing = current_spacing
        proposed = proposed_spacing
        if proposed <= 0:
            return max(spacing, self.min_tick_spacing)

        ratio_first = self.halving_first_trigger_ratio
        ratio_repeat = self.halving_repeat_trigger_ratio
        ratio_first = min(max(ratio_first, 0.0), 1.0)
        ratio_repeat = min(max(ratio_repeat, 0.0), 1.0)

        steps = 0
        if proposed < spacing:
            while spacing > self.min_tick_spacing and steps < self.max_progression_steps:
                trigger = ratio_first if steps == 0 else ratio_repeat
                if trigger <= 0:
                    break
                if proposed <= spacing * trigger:
                    spacing *= 0.5
                    steps += 1
                else:
                    break
            return max(spacing, self.min_tick_spacing)

        if proposed > spacing:
            up_first = self.doubling_first_trigger_ratio
            up_repeat = self.doubling_repeat_trigger_ratio
            if up_first <= 1.0:
                up_first = 1.25
            if up_repeat <= 1.0:
                up_repeat = 2.5
            while steps < self.max_progression_steps:
                trigger = up_first if steps == 0 else up_repeat
                if proposed > spacing * trigger:
                    spacing *= 2.0
                    steps += 1
                    if spacing >= self.halving_enforcement_threshold:
                        break
                else:
                    break
            return spacing

        return spacing

    def can_zoom_in_further(self) -> bool:
        """Determine if additional zoom-in should be allowed based on minimum spacing."""
        try:
            spacing = float(self.current_tick_spacing)
            min_spacing = float(self.min_tick_spacing)
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
        """Serialize coordinate system state for persistence."""
        try:
            visibility = {
                "left_bound": self._safe_float(self.get_visible_left_bound(), -self.width / 2.0),
                "right_bound": self._safe_float(self.get_visible_right_bound(), self.width / 2.0),
                "top_bound": self._safe_float(self.get_visible_top_bound(), self.height / 2.0),
                "bottom_bound": self._safe_float(self.get_visible_bottom_bound(), -self.height / 2.0),
            }
        except Exception:
            visibility = {
                "left_bound": -self.width / 2.0,
                "right_bound": self.width / 2.0,
                "top_bound": self.height / 2.0,
                "bottom_bound": -self.height / 2.0,
            }

        current_spacing = self._safe_float(self.current_tick_spacing, self.default_tick_spacing)
        default_spacing = self._safe_float(self.default_tick_spacing, 100.0)
        min_spacing = self._safe_float(self.min_tick_spacing, 0.0)
        spacing_str = format(current_spacing, ".12g")
        return {
            "Cartesian_System_Visibility": visibility,
            "current_tick_spacing": current_spacing,
            "default_tick_spacing": default_spacing,
            "current_tick_spacing_repr": spacing_str,
            "min_tick_spacing": min_spacing,
            "visible": self.visible,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore coordinate system state from persistence.

        Args:
            state: Dictionary containing coordinate system settings
        """
        if "visible" in state:
            self.visible = bool(state["visible"])
        if "current_tick_spacing" in state:
            self.current_tick_spacing = self._safe_float(
                state["current_tick_spacing"], self.default_tick_spacing
            )
        if "default_tick_spacing" in state:
            self.default_tick_spacing = self._safe_float(
                state["default_tick_spacing"], 100.0
            )

    def _get_axis_origin(self, axis: str) -> float:
        """Get the origin position for the specified axis"""
        origin: Position = self.origin
        return cast(float, origin.x if axis == 'x' else origin.y)

    def _get_axis_boundary(self, axis: str) -> float:
        """Get the boundary (width/height) for the specified axis"""
        return self.width if axis == 'x' else self.height
