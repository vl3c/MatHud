"""
MatHud Coordinate System Manager

Manages the active coordinate system mode (Cartesian or Polar) and provides
access to the appropriate grid object for rendering.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from canvas import Canvas
    from cartesian_system_2axis import Cartesian2Axis
    from polar_grid import PolarGrid


class CoordinateSystemManager:
    """
    Manages coordinate system mode switching between Cartesian and Polar.

    Encapsulates the logic for determining which grid to render and provides
    state persistence for workspace save/restore.

    Attributes:
        canvas: Reference to the parent Canvas
        _mode: Current coordinate system mode ("cartesian" or "polar")
        cartesian_grid: The Cartesian2Axis grid instance
        polar_grid: The PolarGrid instance
    """

    VALID_MODES = ("cartesian", "polar")

    def __init__(self, canvas: "Canvas") -> None:
        """Initialize the coordinate system manager.

        Args:
            canvas: The parent Canvas instance
        """
        from polar_grid import PolarGrid

        self.canvas: "Canvas" = canvas
        self._mode: str = "cartesian"
        self.cartesian_grid: "Cartesian2Axis" = canvas.cartesian2axis
        self.polar_grid: "PolarGrid" = PolarGrid(canvas.coordinate_mapper)

    @property
    def mode(self) -> str:
        """Get the current coordinate system mode."""
        return self._mode

    def set_mode(self, mode: str, redraw: bool = True) -> None:
        """Set the coordinate system mode.

        Args:
            mode: The mode to set ("cartesian" or "polar")
            redraw: Whether to trigger a canvas redraw (default True)

        Raises:
            ValueError: If mode is not "cartesian" or "polar"
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Mode must be one of {self.VALID_MODES}, got '{mode}'")
        self._mode = mode
        if redraw:
            self.canvas.draw()

    def get_active_grid(self) -> Union["Cartesian2Axis", "PolarGrid"]:
        """Get the currently active grid based on mode.

        Returns:
            The Cartesian2Axis if mode is "cartesian", PolarGrid if "polar"
        """
        if self._mode == "cartesian":
            return self.cartesian_grid
        return self.polar_grid

    def is_cartesian(self) -> bool:
        """Check if the current mode is Cartesian."""
        return self._mode == "cartesian"

    def is_polar(self) -> bool:
        """Check if the current mode is Polar."""
        return self._mode == "polar"

    def toggle_mode(self, redraw: bool = True) -> str:
        """Toggle between Cartesian and Polar modes.

        Args:
            redraw: Whether to trigger a canvas redraw (default True)

        Returns:
            The new mode after toggling
        """
        new_mode = "polar" if self._mode == "cartesian" else "cartesian"
        self.set_mode(new_mode, redraw=redraw)
        return new_mode

    def invalidate_cache_on_zoom(self) -> None:
        """Invalidate grid caches when zoom changes."""
        self.cartesian_grid._invalidate_cache_on_zoom()
        self.polar_grid._invalidate_cache_on_zoom()

    def is_grid_visible(self) -> bool:
        """Check if the active grid is visible.

        Returns:
            True if the active grid is visible, False otherwise
        """
        return self.get_active_grid().visible

    def set_grid_visible(self, visible: bool, redraw: bool = True) -> None:
        """Set the visibility of the active grid.

        Args:
            visible: Whether the grid should be visible
            redraw: Whether to trigger a canvas redraw (default True)
        """
        self.get_active_grid().visible = visible
        if redraw:
            self.canvas.draw()

    def get_state(self) -> Dict[str, Any]:
        """Get the state for workspace persistence.

        Returns:
            Dict containing the current mode
        """
        return {"mode": self._mode}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from workspace data.

        Args:
            state: Dict containing the mode to restore
        """
        mode = state.get("mode", "cartesian")
        if mode in self.VALID_MODES:
            self._mode = mode
        else:
            self._mode = "cartesian"
