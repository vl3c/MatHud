"""
Implicit Curve Manager for MatHud

Creates, finds, recolours and deletes implicit curves (see drawables/implicit_curve.py)
and shades inequality regions (drawables/inequality_colored_area.py), with one undo entry
per change. Regions are colored areas: ColoredAreaManager deletes and edits them. The AI
tools bind to this manager directly (FunctionRegistry), so it maps null tool arguments
to their defaults itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from constants import default_area_fill_color, default_area_opacity, default_color
from drawables.implicit_curve import ImplicitCurve
from drawables.inequality_colored_area import InequalityColoredArea
from managers.dependency_removal import remove_drawable_with_dependencies
from utils.implicit_plotter import PlotResult
from utils.implicit_summary import curve_in_view, region_in_view
from utils.style_utils import StyleUtils

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager


class ImplicitCurveManager:
    """Manages implicit curves and inequality regions on the canvas.

    Attributes:
        canvas: The parent Canvas
        drawables: Container for all drawable objects
        dependency_manager: Tracks object dependencies
    """

    def __init__(
        self,
        canvas: "Canvas",
        drawables: "DrawablesContainer",
        dependency_manager: "DrawableDependencyManager",
    ) -> None:
        self.canvas: "Canvas" = canvas
        self.drawables: "DrawablesContainer" = drawables
        self.dependency_manager: "DrawableDependencyManager" = dependency_manager

    # ------------------------------------------------------------------
    # Implicit curves
    # ------------------------------------------------------------------

    def draw_implicit_curve(
        self, equation: str, name: Optional[str] = None, color: Optional[str] = None
    ) -> Dict[str, Any]:
        """Draw the curve of an equation in x and y (one undo entry).

        Returns:
            {"name", "equation", "in_view": what the current view shows of it}

        Raises:
            ValueError: If the equation or color is refused; nothing is then changed
        """
        curve = self.create_implicit_curve(equation, name, color)
        bounds, scale = self._view()
        drawn = self._drawn_plot(curve, scale)
        return {
            "name": curve.name,
            "equation": curve.equation,
            "in_view": curve_in_view(curve.equation, curve.g, curve.factors, bounds, scale, drawn),
        }

    def create_implicit_curve(
        self, equation: str, name: Optional[str] = None, color: Optional[str] = None
    ) -> ImplicitCurve:
        """Add the curve (one undo entry) and return it; also restores saved curves.

        Raises:
            ValueError: If the equation or color is refused; nothing is then changed
        """
        curve = ImplicitCurve(equation, name="c", color=self._checked_color(color, default_color))
        curve.name = self._name(name, "c")
        self._archive_for_undo()
        self.drawables.add(curve)
        self._redraw()
        return curve

    def get_implicit_curve(self, name: str) -> Optional[ImplicitCurve]:
        """The implicit curve with this name, or None."""
        if not name:
            return None
        for curve in self.drawables.ImplicitCurves:
            if getattr(curve, "name", None) == name:
                return curve  # type: ignore[return-value]
        return None

    def delete_implicit_curve(self, name: str) -> bool:
        """Delete the implicit curve with this name; False when there is none."""
        curve = self.get_implicit_curve(name)
        if curve is None:
            return False
        self._archive_for_undo()
        removed = remove_drawable_with_dependencies(self.drawables, self.dependency_manager, curve)
        if removed:
            self._redraw()
        return bool(removed)

    def update_implicit_curve(self, name: str, new_color: Optional[str] = None) -> bool:
        """Recolour an implicit curve; True when it exists, False when there is none.

        Raises:
            ValueError: If the color is not a valid CSS color
        """
        curve = self.get_implicit_curve(name)
        if curve is None:
            return False
        if new_color is None or new_color == curve.color:
            return True
        color = self._checked_color(new_color, curve.color)
        self._archive_for_undo()
        curve.update_color(color)
        self._redraw()
        return True

    # ------------------------------------------------------------------
    # Inequality regions
    # ------------------------------------------------------------------

    def shade_inequality(
        self,
        inequality: str,
        name: Optional[str] = None,
        color: Optional[str] = None,
        opacity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Shade the region where inequalities in x and y hold (one undo entry).

        Returns:
            {"name", "inequality", "in_view": the share of the current view it covers}

        Raises:
            ValueError: If the inequality, color or opacity is refused; nothing is then changed
        """
        area = self.create_inequality_area(inequality, name, color, opacity)
        bounds, _ = self._view()
        return {
            "name": area.name,
            "inequality": area.inequality,
            "in_view": region_in_view(area.inequality, area.contains, bounds),
        }

    def create_inequality_area(
        self,
        inequality: str,
        name: Optional[str] = None,
        color: Optional[str] = None,
        opacity: Optional[float] = None,
    ) -> InequalityColoredArea:
        """Add the region (one undo entry) and return it; also restores saved regions.

        Raises:
            ValueError: If the inequality, color or opacity is refused; nothing is then changed
        """
        fill_opacity = default_area_opacity if opacity is None else opacity
        if not StyleUtils.validate_opacity(fill_opacity):
            raise ValueError("opacity must be a number between 0 and 1.")
        area = InequalityColoredArea(
            inequality,
            name="R",
            color=self._checked_color(color, default_area_fill_color),
            opacity=float(fill_opacity),
        )
        area.name = self._name(name, "R")
        self._archive_for_undo()
        self.drawables.add(area)
        self._redraw()
        return area

    # ------------------------------------------------------------------

    def _name(self, requested: Optional[str], prefix: str) -> str:
        """The requested name made unique, or the first free prefix1, prefix2, ..."""
        if requested:
            return self.drawables.unique_name(str(requested))
        taken = self.drawables.names_in_use()
        number = 1
        while f"{prefix}{number}" in taken:
            number += 1
        return f"{prefix}{number}"

    @staticmethod
    def _checked_color(color: Optional[str], default: str) -> str:
        if color is None or color == "":
            return default
        if not StyleUtils.is_valid_css_color(color):
            raise ValueError(f"Invalid CSS color: {color}")
        return str(color)

    @staticmethod
    def _drawn_plot(drawable: Any, scale: float) -> Optional[PlotResult]:
        """The plot the renderer just made for the drawable at this scale, if any (so the
        tool's result does not plot again)."""
        renderable = getattr(drawable, "_renderable", None)
        result = getattr(renderable, "_result", None)
        key = getattr(renderable, "_key", None)
        if result is None or not key or key[0] != scale:
            return None
        return result  # type: ignore[no-any-return]

    def _view(self) -> Tuple[Tuple[float, float, float, float], float]:
        mapper = self.canvas.coordinate_mapper
        bounds = mapper.get_visible_bounds()
        return (
            (float(bounds["left"]), float(bounds["bottom"]), float(bounds["right"]), float(bounds["top"])),
            float(mapper.scale_factor),
        )

    def _archive_for_undo(self) -> None:
        undo_redo = getattr(self.canvas, "undo_redo_manager", None)
        if undo_redo:
            undo_redo.archive()

    def _redraw(self) -> None:
        if getattr(self.canvas, "draw_enabled", False):
            try:
                self.canvas.draw()
            except Exception:
                pass
