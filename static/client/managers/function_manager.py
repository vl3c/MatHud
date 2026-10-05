"""
MatHud Function Management System

Manages mathematical function creation, modification, and deletion for graph visualization.
Handles function plotting with expression validation, bounds management, and colored area integration.

Core Responsibilities:
    - Function Creation: Creates mathematical function objects from string expressions
    - Function Modification: Updates existing function expressions and bounds
    - Function Deletion: Safe removal with cleanup of associated colored areas
    - Expression Validation: Ensures mathematical expressions are properly formatted

Mathematical Integration:
    - Expression Parsing: Converts string expressions to plottable mathematical functions
    - Bounds Management: Handles left and right domain boundaries for function visualization
    - Domain Validation: Ensures mathematical validity of function domains
    - Function Evaluation: Supports real-time function plotting and computation

Advanced Features:
    - Expression Fixing: Automatic correction of common mathematical notation issues
    - Function Updates: Modifies existing functions without recreation
    - Name Generation: Systematic naming for function identification
    - Colored Area Integration: Automatic cleanup of dependent area visualizations

Integration Points:
    - ExpressionValidator: Mathematical expression parsing and validation
    - ColoredAreaManager: Manages function-bounded area visualizations
    - Canvas: Handles function plotting and visual updates
    - DrawableManager: Coordinates with other geometric objects

Expression Support:
    - Mathematical Functions: sin, cos, tan, log, exp, sqrt, and more
    - Variables: x as primary variable for function expressions
    - Constants: pi, e, and other mathematical constants
    - Operations: Standard arithmetic and advanced mathematical operations

State Management:
    - Undo/Redo: Complete state archiving for function operations
    - Canvas Integration: Immediate visual updates after modifications
    - Dependency Tracking: Maintains relationships with colored areas
    - Expression Persistence: Preserves function expressions across operations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from drawables.function import Function
from managers.dependency_removal import remove_drawable_with_dependencies
from managers.edit_policy import DrawableEditPolicy, EditRule, get_drawable_edit_policy
from utils.function_features import (
    SUPPORTED_FEATURES,
    FeatureReport,
    FunctionFeature,
    find_function_features,
    find_intersections,
    round_report_value,
    sample_count,
)

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from name_generator.drawable import DrawableNameGenerator


class FunctionManager:
    """Manages function drawables for a Canvas with mathematical expression support."""

    def __init__(
        self,
        canvas: "Canvas",
        drawables_container: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
        """
        Initialize the FunctionManager.

        Args:
            canvas: The Canvas object this manager is responsible for
            drawables_container: The container for storing drawables
            name_generator: Generator for drawable names
            dependency_manager: Manager for drawable dependencies
            drawable_manager_proxy: Proxy to the main DrawableManager
        """
        self.canvas: "Canvas" = canvas
        self.drawables: "DrawablesContainer" = drawables_container
        self.name_generator: "DrawableNameGenerator" = name_generator
        self.dependency_manager: "DrawableDependencyManager" = dependency_manager
        self.drawable_manager: "DrawableManagerProxy" = drawable_manager_proxy
        self.function_edit_policy: Optional[DrawableEditPolicy] = get_drawable_edit_policy("Function")

    def get_function(self, name: str) -> Optional[Function]:
        """
        Get a function by its name.

        Searches through all existing functions to find one with the specified name.

        Args:
            name (str): The name of the function to find

        Returns:
            Function: The function with the matching name, or None if not found
        """
        functions = self.drawables.Functions
        for function in functions:
            if function.name == name:
                return function
        return None

    def draw_function(
        self,
        function_string: str,
        name: str,
        left_bound: Optional[float] = None,
        right_bound: Optional[float] = None,
        color: Optional[str] = None,
        undefined_at: Optional[List[float]] = None,
    ) -> Function:
        """
        Draw a function on the canvas.

        Creates a new function or updates an existing one with the specified mathematical
        expression. Handles expression validation, domain bounds, and automatic plotting.
        Archives the state for undo functionality.

        Args:
            function_string (str): Mathematical expression for the function (e.g., "x^2 + 1")
            name (str): Name identifier for the function
            left_bound (float, optional): Left domain boundary for function evaluation
            right_bound (float, optional): Right domain boundary for function evaluation
            color (str, optional): Color for the plotted function
            undefined_at (list, optional): List of x-values where the function is undefined (holes)

        Returns:
            Function: The newly created or updated function object

        Raises:
            ValueError: If the function string cannot be parsed or is invalid, or if
                left_bound equals right_bound

        Reversed bounds (left_bound > right_bound) are swapped. Drawing under the name of
        an existing function redefines it in place: expression, bounds and holes are
        replaced and everything derived from them (asymptotes, discontinuities,
        periodicity) is recomputed; its color is kept unless a new one is given.
        """
        left_bound, right_bound = self.ordered_bounds(left_bound, right_bound)

        # Archive before creation or modification
        self.canvas.undo_redo_manager.archive()

        # Check if the function already exists
        existing_function = self.get_function(name)
        color_value = str(color).strip() if color is not None else ""
        if existing_function:
            existing_function.redefine(function_string, left_bound, right_bound, undefined_at)

            if color_value:
                existing_function.update_color(color_value)

            if self.canvas.draw_enabled:
                self.canvas.draw()
            return existing_function
        else:
            # Generate a proper name if needed
            name = self.name_generator.generate_function_name(name)

            # Create the function (math-only)
            function_kwargs: Dict[str, Any] = {
                "name": name,
                "left_bound": left_bound,
                "right_bound": right_bound,
            }
            if color_value:
                function_kwargs["color"] = color_value
            if undefined_at:
                function_kwargs["undefined_at"] = undefined_at
            new_function = Function(function_string, **function_kwargs)

            # Add to drawables
            self.drawables.add(new_function)

            # Draw the function
            if self.canvas.draw_enabled:
                self.canvas.draw()

            return new_function

    @staticmethod
    def ordered_bounds(
        left_bound: Optional[float], right_bound: Optional[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return the bounds in increasing order; equal bounds leave nothing to plot and are rejected."""
        if left_bound is None or right_bound is None:
            return left_bound, right_bound
        if left_bound == right_bound:
            raise ValueError(
                f"left_bound and right_bound are both {left_bound}; left_bound must be less than right_bound."
            )
        if left_bound > right_bound:
            return right_bound, left_bound
        return left_bound, right_bound

    def delete_function(self, name: str) -> bool:
        """
        Delete a function by its name.

        Finds and removes the function with the specified name, along with
        any associated colored areas. Archives the state for undo functionality.

        Args:
            name (str): The name of the function to delete

        Returns:
            bool: True if the function was found and deleted, False otherwise
        """
        function = self.get_function(name)
        if not function:
            return False

        # Archive before deletion
        self.canvas.undo_redo_manager.archive()

        # Remove the function
        removed = remove_drawable_with_dependencies(self.drawables, self.dependency_manager, function)

        # Also delete any colored areas associated with this function
        self.canvas.drawable_manager.delete_colored_areas_for_function(function, archive=False)

        # Redraw the canvas
        if self.canvas.draw_enabled:
            self.canvas.draw()

        return bool(removed)

    def update_function(
        self,
        function_name: str,
        new_color: Optional[str] = None,
        new_left_bound: Optional[float] = None,
        new_right_bound: Optional[float] = None,
    ) -> bool:
        function = self.get_function(function_name)
        if not function:
            raise ValueError(f"Function '{function_name}' was not found.")

        pending_fields = self._collect_function_fields(new_color, new_left_bound, new_right_bound)
        self._validate_function_policy(list(pending_fields.keys()))
        self._validate_function_payload(function, pending_fields, new_color, new_left_bound, new_right_bound)

        self.canvas.undo_redo_manager.archive()
        self._apply_function_updates(function, pending_fields, new_color, new_left_bound, new_right_bound)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def _collect_function_fields(
        self,
        new_color: Optional[str],
        new_left_bound: Optional[float],
        new_right_bound: Optional[float],
    ) -> Dict[str, str]:
        pending_fields: Dict[str, str] = {}

        if new_color is not None:
            pending_fields["color"] = "color"

        if new_left_bound is not None:
            pending_fields["left_bound"] = "left_bound"

        if new_right_bound is not None:
            pending_fields["right_bound"] = "right_bound"

        if not pending_fields:
            raise ValueError("Provide at least one property to update.")

        return pending_fields

    def _validate_function_policy(self, requested_fields: List[str]) -> Dict[str, EditRule]:
        if not self.function_edit_policy:
            raise ValueError("Edit policy for functions is not configured.")

        validated_rules: Dict[str, EditRule] = {}
        for field in requested_fields:
            rule = self.function_edit_policy.get_rule(field)
            if not rule:
                raise ValueError(f"Editing field '{field}' is not permitted for functions.")
            validated_rules[field] = rule

        return validated_rules

    def _validate_function_payload(
        self,
        function: Function,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
        new_left_bound: Optional[float],
        new_right_bound: Optional[float],
    ) -> None:
        if "color" in pending_fields and (new_color is None or not str(new_color).strip()):
            raise ValueError("Function color cannot be empty.")

        updated_left = function.left_bound
        updated_right = function.right_bound

        if "left_bound" in pending_fields:
            if new_left_bound is None:
                raise ValueError("Function left_bound requires a numeric value.")
            updated_left = float(new_left_bound)

        if "right_bound" in pending_fields:
            if new_right_bound is None:
                raise ValueError("Function right_bound requires a numeric value.")
            updated_right = float(new_right_bound)

        if updated_left is not None and updated_right is not None and updated_left >= updated_right:
            raise ValueError("left_bound must be less than right_bound.")

    def _apply_function_updates(
        self,
        function: Function,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
        new_left_bound: Optional[float],
        new_right_bound: Optional[float],
    ) -> None:
        if "color" in pending_fields and new_color is not None:
            function.update_color(str(new_color))

        if "left_bound" in pending_fields and new_left_bound is not None:
            function.update_left_bound(float(new_left_bound))

        if "right_bound" in pending_fields and new_right_bound is not None:
            function.update_right_bound(float(new_right_bound))

        if "left_bound" in pending_fields or "right_bound" in pending_fields:
            # Asymptotes (e.g. of tan), discontinuities and periodicity depend on the bounds
            function.reanalyze()

    # ------------------- Roots, extrema and intersections -------------------

    def find_function_features(
        self,
        function_names: Sequence[str],
        features: Optional[Sequence[str]] = None,
        left_bound: Optional[float] = None,
        right_bound: Optional[float] = None,
        place_points: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Find the roots and local extrema of one plotted function, or where two of them intersect.

        Works on functions and piecewise functions. The interval is [left_bound, right_bound];
        a missing side defaults to the functions' own bound, else to the visible x range, and
        the interval is clipped to where every function is defined. Sampling is at least as
        fine as two samples per screen pixel. With two functions, ``features`` is ignored.

        With ``place_points`` a point is created at every distinct feature location (an
        existing point there is reused); together they are one undo step.

        Returns:
            Dict with the function names, the interval searched, the features (x, y, kind,
            sorted by x), their count, a truncated flag, a note when nothing was found or the
            list was cut, and point_name on each feature when points are placed.
        """
        curves = self._feature_curves(function_names)
        left, right = self._feature_interval(curves, left_bound, right_bound)
        breakpoints = self._feature_breakpoints(curves)
        samples = self._feature_sample_count(curves, right - left)
        if len(curves) == 1:
            report = find_function_features(
                curves[0].function,
                left,
                right,
                features=list(features) if features else SUPPORTED_FEATURES,
                breakpoints=breakpoints,
                samples=samples,
            )
        else:
            report = find_intersections(
                curves[0].function, curves[1].function, left, right, breakpoints=breakpoints, samples=samples
            )
        result = self._feature_result(curves, left, right, report)
        if place_points and report["features"]:
            placed = self._place_feature_points(report["features"])
            reuse_note = placed.pop("note", None)
            result.update(placed)
            if reuse_note:
                result["note"] = f"{result['note']} {reuse_note}" if result.get("note") else reuse_note
        return result

    def _feature_curves(self, function_names: Sequence[str]) -> List[Any]:
        """Look up one or two plotted functions (plain or piecewise) by name."""
        names = [str(name).strip() for name in (function_names or []) if str(name).strip()]
        if len(names) not in (1, 2):
            raise ValueError(f"Give one function name (roots and extrema) or two (intersections); got {len(names)}.")
        if names[0] == names[-1] and len(names) == 2:
            raise ValueError(f"Give two different functions to intersect; got '{names[0]}' twice.")
        curves: List[Any] = []
        for name in names:
            curve = self._find_plotted_function(name)
            if curve is None:
                raise ValueError(f"No function or piecewise function named '{name}' is plotted.")
            curves.append(curve)
        return curves

    def _find_plotted_function(self, name: str) -> Optional[Any]:
        for curve in list(self.drawables.Functions) + list(self.drawables.PiecewiseFunctions):
            if getattr(curve, "name", None) == name:
                return curve
        return None

    def _feature_interval(
        self, curves: List[Any], left_bound: Optional[float], right_bound: Optional[float]
    ) -> Tuple[float, float]:
        """Requested bounds, else the functions' own bounds, else the view; clipped to the functions' domain."""
        if left_bound is not None and right_bound is not None:
            left_bound, right_bound = self.ordered_bounds(float(left_bound), float(right_bound))
        own_lefts = [float(c.left_bound) for c in curves if getattr(c, "left_bound", None) is not None]
        own_rights = [float(c.right_bound) for c in curves if getattr(c, "right_bound", None) is not None]
        mapper = self.canvas.coordinate_mapper
        if left_bound is not None:
            left = float(left_bound)
        else:
            left = max(own_lefts) if own_lefts else float(mapper.get_visible_left_bound())
        if right_bound is not None:
            right = float(right_bound)
        else:
            right = min(own_rights) if own_rights else float(mapper.get_visible_right_bound())
        if own_lefts:
            left = max(left, max(own_lefts))
        if own_rights:
            right = min(right, min(own_rights))
        if not left < right:
            names = " and ".join(str(c.name) for c in curves)
            raise ValueError(f"Nothing to search: {names} is not defined between x = {left} and x = {right}.")
        return left, right

    @staticmethod
    def _feature_breakpoints(curves: List[Any]) -> List[float]:
        """Vertical asymptotes, point discontinuities and holes of every curve."""
        breakpoints: List[float] = []
        for curve in curves:
            for attribute in ("vertical_asymptotes", "point_discontinuities", "undefined_at"):
                breakpoints.extend(float(x) for x in (getattr(curve, attribute, None) or []))
        return breakpoints

    def _feature_sample_count(self, curves: List[Any], span: float) -> int:
        """Samples for the interval: at least two per screen pixel and 40 per period of a periodic curve."""
        periods = [
            float(c.estimated_period)
            for c in curves
            if getattr(c, "is_periodic", False) and getattr(c, "estimated_period", None)
        ]
        scale = float(getattr(self.canvas.coordinate_mapper, "scale_factor", 0.0) or 0.0)
        return int(sample_count(span, pixel_span=span * scale, period=min(periods) if periods else None))

    @staticmethod
    def _feature_result(curves: List[Any], left: float, right: float, report: FeatureReport) -> Dict[str, Any]:
        found = report["features"]
        interval = [round_report_value(left), round_report_value(right)]
        result: Dict[str, Any] = {
            "function_names": [str(c.name) for c in curves],
            "interval": interval,
            "features": found,
            "count": report["total_found"],
            "truncated": report["truncated"],
        }
        if not found:
            what = "intersections" if len(curves) == 2 else "requested features"
            result["note"] = f"No {what} found for x in [{interval[0]}, {interval[1]}]."
        elif report["truncated"]:
            result["note"] = (
                f"Showing the first {len(found)} of {report['total_found']} features by x; "
                "narrow the interval to see the rest."
            )
        return result

    def _place_feature_points(self, found: List[FunctionFeature]) -> Dict[str, Any]:
        """Create (or reuse) a point at each feature location, all as one undo step.

        Sets ``point_name`` on every feature. Features at the same spot (a touching root and
        its extremum) share one point. A point that already existed at a feature is reused and
        left unchanged; it is reported apart from the created ones, so the caller never
        deletes a point of the user's drawing thinking this call made it.

        Returns:
            point_names (distinct, in x order), created_point_names, reused_point_names, and a
            note when points were reused (merged into the result's note).
        """
        point_manager = self.drawable_manager.point_manager
        undo_manager = self.canvas.undo_redo_manager
        names: List[str] = []
        created: List[str] = []
        reused: List[str] = []
        undo_manager.begin_batch()
        try:
            for feature in found:
                existed_before = point_manager.get_point(feature["x"], feature["y"]) is not None
                point = point_manager.create_point(feature["x"], feature["y"], name="", extra_graphics=False)
                name = str(point.name)
                feature["point_name"] = name
                if name in names:
                    continue
                names.append(name)
                (reused if existed_before else created).append(name)
        finally:
            undo_manager.end_batch()
        placed: Dict[str, Any] = {
            "point_names": names,
            "created_point_names": created,
            "reused_point_names": reused,
        }
        if reused:
            reused_text = ", ".join(reused)
            placed["note"] = (
                f"Points {reused_text} already existed at feature locations and were reused, not created; "
                "to remove the feature points, delete only created_point_names."
            )
        return placed
