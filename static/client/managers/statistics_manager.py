"""Statistics manager for probability distributions and regression plots.

This module provides the StatisticsManager class which handles creation
and management of statistical plots including distributions and regressions.

Key Features:
    - Continuous distribution plots with PDF curves and shaded areas
    - Discrete distribution plots using bar elements
    - Custom bar chart creation with values and labels
    - Regression fitting (linear, polynomial, exponential, etc.)
    - Plot deletion with proper cleanup of constituent drawables
    - Workspace restore via plot materialization methods
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from constants import default_area_fill_color, default_area_opacity
from drawables.bars_plot import BarsPlot
from drawables.continuous_plot import ContinuousPlot
from drawables.bar import Bar
from drawables.discrete_plot import DiscretePlot
from drawables.plot import Plot
from utils.statistics.distributions import default_normal_bounds, normal_pdf_expression
from utils.statistics.regression import fit_regression as _fit_regression, SUPPORTED_MODEL_TYPES

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.colored_area_manager import ColoredAreaManager
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawables_container import DrawablesContainer
    from managers.function_manager import FunctionManager
    from managers.point_manager import PointManager
    from name_generator.drawable import DrawableNameGenerator


class StatisticsManager:
    def __init__(
        self,
        canvas: "Canvas",
        drawables: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        function_manager: "FunctionManager",
        colored_area_manager: "ColoredAreaManager",
    ) -> None:
        self.canvas = canvas
        self.drawables = drawables
        self.name_generator = name_generator
        self.dependency_manager = dependency_manager
        self.function_manager = function_manager
        self.colored_area_manager = colored_area_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_distribution(
        self,
        *,
        name: Optional[str],
        representation: str,
        distribution_type: str,
        distribution_params: Optional[Dict[str, Any]],
        plot_bounds: Optional[Dict[str, Any]],
        shade_bounds: Optional[Dict[str, Any]],
        curve_color: Optional[str],
        fill_color: Optional[str],
        fill_opacity: Optional[float],
        bar_count: Optional[float],
    ) -> Dict[str, Any]:
        dist = str(distribution_type or "").strip().lower()
        if dist != "normal":
            raise ValueError(f"Unsupported distribution_type '{distribution_type}'")

        rep = str(representation or "").strip().lower()
        if rep not in ("continuous", "discrete"):
            raise ValueError("representation must be 'continuous' or 'discrete'")

        mean, sigma = self._parse_normal_params(distribution_params)
        plot_bounds_dict = plot_bounds if isinstance(plot_bounds, dict) else {}
        plot_left_raw = plot_bounds_dict.get("left_bound")
        plot_right_raw = plot_bounds_dict.get("right_bound")
        resolved_left, resolved_right = self._resolve_bounds(mean, sigma, plot_left_raw, plot_right_raw)

        plot_name = self._generate_unique_name(
            self.name_generator.filter_string(name or "") or "normal_plot"
        )

        if rep == "discrete":
            return self._plot_distribution_discrete(
                plot_name=plot_name,
                distribution_type=dist,
                mean=mean,
                sigma=sigma,
                left_bound=resolved_left,
                right_bound=resolved_right,
                curve_color=curve_color,
                fill_color=fill_color,
                fill_opacity=fill_opacity,
                bar_count=bar_count,
            )

        expression = normal_pdf_expression(mean, sigma)

        function_preferred = f"{plot_name}_pdf"
        function_name = self.name_generator.generate_function_name(function_preferred)
        function_obj = self.function_manager.draw_function(
            expression,
            name=function_name,
            left_bound=resolved_left,
            right_bound=resolved_right,
            color=curve_color,
        )
        function_name = getattr(function_obj, "name", function_name)

        shade_left = resolved_left
        shade_right = resolved_right
        shade_bounds_dict = shade_bounds if isinstance(shade_bounds, dict) else {}
        if shade_bounds_dict:
            shade_left_raw = shade_bounds_dict.get("left_bound")
            shade_right_raw = shade_bounds_dict.get("right_bound")
            if shade_left_raw is not None:
                shade_left = float(shade_left_raw)
            if shade_right_raw is not None:
                shade_right = float(shade_right_raw)
            if not math.isfinite(shade_left) or not math.isfinite(shade_right):
                raise ValueError("shade_bounds left_bound and right_bound must be finite")

        # Clamp shade interval into plot interval.
        shade_left = max(resolved_left, shade_left)
        shade_right = min(resolved_right, shade_right)
        if shade_left >= shade_right:
            raise ValueError("shade_bounds must define a non-empty interval within plot_bounds")

        area = self.colored_area_manager.create_colored_area(
            drawable1_name=function_name,
            drawable2_name=None,
            left_bound=shade_left,
            right_bound=shade_right,
            color=self._normalize_fill_color(fill_color),
            opacity=self._normalize_fill_opacity(fill_opacity),
        )

        fill_area_name = self._generate_unique_name(f"{plot_name}_fill")
        try:
            area.name = fill_area_name
        except Exception:
            fill_area_name = getattr(area, "name", fill_area_name)

        plot = ContinuousPlot(
            plot_name,
            plot_type="distribution",
            distribution_type=dist,
            function_name=function_name,
            fill_area_name=fill_area_name,
            distribution_params={"mean": mean, "sigma": sigma},
            bounds={"left": resolved_left, "right": resolved_right},
        )
        self.drawables.add(plot)

        # Non-renderable composite; dependency analyzer currently ignores it but keep call for symmetry.
        try:
            self.dependency_manager.analyze_drawable_for_dependencies(plot)
        except Exception:
            pass

        return {
            "plot_name": plot_name,
            "representation": "continuous",
            "distribution_type": dist,
            "distribution_params": {"mean": mean, "sigma": sigma},
            "bounds": {"left": resolved_left, "right": resolved_right},
            "shade_bounds": {"left": shade_left, "right": shade_right},
            "function_name": function_name,
            "fill_area_name": fill_area_name,
        }

    def plot_bars(
        self,
        *,
        name: Optional[str],
        values: List[float],
        labels_below: List[str],
        labels_above: Optional[List[str]],
        bar_spacing: Optional[float],
        bar_width: Optional[float],
        stroke_color: Optional[str],
        fill_color: Optional[str],
        fill_opacity: Optional[float],
        x_start: Optional[float],
        y_base: Optional[float],
    ) -> Dict[str, Any]:
        if not isinstance(values, list) or not values:
            raise ValueError("values must be a non-empty list")
        if not isinstance(labels_below, list) or len(labels_below) != len(values):
            raise ValueError("labels_below length must match values length")
        if labels_above is not None:
            if not isinstance(labels_above, list) or len(labels_above) != len(values):
                raise ValueError("labels_above length must match values length")

        spacing = 0.2 if bar_spacing is None else float(bar_spacing)
        if not math.isfinite(spacing) or spacing < 0.0:
            raise ValueError("bar_spacing must be a finite number >= 0")

        width = 1.0 if bar_width is None else float(bar_width)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("bar_width must be a finite number > 0")

        x0 = 0.0 if x_start is None else float(x_start)
        if not math.isfinite(x0):
            raise ValueError("x_start must be finite")

        y0 = 0.0 if y_base is None else float(y_base)
        if not math.isfinite(y0):
            raise ValueError("y_base must be finite")

        plot_name = self._generate_unique_name(
            self.name_generator.filter_string(name or "") or "bars_plot"
        )

        plot = BarsPlot(
            plot_name,
            plot_type="bars",
            values=[float(v) for v in values],
            labels_below=[str(v) for v in labels_below],
            labels_above=None if labels_above is None else [str(v) for v in labels_above],
            bar_spacing=spacing,
            bar_width=width,
            x_start=x0,
            y_base=y0,
            stroke_color=None if stroke_color is None else str(stroke_color),
            fill_color=self._normalize_fill_color(fill_color),
            fill_opacity=self._normalize_fill_opacity(fill_opacity),
        )
        self.drawables.add(plot)

        self.materialize_bars_plot(plot)

        if getattr(self.canvas, "draw_enabled", False):
            try:
                self.canvas.draw()
            except Exception:
                pass

        return {
            "plot_name": plot_name,
            "plot_type": "bars",
            "bar_count": len(values),
        }

    def fit_regression(
        self,
        *,
        name: Optional[str],
        x_data: List[float],
        y_data: List[float],
        model_type: str,
        degree: Optional[int],
        plot_bounds: Optional[Dict[str, Any]],
        curve_color: Optional[str],
        show_points: Optional[bool],
        point_color: Optional[str],
    ) -> Dict[str, Any]:
        """
        Fit a regression model to data points and plot the resulting curve.

        Creates a standalone Function for the fitted curve and optionally
        Point markers for the data. No tracking entity is created - delete
        the function with delete_function() and points individually.

        Args:
            name: Optional base name for the function and points
            x_data: List of x values
            y_data: List of y values
            model_type: One of "linear", "polynomial", "exponential",
                       "logarithmic", "power", "logistic", "sinusoidal"
            degree: Polynomial degree (required for polynomial model)
            plot_bounds: Optional bounds {left_bound, right_bound} for plotting
            curve_color: Optional color for the curve
            show_points: Whether to plot the data points (default True)
            point_color: Optional color for data points

        Returns:
            Dict with function_name, expression, coefficients, r_squared,
            model_type, bounds, and optionally point_names
        """
        # Validate model type
        model = str(model_type or "").strip().lower()
        if model not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Supported: {', '.join(SUPPORTED_MODEL_TYPES)}"
            )

        # Validate degree for polynomial
        if model == "polynomial":
            if degree is None:
                raise ValueError("degree is required for polynomial regression")
            if not isinstance(degree, int) or degree < 1:
                raise ValueError("degree must be a positive integer")

        # Fit the regression model
        result = _fit_regression(x_data, y_data, model, degree)
        expression = result["expression"]
        coefficients = result["coefficients"]
        r_squared = result["r_squared"]

        # Generate base name for function and points
        default_name = f"{model}_fit"
        base_name = self.name_generator.filter_string(name or "") or default_name

        # Determine plot bounds
        x_min = min(x_data)
        x_max = max(x_data)
        x_range = x_max - x_min
        padding = x_range * 0.1 if x_range > 0 else 1.0

        left_bound = x_min - padding
        right_bound = x_max + padding

        if plot_bounds is not None and isinstance(plot_bounds, dict):
            if plot_bounds.get("left_bound") is not None:
                left_bound = float(plot_bounds["left_bound"])
            if plot_bounds.get("right_bound") is not None:
                right_bound = float(plot_bounds["right_bound"])

        if not math.isfinite(left_bound) or not math.isfinite(right_bound):
            raise ValueError("plot_bounds must contain finite values")
        if left_bound >= right_bound:
            raise ValueError("left_bound must be less than right_bound")

        # Draw the fitted function
        function_name = self.name_generator.generate_function_name(base_name)
        function_obj = self.function_manager.draw_function(
            expression,
            name=function_name,
            left_bound=left_bound,
            right_bound=right_bound,
            color=curve_color,
        )
        function_name = getattr(function_obj, "name", function_name)

        # Optionally plot the data points
        point_names: List[str] = []
        should_show_points = show_points if show_points is not None else True

        if should_show_points:
            point_manager = self._get_point_manager()
            if point_manager is not None:
                for i, (x, y) in enumerate(zip(x_data, y_data)):
                    point_preferred = f"{base_name}_pt{i}"
                    try:
                        created_point = point_manager.create_point(
                            x=x,
                            y=y,
                            name=point_preferred,
                            color=point_color,
                            extra_graphics=False,
                        )
                        if created_point is not None:
                            point_names.append(created_point.name)
                    except Exception as e:
                        # Log point creation failure for debugging
                        try:
                            logger = getattr(self.canvas, "logger", None)
                            if logger is not None:
                                logger.debug(
                                    f"Failed to create regression point {point_preferred}: {e}"
                                )
                        except Exception:
                            pass

        # Trigger redraw
        if getattr(self.canvas, "draw_enabled", False):
            try:
                self.canvas.draw()
            except Exception:
                pass

        result_dict: Dict[str, Any] = {
            "function_name": function_name,
            "expression": expression,
            "coefficients": coefficients,
            "r_squared": r_squared,
            "model_type": model,
            "bounds": {"left": left_bound, "right": right_bound},
        }

        if point_names:
            result_dict["point_names"] = point_names

        return result_dict

    def _get_point_manager(self) -> Optional["PointManager"]:
        """Get the point manager from the canvas."""
        try:
            dm = getattr(self.canvas, "drawable_manager", None)
            if dm is not None:
                return getattr(dm, "point_manager", None)
        except Exception:
            pass
        return None

    def delete_plot(self, name: str) -> bool:
        plot = self._get_plot_by_name(name)
        if plot is None:
            return False

        plot_class = plot.get_class_name() if hasattr(plot, "get_class_name") else ""

        if plot_class == "DiscretePlot":
            self._delete_discrete_plot(plot)
        elif plot_class == "BarsPlot":
            self._delete_bars_plot(plot)
        elif plot_class == "ContinuousPlot":
            self._delete_continuous_plot(plot)
        else:
            # Legacy best-effort deletion.
            self._delete_legacy_plot(plot)

        try:
            self.drawables.remove(plot)
        except Exception:
            pass

        try:
            self.dependency_manager.remove_drawable(plot)
        except Exception:
            pass

        if getattr(self.canvas, "draw_enabled", False):
            try:
                self.canvas.draw()
            except Exception:
                pass

        return True

    def materialize_discrete_plot(self, plot: Any) -> None:
        """
        Create Bar drawables for a DiscretePlot from its stored parameters.

        This is intended for workspace restore and does not archive undo history.
        """
        if plot is None:
            return
        plot_name = str(getattr(plot, "name", "") or "")
        if not plot_name:
            return
        bounds = getattr(plot, "bounds", None)
        if not isinstance(bounds, dict):
            return
        left = bounds.get("left")
        right = bounds.get("right")
        try:
            left_f = float(left)
            right_f = float(right)
        except Exception:
            return
        if not math.isfinite(left_f) or not math.isfinite(right_f) or left_f >= right_f:
            return

        params = getattr(plot, "distribution_params", None)
        params = params if isinstance(params, dict) else {}
        try:
            mean = float(params.get("mean", 0.0))
            sigma = float(params.get("sigma", 1.0))
        except Exception:
            return
        if not math.isfinite(mean) or not math.isfinite(sigma) or sigma <= 0.0:
            return

        bar_count = getattr(plot, "bar_count", None)
        try:
            n = int(bar_count) if bar_count is not None else 0
        except Exception:
            n = 0
        if n <= 0:
            return

        labels = getattr(plot, "bar_labels", None)
        labels_list: Optional[List[str]] = None
        if isinstance(labels, list):
            labels_list = [str(value) for value in labels]

        curve_color = getattr(plot, "curve_color", None)
        fill_color = getattr(plot, "fill_color", None)
        fill_opacity = getattr(plot, "fill_opacity", None)

        bar_manager = getattr(getattr(self.canvas, "drawable_manager", None), "bar_manager", None)
        for i in range(n):
            bar_name = f"{plot_name}_bar_{i}"
            if self._drawable_exists(bar_name, "Bar"):
                continue
            x_left = left_f + i * ((right_f - left_f) / float(n))
            x_right = left_f + (i + 1) * ((right_f - left_f) / float(n))
            x_mid = (x_left + x_right) / 2.0
            y_top = self._normal_pdf_value(x_mid, mean, sigma)

            label_text = None
            if labels_list is not None and i < len(labels_list):
                label_text = labels_list[i]

            if bar_manager is not None and hasattr(bar_manager, "create_bar"):
                bar_manager.create_bar(
                    name=bar_name,
                    x_left=x_left,
                    x_right=x_right,
                    y_bottom=0.0,
                    y_top=y_top,
                    stroke_color=curve_color,
                    fill_color=self._normalize_fill_color(fill_color),
                    fill_opacity=self._normalize_fill_opacity(fill_opacity),
                    label_above_text=label_text,
                    label_below_text=None,
                    archive=False,
                    redraw=False,
                )
            else:
                self.drawables.add(
                    Bar(
                        name=bar_name,
                        x_left=x_left,
                        x_right=x_right,
                        y_bottom=0.0,
                        y_top=y_top,
                        stroke_color=curve_color,
                        fill_color=self._normalize_fill_color(fill_color),
                        fill_opacity=self._normalize_fill_opacity(fill_opacity),
                        label_text=label_text,
                    )
                )

    def materialize_bars_plot(self, plot: Any) -> None:
        """
        Create Bar drawables for a BarsPlot from its stored parameters.

        This is intended for workspace restore and does not archive undo history.
        """
        if plot is None:
            return
        plot_name = str(getattr(plot, "name", "") or "")
        if not plot_name:
            return

        values = getattr(plot, "values", None)
        labels_below = getattr(plot, "labels_below", None)
        labels_above = getattr(plot, "labels_above", None)
        if not isinstance(values, list) or not values:
            return
        if not isinstance(labels_below, list) or len(labels_below) != len(values):
            return
        if labels_above is not None:
            if not isinstance(labels_above, list) or len(labels_above) != len(values):
                return

        try:
            spacing = float(getattr(plot, "bar_spacing", 0.2) or 0.2)
            width = float(getattr(plot, "bar_width", 1.0) or 1.0)
            x0 = float(getattr(plot, "x_start", 0.0) or 0.0)
            y0 = float(getattr(plot, "y_base", 0.0) or 0.0)
        except Exception:
            return
        if not math.isfinite(spacing) or spacing < 0.0:
            return
        if not math.isfinite(width) or width <= 0.0:
            return
        if not math.isfinite(x0) or not math.isfinite(y0):
            return

        stroke_color = getattr(plot, "stroke_color", None)
        fill_color = getattr(plot, "fill_color", None)
        fill_opacity = getattr(plot, "fill_opacity", None)

        bar_manager = getattr(getattr(self.canvas, "drawable_manager", None), "bar_manager", None)
        for i in range(len(values)):
            bar_name = f"{plot_name}_bar_{i}"
            if self._drawable_exists(bar_name, "Bar"):
                continue
            try:
                height = float(values[i])
            except Exception:
                continue
            if not math.isfinite(height):
                continue
            x_left = x0 + float(i) * (width + spacing)
            x_right = x_left + width
            y_bottom = y0
            y_top = y0 + height

            above_text = None
            try:
                if labels_above is not None and i < len(labels_above):
                    above_text = str(labels_above[i])
            except Exception:
                above_text = None

            below_text = None
            try:
                if i < len(labels_below):
                    below_text = str(labels_below[i])
            except Exception:
                below_text = None

            if bar_manager is not None and hasattr(bar_manager, "create_bar"):
                bar_manager.create_bar(
                    name=bar_name,
                    x_left=x_left,
                    x_right=x_right,
                    y_bottom=y_bottom,
                    y_top=y_top,
                    stroke_color=stroke_color,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity,
                    label_above_text=above_text,
                    label_below_text=below_text,
                    archive=False,
                    redraw=False,
                )
            else:
                self.drawables.add(
                    Bar(
                        name=bar_name,
                        x_left=x_left,
                        x_right=x_right,
                        y_bottom=y_bottom,
                        y_top=y_top,
                        stroke_color=stroke_color,
                        fill_color=fill_color,
                        fill_opacity=fill_opacity,
                        label_above_text=above_text,
                        label_below_text=below_text,
                    )
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_plot_by_name(self, name: str) -> Optional[Plot]:
        for class_name in ("ContinuousPlot", "DiscretePlot", "BarsPlot", "Plot"):
            for d in self.drawables.get_by_class_name(class_name):
                try:
                    if getattr(d, "name", None) == name:
                        return d  # type: ignore[return-value]
                except Exception:
                    continue
        return None

    def _plot_distribution_discrete(
        self,
        *,
        plot_name: str,
        distribution_type: str,
        mean: float,
        sigma: float,
        left_bound: float,
        right_bound: float,
        curve_color: Optional[str],
        fill_color: Optional[str],
        fill_opacity: Optional[float],
        bar_count: Optional[float],
    ) -> Dict[str, Any]:
        n = self._resolve_bar_count(bar_count)
        width = (right_bound - left_bound) / float(n)
        if width <= 0.0:
            raise ValueError("Invalid bounds for discrete plot")

        bar_manager = getattr(getattr(self.canvas, "drawable_manager", None), "bar_manager", None)
        for i in range(n):
            x_left = left_bound + i * width
            x_right = x_left + width
            x_mid = (x_left + x_right) / 2.0
            height = self._normal_pdf_value(x_mid, mean, sigma)

            bar_name = f"{plot_name}_bar_{i}"
            label_text = None

            if bar_manager is not None and hasattr(bar_manager, "create_bar"):
                bar_manager.create_bar(
                    name=bar_name,
                    x_left=x_left,
                    x_right=x_right,
                    y_bottom=0.0,
                    y_top=height,
                    stroke_color=curve_color,
                    fill_color=self._normalize_fill_color(fill_color),
                    fill_opacity=self._normalize_fill_opacity(fill_opacity),
                    label_above_text=None,
                    label_below_text=None,
                    archive=False,
                    redraw=False,
                )
            else:
                self.drawables.add(
                    Bar(
                        name=bar_name,
                        x_left=x_left,
                        x_right=x_right,
                        y_bottom=0.0,
                        y_top=height,
                        stroke_color=curve_color,
                        fill_color=self._normalize_fill_color(fill_color),
                        fill_opacity=self._normalize_fill_opacity(fill_opacity),
                        label_text=None,
                    )
                )

        plot = DiscretePlot(
            plot_name,
            plot_type="distribution",
            distribution_type=distribution_type,
            bar_count=n,
            bar_labels=None,
            curve_color=None if curve_color is None else str(curve_color),
            fill_color=self._normalize_fill_color(fill_color),
            fill_opacity=self._normalize_fill_opacity(fill_opacity),
            distribution_params={"mean": mean, "sigma": sigma},
            bounds={"left": left_bound, "right": right_bound},
            metadata={"bar_count": n},
        )
        self.drawables.add(plot)

        return {
            "plot_name": plot_name,
            "representation": "discrete",
            "distribution_type": distribution_type,
            "distribution_params": {"mean": mean, "sigma": sigma},
            "bounds": {"left": left_bound, "right": right_bound},
            "bar_count": n,
        }

    def _resolve_bar_count(self, bar_count: Optional[float]) -> int:
        if bar_count is None:
            return 24
        value = float(bar_count)
        if not math.isfinite(value):
            raise ValueError("bar_count must be finite")
        if abs(value - round(value)) > 1e-9:
            raise ValueError("bar_count must be an integer")
        n = int(round(value))
        if n <= 0:
            raise ValueError("bar_count must be > 0")
        return n

    def _normal_pdf_value(self, x: float, mean: float, sigma: float) -> float:
        # f(x) = 1/(sigma*sqrt(2*pi)) * exp(-(x-mean)^2/(2*sigma^2))
        coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        exponent = -((x - mean) ** 2) / (2.0 * (sigma ** 2))
        return coeff * math.exp(exponent)

    def _delete_continuous_plot(self, plot: Any) -> None:
        fill_area_name = getattr(plot, "fill_area_name", None)
        if fill_area_name:
            try:
                self.colored_area_manager.delete_colored_area(fill_area_name)
            except Exception:
                pass
        function_name = getattr(plot, "function_name", None)
        if function_name:
            try:
                self.function_manager.delete_function(function_name)
            except Exception:
                pass

    def _delete_discrete_plot(self, plot: Any) -> None:
        # New-style discrete plots: delete Bars by deterministic names.
        plot_name = str(getattr(plot, "name", "") or "")
        bar_count = getattr(plot, "bar_count", None)
        try:
            n = int(bar_count) if bar_count is not None else 0
        except Exception:
            n = 0
        if plot_name and n > 0:
            bar_manager = getattr(getattr(self.canvas, "drawable_manager", None), "bar_manager", None)
            for i in range(n):
                bar_name = f"{plot_name}_bar_{i}"
                if bar_manager is not None and hasattr(bar_manager, "delete_bar"):
                    try:
                        bar_manager.delete_bar(bar_name, archive=False, redraw=False)
                    except Exception:
                        self._delete_drawable_by_name(bar_name, "Bar")
                else:
                    self._delete_drawable_by_name(bar_name, "Bar")
            return

    def _delete_bars_plot(self, plot: Any) -> None:
        plot_name = str(getattr(plot, "name", "") or "")
        if not plot_name:
            return
        values = getattr(plot, "values", None)
        if not isinstance(values, list) or not values:
            return
        bar_manager = getattr(getattr(self.canvas, "drawable_manager", None), "bar_manager", None)
        for i in range(len(values)):
            bar_name = f"{plot_name}_bar_{i}"
            if bar_manager is not None and hasattr(bar_manager, "delete_bar"):
                try:
                    bar_manager.delete_bar(bar_name, archive=False, redraw=False)
                    continue
                except Exception:
                    pass
            self._delete_drawable_by_name(bar_name, "Bar")

        # Legacy discrete plots: delete colored areas and rectangles.
        fill_names = getattr(plot, "fill_area_names", []) or []
        for fill_name in list(fill_names):
            if not fill_name:
                continue
            try:
                self.colored_area_manager.delete_colored_area(str(fill_name))
            except Exception:
                pass

        rect_names = getattr(plot, "rectangle_names", []) or []
        for rect_name in list(rect_names):
            if not rect_name:
                continue
            try:
                self.canvas.delete_polygon(name=str(rect_name), polygon_type="rectangle")
            except Exception:
                pass

    def _delete_legacy_plot(self, plot: Any) -> None:
        # Best-effort support for older plot objects that stored component names directly.
        if getattr(plot, "fill_area_name", None) or getattr(plot, "function_name", None):
            self._delete_continuous_plot(plot)
            return
        if getattr(plot, "fill_area_names", None) or getattr(plot, "rectangle_names", None):
            self._delete_discrete_plot(plot)
            return

    def _drawable_exists(self, name: str, class_name: str) -> bool:
        if not name:
            return False
        try:
            for d in self.drawables.get_by_class_name(class_name):
                if getattr(d, "name", None) == name:
                    return True
        except Exception:
            return False
        return False

    def _delete_drawable_by_name(self, name: str, class_name: str) -> None:
        if not name:
            return
        try:
            for d in list(self.drawables.get_by_class_name(class_name)):
                if getattr(d, "name", None) == name:
                    self.drawables.remove(d)
                    try:
                        self.dependency_manager.remove_drawable(d)
                    except Exception:
                        pass
                    return
        except Exception:
            return

    def _generate_unique_name(self, base: str) -> str:
        base = str(base or "").strip()
        if not base:
            base = "plot"

        existing_names = set()
        try:
            for drawable in self.drawables.get_all():
                dname = getattr(drawable, "name", "")
                if dname:
                    existing_names.add(str(dname))
        except Exception:
            existing_names = set()

        if base not in existing_names:
            return base

        idx = 1
        while True:
            candidate = f"{base}_{idx}"
            if candidate not in existing_names:
                return candidate
            idx += 1

    def _parse_normal_params(self, params: Optional[Dict[str, Any]]) -> tuple[float, float]:
        params = params if isinstance(params, dict) else {}

        mean_raw = params.get("mean")
        sigma_raw = params.get("sigma")

        mean = 0.0 if mean_raw is None else float(mean_raw)
        sigma = 1.0 if sigma_raw is None else float(sigma_raw)

        if not math.isfinite(mean):
            raise ValueError("mean must be finite")
        if not math.isfinite(sigma):
            raise ValueError("sigma must be finite")
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0")

        return mean, sigma

    def _resolve_bounds(
        self,
        mean: float,
        sigma: float,
        left_bound: Optional[float],
        right_bound: Optional[float],
    ) -> tuple[float, float]:
        default_left, default_right = default_normal_bounds(mean, sigma, k=4.0)
        left = default_left if left_bound is None else float(left_bound)
        right = default_right if right_bound is None else float(right_bound)

        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("left_bound and right_bound must be finite")
        if left >= right:
            raise ValueError("left_bound must be less than right_bound")

        return left, right

    def _normalize_fill_color(self, fill_color: Optional[str]) -> str:
        if fill_color is None:
            return default_area_fill_color
        value = str(fill_color).strip()
        return value if value else default_area_fill_color

    def _normalize_fill_opacity(self, fill_opacity: Optional[float]) -> float:
        if fill_opacity is None:
            return float(default_area_opacity)
        value = float(fill_opacity)
        if not math.isfinite(value):
            return float(default_area_opacity)
        return max(0.0, min(value, 1.0))


