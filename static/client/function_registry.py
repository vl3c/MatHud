"""
MatHud Function Registry System

Central registry for mapping AI function names to their implementations.
Manages available functions and defines which operations support undo/redo functionality.

Function Categories:
    - Canvas operations: reset, clear, undo, redo
    - Geometric shapes: points, segments, vectors, triangles, rectangles, circles, ellipses
    - Mathematical functions: plotting, evaluation, symbolic computation
    - Object transformations: translate, rotate
    - Workspace management: save, load, list, delete
    - Special features: colored areas, angle measurement, testing

Dependencies:
    - utils.math_utils: Mathematical computation functions
    - process_function_calls: Expression evaluation facade
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, cast

from utils.math_utils import MathUtils
from process_function_calls import ProcessFunctionCalls

if TYPE_CHECKING:
    from ai_interface import AIInterface
    from canvas import Canvas
    from workspace_manager import WorkspaceManager


class FunctionRegistry:
    """Static registry mapping function names to implementations and undo capabilities.

    Maintains the central mapping between AI function names and their Python implementations,
    and defines which functions support undo/redo operations for state management.
    """

    @staticmethod
    def _convert_coordinates(coord1: float, coord2: float, from_system: str, to_system: str) -> Dict[str, Any]:
        """Convert coordinates between rectangular/cartesian and polar systems.

        Args:
            coord1: First coordinate (x for rect-to-polar, r for polar-to-rect)
            coord2: Second coordinate (y for rect-to-polar, theta for polar-to-rect)
            from_system: Source system ("rectangular", "cartesian", or "polar")
            to_system: Target system ("rectangular", "cartesian", or "polar")

        Returns:
            Dict with converted coordinates
        """
        # Normalize "cartesian" to "rectangular" for consistency
        if from_system == "cartesian":
            from_system = "rectangular"
        if to_system == "cartesian":
            to_system = "rectangular"

        if from_system == to_system:
            return {"coord1": coord1, "coord2": coord2, "system": to_system}

        if from_system == "rectangular" and to_system == "polar":
            r, theta = MathUtils.rectangular_to_polar(coord1, coord2)
            return {"r": r, "theta": theta, "theta_degrees": theta * 180 / 3.141592653589793}
        elif from_system == "polar" and to_system == "rectangular":
            x, y = MathUtils.polar_to_rectangular(coord1, coord2)
            return {"x": x, "y": y}
        else:
            return {"error": f"Invalid conversion: {from_system} to {to_system}"}

    @staticmethod
    def get_available_functions(canvas: "Canvas", workspace_manager: "WorkspaceManager", ai_interface: Optional["AIInterface"] = None) -> Dict[str, Any]:
        """Get the complete dictionary of all available functions with their implementations.

        Creates the mapping between AI function names and their bound Python methods,
        organizing functions by category for easier maintenance.

        Args:
            canvas (Canvas): Canvas instance for geometric and mathematical operations
            workspace_manager (WorkspaceManager): Manager for workspace persistence operations
            ai_interface (AIInterface, optional): Interface for testing functions

        Returns:
            dict: Complete mapping of function names to their bound implementations
        """
        functions: Dict[str, Any] = {
            # ===== CANVAS OPERATIONS =====
            "reset_canvas": canvas.reset,
            "clear_canvas": canvas.clear,
            "zoom": canvas.zoom,
            "get_current_canvas_state": lambda drawable_types=None, object_names=None, include_computations=True: {
                "type": "canvas_state",
                "value": canvas.get_canvas_state_filtered(
                    drawable_types=drawable_types,
                    object_names=object_names,
                    include_computations=include_computations,
                ),
            },

            # ===== POINT OPERATIONS =====
            "create_point": canvas.create_point,
            "delete_point": canvas.delete_point,
            "update_point": canvas.update_point,

            # ===== SEGMENT OPERATIONS =====
            "create_segment": canvas.create_segment,
            "delete_segment": canvas.delete_segment,
            "update_segment": canvas.update_segment,

            # ===== VECTOR OPERATIONS =====
            "create_vector": canvas.create_vector,
            "delete_vector": canvas.delete_vector,
            "update_vector": canvas.update_vector,

            # ===== POLYGON OPERATIONS =====
            "create_polygon": canvas.create_polygon,
            "delete_polygon": canvas.delete_polygon,
            "update_polygon": canvas.update_polygon,

            # ===== CIRCLE OPERATIONS =====
            "create_circle": canvas.create_circle,
            "delete_circle": canvas.delete_circle,
            "update_circle": canvas.update_circle,

            # ===== CIRCLE ARC OPERATIONS =====
            "create_circle_arc": canvas.create_circle_arc,
            "delete_circle_arc": canvas.delete_circle_arc,
            "update_circle_arc": canvas.update_circle_arc,

            # ===== ELLIPSE OPERATIONS =====
            "create_ellipse": canvas.create_ellipse,
            "delete_ellipse": canvas.delete_ellipse,
            "update_ellipse": canvas.update_ellipse,

            # ===== LABEL OPERATIONS =====
            "create_label": canvas.create_label,
            "delete_label": canvas.delete_label,
            "update_label": canvas.update_label,

            # ===== FUNCTION PLOTTING =====
            "draw_function": canvas.draw_function,
            "delete_function": canvas.delete_function,
            "update_function": canvas.update_function,

            # ===== PIECEWISE FUNCTION PLOTTING =====
            "draw_piecewise_function": canvas.draw_piecewise_function,
            "delete_piecewise_function": canvas.delete_piecewise_function,
            "update_piecewise_function": canvas.update_piecewise_function,

            # ===== PARAMETRIC FUNCTION PLOTTING =====
            "draw_parametric_function": canvas.draw_parametric_function,
            "delete_parametric_function": canvas.delete_parametric_function,
            "update_parametric_function": canvas.update_parametric_function,

            # ===== TANGENT AND NORMAL LINES =====
            "draw_tangent_line": canvas.create_tangent_line,
            "draw_normal_line": canvas.create_normal_line,

            # ===== GEOMETRIC CONSTRUCTIONS =====
            "construct_midpoint": canvas.create_midpoint,
            "construct_perpendicular_bisector": canvas.create_perpendicular_bisector,
            "construct_perpendicular_from_point": canvas.create_perpendicular_from_point,
            "construct_angle_bisector": canvas.create_angle_bisector,
            "construct_parallel_line": canvas.create_parallel_line,
            "construct_circumcircle": canvas.create_circumcircle,
            "construct_incircle": canvas.create_incircle,

            # ===== OBJECT TRANSFORMATIONS =====
            "translate_object": canvas.translate_object,
            "rotate_object": canvas.rotate_object,

            # ===== MATHEMATICAL OPERATIONS =====
            "evaluate_expression": ProcessFunctionCalls.evaluate_expression,
            "evaluate_linear_algebra_expression": ProcessFunctionCalls.evaluate_linear_algebra_expression,
            "convert": MathUtils.convert,
            "limit": MathUtils.limit,
            "derive": MathUtils.derivative,
            "integrate": MathUtils.integral,
            "numeric_integrate": MathUtils.numeric_integrate,
            "simplify": MathUtils.simplify,
            "expand": MathUtils.expand,
            "factor": MathUtils.factor,
            "solve": MathUtils.solve,
            "solve_system_of_equations": MathUtils.solve_system_of_equations,
            "solve_numeric": MathUtils.solve_numeric,

            # ===== CANVAS HISTORY =====
            "undo": canvas.undo,
            "redo": canvas.redo,

            # ===== WORKSPACE OPERATIONS =====
            "save_workspace": workspace_manager.save_workspace,
            "load_workspace": workspace_manager.load_workspace,
            "list_workspaces": workspace_manager.list_workspaces,
            "delete_workspace": workspace_manager.delete_workspace,

            # ===== COLORED AREA OPERATIONS =====
            "create_colored_area": canvas.create_colored_area,
            "create_region_colored_area": canvas.create_region_colored_area,
            "delete_colored_area": canvas.delete_colored_area,
            "update_colored_area": canvas.update_colored_area,

            # ===== GRAPH OPERATIONS =====
            "generate_graph": canvas.generate_graph,
            "delete_graph": canvas.delete_graph,
            "analyze_graph": canvas.analyze_graph,

            # ===== PLOT OPERATIONS =====
            "plot_distribution": canvas.plot_distribution,
            "plot_bars": canvas.plot_bars,
            "delete_plot": canvas.delete_plot,
            "fit_regression": canvas.fit_regression,

            # ===== ANGLE OPERATIONS =====
            "create_angle": canvas.create_angle,
            "delete_angle": canvas.delete_angle,
            "update_angle": canvas.update_angle,

            # ===== AREA CALCULATION =====
            "calculate_area": lambda expression: ProcessFunctionCalls.calculate_area(expression, canvas),

            # ===== COORDINATE SYSTEM OPERATIONS =====
            "set_coordinate_system": canvas.set_coordinate_system,
            "convert_coordinates": FunctionRegistry._convert_coordinates,
            "set_grid_visible": canvas.set_grid_visible,
        }

        # Add testing functions if ai_interface is provided
        if ai_interface is not None:
            functions["run_tests"] = ai_interface.run_tests

        # Add tool search function (makes request to backend)
        functions["search_tools"] = FunctionRegistry._create_search_tools_handler()

        return functions

    @staticmethod
    def _create_search_tools_handler() -> Callable[..., Dict[str, Any]]:
        """Create a handler for the search_tools function.

        Returns a function that makes a request to the backend /search_tools endpoint.
        """
        def search_tools(query: str, max_results: int | None = None) -> dict:
            """Search for tools matching a query description.

            Args:
                query: Description of what the user wants to accomplish.
                max_results: Maximum number of tools to return (default: 10, max: 20).

            Returns:
                Dict with matching tool definitions.
            """
            # Handle None or invalid max_results
            if max_results is None or not isinstance(max_results, int):
                max_results = 10

            # Default result structure
            default_result: dict = {"tools": [], "count": 0, "query": query, "error": None}

            # In Brython environment, use browser.ajax
            try:
                from browser import ajax, document
                import json as json_module

                # Get current AI model to use same provider for search
                ai_model = None
                try:
                    ai_model = document["ai-model-selector"].value
                except Exception:
                    pass  # Model selector not available

                # Use XMLHttpRequest directly for synchronous request (more reliable)
                req = ajax.Ajax()
                req.open("POST", "/search_tools", False)  # Synchronous
                req.set_header("Content-Type", "application/json")

                payload = {"query": query, "max_results": max_results}
                if ai_model:
                    payload["ai_model"] = ai_model

                try:
                    req.send(json_module.dumps(payload))
                except Exception as e:
                    default_result["error"] = f"Request send failed: {e}"
                    return default_result

                # After synchronous send(), we can directly access status and text
                if req.status == 200:
                    try:
                        response = json_module.loads(req.text)
                        if response.get("status") == "success" and response.get("data"):
                            return cast(Dict[str, Any], response["data"])
                        else:
                            default_result["error"] = response.get("message", "Unknown error")
                    except Exception as e:
                        default_result["error"] = f"Response parse error: {e}"
                else:
                    default_result["error"] = f"Request failed with status {req.status}"

                return default_result

            except ImportError:
                # Not in Brython environment, return placeholder
                default_result["error"] = "search_tools only available in browser environment"
                return default_result

        return search_tools

    @staticmethod
    def get_undoable_functions() -> Tuple[str, ...]:
        """Get the tuple of function names that support undo/redo operations.

        Defines which operations modify canvas state and can be reversed through
        the undo/redo system for user experience and error recovery.

        Returns:
            tuple: Function names that support undo/redo operations
        """
        return (
            # Canvas operations that modify state
            "clear_canvas",
            "reset_canvas",
            "zoom",

            # Point operations
            "create_point",
            "delete_point",
            "update_point",

            # Segment operations
            "create_segment",
            "delete_segment",
            "update_segment",

            # Vector operations
            "create_vector",
            "delete_vector",
            "update_vector",

            # Polygon operations
            "create_polygon",
            "delete_polygon",
            "update_polygon",

            # Circle operations
            "create_circle",
            "delete_circle",
            "update_circle",

            # Circle arc operations
            "create_circle_arc",
            "delete_circle_arc",
            "update_circle_arc",

            # Ellipse operations
            "create_ellipse",
            "delete_ellipse",
            "update_ellipse",

            # Label operations
            "create_label",
            "delete_label",
            "update_label",

            # Function operations
            "draw_function",
            "delete_function",
            "update_function",

            # Piecewise function operations
            "draw_piecewise_function",
            "delete_piecewise_function",
            "update_piecewise_function",

            # Parametric function operations
            "draw_parametric_function",
            "delete_parametric_function",
            "update_parametric_function",

            # Tangent and normal line operations
            "draw_tangent_line",
            "draw_normal_line",

            # Geometric construction operations
            "construct_midpoint",
            "construct_perpendicular_bisector",
            "construct_perpendicular_from_point",
            "construct_angle_bisector",
            "construct_parallel_line",
            "construct_circumcircle",
            "construct_incircle",

            # Object transformations
            "translate_object",
            "rotate_object",

            # Colored area operations
            "create_colored_area",
            "create_region_colored_area",
            "delete_colored_area",
            "update_colored_area",

            # Graph operations
            "generate_graph",
            "delete_graph",

            # Plot operations
            "plot_distribution",
            "plot_bars",
            "delete_plot",
            # Note: fit_regression is NOT undoable - it returns stats to the AI

            # Angle operations
            "create_angle",
            "delete_angle",
            "update_angle"
        )
