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

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

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
            "get_current_canvas_state": lambda: {"type": "canvas_state", "value": canvas.get_canvas_state()},
            
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
            "simplify": MathUtils.simplify,
            "expand": MathUtils.expand,
            "factor": MathUtils.factor,
            "solve": MathUtils.solve,
            "solve_system_of_equations": MathUtils.solve_system_of_equations,
            
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

            # ===== ANGLE OPERATIONS =====
            "create_angle": canvas.create_angle,
            "delete_angle": canvas.delete_angle,
            "update_angle": canvas.update_angle,
            
            # ===== AREA CALCULATION =====
            "calculate_area": lambda expression: ProcessFunctionCalls.calculate_area(expression, canvas),
        }

        # Add testing functions if ai_interface is provided
        if ai_interface is not None:
            functions["run_tests"] = ai_interface.run_tests
            
        return functions

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

            # Angle operations
            "create_angle",
            "delete_angle",
            "update_angle"
        ) 