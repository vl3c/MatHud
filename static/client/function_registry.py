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

from utils.math_utils import MathUtils
from process_function_calls import ProcessFunctionCalls


class FunctionRegistry:
    """Static registry mapping function names to implementations and undo capabilities.
    
    Maintains the central mapping between AI function names and their Python implementations,
    and defines which functions support undo/redo operations for state management.
    """

    @staticmethod
    def get_available_functions(canvas, workspace_manager, ai_interface=None):
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
        functions = {
            # ===== CANVAS OPERATIONS =====
            "reset_canvas": canvas.reset,
            "clear_canvas": canvas.clear,
            "zoom_to_bounds": canvas.zoom_to_bounds,
            
            # ===== POINT OPERATIONS =====
            "create_point": canvas.create_point,
            "delete_point": canvas.delete_point,
            
            # ===== SEGMENT OPERATIONS =====
            "create_segment": canvas.create_segment,
            "delete_segment": canvas.delete_segment,
            
            # ===== VECTOR OPERATIONS =====
            "create_vector": canvas.create_vector,
            "delete_vector": canvas.delete_vector,
            
            # ===== TRIANGLE OPERATIONS =====
            "create_triangle": canvas.create_triangle,
            "delete_triangle": canvas.delete_triangle,
            
            # ===== RECTANGLE OPERATIONS =====
            "create_rectangle": canvas.create_rectangle,
            "delete_rectangle": canvas.delete_rectangle,
            
            # ===== CIRCLE OPERATIONS =====
            "create_circle": canvas.create_circle,
            "delete_circle": canvas.delete_circle,
            
            # ===== ELLIPSE OPERATIONS =====
            "create_ellipse": canvas.create_ellipse,
            "delete_ellipse": canvas.delete_ellipse,
            
            # ===== FUNCTION PLOTTING =====
            "draw_function": canvas.draw_function,
            "delete_function": canvas.delete_function,
            
            # ===== OBJECT TRANSFORMATIONS =====
            "translate_object": canvas.translate_object,
            "rotate_object": canvas.rotate_object,
            
            # ===== MATHEMATICAL OPERATIONS =====
            "evaluate_expression": ProcessFunctionCalls.evaluate_expression,
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
            "delete_colored_area": canvas.delete_colored_area,
            
            # ===== ANGLE OPERATIONS =====
            "create_angle": canvas.create_angle,
            "delete_angle": canvas.delete_angle,
            "update_angle_properties": canvas.update_angle_properties,
        }

        # Add testing functions if ai_interface is provided
        if ai_interface is not None:
            functions["run_tests"] = ai_interface.run_tests
            
        return functions

    @staticmethod
    def get_undoable_functions():
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
            "zoom_to_bounds",
            
            # Point operations
            "create_point",
            "delete_point",
            
            # Segment operations
            "create_segment",
            "delete_segment",
            
            # Vector operations
            "create_vector", 
            "delete_vector",
            
            # Triangle operations
            "create_triangle",
            "delete_triangle",
            
            # Rectangle operations
            "create_rectangle",
            "delete_rectangle",
            
            # Circle operations
            "create_circle",
            "delete_circle",
            
            # Ellipse operations
            "create_ellipse",
            "delete_ellipse",
            
            # Function operations
            "draw_function",
            "delete_function",
            
            # Object transformations
            "translate_object",
            "rotate_object",
            
            # Colored area operations
            "create_colored_area",
            "delete_colored_area",
            
            # Angle operations
            "create_angle",
            "delete_angle",
            "update_angle_properties"
        ) 