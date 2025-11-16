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

from typing import TYPE_CHECKING, Dict, List, Optional

from drawables.function import Function
from expression_validator import ExpressionValidator
from managers.edit_policy import DrawableEditPolicy, EditRule, get_drawable_edit_policy

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
        
    def draw_function(self, function_string: str, name: str, left_bound: Optional[float] = None, right_bound: Optional[float] = None) -> Function:
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
            
        Returns:
            Function: The newly created or updated function object
            
        Raises:
            ValueError: If the function string cannot be parsed or is invalid
        """
        # Archive before creation or modification
        self.canvas.undo_redo_manager.archive()
        
        # Check if the function already exists
        existing_function = self.get_function(name)
        if existing_function:
            # If it exists, update its expression
            try:
                existing_function.function_string = ExpressionValidator.fix_math_expression(function_string)
                existing_function.function = ExpressionValidator.parse_function_string(function_string, use_mathjs=False)
            except Exception as e:
                raise ValueError(f"Failed to parse function string '{function_string}': {str(e)}")
            # Update the bounds
            existing_function.left_bound = left_bound
            existing_function.right_bound = right_bound
            
            if self.canvas.draw_enabled:
                self.canvas.draw()
            return existing_function
        else:
            # Generate a proper name if needed
            name = self.name_generator.generate_function_name(name)
                
            # Create the function (math-only)
            new_function = Function(function_string, name=name, left_bound=left_bound, right_bound=right_bound)
            
            # Add to drawables
            self.drawables.add(new_function)
            
            # Draw the function
            if self.canvas.draw_enabled:
                self.canvas.draw()
                
            return new_function
        
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
        self.drawables.remove(function)
            
        # Also delete any colored areas associated with this function
        self.canvas.drawable_manager.delete_colored_areas_for_function(function)
        
        # Redraw the canvas
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return True 

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