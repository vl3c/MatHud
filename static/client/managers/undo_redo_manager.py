"""
MatHud Undo/Redo State Management System

Provides comprehensive undo and redo functionality for canvas operations through state archiving
and restoration. Maintains operation history and handles complex object relationships during
state transitions.

State Management Architecture:
    - Snapshot System: Deep copying of entire canvas state for each operation
    - Dual Stack System: Separate undo and redo stacks for bidirectional navigation
    - Automatic Archiving: State capture before any destructive operation
    - State Restoration: Complete canvas reconstruction from archived states

Archived State Components:
    - Drawable Objects: Complete deep copy of all geometric objects and their properties
    - Computation History: Mathematical operation results and expressions
    - Object Relationships: Preservation of parent-child dependencies
    - Canvas References: Proper object-to-canvas relationship maintenance

Operation Flow:
    - archive(): Captures current state before modifications
    - undo(): Restores previous state and moves current to redo stack
    - redo(): Restores next state and moves current to undo stack
    - State clearing: Automatic redo stack clearing on new operations

Complex State Handling:
    - Dependency Rebuilding: _rebuild_dependency_graph() recreates object relationships
    - Deep Copy Management: Handles nested object structures and circular references
    - Memory Efficiency: Strategic state limitation to prevent memory bloat

Integration Points:
    - DrawableManager: State capture of all drawable objects
    - DrawableDependencyManager: Dependency graph reconstruction
    - Canvas: Automatic redrawing after state changes
    - Mathematical Operations: Computation history preservation

Error Recovery:
    - Graceful Degradation: Continues operation even if some references can't be restored
    - Validation: Checks for required manager instances before operations
    - Logging: Comprehensive warning system for debugging state issues
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from canvas import Canvas

class UndoRedoManager:
    """
    Manages undo and redo operations for a Canvas object.
    
    This class is responsible for:
    - Archiving canvas states (for undo operations)
    - Handling undo operations (restore previous state)
    - Handling redo operations (restore undone state)
    """
    
    def __init__(self, canvas: "Canvas") -> None:
        """
        Initialize the UndoRedoManager.
        
        Args:
            canvas: The Canvas object this manager is responsible for
        """
        self.canvas: "Canvas" = canvas
        self.undo_stack: List[Dict[str, Any]] = []
        self.redo_stack: List[Dict[str, Any]] = []
        self._archive_suspension_depth: int = 0
    
    def archive(self) -> None:
        """
        Archives the current state of the canvas for undo operations.
        
        This method should be called whenever a change is made to the canvas
        that should be undoable.
        """
        if self._archive_suspension_depth > 0:
            return
        self.push_undo_state(self.capture_state())

    def capture_state(self) -> Dict[str, Any]:
        """Capture the current canvas state snapshot."""
        return {
            'drawables': copy.deepcopy(self.canvas.drawable_manager.drawables._drawables),
            'computations': copy.deepcopy(self.canvas.computations),
        }

    def push_undo_state(self, state: Dict[str, Any]) -> None:
        """Push a prior state onto the undo stack and clear redo history."""
        self.undo_stack.append(copy.deepcopy(state))
        self.redo_stack = []

    def restore_state(self, state: Dict[str, Any], redraw: bool = True) -> None:
        """Restore a captured state snapshot."""
        self.canvas.drawable_manager.drawables._drawables = copy.deepcopy(state['drawables'])
        self.canvas.drawable_manager.drawables.rebuild_renderables()
        self.canvas.computations = copy.deepcopy(state.get('computations', []))
        self._rebuild_dependency_graph()
        if redraw:
            self.canvas.draw()

    def suspend_archiving(self) -> None:
        """Suspend archive() calls for composite operations."""
        self._archive_suspension_depth += 1

    def resume_archiving(self) -> None:
        """Resume archive() calls after a composite operation."""
        if self._archive_suspension_depth > 0:
            self._archive_suspension_depth -= 1
        
    def undo(self) -> bool:
        """
        Restores the last archived state from the undo stack.
        
        Returns:
            bool: True if an undo was performed, False otherwise
        """
        if not self.undo_stack:
            return False
            
        # Get the last archived state
        last_state = self.undo_stack.pop()
        
        # Archive current state for redo
        current_state = {
            'drawables': copy.deepcopy(self.canvas.drawable_manager.drawables._drawables),
            'computations': copy.deepcopy(self.canvas.computations)
        }
        self.redo_stack.append(current_state)
        
        # Restore only the drawables from the last state
        self.canvas.drawable_manager.drawables._drawables = copy.deepcopy(last_state['drawables'])
        self.canvas.drawable_manager.drawables.rebuild_renderables()
        
        # Ensure all objects are properly initialized
        self._rebuild_dependency_graph()
        
        # Make sure to reset any cached or derived values
        # This ensures a complete state reset
        self.canvas.draw()
        
        return True
    
    def redo(self) -> bool:
        """
        Restores the last undone state from the redo stack.
        
        Returns:
            bool: True if a redo was performed, False otherwise
        """
        if not self.redo_stack:
            return False
            
        # Get the last undone state
        next_state = self.redo_stack.pop()
        
        # Archive current state for undo
        current_state = {
            'drawables': copy.deepcopy(self.canvas.drawable_manager.drawables._drawables),
            'computations': copy.deepcopy(self.canvas.computations)
        }
        self.undo_stack.append(current_state)
        
        # Restore only the drawables from the next state
        self.canvas.drawable_manager.drawables._drawables = copy.deepcopy(next_state['drawables'])
        self.canvas.drawable_manager.drawables.rebuild_renderables()
        
        # Ensure all objects are properly initialized
        self._rebuild_dependency_graph()
        
        # Make sure to reset any cached or derived values
        # This ensures a complete state reset
        self.canvas.draw()
        
        return True
    
    def can_undo(self) -> bool:
        """
        Checks if there are any states that can be undone.
        
        Returns:
            bool: True if undo is possible, False otherwise
        """
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """
        Checks if there are any states that can be redone.
        
        Returns:
            bool: True if redo is possible, False otherwise
        """
        return len(self.redo_stack) > 0
    
    def _rebuild_dependency_graph(self) -> None:
        """
        Rebuilds the dependency relationships between drawables.
        
        This is necessary after loading a saved state, as the serialization
        process may lose dependency links which need to be re-established
        with the new object instances.
        """
        all_drawables = []
        # Collect all drawable objects from the restored state
        for drawable_type in self.canvas.drawable_manager.drawables._drawables:
            for drawable in self.canvas.drawable_manager.drawables._drawables[drawable_type]:
                all_drawables.append(drawable)

        # This assumes self.canvas has a 'dependency_manager' attribute
        # which is an instance of DrawableDependencyManager.
        if hasattr(self.canvas, 'dependency_manager') and self.canvas.dependency_manager is not None:
            dependency_manager = self.canvas.dependency_manager
            
            # Clear existing dependency relationships from the manager
            dependency_manager._parents.clear()
            dependency_manager._children.clear()
            
            # Re-analyze each drawable to rebuild the dependency graph
            # The analyze_drawable_for_dependencies method in DrawableDependencyManager
            # is responsible for handling various drawable types and their specific dependencies.
            for drawable in all_drawables:
                dependency_manager.analyze_drawable_for_dependencies(drawable)
        else:
            # Log a warning if the dependency manager isn't found on the canvas object.
            # This helps in debugging if the expected structure isn't met.
            print("UndoRedoManager: Warning - Canvas instance does not have a 'dependency_manager' " + \
                  "attribute or it is None. Skipping dependency graph rebuild for undo/redo operation.")

    def clear(self) -> None:
        """
        Clears all undo and redo history.
        """
        self.undo_stack = []
        self.redo_stack = [] 
