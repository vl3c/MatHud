"""
MatHud Drawable Dependency Management System

Maintains hierarchical relationships between drawable objects to preserve geometric integrity
and enable intelligent cascading operations. Tracks parent-child dependencies and manages
the propagation of changes through the dependency graph.

Dependency Architecture:
    - Hierarchical Relationships: Parent-child tracking between geometric objects
    - Type-Based Hierarchy: Points → Segments → Triangles/Rectangles → Complex Objects
    - Bidirectional Mapping: Efficient lookup of both parents and children
    - Transitive Closure: Recursive traversal of entire dependency chains

Core Dependency Rules:
    - Segments depend on their endpoint Points
    - Vectors depend on their origin and tip Points  
    - Triangles depend on their three Segments and six Points
    - Rectangles depend on their four Segments and four Points
    - Circles/Ellipses depend on their center Points
    - ColoredAreas depend on their boundary Functions and/or Segments
    - Angles depend on their vertex Point and two arm Segments

Change Propagation:
    - Canvas Reference Updates: Ensures all objects maintain proper canvas references
    - Dependency Analysis: Automatic detection of relationships during object creation
    - Cascading Operations: Moving/deleting parents affects all children
    - Integrity Maintenance: Prevents orphaned objects and broken references

Graph Operations:
    - Dependency Registration: register_dependency(child, parent)
    - Relationship Queries: get_parents(), get_children(), get_all_parents(), get_all_children()
    - Graph Cleanup: remove_drawable() removes all references
    - Topological Sorting: resolve_dependency_order() for proper operation sequencing

State Management:
    - Deep Copying Support: Handles state serialization for undo/redo
    - Reference Restoration: Rebuilds dependency graph after state restoration
    - Validation: Ensures all objects have required methods (get_class_name)
    - Error Recovery: Graceful handling of missing or invalid dependencies

Mathematical Integration:
    - Coordinate Matching: Uses MathUtils for floating-point coordinate comparisons
    - Geometric Validation: Ensures relationships match geometric reality
    - Tolerance Handling: Robust matching with mathematical precision considerations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from utils.math_utils import MathUtils

if TYPE_CHECKING:
    from drawables.drawable import Drawable
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from canvas import Canvas

class DrawableDependencyManager:
    """
    Manages dependencies between drawable objects to maintain hierarchical structure.
    
    This class:
    - Tracks parent-child relationships between drawables
    - Resolves dependency chains
    - Handles propagation of changes (like canvas references)
    """
    
    def __init__(self, drawable_manager_proxy: Optional["DrawableManagerProxy"] = None) -> None:
        """Initialize the dependency manager"""
        self.drawable_manager: Optional["DrawableManagerProxy"] = drawable_manager_proxy # Store the proxy
        # Re-add internal state maps needed by other methods
        self._parents: Dict[int, Set[int]] = {}
        self._children: Dict[int, Set[int]] = {}
        self._object_lookup: Dict[int, "Drawable"] = {}
        # Type hierarchy - which types depend on which other types
        self._type_hierarchy: Dict[str, List[str]] = {
            'Point': [],
            'Segment': ['Point'],
            'Vector': ['Point'],
            'Triangle': ['Segment', 'Point'],
            'Rectangle': ['Segment', 'Point'],
            'Circle': ['Point'],
            'Ellipse': ['Point'],
            'Angle': ['Segment', 'Point'],
            'CircleArc': ['Point', 'Circle'],
            'Function': [],
            'ColoredArea': ['Function', 'Segment'],
            'SegmentsBoundedColoredArea': ['Segment'],
            'FunctionSegmentBoundedColoredArea': ['Function', 'Segment'],
            'FunctionsBoundedColoredArea': ['Function'],
            'ClosedShapeColoredArea': ['Segment', 'Circle', 'Ellipse'],
            'Graph': ['Segment', 'Vector', 'Point'],
            'DirectedGraph': ['Vector', 'Point'],
            'UndirectedGraph': ['Segment', 'Point'],
            'Tree': ['Segment', 'Point'],
        }
    
    def _should_skip_point_point_dependency(self, child: "Drawable", parent: "Drawable") -> bool:
        """Check if a dependency registration should be skipped (e.g., Point as child of Point)."""
        is_child_point = hasattr(child, 'get_class_name') and child.get_class_name() == 'Point'
        is_parent_point = hasattr(parent, 'get_class_name') and parent.get_class_name() == 'Point'
        return is_child_point and is_parent_point

    def register_dependency(self, child: "Drawable", parent: "Drawable") -> None:
        """
        Register a child-parent dependency
        
        Args:
            child: The child drawable that depends on the parent
            parent: The parent drawable that child depends on
        """
        # Prevent Point from being registered as a child of another Point using the helper method
        if self._should_skip_point_point_dependency(child, parent):
            return
        
        # Verify objects have get_class_name
        self._verify_get_class_name_method(child, "Child")
        self._verify_get_class_name_method(parent, "Parent")
        
        child_id = id(child)
        parent_id = id(parent)

        self._object_lookup[child_id] = child
        self._object_lookup[parent_id] = parent

        if child_id not in self._parents:
            self._parents[child_id] = set()
        if parent_id not in self._children:
            self._children[parent_id] = set()

        self._parents[child_id].add(parent_id)
        self._children[parent_id].add(child_id)
    
    def unregister_dependency(self, child: Optional["Drawable"], parent: Optional["Drawable"]) -> None:
        """
        Unregister a specific child-parent dependency.

        Args:
            child: The child drawable.
            parent: The parent drawable.
        """
        if child is None or parent is None:
            # print("Warning: Trying to unregister dependency with None drawable(s).") # Retaining print for now, can be removed if too verbose
            return

        child_id = id(child)
        parent_id = id(parent)

        if child_id in self._parents:
            self._parents[child_id].discard(parent_id)
        
        if parent_id in self._children:
            self._children[parent_id].discard(child_id)

    def _verify_get_class_name_method(self, obj: Any, obj_type_name: str) -> None:
        """
        Verify that an object has the get_class_name method
        
        Args:
            obj: The object to verify
            obj_type_name: A string indicating the type of object (e.g., "Child", "Parent")
        """
        if not hasattr(obj, 'get_class_name'):
            print(f"WARNING: {obj_type_name} {obj} is missing get_class_name method")
            # If missing, let's make sure we can still identify the object
            print(f"{obj_type_name} object type: {type(obj)}")
    
    def get_parents(self, drawable: Optional["Drawable"]) -> Set["Drawable"]:
        """
        Get all direct parents of a drawable
        
        Args:
            drawable: The drawable to find parents for
            
        Returns:
            set: Set of parent drawables
        """
        if drawable is None:
            print("Warning: Trying to get parents for None drawable")
            return set()
        
        drawable_id = id(drawable)
        return {self._object_lookup[parent_id] for parent_id in self._parents.get(drawable_id, set()) if parent_id in self._object_lookup}
    
    def get_children(self, drawable: Optional["Drawable"]) -> Set["Drawable"]:
        """
        Get all direct children of a drawable
        
        Args:
            drawable: The drawable to find children for
            
        Returns:
            set: Set of child drawables
        """
        if drawable is None:
            print("Warning: Trying to get children for None drawable")
            return set()
        
        drawable_id = id(drawable)
        return {self._object_lookup[child_id] for child_id in self._children.get(drawable_id, set()) if child_id in self._object_lookup}
    
    def get_all_parents(self, drawable: Optional["Drawable"]) -> Set["Drawable"]:
        """
        Get all parents recursively (transitive closure)
        
        Args:
            drawable: The drawable to find all parents for
            
        Returns:
            set: Set of all parent drawables
        """
        if drawable is None:
            print("Warning: Trying to get parents for None drawable")
            return set()
        
        all_parents = set()
        # Operate on a copy to avoid modifying the original set
        to_process = self.get_parents(drawable).copy() 
        
        while to_process:
            parent = to_process.pop()
            if parent is None:
                print("Warning: Found None parent in dependency tree")
                continue
            
            if parent not in all_parents:
                all_parents.add(parent)
                to_process.update(self.get_parents(parent))
            
        return all_parents
    
    def get_all_children(self, drawable: Optional["Drawable"]) -> Set["Drawable"]:
        """
        Get all children recursively (transitive closure)
        
        Args:
            drawable: The drawable to find all children for
            
        Returns:
            set: Set of all child drawables
        """
        if drawable is None:
            print("Warning: Trying to get children for None drawable")
            return set()
        
        all_children = set()
        to_process = self.get_children(drawable)
        
        while to_process:
            child = to_process.pop()
            if child is None:
                print("Warning: Found None child in dependency tree")
                continue
            
            if child not in all_children:
                all_children.add(child)
                to_process.update(self.get_children(child))
            
        return all_children
    
    def remove_drawable(self, drawable: "Drawable") -> None:
        """
        Remove a drawable from the dependency graph
        
        Args:
            drawable: The drawable to remove
        """
        drawable_id = id(drawable)
        drawable_class = drawable.get_class_name() if hasattr(drawable, 'get_class_name') else ""

        # Notify children (e.g., graphs) to remove references to this drawable
        for child_id in self._children.get(drawable_id, set()).copy():
            child = self._object_lookup.get(child_id)
            if child:
                self._notify_child_of_parent_removal(child, drawable, drawable_class)
            parents = self._parents.get(child_id)
            if parents and drawable_id in parents:
                parents.discard(drawable_id)
                
        # Remove from parents' children
        for parent_id in self._parents.get(drawable_id, set()).copy():
            children = self._children.get(parent_id)
            if children and drawable_id in children:
                children.discard(drawable_id)
                
        # Remove drawable's entries
        if drawable_id in self._parents:
            del self._parents[drawable_id]
        if drawable_id in self._children:
            del self._children[drawable_id]
        if drawable_id in self._object_lookup:
            del self._object_lookup[drawable_id]

    def _notify_child_of_parent_removal(self, child: "Drawable", parent: "Drawable", parent_class: str) -> None:
        """Notify a child drawable that one of its parents has been removed."""
        child_class = child.get_class_name() if hasattr(child, 'get_class_name') else ""
        
        # Handle graph types - remove the reference from internal lists
        if child_class in ('Graph', 'DirectedGraph', 'UndirectedGraph', 'Tree'):
            if parent_class == 'Segment' and hasattr(child, 'remove_segment'):
                child.remove_segment(parent)
            elif parent_class == 'Vector' and hasattr(child, 'remove_vector'):
                child.remove_vector(parent)
            elif parent_class == 'Point' and hasattr(child, 'remove_point'):
                child.remove_point(parent)
    
    
    def analyze_drawable_for_dependencies(self, drawable: "Drawable") -> List["Drawable"]:
        """
        Analyze a drawable to find and register its dependencies
        
        Args:
            drawable: The drawable to analyze
            
        Returns:
            list: List of identified dependencies
        """
        dependencies: List["Drawable"] = []
        
        # Verify drawable has get_class_name method
        self._verify_get_class_name_method(drawable, "Drawable")
        
        # Get class name safely
        if not hasattr(drawable, 'get_class_name'):
            print(f"Cannot analyze dependencies for {drawable} without get_class_name method")
            return dependencies
            
        class_name = drawable.get_class_name()
        
        # Handle different drawable types
        if class_name == 'Point':
            # Points don't have dependencies
            pass
            
        elif class_name == 'Segment':
            if hasattr(drawable, 'point1'):
                dependencies.append(drawable.point1)
                self.register_dependency(drawable, drawable.point1)
            if hasattr(drawable, 'point2'):
                dependencies.append(drawable.point2)
                self.register_dependency(drawable, drawable.point2)
                
        elif class_name == 'Vector':
            if hasattr(drawable, 'segment'):
                dependencies.append(drawable.segment)
                self.register_dependency(drawable, drawable.segment)
                
        elif class_name == 'Triangle':
            # Check for individual segment attributes
            for i in range(1, 4):
                segment_attr = f'segment{i}'
                if hasattr(drawable, segment_attr):
                    segment = getattr(drawable, segment_attr)
                    dependencies.append(segment)
                    self.register_dependency(drawable, segment)
                    
        elif class_name == 'Rectangle':
            # Check for individual segment attributes
            for i in range(1, 5):
                segment_attr = f'segment{i}'
                if hasattr(drawable, segment_attr):
                    segment = getattr(drawable, segment_attr)
                    dependencies.append(segment)
                    self.register_dependency(drawable, segment)
                    
        elif class_name == 'Circle':
            if hasattr(drawable, 'center'):
                dependencies.append(drawable.center)
                self.register_dependency(drawable, drawable.center)
                
        elif class_name == 'Ellipse':
            if hasattr(drawable, 'center'):
                dependencies.append(drawable.center)
                self.register_dependency(drawable, drawable.center)
                
        elif class_name == 'Function':
            # Functions typically don't have drawable dependencies
            pass
            
        elif class_name == 'SegmentsBoundedColoredArea':
            if hasattr(drawable, 'segment1') and drawable.segment1:
                dependencies.append(drawable.segment1)
                self.register_dependency(drawable, drawable.segment1)
            if hasattr(drawable, 'segment2') and drawable.segment2:
                dependencies.append(drawable.segment2)
                self.register_dependency(drawable, drawable.segment2)
                
        elif class_name == 'FunctionSegmentBoundedColoredArea':
            if hasattr(drawable, 'func') and drawable.func and hasattr(drawable.func, 'get_class_name'):
                dependencies.append(drawable.func)
                self.register_dependency(drawable, drawable.func)
            if hasattr(drawable, 'segment'):
                dependencies.append(drawable.segment)
                self.register_dependency(drawable, drawable.segment)
                
        elif class_name == 'FunctionsBoundedColoredArea':
            if hasattr(drawable, 'func1') and drawable.func1 and hasattr(drawable.func1, 'get_class_name'):
                dependencies.append(drawable.func1)
                self.register_dependency(drawable, drawable.func1)
            if hasattr(drawable, 'func2') and drawable.func2 and hasattr(drawable.func2, 'get_class_name'):
                dependencies.append(drawable.func2)
                self.register_dependency(drawable, drawable.func2)
                
        elif class_name == 'Angle':
            # Angles depend on their constituent segments and points
            if hasattr(drawable, 'segment1') and drawable.segment1:
                dependencies.append(drawable.segment1)
                self.register_dependency(drawable, drawable.segment1)
            if hasattr(drawable, 'segment2') and drawable.segment2:
                dependencies.append(drawable.segment2)
                self.register_dependency(drawable, drawable.segment2)
            if hasattr(drawable, 'vertex_point') and drawable.vertex_point:
                dependencies.append(drawable.vertex_point)
                self.register_dependency(drawable, drawable.vertex_point)
            if hasattr(drawable, 'arm1_point') and drawable.arm1_point:
                dependencies.append(drawable.arm1_point)
                self.register_dependency(drawable, drawable.arm1_point)
            if hasattr(drawable, 'arm2_point') and drawable.arm2_point:
                dependencies.append(drawable.arm2_point)
                self.register_dependency(drawable, drawable.arm2_point)

        elif class_name == 'CircleArc':
            if hasattr(drawable, 'point1') and drawable.point1:
                dependencies.append(drawable.point1)
                self.register_dependency(drawable, drawable.point1)
            if hasattr(drawable, 'point2') and drawable.point2:
                dependencies.append(drawable.point2)
                self.register_dependency(drawable, drawable.point2)
            if hasattr(drawable, 'circle') and drawable.circle:
                dependencies.append(drawable.circle)
                self.register_dependency(drawable, drawable.circle)

        elif class_name == 'ClosedShapeColoredArea':
            if hasattr(drawable, 'segments'):
                for segment in drawable.segments:
                    if segment:
                        dependencies.append(segment)
                        self.register_dependency(drawable, segment)
            if hasattr(drawable, 'circle') and drawable.circle:
                dependencies.append(drawable.circle)
                self.register_dependency(drawable, drawable.circle)
            if hasattr(drawable, 'ellipse') and drawable.ellipse:
                dependencies.append(drawable.ellipse)
                self.register_dependency(drawable, drawable.ellipse)
            if hasattr(drawable, 'chord_segment') and drawable.chord_segment:
                dependencies.append(drawable.chord_segment)
                self.register_dependency(drawable, drawable.chord_segment)
                
        elif class_name == 'ColoredArea':
            # Base ColoredArea type
            if hasattr(drawable, 'function'):
                dependencies.append(drawable.function)
                self.register_dependency(drawable, drawable.function)
            if hasattr(drawable, 'segments'):
                for segment in drawable.segments:
                    dependencies.append(segment)
                    self.register_dependency(drawable, segment)
                    
        elif class_name.endswith('ColoredArea'):
            # Generic case for other ColoredArea types
            if hasattr(drawable, 'function'):
                dependencies.append(drawable.function)
                self.register_dependency(drawable, drawable.function)
            if hasattr(drawable, 'segments'):
                for segment in drawable.segments:
                    dependencies.append(segment)
                    self.register_dependency(drawable, segment)

        elif class_name in ('Graph', 'DirectedGraph', 'UndirectedGraph', 'Tree'):
            # Graphs depend on their segments, vectors, and isolated points
            if hasattr(drawable, '_segments'):
                for segment in drawable._segments:
                    if segment:
                        dependencies.append(segment)
                        self.register_dependency(drawable, segment)
            if hasattr(drawable, '_vectors'):
                for vector in drawable._vectors:
                    if vector:
                        dependencies.append(vector)
                        self.register_dependency(drawable, vector)
            if hasattr(drawable, '_isolated_points'):
                for point in drawable._isolated_points:
                    if point:
                        dependencies.append(point)
                        self.register_dependency(drawable, point)
            
        return dependencies
    
    def _find_segment_children(self, segment: Optional["Drawable"]) -> List["Drawable"]:
        """Finds children geometrically by iterating through all segments."""
        # Safety check for segment and its points
        if not segment or not hasattr(segment, 'point1') or not hasattr(segment, 'point2'):
            return []
            
        # Safety check for points
        if not segment.point1 or not segment.point2:
            return []
            
        sp1x, sp1y = segment.point1.x, segment.point1.y
        sp2x, sp2y = segment.point2.x, segment.point2.y
        children: List["Drawable"] = []
        
        # Access segments via the proxy
        if self.drawable_manager and self.drawable_manager.drawables:
            all_segments = self.drawable_manager.drawables.Segments
            for s in all_segments:
                if s == segment:
                    continue
                if not hasattr(s, 'point1') or not hasattr(s, 'point2'): # Safety check
                    continue 
                if not s.point1 or not s.point2:
                    continue
                
                p1x, p1y = s.point1.x, s.point1.y
                p2x, p2y = s.point2.x, s.point2.y
                # Check if s is geometrically within segment
                if MathUtils.is_point_on_segment(p1x, p1y, sp1x, sp1y, sp2x, sp2y) and \
                   MathUtils.is_point_on_segment(p2x, p2y, sp1x, sp1y, sp2x, sp2y):
                    children.append(s)
        return children

    def resolve_dependency_order(self, drawables: List["Drawable"]) -> List["Drawable"]:
        """
        Determine the correct order to process drawables based on dependencies
        
        Args:
            drawables: List of drawables to process
            
        Returns:
            list: Ordered list of drawables (parents before children)
        """
        # Filter out None values
        filtered_drawables: List["Drawable"] = [d for d in drawables if d is not None]
        
        # Simple topological sort
        result: List["Drawable"] = []
        visited: Set["Drawable"] = set()
        
        def visit(drawable: "Drawable") -> None:
            if drawable in visited:
                return
                
            visited.add(drawable)
            
            # Visit all parents first
            for parent in self.get_parents(drawable):
                visit(parent)
                
            result.append(drawable)
            
        # Process all drawables
        for drawable in filtered_drawables:
            visit(drawable)
            
        return result 