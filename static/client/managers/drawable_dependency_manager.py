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

from utils.math_utils import MathUtils

class DrawableDependencyManager:
    """
    Manages dependencies between drawable objects to maintain hierarchical structure.
    
    This class:
    - Tracks parent-child relationships between drawables
    - Resolves dependency chains
    - Handles propagation of changes (like canvas references)
    """
    
    def __init__(self, drawable_manager_proxy=None):
        """Initialize the dependency manager"""
        self.drawable_manager = drawable_manager_proxy # Store the proxy
        # Re-add internal state maps needed by other methods
        self._parents = {}
        self._children = {}
        # Type hierarchy - which types depend on which other types
        self._type_hierarchy = {
            'Point': [],
            'Segment': ['Point'],
            'Vector': ['Point'],
            'Triangle': ['Segment', 'Point'],
            'Rectangle': ['Segment', 'Point'],
            'Circle': ['Point'],
            'Ellipse': ['Point'],
            'Angle': ['Segment', 'Point'],
            'Function': [],
            'ColoredArea': ['Function', 'Segment'],
            'SegmentsBoundedColoredArea': ['Segment'],
            'FunctionSegmentBoundedColoredArea': ['Function', 'Segment'],
            'FunctionsBoundedColoredArea': ['Function']
        }
    
    def _should_skip_point_point_dependency(self, child, parent):
        """Check if a dependency registration should be skipped (e.g., Point as child of Point)."""
        is_child_point = hasattr(child, 'get_class_name') and child.get_class_name() == 'Point'
        is_parent_point = hasattr(parent, 'get_class_name') and parent.get_class_name() == 'Point'
        return is_child_point and is_parent_point

    def register_dependency(self, child, parent):
        """
        Register a child-parent dependency
        
        Args:
            child: The child drawable that depends on the parent
            parent: The parent drawable that child depends on
        """
        # Get names for logging, fall back to str(obj) if no name attribute
        child_name_for_log = child.name if hasattr(child, 'name') and child.name else str(child)
        parent_name_for_log = parent.name if hasattr(parent, 'name') and parent.name else str(parent)

        # Prevent Point from being registered as a child of another Point using the helper method
        if self._should_skip_point_point_dependency(child, parent):
            print(f"### DDManager: SKIPPING Point-as-child-of-Point registration: Child Point '{child_name_for_log}' with Parent Point '{parent_name_for_log}'.")
            return # Exit early
        
        # Verify objects have get_class_name
        self._verify_get_class_name_method(child, "Child")
        self._verify_get_class_name_method(parent, "Parent")
        
        # Ensure the objects are in the maps
        if child not in self._parents:
            self._parents[child] = set()
        if parent not in self._children:
            self._children[parent] = set()
            
        # Add the relationship
        self._parents[child].add(parent)
        self._children[parent].add(child)
    
    def unregister_dependency(self, child, parent):
        """
        Unregister a specific child-parent dependency.

        Args:
            child: The child drawable.
            parent: The parent drawable.
        """
        if child is None or parent is None:
            # print("Warning: Trying to unregister dependency with None drawable(s).") # Retaining print for now, can be removed if too verbose
            return

        if child in self._parents:
            self._parents[child].discard(parent)
        
        if parent in self._children:
            self._children[parent].discard(child)

    def _verify_get_class_name_method(self, obj, obj_type_name):
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
    
    def get_parents(self, drawable):
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
        
        return self._parents.get(drawable, set())
    
    def get_children(self, drawable):
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
        
        return self._children.get(drawable, set())
    
    def get_all_parents(self, drawable):
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
    
    def get_all_children(self, drawable):
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
    
    def remove_drawable(self, drawable):
        """
        Remove a drawable from the dependency graph
        
        Args:
            drawable: The drawable to remove
        """
        # Remove from children's parents
        for child in self.get_children(drawable):
            if drawable in self._parents.get(child, set()):
                self._parents[child].remove(drawable)
                
        # Remove from parents' children
        for parent in self.get_parents(drawable):
            if drawable in self._children.get(parent, set()):
                self._children[parent].remove(drawable)
                
        # Remove drawable's entries
        if drawable in self._parents:
            del self._parents[drawable]
        if drawable in self._children:
            del self._children[drawable]
    
    def update_canvas_references(self, drawable, canvas):
        """
        Update canvas references for a drawable and its dependencies
        
        Args:
            drawable: The drawable to update
            canvas: The canvas reference to set
        """
        if drawable is None:
            print("Warning: Trying to update canvas references for None drawable")
            return
            
        print(f"Starting canvas update for {drawable}")
        
        # Track all visited objects to avoid cycles
        visited = set()
        
        # Use a queue for breadth-first traversal
        queue = [drawable]
        
        # Breadth-first traversal to update all connected objects
        while queue:
            current = queue.pop(0)
            
            # Skip if already visited or None
            if current in visited or current is None:
                continue
                
            visited.add(current)
            
            # Update the current object's canvas
            if hasattr(current, 'canvas'):
                current.canvas = canvas
                print(f"Updated {current}.canvas = {canvas}")
            
            # Add children (objects that depend on this one)
            children = self.get_children(current)
            for child in children:
                if child is not None and child not in visited:
                    queue.append(child)
                    
            # Add parents (objects this one depends on)
            parents = self.get_parents(current)
            for parent in parents:
                if parent is not None and parent not in visited:
                    queue.append(parent)
                    
            # Special handling for segments (they already have the points as parents,
            # but this ensures we handle the direct references too)
            if hasattr(current, 'get_class_name') and current.get_class_name() == 'Segment':
                if hasattr(current, 'point1') and current.point1 is not None:
                    if current.point1 not in visited:
                        queue.append(current.point1)
                        
                if hasattr(current, 'point2') and current.point2 is not None:
                    if current.point2 not in visited:
                        queue.append(current.point2)
    
    def analyze_drawable_for_dependencies(self, drawable):
        """
        Analyze a drawable to find and register its dependencies
        
        Args:
            drawable: The drawable to analyze
            
        Returns:
            list: List of identified dependencies
        """
        dependencies = []
        
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
            
        return dependencies
    
    def _find_segment_children(self, segment):
        """Finds children geometrically by iterating through all segments."""
        # Safety check for segment and its points
        if not segment or not hasattr(segment, 'point1') or not hasattr(segment, 'point2'):
            return []
            
        # Safety check for points
        if not segment.point1 or not segment.point2:
            return []
            
        sp1x, sp1y = segment.point1.x, segment.point1.y
        sp2x, sp2y = segment.point2.x, segment.point2.y
        children = []
        
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

    def resolve_dependency_order(self, drawables):
        """
        Determine the correct order to process drawables based on dependencies
        
        Args:
            drawables: List of drawables to process
            
        Returns:
            list: Ordered list of drawables (parents before children)
        """
        # Filter out None values
        filtered_drawables = [d for d in drawables if d is not None]
        
        # Simple topological sort
        result = []
        visited = set()
        
        def visit(drawable):
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