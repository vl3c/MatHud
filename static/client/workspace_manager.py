"""
MatHud Client-Side Workspace Persistence and State Management System

Client-side workspace manager that runs in the browser using Brython. Manages workspace 
operations including saving, loading, listing and deleting workspaces through AJAX 
communication with the Flask backend server.

Key Features:
    - Canvas state serialization with complete object preservation
    - AJAX-based communication with backend workspace API
    - Incremental object restoration with dependency resolution
    - Error handling and validation for workspace operations
    - Support for all drawable object types and relationships
    - Computation history preservation and restoration

Workspace Operations:
    - Save: Serializes current canvas state and sends to server via AJAX
    - Load: Requests workspace data from server and restores canvas state
    - List: Retrieves available workspace names from server storage
    - Delete: Removes workspace files from server persistent storage

Object Restoration:
    - Points: Coordinate-based geometric primitives
    - Segments: Line segments with endpoint dependency resolution
    - Vectors: Directed segments with origin/tip point relationships
    - Triangles: Three-vertex polygons with automatic edge detection
    - Rectangles: Four-corner polygons with diagonal point calculation and segment reuse
    - Circles: Circular objects with center point dependencies
    - Ellipses: Elliptical objects with center and rotation parameters
    - Functions: Mathematical function expressions with domain settings
    - Colored Areas: Bounded regions with drawable object relationships
    - Angles: Angular measurements with vertex and arm dependencies

Dependencies:
    - browser: AJAX communication for backend workspace operations
    - utils.math_utils: Geometric calculations for object restoration
    - json: State serialization and deserialization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import json
from itertools import permutations

from browser import ajax, document
from utils.math_utils import MathUtils

if TYPE_CHECKING:
    from canvas import Canvas
    from drawables.point import Point
    from drawables.segment import Segment

class WorkspaceManager:
    """
    Client-side workspace manager that handles workspace operations via AJAX communication.

    This class handles all workspace-related operations and their associated error handling.
    It works with the canvas to save and restore workspace states, including all geometric
    objects and computations. Runs in the browser using Brython.

    Attributes:
        canvas: The canvas instance to manage workspaces for.
    """
    
    def __init__(self, canvas: "Canvas") -> None:
        """Initialize workspace manager with canvas reference."""
        self.canvas: "Canvas" = canvas

    def save_workspace(self, name: Optional[str] = None) -> str:
        """
        Save the current workspace state to server via AJAX.
        
        Serializes the complete canvas state including all geometric objects,
        computations, and settings, then sends it to the Flask backend server
        for persistent storage.
        
        Args:
            name (str, optional): Name for the workspace. If None, saves as "current".
            
        Returns:
            str: Success or error message from the save operation.
        """
        def on_complete(req: Any) -> str:
            try:
                response: Dict[str, Any] = json.loads(req.text)
                if response.get('status') == 'success':
                    return f'Workspace "{name if name else "current"}" saved successfully.'
                return f'Error saving workspace: {response.get("message")}'
            except Exception as e:
                return f'Error saving workspace: {str(e)}'

        req: Any = ajax.Ajax()
        req.bind('complete', on_complete)
        req.bind('error', lambda e: f'Error saving workspace: {e.text}')
        
        data: Dict[str, Any] = {
            'state': self.canvas.get_canvas_state(),
            'name': name
        }
        
        req.open('POST', '/save_workspace', False)  # Set to synchronous
        req.set_header('Content-Type', 'application/json')
        req.send(json.dumps(data))
        return on_complete(req)

    def _create_points(self, state: Dict[str, Any]) -> None:
        """Create points from workspace state."""
        if "Points" not in state:
            return
        for item_state in state["Points"]:
            self.canvas.create_point(
                item_state["args"]["position"]["x"], 
                item_state["args"]["position"]["y"],
                name=item_state.get("name", "")
            )

    def _create_segments(self, state: Dict[str, Any]) -> None:
        """Create segments from workspace state."""
        if "Segments" not in state:
            return
        for item_state in state["Segments"]:
            p1: Optional["Point"] = self.canvas.get_point_by_name(item_state["args"]["p1"])
            p2: Optional["Point"] = self.canvas.get_point_by_name(item_state["args"]["p2"])
            if p1 and p2:
                self.canvas.create_segment(
                    p1.x,
                    p1.y,
                    p2.x,
                    p2.y,
                    name=item_state.get("name", "")
                )

    def _create_vectors(self, state: Dict[str, Any]) -> None:
        """Create vectors from workspace state."""
        if "Vectors" not in state:
            return
        for item_state in state["Vectors"]:
            origin_point_name: Optional[str] = item_state["args"].get("origin")
            tip_point_name: Optional[str] = item_state["args"].get("tip")

            if not origin_point_name or not tip_point_name:
                print(f"Warning: Vector '{item_state.get('name', 'Unnamed')}' is missing origin or tip point name in its state. Skipping.")
                continue

            origin_point: Optional["Point"] = self.canvas.get_point_by_name(origin_point_name)
            tip_point: Optional["Point"] = self.canvas.get_point_by_name(tip_point_name)

            if not origin_point or not tip_point:
                print(f"Warning: Could not find origin ('{origin_point_name}') or tip ('{tip_point_name}') point for vector '{item_state.get('name', 'Unnamed')}' in the canvas. Skipping.")
                continue
            
            self.canvas.create_vector(
                origin_point.x,
                origin_point.y, 
                tip_point.x,
                tip_point.y,
                name=item_state.get("name", "")
            )

    def _create_triangles(self, state: Dict[str, Any]) -> None:
        """Create triangles from workspace state."""
        if "Triangles" not in state:
            return
        for item_state in state["Triangles"]:
            p1: Optional["Point"] = self.canvas.get_point_by_name(item_state["args"]["p1"])
            p2: Optional["Point"] = self.canvas.get_point_by_name(item_state["args"]["p2"])
            p3: Optional["Point"] = self.canvas.get_point_by_name(item_state["args"]["p3"])
            if p1 and p2 and p3:
                self.canvas.create_triangle(
                    p1.x,
                    p1.y,
                    p2.x,
                    p2.y,
                    p3.x,
                    p3.y,
                    name=item_state.get("name", "")
                )

    def _create_rectangles(self, state: Dict[str, Any]) -> None:
        """
        Create rectangles from workspace state with diagonal point calculation and segment reuse.
        
        Handles complex rectangle restoration by finding existing segments that form
        rectangles and reusing them when possible. Falls back to standard rectangle
        creation if existing segments don't form valid rectangles.
        
        Args:
            state (dict): Workspace state containing rectangle definitions.
        """
        if "Rectangles" not in state:
            return
        
        for item_state in state["Rectangles"]:
            rect_name: str = item_state.get("name", "UnnamedRectangle")
            
            arg_point_names: List[Optional[str]] = [
                item_state["args"].get("p1"),
                item_state["args"].get("p2"),
                item_state["args"].get("p3"),
                item_state["args"].get("p4")
            ]

            if not all(arg_point_names):
                print(f"Warning: Rectangle '{rect_name}' is missing one or more point names (p1, p2, p3, p4) in its state. Skipping.")
                continue

            points: List[Optional["Point"]] = [self.canvas.get_point_by_name(name) for name in arg_point_names]

            if not all(points):
                missing_names: List[str] = [str(arg_point_names[i]) for i, p in enumerate(points) if not p]
                print(f"Warning: Could not find one or more points ({', '.join(missing_names)}) for rectangle '{rect_name}' in the canvas. Skipping.")
                continue
            
            p_diag1: Optional["Point"]
            p_diag2: Optional["Point"]
            p_diag1, p_diag2 = MathUtils.find_diagonal_points(points, rect_name)
            
            if not p_diag1 or not p_diag2:
                print(f"Warning: Could not determine diagonal points for rectangle '{rect_name}'. Skipping.")
                continue
            
            try:
                # Instead of using canvas.create_rectangle (which creates new segments),
                # find the existing segments that form this rectangle and create it directly
                rect_segments: Optional[List["Segment"]] = self._find_rectangle_segments(points)
                
                if not rect_segments:
                    # Fall back to standard creation if segments not found
                    created_rect: Any = self.canvas.create_rectangle(
                        p_diag1.x,
                        p_diag1.y,
                        p_diag2.x,
                        p_diag2.y,
                        name=item_state.get("name", "")
                    )
                else:
                    # Existing segments found that connect the rectangle's points
                    # Create properly oriented segments for Rectangle constructor
                    properly_oriented_segments: Optional[List["Segment"]] = self._get_properly_oriented_rectangle_segments(rect_segments, points)
                    
                    if properly_oriented_segments:
                        # Create rectangle directly using properly oriented segments
                        from drawables.rectangle import Rectangle
                        try:
                            created_rect = Rectangle(
                                properly_oriented_segments[0], properly_oriented_segments[1], 
                                properly_oriented_segments[2], properly_oriented_segments[3], 
                                self.canvas
                            )
                            
                            # Set the name if provided
                            if item_state.get("name"):
                                created_rect.name = item_state.get("name")
                            
                            # Add to drawables
                            self.canvas.drawable_manager.drawables.add(created_rect)
                            
                            # Register dependencies
                            self.canvas.drawable_manager.dependency_manager.analyze_drawable_for_dependencies(created_rect)
                            
                        except ValueError as e:
                            # If Rectangle constructor still fails, fall back to standard creation
                            created_rect = self.canvas.create_rectangle(
                                p_diag1.x,
                                p_diag1.y,
                                p_diag2.x,
                                p_diag2.y,
                                name=item_state.get("name", "")
                            )
                    else:
                        # Fall back to standard rectangle creation
                        created_rect = self.canvas.create_rectangle(
                            p_diag1.x,
                            p_diag1.y,
                            p_diag2.x,
                            p_diag2.y,
                            name=item_state.get("name", "")
                        )
                
            except ValueError as e:
                if "The segments do not form a rectangle" in str(e):
                    print(f"Warning: Skipping rectangle '{rect_name}' - segments do not form a valid rectangle topology. This may be due to inconsistent workspace data.")
                    continue
                else:
                    # Re-raise if it's a different ValueError
                    raise
            except Exception as e:
                raise

    def _find_rectangle_segments(self, points: List["Point"]) -> Optional[List["Segment"]]:
        """
        Find existing segments that form a rectangle from the given points.
        
        Searches for existing segments connecting pairs of the four rectangle points
        and attempts to arrange them in proper rectangular order forming a closed loop.
        
        Args:
            points (list): List of 4 Point objects that should form a rectangle.
            
        Returns:
            list or None: List of 4 segments in proper rectangle order, or None if not found.
        """
        if len(points) != 4:
            return None
            
        # Find all segments that connect pairs of these points
        connecting_segments: List["Segment"] = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1: "Point" = points[i]
                p2: "Point" = points[j]
                # Look for segment connecting these two points (regardless of direction)
                segment: Optional["Segment"] = self.canvas.get_segment_by_points(p1, p2)
                if not segment:
                    # Try the other direction
                    segment = self.canvas.get_segment_by_points(p2, p1)
                if segment:
                    connecting_segments.append(segment)
        
        # For a rectangle, we need exactly 4 segments
        if len(connecting_segments) != 4:
            return None
            
        # Try to arrange segments in proper rectangle order
        # A rectangle needs segments that form a closed loop
        return self._arrange_segments_in_rectangle_order(connecting_segments)
    
    def _arrange_segments_in_rectangle_order(self, segments: List["Segment"]) -> Optional[List["Segment"]]:
        """Arrange segments in proper rectangle order (each segment's end connects to next segment's start)."""
        if len(segments) != 4:
            return None
        
        # Try different starting segments and arrangements
        for start_seg in segments:
            remaining: List["Segment"] = segments.copy()
            remaining.remove(start_seg)
            
            # Try both directions for the starting segment
            for start_direction in [True, False]:  # True = normal direction, False = reversed
                ordered: List["Segment"] = [start_seg]
                current_end_point: "Point"
                if start_direction:
                    current_end_point = start_seg.point2
                else:
                    current_end_point = start_seg.point1  
                
                remaining_copy: List["Segment"] = remaining.copy()
                
                # Try to build a path
                while remaining_copy and len(ordered) < 4:
                    found_next: bool = False
                    for seg in remaining_copy:
                        if seg.point1 == current_end_point:
                            ordered.append(seg)
                            current_end_point = seg.point2
                            remaining_copy.remove(seg)
                            found_next = True
                            break
                        elif seg.point2 == current_end_point:
                            # We conceptually reverse this segment for path building
                            ordered.append(seg)
                            current_end_point = seg.point1
                            remaining_copy.remove(seg) 
                            found_next = True
                            break
                    
                    if not found_next:
                        break
                
                # Check if we have a closed rectangle (4 segments, last connects back to first)
                if len(ordered) == 4:
                    # Check if the path closes properly
                    start_point: "Point"
                    if start_direction:
                        start_point = start_seg.point1
                    else:
                        start_point = start_seg.point2
                    
                    closes_properly: bool = (current_end_point == start_point)
                    
                    if closes_properly:
                        return ordered
        
        return None

    def _create_circles(self, state: Dict[str, Any]) -> None:
        """Create circles from workspace state."""
        if "Circles" not in state:
            return
        for item_state in state["Circles"]:
            center_point: Optional["Point"] = self.canvas.get_point_by_name(item_state["args"]["center"])
            if center_point:
                self.canvas.create_circle(
                    center_point.x,
                    center_point.y,
                    item_state["args"]["radius"],
                    name=item_state.get("name", "")
                )

    def _create_ellipses(self, state: Dict[str, Any]) -> None:
        """Create ellipses from workspace state."""
        if "Ellipses" not in state:
            return
        for item_state in state["Ellipses"]:
            center_point: Optional["Point"] = self.canvas.get_point_by_name(item_state["args"]["center"])
            if center_point:
                self.canvas.create_ellipse(
                    center_point.x,
                    center_point.y,
                    item_state["args"]["radius_x"],
                    item_state["args"]["radius_y"],
                    rotation_angle=item_state["args"].get("rotation_angle", 0),
                    name=item_state.get("name", "")
                )

    def _create_functions(self, state: Dict[str, Any]) -> None:
        """Create functions from workspace state."""
        if "Functions" not in state:
            return
        for item_state in state["Functions"]:
            self.canvas.draw_function(
                item_state["args"]["function_string"],
                name=item_state.get("name", ""),
                left_bound=item_state["args"].get("left_bound"),
                right_bound=item_state["args"].get("right_bound")
            )

    def _create_colored_areas(self, state: Dict[str, Any]) -> None:
        """Create colored areas from workspace state."""
        # Handle different types of colored areas
        colored_area_types: List[str] = [
            "FunctionsBoundedColoredAreas",
            "SegmentsBoundedColoredAreas", 
            "FunctionSegmentBoundedColoredAreas",
            "ColoredAreas"
        ]
        
        for area_type in colored_area_types:
            if area_type not in state:
                continue
                
            for item_state in state[area_type]:
                try:
                    # Extract parameters based on area type
                    drawable1_name: Optional[str]
                    drawable2_name: Optional[str]
                    if area_type == "FunctionsBoundedColoredAreas":
                        # Functions bounded colored areas use func1/func2
                        drawable1_name = item_state["args"]["func1"]
                        drawable2_name = item_state["args"].get("func2")
                    else:
                        # Other types might use different field names
                        drawable1_name = item_state["args"].get("drawable1_name") or item_state["args"].get("segment1") or item_state["args"].get("func1")
                        drawable2_name = item_state["args"].get("drawable2_name") or item_state["args"].get("segment2") or item_state["args"].get("func2")
                    
                    # Common parameters
                    left_bound: Optional[float] = item_state["args"].get("left_bound")
                    right_bound: Optional[float] = item_state["args"].get("right_bound")
                    color: str = item_state["args"].get("color", "lightblue")
                    opacity: float = item_state["args"].get("opacity", 0.3)
                    name: str = item_state.get("name", "")
                    
                    # Create the colored area
                    created_area: Any = self.canvas.create_colored_area(
                        drawable1_name=drawable1_name,
                        drawable2_name=drawable2_name,
                        left_bound=left_bound,
                        right_bound=right_bound,
                        color=color,
                        opacity=opacity
                    )
                    
                    # Set the name if specified and area was created successfully
                    if created_area and name:
                        created_area.name = name
                        
                except Exception as e:
                    print(f"Warning: Could not restore colored area '{item_state.get('name', 'Unnamed')}': {e}")
                    continue

    def _create_angles(self, state: Dict[str, Any]) -> None:
        """Create angles from workspace state."""
        if "Angles" not in state:
            return
        
        # Use the angle manager's load_angles method
        if hasattr(self.canvas, 'drawable_manager') and \
           hasattr(self.canvas.drawable_manager, 'angle_manager') and \
           self.canvas.drawable_manager.angle_manager:
            try:
                self.canvas.drawable_manager.angle_manager.load_angles(state["Angles"])
            except Exception as e:
                print(f"Warning: Could not restore angles: {e}")
        else:
            print("Warning: Angle manager not available for loading angles")

    def _create_circle_arcs(self, state: Dict[str, Any]) -> None:
        """Create circle arcs from workspace state."""
        arcs_data = state.get("CircleArcs")
        if not arcs_data:
            return

        for arc_state in arcs_data:
            args = arc_state.get("args", {})
            point1_name = args.get("point1_name")
            point2_name = args.get("point2_name")
            point1 = self.canvas.get_point_by_name(point1_name) if point1_name else None
            point2 = self.canvas.get_point_by_name(point2_name) if point2_name else None
            if not point1 or not point2:
                continue

            try:
                self.canvas.create_circle_arc(
                    point1_x=point1.x,
                    point1_y=point1.y,
                    point2_x=point2.x,
                    point2_y=point2.y,
                    point1_name=point1.name,
                    point2_name=point2.name,
                    point3_x=None,
                    point3_y=None,
                    point3_name=None,
                    center_point_choice=None,
                    circle_name=args.get("circle_name"),
                    center_x=args.get("center_x"),
                    center_y=args.get("center_y"),
                    radius=args.get("radius"),
                    arc_name=arc_state.get("name"),
                    color=args.get("color"),
                    use_major_arc=args.get("use_major_arc", False),
                    extra_graphics=False,
                )
            except Exception as exc:
                print(f"Warning: Could not restore circle arc '{arc_state.get('name', '')}': {exc}")

    def _restore_computations(self, state: Dict[str, Any]) -> None:
        """Restore computations from workspace state."""
        if "computations" not in state:
            return
        for comp in state["computations"]:
            # Skip workspace management functions
            if comp["expression"].startswith("list_workspaces") or \
               comp["expression"].startswith("save_workspace") or \
               comp["expression"].startswith("load_workspace"):
                continue
            self.canvas.add_computation(comp["expression"], comp["result"])

    def _restore_workspace_state(self, state: Dict[str, Any]) -> None:
        """
        Main restoration orchestrator for complete workspace state.
        
        Restores all geometric objects and computations in the correct dependency
        order to ensure proper relationships between objects. Clears the canvas
        first, then creates objects from points to complex shapes.
        
        Args:
            state (dict): Workspace state dictionary containing all object data.
        """
        self.canvas.clear()
        
        # Create objects in the correct dependency order
        self._create_points(state)
        self._create_segments(state)
        self._create_vectors(state)
        self._create_triangles(state)
        self._create_rectangles(state)
        self._create_circles(state)
        self._create_circle_arcs(state)
        self._create_ellipses(state)
        self._create_functions(state)
        # Create colored areas after functions since they may depend on functions
        self._create_colored_areas(state)
        # Create angles after segments since they depend on segments  
        self._create_angles(state)
        self._restore_computations(state)

    def load_workspace(self, name: Optional[str] = None) -> str:
        """
        Load and restore workspace state from server.
        
        Requests workspace data from the Flask backend server via AJAX and
        restores the complete canvas state including all geometric objects
        and computations in the correct dependency order.
        
        Args:
            name (str, optional): Name of the workspace to load. If None, loads default.
            
        Returns:
            str: Success or error message from the load operation.
        """
        def on_complete(req: Any) -> str:
            try:
                response: Dict[str, Any] = json.loads(req.text)
                if response.get('status') == 'success':
                    state: Optional[Dict[str, Any]] = response.get('data', {}).get('state')
                    if not state:
                        return f'Error loading workspace: No state data found in response'
                    
                    self._restore_workspace_state(state)
                    return f'Workspace "{name if name else "current"}" loaded successfully.'
                else:
                    return f'Error loading workspace: {response.get("message")}'
            except Exception as e:
                return f'Error loading workspace: {str(e)}'

        req: Any = ajax.Ajax()
        req.bind('complete', on_complete)
        req.bind('error', lambda e: f'Error loading workspace: {e.text}')
        
        url: str = f'/load_workspace?name={name}' if name else '/load_workspace'
        req.open('GET', url, False)  # Set to synchronous
        req.send()
        return on_complete(req)

    def list_workspaces(self) -> str:
        """
        Retrieve list of available workspace names from server storage.
        
        Sends an AJAX request to the Flask backend to get all available
        workspace names that can be loaded.
        
        Returns:
            str: Comma-separated list of workspace names, or 'None' if empty.
        """
        def on_complete(req: Any) -> str:
            try:
                response: Dict[str, Any] = json.loads(req.text)
                if response.get('status') == 'success':
                    workspaces: List[str] = response.get('data', [])
                    return ', '.join(workspaces) if workspaces else 'None'
                return f'Error listing workspaces: {response.get("message")}'
            except Exception as e:
                return f'Error listing workspaces: {str(e)}'
                
        req: Any = ajax.Ajax()
        req.bind('complete', on_complete)
        req.bind('error', lambda e: f'Error listing workspaces: {e.text}')
        
        req.open('GET', '/list_workspaces', False)  # Set to synchronous
        req.send()
        return on_complete(req)

    def delete_workspace(self, name: str) -> str:
        """
        Remove workspace from server persistent storage.
        
        Sends an AJAX request to the Flask backend to permanently delete
        the specified workspace from storage.
        
        Args:
            name (str): Name of the workspace to delete.
            
        Returns:
            str: Success or error message from the delete operation.
        """
        def on_complete(req: Any) -> str:
            try:
                response: Dict[str, Any] = json.loads(req.text)
                if response.get('status') == 'success':
                    return f'Workspace "{name}" deleted successfully.'
                return f'Error deleting workspace: {response.get("message")}'
            except Exception as e:
                return f'Error deleting workspace: {str(e)}'
                
        req: Any = ajax.Ajax()
        req.bind('complete', on_complete)
        req.bind('error', lambda e: f'Error deleting workspace: {e.text}')
        
        url: str = f'/delete_workspace?name={name}'
        req.open('GET', url, False)  # Set to synchronous
        req.send()
        return on_complete(req)

    def _get_properly_oriented_rectangle_segments(self, segments: List["Segment"], points: List["Point"]) -> Optional[List["Segment"]]:
        """Create properly oriented segments that satisfy Rectangle constructor connectivity.
        Since existing segments may have orientations that don't match Rectangle requirements,
        we create new segment objects with correct orientations."""
        if len(segments) != 4 or len(points) != 4:
            return None
        
        from drawables.segment import Segment
        
        # Try all possible rectangular paths through the 4 points
        for point_perm in permutations(points):
            p1: "Point"
            p2: "Point"
            p3: "Point"
            p4: "Point"
            p1, p2, p3, p4 = point_perm
            
            # Check if this forms a valid rectangle path: p1 -> p2 -> p3 -> p4 -> p1
            # Verify we have segments connecting each pair of consecutive points
            has_p1_p2: bool = any(self._segments_connect_points(seg, p1, p2) for seg in segments)
            has_p2_p3: bool = any(self._segments_connect_points(seg, p2, p3) for seg in segments)
            has_p3_p4: bool = any(self._segments_connect_points(seg, p3, p4) for seg in segments)
            has_p4_p1: bool = any(self._segments_connect_points(seg, p4, p1) for seg in segments)
            
            if has_p1_p2 and has_p2_p3 and has_p3_p4 and has_p4_p1:
                # This is a valid rectangular path
                # Create new segments with correct orientations for Rectangle constructor
                try:
                    # Create segments with proper connectivity: each segment's end connects to next segment's start
                    seg1: "Segment" = Segment(p1, p2)  # p1 -> p2
                    seg2: "Segment" = Segment(p2, p3)  # p2 -> p3  
                    seg3: "Segment" = Segment(p3, p4)  # p3 -> p4
                    seg4: "Segment" = Segment(p4, p1)  # p4 -> p1
                    
                    # Verify this satisfies Rectangle constructor requirements
                    if (seg1.point2 == seg2.point1 and 
                        seg2.point2 == seg3.point1 and 
                        seg3.point2 == seg4.point1 and 
                        seg4.point2 == seg1.point1):
                        
                        return [seg1, seg2, seg3, seg4]
                        
                except Exception:
                    continue
        
        return None
    
    def _segments_connect_points(self, segment: "Segment", point1: "Point", point2: "Point") -> bool:
        """Check if a segment connects two points (in either direction)."""
        return bool((segment.point1 == point1 and segment.point2 == point2) or 
                (segment.point1 == point2 and segment.point2 == point1)) 