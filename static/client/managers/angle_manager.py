"""
MatHud Angle Management System

Manages angle creation, retrieval, and deletion for angular measurement and visualization.
Handles angle construction from vertex points and arm segments with reflex angle support.

Core Responsibilities:
    - Angle Creation: Creates angles from vertex and arm-defining points
    - Angle Retrieval: Lookup by name, constituent segments, or defining points
    - Angle Deletion: Safe removal with dependency cleanup
    - Property Management: Updates angle colors and visual properties

Geometric Construction:
    - Vertex-Based Definition: Uses common vertex point and two arm-defining points
    - Automatic Segment Creation: Creates underlying segments for angle arms
    - Reflex Angle Support: Handles both standard and reflex angle measurements
    - Point Integration: Seamless creation and management of defining points

Advanced Features:
    - Dependency Tracking: Maintains relationships with constituent segments
    - State Management: Preserves angle properties during operations
    - Collision Detection: Prevents creation of duplicate angles
    - Event Handling: Responds to segment updates and deletions

Integration Points:
    - PointManager: Creates and manages vertex and arm-defining points
    - SegmentManager: Creates and manages angle arm segments
    - DependencyManager: Tracks angle relationships with constituent elements
    - Canvas: Handles angle rendering and visual updates

Mathematical Properties:
    - Angle Measurement: Supports angular calculations and degree measurements
    - Orientation Detection: Handles clockwise and counterclockwise orientations
    - Reflex Calculations: Manages angles greater than 180 degrees
    - Collinearity Validation: Prevents creation of degenerate angles

State Persistence:
    - Serialization: Saves and loads angle states for workspace management
    - Property Updates: Modifies angle properties without recreation
    - Cleanup Logic: Intelligent removal when constituent segments are deleted
    - Undo/Redo: Complete state archiving for angle operations
"""

from drawables.angle import Angle
from drawables.point import Point
from drawables.segment import Segment


class AngleManager:
    """
    Manages Angle drawables for a Canvas.
    This class is responsible for:
    - Creating Angle objects, including their underlying points and segments if needed.
    - Retrieving Angle objects by name, constituent segments, or defining points.
    - (Future) Deleting Angle objects and managing their dependencies.
    """

    def __init__(self, canvas, drawables_container, name_generator, 
                 dependency_manager, point_manager, segment_manager, drawable_manager_proxy):
        """
        Initialize the AngleManager.

        Args:
            canvas: The Canvas object this manager is responsible for.
            drawables_container: The container for storing drawables (angles will be in .Angles).
            name_generator: Utility for generating unique names for drawables.
            dependency_manager: Manages dependencies between drawables.
            point_manager: Manager for Point drawables.
            segment_manager: Manager for Segment drawables.
            drawable_manager_proxy: Proxy to the main DrawableManager or for inter-manager calls.
        """
        self.canvas = canvas
        self.drawables = drawables_container
        self.name_generator = name_generator
        self.dependency_manager = dependency_manager
        self.point_manager = point_manager
        self.segment_manager = segment_manager
        self.drawable_manager = drawable_manager_proxy

    def create_angle(self, vx: float, vy: float, p1x: float, p1y: float, p2x: float, p2y: float, 
                     color: str = None, 
                     angle_name: str = None, 
                     is_reflex: bool = False,
                     extra_graphics: bool = True) -> Angle | None:
        """
        Creates an angle defined by a vertex point (vx, vy) and two other points 
        (p1x, p1y) and (p2x, p2y) that define the arms of the angle.
        Points and segments will be created if they don't exist, or existing ones will be used.

        Args:
            vx, vy: Coordinates of the common vertex point.
            p1x, p1y: Coordinates of a point on the first arm.
            p2x, p2y: Coordinates of a point on the second arm.
            color: Optional color for the angle.
            angle_name: Optional name for the angle. If not provided, Angle class may generate one.
            is_reflex (bool): Optional. If True, creates the reflex angle. Defaults to False.
            extra_graphics: Whether to trigger creation of related drawables.

        Returns:
            The created or existing Angle object, or None if creation failed.
        """
        self.canvas.undo_redo_manager.archive() # TODO: Ensure this call is correct as per manager patterns

        # 1. Create/Retrieve defining points
        # Point names can be derived if angle_name follows a convention, e.g., "angle_ABC"
        # For now, let PointManager handle default naming if specific names aren't easily parsed.
        vertex_point_obj = self.point_manager.create_point(vx, vy, extra_graphics=False)
        arm1_defining_point_obj = self.point_manager.create_point(p1x, p1y, extra_graphics=False)
        arm2_defining_point_obj = self.point_manager.create_point(p2x, p2y, extra_graphics=False)

        if not all([vertex_point_obj, arm1_defining_point_obj, arm2_defining_point_obj]):
            print(f"AngleManager: Failed to create one or more defining points for the angle.")
            # TODO: Consider if self.canvas.undo_redo_manager.revert() is needed if part-way through
            return None

        # 2. Create/Retrieve segments using these points
        # SegmentManager.create_segment takes coordinates, not point objects directly.
        segment1 = self.segment_manager.create_segment(
            vertex_point_obj.x, vertex_point_obj.y,
            arm1_defining_point_obj.x, arm1_defining_point_obj.y,
            extra_graphics=False
        )
        segment2 = self.segment_manager.create_segment(
            vertex_point_obj.x, vertex_point_obj.y,
            arm2_defining_point_obj.x, arm2_defining_point_obj.y,
            extra_graphics=False
        )

        if not segment1 or not segment2:
            print(f"AngleManager: Failed to create one or more segments for the angle.")
            return None

        # 3. Check if an angle with these exact segments AND same reflex state already exists
        existing_angle = self.get_angle_by_segments(segment1, segment2, is_reflex)
        if existing_angle:
            return existing_angle

        # 4. Instantiate the Angle
        new_angle = None
        try:
            # Handle color parameter - if None, let Angle constructor use its default
            angle_kwargs = {
                'segment1': segment1,
                'segment2': segment2,
                'canvas': self.canvas,
                'is_reflex': is_reflex
            }
            if color is not None:
                angle_kwargs['color'] = color
            
            new_angle = Angle(**angle_kwargs)
        except ValueError as e:
            print(f"AngleManager: Error creating Angle - {e}")
            # Segments might be valid, but not form a valid angle (e.g., collinear)
            return None

        # 5. Add to drawables container
        self.drawables.add(new_angle)

        # 6. Register dependencies
        self.dependency_manager.register_dependency(child=new_angle, parent=segment1)
        self.dependency_manager.register_dependency(child=new_angle, parent=segment2)
        # Also register dependencies on the constituent points
        self.dependency_manager.register_dependency(child=new_angle, parent=vertex_point_obj)
        self.dependency_manager.register_dependency(child=new_angle, parent=arm1_defining_point_obj)
        self.dependency_manager.register_dependency(child=new_angle, parent=arm2_defining_point_obj)

        # 7. Handle extra graphics if requested
        if extra_graphics:
            self.drawable_manager.create_drawables_from_new_connections()

        # 8. Draw the canvas
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return new_angle

    def get_angle_by_name(self, name: str) -> Angle | None:
        """
        Retrieves an Angle by its unique name.

        Args:
            name: The name of the angle.

        Returns:
            The Angle object if found, otherwise None.
        """
        # Ensure self.drawables.Angles exists and is iterable
        if not hasattr(self.drawables, 'Angles') or not isinstance(self.drawables.Angles, list):
            # print("AngleManager: DrawablesContainer has no 'Angles' list or it's not a list.")
            return None
        for angle in self.drawables.Angles: 
            if angle.name == name:
                return angle
        return None

    def get_angle_by_segments(self, segment1: Segment, segment2: Segment, is_reflex_filter: bool = None) -> Angle | None:
        """
        Retrieves an Angle by its two defining Segment objects.
        The order of segments does not matter.
        If is_reflex_filter is specified, it will also filter by the angle's is_reflex state.

        Args:
            segment1: The first segment object.
            segment2: The second segment object.
            is_reflex_filter (bool, optional): If provided, filters for angles with this is_reflex state.

        Returns:
            The Angle object if found, otherwise None.
        """
        if not hasattr(self.drawables, 'Angles') or not isinstance(self.drawables.Angles, list):
            return None
        if not segment1 or not segment2: # Ensure segments themselves are not None
            return None
            
        for angle in self.drawables.Angles:
            if not (hasattr(angle, 'segment1') and hasattr(angle, 'segment2') and hasattr(angle, 'is_reflex')):
                continue # Skip angles that don't have segment or is_reflex attributes properly set up
            if not (angle.segment1 and angle.segment2):
                continue

            match_segments = (angle.segment1 is segment1 and angle.segment2 is segment2) or \
                             (angle.segment1 is segment2 and angle.segment2 is segment1)
            
            if match_segments:
                if is_reflex_filter is None: # No reflex filter, first match is fine
                    return angle
                elif angle.is_reflex == is_reflex_filter: # Reflex state also matches
                    return angle
        return None

    def get_angle_by_points(self, vertex_point: Point, arm1_point: Point, arm2_point: Point, is_reflex_filter: bool = None) -> Angle | None:
        """
        Retrieves an Angle defined by three Point objects: a common vertex, 
        and one point on each arm. The order of arm1_point and arm2_point does not matter.
        If is_reflex_filter is specified, it will also filter by the angle's is_reflex state.

        Args:
            vertex_point: The common vertex Point object.
            arm1_point: A Point object on the first arm.
            arm2_point: A Point object on the second arm.
            is_reflex_filter (bool, optional): If provided, filters for angles with this is_reflex state.

        Returns:
            The Angle object if found, otherwise None.
        """
        if not hasattr(self.drawables, 'Angles') or not isinstance(self.drawables.Angles, list):
            return None
        if not all([vertex_point, arm1_point, arm2_point]):
            return None
            
        for angle in self.drawables.Angles:
            if not all([hasattr(angle, 'vertex_point'), hasattr(angle, 'arm1_point'), 
                        hasattr(angle, 'arm2_point'), hasattr(angle, 'is_reflex')]):
                continue # Skip angles missing point or is_reflex attributes
            if not all([angle.vertex_point, angle.arm1_point, angle.arm2_point]):
                continue

            if angle.vertex_point is not vertex_point:
                continue

            angle_arm_points = {angle.arm1_point, angle.arm2_point}
            input_arm_points = {arm1_point, arm2_point}

            if angle_arm_points == input_arm_points:
                if is_reflex_filter is None: # No reflex filter, first match is fine
                    return angle
                elif angle.is_reflex == is_reflex_filter: # Reflex state also matches
                    return angle
        return None 

    def delete_angle(self, angle_name: str) -> bool:
        """
        Removes an angle by its name. Also attempts to remove its constituent segments 
        if they are no longer needed by other drawables (handled by SegmentManager).

        Args:
            angle_name: The name of the angle to remove.

        Returns:
            True if the angle was found and processed for removal, False otherwise.
        """
        self.canvas.undo_redo_manager.archive()
        
        angle_to_delete = self.get_angle_by_name(angle_name)
        if not angle_to_delete:
            return False

        segment1 = angle_to_delete.segment1
        segment2 = angle_to_delete.segment2
        vertex_point = getattr(angle_to_delete, 'vertex_point', None)
        arm1_point = getattr(angle_to_delete, 'arm1_point', None)
        arm2_point = getattr(angle_to_delete, 'arm2_point', None)

        # 1. Unregister angle's dependencies on its parent segments
        if segment1:
            self.dependency_manager.unregister_dependency(child=angle_to_delete, parent=segment1)
        if segment2:
            self.dependency_manager.unregister_dependency(child=angle_to_delete, parent=segment2)
        
        # Also unregister dependencies on the constituent points
        if vertex_point:
            self.dependency_manager.unregister_dependency(child=angle_to_delete, parent=vertex_point)
        if arm1_point:
            self.dependency_manager.unregister_dependency(child=angle_to_delete, parent=arm1_point)
        if arm2_point:
            self.dependency_manager.unregister_dependency(child=angle_to_delete, parent=arm2_point)
        
        # 2. Remove angle from dependency manager's own tracking
        self.dependency_manager.remove_drawable(angle_to_delete)
        
        # 3. Remove angle from the drawables container
        # This is the primary step that ensures it won't be drawn in the next canvas redraw.
        try:
            if hasattr(self.drawables, 'Angles') and angle_to_delete in self.drawables.Angles:
                self.drawables.Angles.remove(angle_to_delete)
            elif hasattr(self.drawables, 'remove') and callable(self.drawables.remove): # Generic remove
                 self.drawables.remove(angle_to_delete)
            else:
                print(f"AngleManager: Warning - Could not remove angle '{angle_name}' from drawables container.")
        except ValueError:
            print(f"AngleManager: Warning - Angle '{angle_name}' not found in Angles list for direct removal.")

        # 4. Attempt to delete the constituent segments (SegmentManager handles if they are still in use)
        if segment1 and hasattr(segment1, 'point1') and hasattr(segment1, 'point2'):
            self.segment_manager.delete_segment(
                segment1.point1.x, segment1.point1.y,
                segment1.point2.x, segment1.point2.y
            )
        if segment2 and hasattr(segment2, 'point1') and hasattr(segment2, 'point2'):
            self.segment_manager.delete_segment(
                segment2.point1.x, segment2.point1.y,
                segment2.point2.x, segment2.point2.y
            )
            
        # 5. Draw the canvas
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return True

    def update_angle_properties(self, angle_name: str, new_color: str = None) -> bool:
        """
        Updates the properties (color) of an existing angle.

        Args:
            angle_name: The name of the angle to update.
            new_color: The new color for the angle. If None, color is not changed.

        Returns:
            True if the angle was found and updated, False otherwise.
        """
        self.canvas.undo_redo_manager.archive()
        angle_to_update = self.get_angle_by_name(angle_name)

        if not angle_to_update:
            # print(f"AngleManager: Angle '{angle_name}' not found for update.")
            return False

        updated = False
        if new_color is not None and hasattr(angle_to_update, 'color'):
            angle_to_update.color = new_color
            updated = True

        if updated:
            # The Angle's draw method should handle re-rendering with new properties.
            # A general canvas draw will trigger this.
            if hasattr(angle_to_update, 'remove_svg_elements') and callable(angle_to_update.remove_svg_elements):
                 angle_to_update.remove_svg_elements() # Remove old before redrawing with new properties
            
            if self.canvas.draw_enabled:
                self.canvas.draw() # This should cause angle_to_update.draw() to be called.
        
        return True # Returns True if angle found, even if no properties changed

    def handle_segment_updated(self, updated_segment_name: str):
        """
        Called when a segment (that an angle might depend on) is updated.
        This method should find all angles dependent on this segment and trigger their update/re-initialization.

        Args:
            updated_segment_name: The name of the segment that was updated.
        """
        if not hasattr(self.drawables, 'Angles') or not isinstance(self.drawables.Angles, list):
            return

        needs_redraw = False
        for angle in list(self.drawables.Angles): # Iterate over a copy in case of modification
            if not (hasattr(angle, 'segment1') and hasattr(angle, 'segment2')):
                continue
            if not (angle.segment1 and angle.segment2): 
                continue
            
            if angle.segment1.name == updated_segment_name or angle.segment2.name == updated_segment_name:
                if hasattr(angle, '_initialize') and callable(angle._initialize):
                    try:
                        # Before re-initializing, remove old SVG elements
                        if hasattr(angle, 'remove_svg_elements') and callable(angle.remove_svg_elements):
                            angle.remove_svg_elements()
                        angle._initialize()
                        needs_redraw = True
                    except ValueError as e:
                        # If _initialize fails (e.g., angle becomes invalid), remove the angle
                        print(f"AngleManager: Angle '{angle.name}' became invalid after segment '{updated_segment_name}' update. Error: {e}. Removing angle.")
                        self.delete_angle(angle.name) # This will handle its own draw call
                        needs_redraw = True # Ensure redraw happens even if this one is removed
                else:
                    print(f"AngleManager: Warning - Angle '{angle.name}' does not have _initialize method for update.")

        if needs_redraw and self.canvas.draw_enabled:
            # If delete_angle was called, it might have drawn. A final draw ensures overall consistency.
            self.canvas.draw()

    def handle_segment_removed(self, removed_segment_name: str):
        """
        Called when a segment (that an angle might depend on) is removed.
        This method should find all angles dependent on this segment and remove them.

        Args:
            removed_segment_name: The name of the segment that was removed.
        """
        if not hasattr(self.drawables, 'Angles') or not isinstance(self.drawables.Angles, list):
            return

        # Collect names first to avoid modification issues while iterating
        angles_to_remove_names = []
        for angle in self.drawables.Angles: # No need for list copy if just collecting names
            if not (hasattr(angle, 'segment1') and hasattr(angle, 'segment2')):
                continue
            if not (angle.segment1 and angle.segment2): 
                continue
            if angle.segment1.name == removed_segment_name or angle.segment2.name == removed_segment_name:
                angles_to_remove_names.append(angle.name)

        if angles_to_remove_names:
            # Archiving once before multiple removals might be better, 
            # but delete_angle handles its own archiving.
            # self.canvas.undo_redo_manager.archive() 
            
            for angle_name in angles_to_remove_names:
                print(f"AngleManager: Segment '{removed_segment_name}' was removed. Removing dependent angle '{angle_name}'.")
                self.delete_angle(angle_name) # This handles its own draw and archive
            
            # A final draw might not be necessary if delete_angle always draws and is the last action.
            # if self.canvas.draw_enabled:
            #     self.canvas.draw()

    def load_angles(self, angles_data: list[dict]):
        """
        Loads angles from a list of state dictionaries (e.g., from a saved workspace).

        Args:
            angles_data: A list of dictionaries, where each dict is the state of an Angle object.
        """
        # It's assumed that dependent segments (and their points) should already exist or be loaded
        # by their respective managers before angles are loaded.
        # Angle.from_state will use drawable_manager to find segments by name.

        # self.canvas.undo_redo_manager.archive() # Usually part of a larger workspace load operation

        for angle_state in angles_data:
            if not isinstance(angle_state, dict):
                print(f"AngleManager: Skipping invalid angle state data (not a dict): {angle_state}")
                continue
            
            # The Angle.from_state method requires the drawable_manager to find segments.
            new_angle = Angle.from_state(angle_state, self.canvas)
            
            if new_angle:
                # Check for duplicates by name before adding, though from_state might handle this
                # or Angle's name generation from segments might lead to an existing object.
                # For simplicity, assume from_state returns a valid, potentially new, object.
                # A robust system might use get_angle_by_segments with segments derived from names in state.
                
                self.drawables.add(new_angle) # Add to Angles list
                if hasattr(new_angle, 'segment1') and new_angle.segment1 and \
                   hasattr(new_angle, 'segment2') and new_angle.segment2:
                    self.dependency_manager.register_dependency(child=new_angle, parent=new_angle.segment1)
                    self.dependency_manager.register_dependency(child=new_angle, parent=new_angle.segment2)
                    
                    # Also register dependencies on the constituent points
                    if hasattr(new_angle, 'vertex_point') and new_angle.vertex_point:
                        self.dependency_manager.register_dependency(child=new_angle, parent=new_angle.vertex_point)
                    if hasattr(new_angle, 'arm1_point') and new_angle.arm1_point:
                        self.dependency_manager.register_dependency(child=new_angle, parent=new_angle.arm1_point)
                    if hasattr(new_angle, 'arm2_point') and new_angle.arm2_point:
                        self.dependency_manager.register_dependency(child=new_angle, parent=new_angle.arm2_point)
                else:
                    print(f"AngleManager: Warning - Loaded angle '{new_angle.name}' from state but segments are missing. Cannot register dependencies.")
            else:
                angle_name_in_state = angle_state.get('name', '[Unknown Name]')
                print(f"AngleManager: Failed to load angle from state: '{angle_name_in_state}'.")

        if self.canvas.draw_enabled:
            self.canvas.draw()

    def get_angles_state(self) -> list[dict]:
        """
        Returns a list of state dictionaries for all managed angles, for saving to workspace.

        Returns:
            A list of dictionaries, where each dict represents the state of an Angle.
        """
        if not hasattr(self.drawables, 'Angles') or not isinstance(self.drawables.Angles, list):
            return []
        return [angle.get_state() for angle in self.drawables.Angles if hasattr(angle, 'get_state') and callable(angle.get_state)]

    def clear_angles(self):
        """
        Removes all angles managed by this manager.
        """
        self.canvas.undo_redo_manager.archive()
        
        if not hasattr(self.drawables, 'Angles') or not isinstance(self.drawables.Angles, list):
            return

        angle_names_to_remove = [angle.name for angle in list(self.drawables.Angles) if hasattr(angle, 'name')]
        
        for angle_name in angle_names_to_remove:
            self.delete_angle(angle_name) # This will handle individual draws and further dependencies.
        
        # A final draw call might be redundant if delete_angle always draws and handles all cleanup.
        # However, if delete_angle was optimized not to draw, this would be needed.
        if self.canvas.draw_enabled and not angle_names_to_remove: # Only draw if nothing was removed (and drawn individually)
            self.canvas.draw()
        elif self.canvas.draw_enabled and angle_names_to_remove: # If angles were removed, they handled their draw, one final comprehensive draw
             self.canvas.draw() 