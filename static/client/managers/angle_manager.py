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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from drawables.angle import Angle
from drawables.point import Point
from drawables.segment import Segment
from managers.edit_policy import DrawableEditPolicy, EditRule, get_drawable_edit_policy

if TYPE_CHECKING:
    from drawables.drawable import Drawable
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from managers.point_manager import PointManager
    from managers.segment_manager import SegmentManager
    from name_generator.drawable import DrawableNameGenerator

class AngleManager:
    """
    Manages Angle drawables for a Canvas.
    This class is responsible for:
    - Creating Angle objects, including their underlying points and segments if needed.
    - Retrieving Angle objects by name, constituent segments, or defining points.
    - (Future) Deleting Angle objects and managing their dependencies.
    """

    def __init__(
        self,
        canvas: "Canvas",
        drawables_container: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        point_manager: "PointManager",
        segment_manager: "SegmentManager",
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
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
        self.canvas: "Canvas" = canvas
        self.drawables: "DrawablesContainer" = drawables_container
        self.name_generator: "DrawableNameGenerator" = name_generator
        self.dependency_manager: "DrawableDependencyManager" = dependency_manager
        self.point_manager: "PointManager" = point_manager
        self.segment_manager: "SegmentManager" = segment_manager
        self.drawable_manager: "DrawableManagerProxy" = drawable_manager_proxy
        self.angle_edit_policy: Optional[DrawableEditPolicy] = get_drawable_edit_policy("Angle")

    def create_angle(
        self,
        vx: float,
        vy: float,
        p1x: float,
        p1y: float,
        p2x: float,
        p2y: float,
        color: Optional[str] = None,
        angle_name: Optional[str] = None,
        is_reflex: bool = False,
        extra_graphics: bool = True,
    ) -> Optional[Angle]:
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
        undo_manager = self.canvas.undo_redo_manager
        baseline_state = undo_manager.capture_state()
        undo_manager.suspend_archiving()

        try:
            # 1. Create/Retrieve defining points
            vertex_point_obj = self.point_manager.create_point(vx, vy, extra_graphics=False)
            arm1_defining_point_obj = self.point_manager.create_point(p1x, p1y, extra_graphics=False)
            arm2_defining_point_obj = self.point_manager.create_point(p2x, p2y, extra_graphics=False)

            if not all([vertex_point_obj, arm1_defining_point_obj, arm2_defining_point_obj]):
                print("AngleManager: Failed to create one or more defining points for the angle.")
                undo_manager.restore_state(baseline_state, redraw=self.canvas.draw_enabled)
                return None

            # 2. Create/Retrieve segments using these points
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
                print("AngleManager: Failed to create one or more segments for the angle.")
                undo_manager.restore_state(baseline_state, redraw=self.canvas.draw_enabled)
                return None

            # 3. Check if an angle with these exact segments AND same reflex state already exists
            existing_angle = self.get_angle_by_segments(segment1, segment2, is_reflex)
            if existing_angle:
                undo_manager.restore_state(baseline_state, redraw=False)
                return existing_angle

            # 4. Instantiate the Angle (let Angle compute deterministic name if none provided)
            angle_kwargs: Dict[str, Any] = {'is_reflex': is_reflex}
            if color is not None:
                angle_kwargs['color'] = color
            if angle_name is not None:
                angle_kwargs['name'] = angle_name
            try:
                new_angle = Angle(segment1, segment2, **angle_kwargs)
            except ValueError as e:
                print(f"AngleManager: Error creating Angle - {e}")
                undo_manager.restore_state(baseline_state, redraw=self.canvas.draw_enabled)
                return None

            # 5. Add to drawables container
            self.drawables.add(new_angle)

            # 6. Register dependencies
            self.dependency_manager.register_dependency(child=new_angle, parent=segment1)
            self.dependency_manager.register_dependency(child=new_angle, parent=segment2)
            self.dependency_manager.register_dependency(child=new_angle, parent=vertex_point_obj)
            self.dependency_manager.register_dependency(child=new_angle, parent=arm1_defining_point_obj)
            self.dependency_manager.register_dependency(child=new_angle, parent=arm2_defining_point_obj)

            # 7. Handle extra graphics if requested
            if extra_graphics:
                self.drawable_manager.create_drawables_from_new_connections()

            # Commit as a single undo step once creation fully succeeds.
            undo_manager.push_undo_state(baseline_state)

            # 8. Draw the canvas
            if self.canvas.draw_enabled:
                self.canvas.draw()

            return new_angle
        except Exception:
            undo_manager.restore_state(baseline_state, redraw=self.canvas.draw_enabled)
            raise
        finally:
            undo_manager.resume_archiving()

    def get_angle_by_name(self, name: str) -> Optional[Angle]:
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

    def get_angle_by_segments(self, segment1: Segment, segment2: Segment, is_reflex_filter: Optional[bool] = None) -> Optional[Angle]:
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

    def get_angle_by_points(self, vertex_point: Point, arm1_point: Point, arm2_point: Point, is_reflex_filter: Optional[bool] = None) -> Optional[Angle]:
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

    def update_angle(
        self,
        angle_name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        """
        Update editable properties of an existing angle based on the policy catalog.

        Args:
            angle_name: Name of the angle to edit.
            new_color: Optional CSS color for the arc.
        """
        angle = self._get_angle_or_raise(angle_name)
        pending_fields = self._collect_angle_requested_fields(new_color)
        self._validate_angle_policy(list(pending_fields.keys()))
        self._validate_color_request(pending_fields, new_color)

        self.canvas.undo_redo_manager.archive()
        self._apply_angle_updates(angle, pending_fields, new_color)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def _get_angle_or_raise(self, angle_name: str) -> Angle:
        angle = self.get_angle_by_name(angle_name)
        if not angle:
            raise ValueError(f"Angle '{angle_name}' was not found.")
        return angle

    def _collect_angle_requested_fields(
        self,
        new_color: Optional[str],
    ) -> Dict[str, str]:
        pending_fields: Dict[str, str] = {}
        if new_color is not None:
            pending_fields["color"] = new_color

        if not pending_fields:
            raise ValueError("Provide at least one property to update.")

        return pending_fields

    def _validate_angle_policy(self, requested_fields: List[str]) -> Dict[str, EditRule]:
        if not self.angle_edit_policy:
            raise ValueError("Edit policy for angles is not configured.")

        validated_rules: Dict[str, EditRule] = {}
        for field in requested_fields:
            rule = self.angle_edit_policy.get_rule(field)
            if not rule:
                raise ValueError(f"Editing field '{field}' is not permitted for angles.")
            validated_rules[field] = rule

        return validated_rules

    def _validate_color_request(
        self,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
    ) -> None:
        if "color" in pending_fields and (new_color is None or not str(new_color).strip()):
            raise ValueError("Angle color cannot be empty.")

    def _apply_angle_updates(
        self,
        angle: Angle,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
    ) -> None:
        if "color" in pending_fields and new_color is not None:
            if hasattr(angle, "update_color") and callable(getattr(angle, "update_color")):
                angle.update_color(str(new_color))
            else:
                angle.color = str(new_color)

        remover = getattr(angle, "remove_svg_elements", None)
        if callable(remover):
            try:
                remover()
            except Exception:
                pass

    def handle_segment_updated(self, updated_segment_name: str) -> None:
        """
        Called when a segment (that an angle might depend on) is updated.
        This method should find all angles dependent on this segment and trigger their update/re-initialization.

        Args:
            updated_segment_name: The name of the segment that was updated.
        """
        if not hasattr(self.drawables, 'Angles') or not isinstance(self.drawables.Angles, list):
            return

        needs_redraw: bool = False
        for angle in cast(List["Drawable"], list(self.drawables.Angles)): # Iterate over a copy in case of modification
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

    def handle_segment_removed(self, removed_segment_name: str) -> None:
        """
        Called when a segment (that an angle might depend on) is removed.
        This method should find all angles dependent on this segment and remove them.

        Args:
            removed_segment_name: The name of the segment that was removed.
        """
        if not hasattr(self.drawables, 'Angles') or not isinstance(self.drawables.Angles, list):
            return

        # Collect names first to avoid modification issues while iterating
        angles_to_remove_names: List[str] = []
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

    def load_angles(self, angles_data: List[Dict[str, Any]]) -> None:
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
            
            # Resolve segments via drawable_manager, then construct Angle without model canvas logic
            args: Dict[str, Any] = angle_state.get('args', {})
            seg1_name = args.get('segment1_name')
            seg2_name = args.get('segment2_name')
            if not seg1_name or not seg2_name:
                continue
            segment1 = self.drawable_manager.get_segment_by_name(seg1_name)
            segment2 = self.drawable_manager.get_segment_by_name(seg2_name)
            if not segment1 or not segment2:
                continue

            angle_kwargs: Dict[str, Any] = {'is_reflex': args.get('is_reflex', False)}
            if 'color' in args:
                angle_kwargs['color'] = args['color']

            new_angle = Angle(segment1, segment2, **angle_kwargs)
            
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

    def get_angles_state(self) -> List[Dict[str, Any]]:
        """
        Returns a list of state dictionaries for all managed angles, for saving to workspace.

        Returns:
            A list of dictionaries, where each dict represents the state of an Angle.
        """
        if not hasattr(self.drawables, 'Angles') or not isinstance(self.drawables.Angles, list):
            return []
        return [angle.get_state() for angle in self.drawables.Angles if hasattr(angle, 'get_state') and callable(angle.get_state)]

    def clear_angles(self) -> None:
        """
        Removes all angles managed by this manager.
        """
        self.canvas.undo_redo_manager.archive()
        
        if not hasattr(self.drawables, 'Angles') or not isinstance(self.drawables.Angles, list):
            return

        angle_names_to_remove: List[str] = [angle.name for angle in cast(List["Drawable"], list(self.drawables.Angles)) if hasattr(angle, 'name')]
        
        for angle_name in angle_names_to_remove:
            self.delete_angle(angle_name) # This will handle individual draws and further dependencies.
        
        # A final draw call might be redundant if delete_angle always draws and handles all cleanup.
        # However, if delete_angle was optimized not to draw, this would be needed.
        if self.canvas.draw_enabled and not angle_names_to_remove: # Only draw if nothing was removed (and drawn individually)
            self.canvas.draw()
        elif self.canvas.draw_enabled and angle_names_to_remove: # If angles were removed, they handled their draw, one final comprehensive draw
             self.canvas.draw() 
