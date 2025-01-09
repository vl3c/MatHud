import os
import json
from datetime import datetime

WORKSPACES_DIR = "workspaces"

def ensure_workspaces_dir(test_dir=None):
    """Ensure the workspaces directory exists.
    
    Args:
        test_dir: Optional test directory path relative to WORKSPACES_DIR
    """
    target_dir = WORKSPACES_DIR if test_dir is None else os.path.join(WORKSPACES_DIR, test_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir

def get_workspace_path(name=None, test_dir=None):
    """Get the path for a workspace file.
    
    Args:
        name: Optional name for the workspace file
        test_dir: Optional test directory path relative to WORKSPACES_DIR
    """
    target_dir = ensure_workspaces_dir(test_dir)
    if name is None:
        # For current workspace, use a special name
        return os.path.join(target_dir, "current_workspace.json")
    return os.path.join(target_dir, f"{name}.json")

def save_workspace(canvas, name=None, test_dir=None):
    """Save the current workspace state to a file.
    
    Args:
        canvas: The Canvas instance containing the current state
        name: Optional name for the workspace. If None, saves to current workspace.
        test_dir: Optional test directory path relative to WORKSPACES_DIR
        
    Returns:
        bool: True if save was successful, False otherwise.
    """
    try:
        # Prepare workspace data
        workspace_data = {
            "metadata": {
                "name": name or "current_workspace",
                "last_modified": datetime.now().isoformat(),
            },
            "state": canvas.get_canvas_state()
        }
        
        # Save to file
        file_path = get_workspace_path(name, test_dir)
        with open(file_path, 'w') as f:
            json.dump(workspace_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving workspace: {str(e)}")
        return False

def load_workspace(canvas, name=None, test_dir=None):
    """Load a workspace from a file.
    
    Args:
        canvas: The Canvas instance to load the state into
        name: Optional name of the workspace to load. If None, loads current workspace.
        test_dir: Optional test directory path relative to WORKSPACES_DIR
    """
    file_path = get_workspace_path(name, test_dir)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No workspace found at {file_path}")
    
    with open(file_path, 'r') as f:
        workspace_data = json.load(f)
    
    # Clear current canvas state
    canvas.clear()
    
    # Load the state
    state = workspace_data["state"]
    
    # Restore drawables
    for category, items in state.items():
        if category == "computations":
            for comp in items:
                canvas.add_computation(comp["expression"], comp["result"])
            continue
            
        if not category.endswith('s'):  # Skip non-collection keys
            continue
            
        category_name = category[:-1]  # Remove 's' to get the class name
        for item_state in items:
            if category_name == "Point":
                canvas.create_point(
                    item_state["x"], 
                    item_state["y"],
                    name=item_state.get("name", "")
                )
            elif category_name == "Segment":
                canvas.create_segment(
                    item_state["point1"]["x"],
                    item_state["point1"]["y"],
                    item_state["point2"]["x"],
                    item_state["point2"]["y"],
                    name=item_state.get("name", "")
                )
            elif category_name == "Triangle":
                canvas.create_triangle(
                    item_state["point1"]["x"],
                    item_state["point1"]["y"],
                    item_state["point2"]["x"],
                    item_state["point2"]["y"],
                    item_state["point3"]["x"],
                    item_state["point3"]["y"],
                    name=item_state.get("name", "")
                )
            elif category_name == "Rectangle":
                canvas.create_rectangle(
                    item_state["point1"]["x"],
                    item_state["point1"]["y"],
                    item_state["point3"]["x"],
                    item_state["point3"]["y"],
                    name=item_state.get("name", "")
                )
            elif category_name == "Circle":
                canvas.create_circle(
                    item_state["center"]["x"],
                    item_state["center"]["y"],
                    item_state["radius"],
                    name=item_state.get("name", "")
                )
            elif category_name == "Ellipse":
                canvas.create_ellipse(
                    item_state["center"]["x"],
                    item_state["center"]["y"],
                    item_state["radius_x"],
                    item_state["radius_y"],
                    rotation_angle=item_state.get("rotation_angle", 0),
                    name=item_state.get("name", "")
                )
            elif category_name == "Function":
                canvas.draw_function(
                    item_state["function_string"],
                    name=item_state.get("name", ""),
                    left_bound=item_state.get("left_bound"),
                    right_bound=item_state.get("right_bound")
                )
    
    return f"Workspace loaded from {file_path}"

def list_workspaces(test_dir=None):
    """List all saved workspaces.
    
    Args:
        test_dir: Optional test directory path relative to WORKSPACES_DIR
    
    Returns:
        A list of workspace names (without .json extension)
    """
    target_dir = ensure_workspaces_dir(test_dir)
    workspaces = []
    
    for filename in os.listdir(target_dir):
        if filename.endswith('.json'):
            name = filename[:-5]  # Remove .json extension
            if name != "current_workspace":  # Don't include the current workspace in the list
                workspaces.append(name)
    
    return workspaces 