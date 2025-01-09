import os
import json
from datetime import datetime

class WorkspaceManager:
    def __init__(self, workspaces_dir="workspaces"):
        """Initialize the workspace manager.
        
        Args:
            workspaces_dir: Base directory for storing workspaces
        """
        self.workspaces_dir = workspaces_dir
        self.ensure_workspaces_dir()

    def ensure_workspaces_dir(self, test_dir=None):
        """Ensure the workspaces directory exists.
        
        Args:
            test_dir: Optional test directory path relative to workspaces_dir
        """
        target_dir = self.workspaces_dir if test_dir is None else os.path.join(self.workspaces_dir, test_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        return target_dir

    def get_workspace_path(self, name=None, test_dir=None):
        """Get the path for a workspace file.
        
        Args:
            name: Optional name for the workspace file
            test_dir: Optional test directory path relative to workspaces_dir
        """
        target_dir = self.ensure_workspaces_dir(test_dir)
        if name is None:
            # For current workspace, use a special name
            return os.path.join(target_dir, "current_workspace.json")
        return os.path.join(target_dir, f"{name}.json")

    def save_workspace(self, state, name=None, test_dir=None):
        """Save a workspace state to a file.
        
        Args:
            state: The state data to save
            name: Optional name for the workspace
            test_dir: Optional test directory path
            
        Returns:
            bool: True if save was successful, False otherwise.
        """
        try:
            workspace_data = {
                "metadata": {
                    "name": name or "current_workspace",
                    "last_modified": datetime.now().isoformat(),
                },
                "state": state
            }
            
            file_path = self.get_workspace_path(name, test_dir)
            with open(file_path, 'w') as f:
                json.dump(workspace_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving workspace: {str(e)}")
            return False

    def load_workspace(self, name=None, test_dir=None):
        """Load a workspace state from a file.
        
        Args:
            name: Optional name of the workspace to load
            test_dir: Optional test directory path
            
        Returns:
            dict: The loaded state data
        """
        file_path = self.get_workspace_path(name, test_dir)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No workspace found at {file_path}")
        
        with open(file_path, 'r') as f:
            workspace_data = json.load(f)
        
        return workspace_data["state"]

    def list_workspaces(self, test_dir=None):
        """List all saved workspaces.
        
        Args:
            test_dir: Optional test directory path relative to workspaces_dir
        
        Returns:
            A list of workspace names (without .json extension)
        """
        target_dir = self.ensure_workspaces_dir(test_dir)
        workspaces = []
        
        for filename in os.listdir(target_dir):
            if filename.endswith('.json'):
                name = filename[:-5]  # Remove .json extension
                if name != "current_workspace":  # Don't include the current workspace in the list
                    workspaces.append(name)
        
        return workspaces 