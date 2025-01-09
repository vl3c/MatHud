import os
import json
import re
from datetime import datetime

class WorkspaceManager:
    def __init__(self, workspaces_dir="workspaces"):
        """Initialize the workspace manager.
        
        Args:
            workspaces_dir: Base directory for storing workspaces
        """
        self.workspaces_dir = os.path.abspath(workspaces_dir)
        self.ensure_workspaces_dir()

    def _is_safe_workspace_name(self, name):
        """Check if a workspace name is safe to use.
        
        Args:
            name: The workspace name to check
            
        Returns:
            bool: True if the name is safe, False otherwise
        """
        if not name or not isinstance(name, str):
            return False
            
        # Only allow alphanumeric characters, underscores, and hyphens
        if not re.match(r'^[\w\-]+$', name):
            return False
            
        return True

    def _is_path_in_workspace_dir(self, path):
        """Check if a path is within the workspaces directory.
        
        Args:
            path: The path to check
            
        Returns:
            bool: True if the path is within workspaces directory, False otherwise
        """
        # Convert both paths to absolute and normalize them
        abs_path = os.path.abspath(path)
        abs_workspace_dir = os.path.abspath(self.workspaces_dir)
        
        # Check if the normalized path starts with the workspace directory
        try:
            os.path.commonpath([abs_path, abs_workspace_dir])
            return abs_path.startswith(abs_workspace_dir)
        except ValueError:  # Different drives or invalid path
            return False

    def ensure_workspaces_dir(self, test_dir=None):
        """Ensure the workspaces directory exists.
        
        Args:
            test_dir: Optional test directory path relative to workspaces_dir
        """
        if test_dir is not None and not self._is_safe_workspace_name(test_dir):
            raise ValueError("Invalid test directory name")
            
        target_dir = self.workspaces_dir if test_dir is None else os.path.join(self.workspaces_dir, test_dir)
        target_dir = os.path.abspath(target_dir)
        
        if not self._is_path_in_workspace_dir(target_dir):
            raise ValueError("Target directory must be within workspace directory")
            
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        return target_dir

    def get_workspace_path(self, name=None, test_dir=None):
        """Get the path for a workspace file.
        
        Args:
            name: Optional name for the workspace file
            test_dir: Optional test directory path relative to workspaces_dir
        """
        if name is not None and not self._is_safe_workspace_name(name):
            raise ValueError("Invalid workspace name")
            
        target_dir = self.ensure_workspaces_dir(test_dir)
        
        if name is None:
            # For current workspace, use a special name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(target_dir, f"current_workspace_{timestamp}.json")
        else:
            file_path = os.path.join(target_dir, f"{name}.json")
            
        # Final safety check
        if not self._is_path_in_workspace_dir(file_path):
            raise ValueError("Workspace path must be within workspace directory")
            
        return file_path

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
            if state is None:
                return False
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_data = {
                "metadata": {
                    "name": name or f"current_workspace_{timestamp}",
                    "last_modified": datetime.now().isoformat(),
                },
                "state": state
            }
            
            file_path = self.get_workspace_path(name, test_dir)  # This now includes security checks
            with open(file_path, 'w') as f:
                json.dump(workspace_data, f, indent=2)
            
            return True
        except (ValueError, OSError) as e:
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
        file_path = self.get_workspace_path(name, test_dir)  # This now includes security checks
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
        target_dir = self.ensure_workspaces_dir(test_dir)  # This now includes security checks
        workspaces = []
        
        for filename in os.listdir(target_dir):
            if filename.endswith('.json'):
                name = filename[:-5]  # Remove .json extension
                # Only include filenames that would be valid workspace names
                if self._is_safe_workspace_name(name):
                    workspaces.append(name)
        
        return sorted(workspaces)  # Sort the list for consistent ordering 

    def delete_workspace(self, name, test_dir=None):
        """Delete a workspace file.
        
        Args:
            name: Name of the workspace to delete
            test_dir: Optional test directory path
            
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            if not self._is_safe_workspace_name(name):
                return False
                
            file_path = self.get_workspace_path(name, test_dir)  # This now includes security checks
            if not os.path.exists(file_path):
                return False
                
            # Final safety check before deletion
            if not self._is_path_in_workspace_dir(file_path):
                return False
                
            os.remove(file_path)
            return True
        except (ValueError, OSError) as e:
            print(f"Error deleting workspace: {str(e)}")
            return False 