"""
MatHud Server-Side Workspace Management

Handles workspace file operations for saving and loading canvas states.
Provides secure file operations with path validation and JSON-based storage.

Dependencies:
    - os: File system operations and path validation
    - json: Workspace state serialization and deserialization
    - re: Workspace name validation with regex
    - datetime: Timestamp generation for metadata
"""

import os
import json
import re
from datetime import datetime

WORKSPACES_DIR = "workspaces"

class WorkspaceManager:
    """Server-side workspace file operations manager.
    
    Manages saving, loading, listing, and deleting workspace files with
    security validation and JSON-based state storage with metadata.
    """
    
    def __init__(self, workspaces_dir=WORKSPACES_DIR):
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
            
        if not re.match(r'^[\w-]+\Z$', name):  # Only allow alphanumeric characters, underscores, and hyphens
            return False
            
        return True

    def _is_path_in_workspace_dir(self, path):
        """Check if a path is within the workspaces directory.
        
        Args:
            path: The path to check
            
        Returns:
            bool: True if the path is within workspaces directory, False otherwise
        """
        abs_path = os.path.abspath(path)
        abs_workspace_dir = os.path.abspath(self.workspaces_dir)
        
        try:
            os.path.commonpath([abs_path, abs_workspace_dir])
            return abs_path.startswith(abs_workspace_dir)
        except ValueError:
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(target_dir, f"current_workspace_{timestamp}.json")
        else:
            file_path = os.path.join(target_dir, f"{name}.json")
            
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
            
            file_path = self.get_workspace_path(name, test_dir)
            with open(file_path, 'w') as f:
                json.dump(workspace_data, f, indent=2)
            
            return True
        except (ValueError, OSError) as e:
            print(f"Error saving workspace: {str(e)}")
            return False

    def _get_most_recent_current_workspace(self, test_dir=None):
        """Get the path of the most recent current workspace.
        
        Args:
            test_dir: Optional test directory path
            
        Returns:
            str: Path to the most recent current workspace file
            
        Raises:
            FileNotFoundError: If no current workspace exists
        """
        target_dir = self.ensure_workspaces_dir(test_dir)
        current_workspaces = []
        
        for filename in os.listdir(target_dir):
            if filename.startswith('current_workspace_') and filename.endswith('.json'):
                file_path = os.path.join(target_dir, filename)
                current_workspaces.append(file_path)
        
        if not current_workspaces:
            raise FileNotFoundError("No current workspace found")
            
        current_workspaces.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return current_workspaces[0]

    def load_workspace(self, name=None, test_dir=None):
        """Load a workspace state from a file.
        
        Args:
            name: Optional name of the workspace to load
            test_dir: Optional test directory path
            
        Returns:
            dict: The loaded state data
        """
        try:
            if name is None:
                file_path = self._get_most_recent_current_workspace(test_dir)
            else:
                file_path = self.get_workspace_path(name, test_dir)
                
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No workspace found at {file_path}")
            
            with open(file_path, 'r') as f:
                workspace_data = json.load(f)
            
            return workspace_data["state"]
        except FileNotFoundError:
            raise
        except Exception as e:
            raise ValueError(f"Error loading workspace: {str(e)}")

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
            if filename.endswith(".json"):
                name_without_extension = filename[:-5]
                if name_without_extension.startswith("current_workspace_"):
                    continue

                file_path = os.path.join(target_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "state" in data and "metadata" in data:
                        if data.get("metadata", {}).get("name") == name_without_extension:
                            workspaces.append(name_without_extension)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"Skipping file {filename} due to error: {e}")
                    pass
        return workspaces

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
                
            file_path = self.get_workspace_path(name, test_dir)
            if not os.path.exists(file_path):
                return False
                
            if not self._is_path_in_workspace_dir(file_path):
                return False
                
            os.remove(file_path)
            return True
        except (ValueError, OSError) as e:
            print(f"Error deleting workspace: {str(e)}")
            return False 