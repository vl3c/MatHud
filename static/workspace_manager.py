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

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Union, cast

WORKSPACES_DIR = "workspaces"
CURRENT_WORKSPACE_SCHEMA_VERSION = 1

JsonPrimitive = Union[str, int, float, bool, None]
JsonValue = Union[JsonPrimitive, Dict[str, "JsonValue"], List["JsonValue"]]
JsonObject = Dict[str, JsonValue]
WorkspaceState = JsonValue


class WorkspaceMetadata(TypedDict):
    name: str
    last_modified: str
    schema_version: int


class WorkspaceRecord(TypedDict):
    metadata: WorkspaceMetadata
    state: WorkspaceState


class WorkspaceManager:
    """Server-side workspace file operations manager.

    Manages saving, loading, listing, and deleting workspace files with
    security validation and JSON-based state storage with metadata.
    """

    def __init__(self, workspaces_dir: str = WORKSPACES_DIR):
        """Initialize the workspace manager.

        Args:
            workspaces_dir: Base directory for storing workspaces
        """
        self.workspaces_dir = os.path.abspath(workspaces_dir)
        self.ensure_workspaces_dir()

    def _is_safe_workspace_name(self, name: Optional[str]) -> bool:
        """Check if a workspace name is safe to use.

        Args:
            name: The workspace name to check

        Returns:
            bool: True if the name is safe, False otherwise
        """
        if not name or not isinstance(name, str):
            return False

        if not re.match(r"^[\w-]+\Z$", name):  # Only allow alphanumeric characters, underscores, and hyphens
            return False

        return True

    def _is_path_in_workspace_dir(self, path: str) -> bool:
        """Check if a path is within the workspaces directory.

        Args:
            path: The path to check

        Returns:
            bool: True if the path is within workspaces directory, False otherwise
        """
        abs_path = os.path.abspath(path)
        abs_workspace_dir = os.path.abspath(self.workspaces_dir)

        try:
            common_prefix = os.path.commonpath([abs_path, abs_workspace_dir])
        except ValueError:
            return False
        return common_prefix == abs_workspace_dir

    def ensure_workspaces_dir(self, test_dir: Optional[str] = None) -> str:
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

    def get_workspace_path(self, name: Optional[str] = None, test_dir: Optional[str] = None) -> str:
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

    def save_workspace(
        self,
        state: WorkspaceState,
        name: Optional[str] = None,
        test_dir: Optional[str] = None,
    ) -> bool:
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
            workspace_data: WorkspaceRecord = {
                "metadata": {
                    "name": name or f"current_workspace_{timestamp}",
                    "last_modified": datetime.now().isoformat(),
                    "schema_version": CURRENT_WORKSPACE_SCHEMA_VERSION,
                },
                "state": state,
            }

            file_path = self.get_workspace_path(name, test_dir)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(workspace_data, f, indent=2)

            return True
        except (ValueError, OSError) as e:
            print(f"Error saving workspace: {str(e)}")
            return False

    def _get_most_recent_current_workspace(self, test_dir: Optional[str] = None) -> str:
        """Get the path of the most recent current workspace.

        Args:
            test_dir: Optional test directory path

        Returns:
            str: Path to the most recent current workspace file

        Raises:
            FileNotFoundError: If no current workspace exists
        """
        target_dir = self.ensure_workspaces_dir(test_dir)
        current_workspaces: List[str] = []

        for filename in os.listdir(target_dir):
            if filename.startswith("current_workspace_") and filename.endswith(".json"):
                file_path = os.path.join(target_dir, filename)
                current_workspaces.append(file_path)

        if not current_workspaces:
            raise FileNotFoundError("No current workspace found")

        current_workspaces.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return current_workspaces[0]

    def load_workspace(self, name: Optional[str] = None, test_dir: Optional[str] = None) -> WorkspaceState:
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

            with open(file_path, "r", encoding="utf-8") as f:
                workspace_data_raw: JsonValue = json.load(f)

            if not isinstance(workspace_data_raw, dict):
                raise ValueError("Workspace file is not a JSON object")
            normalized = self._normalize_and_migrate_workspace_record(
                workspace_data_raw,
                workspace_name=(name or "current"),
            )
            return normalized["state"]
        except FileNotFoundError:
            raise
        except Exception as e:
            raise ValueError(f"Error loading workspace: {str(e)}")

    def _normalize_and_migrate_workspace_record(
        self,
        workspace_data_raw: JsonObject,
        workspace_name: str,
    ) -> WorkspaceRecord:
        """Normalize workspace schema and migrate legacy versions to current."""
        if "state" not in workspace_data_raw:
            if "metadata" in workspace_data_raw:
                raise ValueError("Workspace record missing 'state'")
            # Legacy format: top-level object is the state payload.
            state = cast(WorkspaceState, workspace_data_raw)
            metadata: Dict[str, JsonValue] = {}
            schema_version = 0
        else:
            state = cast(WorkspaceState, workspace_data_raw["state"])
            metadata_raw = workspace_data_raw.get("metadata")
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            schema_version = self._parse_schema_version(metadata.get("schema_version"))

        migrated_state = self._migrate_state(state, schema_version)

        metadata_name_raw = metadata.get("name")
        metadata_name = (
            metadata_name_raw if isinstance(metadata_name_raw, str) and metadata_name_raw else workspace_name
        )
        metadata_last_modified_raw = metadata.get("last_modified")
        metadata_last_modified = (
            metadata_last_modified_raw
            if isinstance(metadata_last_modified_raw, str) and metadata_last_modified_raw
            else datetime.now().isoformat()
        )

        return {
            "metadata": {
                "name": metadata_name,
                "last_modified": metadata_last_modified,
                "schema_version": CURRENT_WORKSPACE_SCHEMA_VERSION,
            },
            "state": migrated_state,
        }

    def _parse_schema_version(self, schema_version: JsonValue) -> int:
        """Parse schema_version from metadata, defaulting to 0 for legacy files."""
        if isinstance(schema_version, int):
            return schema_version
        if isinstance(schema_version, str):
            try:
                return int(schema_version.strip())
            except ValueError:
                return 0
        return 0

    def _migrate_state(self, state: WorkspaceState, schema_version: int) -> WorkspaceState:
        """Apply forward migrations from older schema versions."""
        if schema_version <= 0:
            return state
        if schema_version > CURRENT_WORKSPACE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported workspace schema_version: {schema_version} (current: {CURRENT_WORKSPACE_SCHEMA_VERSION})"
            )
        return state

    def list_workspaces(self, test_dir: Optional[str] = None) -> List[str]:
        """List all saved workspaces.

        Args:
            test_dir: Optional test directory path relative to workspaces_dir

        Returns:
            A list of workspace names (without .json extension)
        """
        target_dir = self.ensure_workspaces_dir(test_dir)
        workspaces: List[str] = []

        for filename in os.listdir(target_dir):
            if filename.endswith(".json"):
                name_without_extension = filename[:-5]
                if name_without_extension.startswith("current_workspace_"):
                    continue

                file_path = os.path.join(target_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data_raw: JsonValue = json.load(f)
                    if isinstance(data_raw, dict):
                        metadata_candidate = data_raw.get("metadata")
                        has_state_key = "state" in data_raw
                        if (
                            isinstance(metadata_candidate, dict)
                            and "name" in metadata_candidate
                            and has_state_key
                            and metadata_candidate.get("name") == name_without_extension
                        ):
                            workspaces.append(name_without_extension)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"Skipping file {filename} due to error: {e}")
                    pass
        return workspaces

    def delete_workspace(self, name: str, test_dir: Optional[str] = None) -> bool:
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
