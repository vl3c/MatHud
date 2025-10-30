from __future__ import annotations

import json
import os
import unittest
from datetime import datetime
from typing import Any, Dict, cast

from server_tests.test_mocks import CanvasStateDict, MockCanvas
from static.workspace_manager import WORKSPACES_DIR, WorkspaceManager, WorkspaceState

TEST_DIR = "Test"


class TestWorkspaceManagement(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test environment before each test."""
        self.canvas: MockCanvas = MockCanvas(500, 500, draw_enabled=False)
        self.workspace_manager: WorkspaceManager = WorkspaceManager(WORKSPACES_DIR)
        self.workspace_manager.ensure_workspaces_dir(TEST_DIR)
        self.cleanup_test_workspaces()

    def tearDown(self) -> None:
        """Clean up after each test."""
        self.cleanup_test_workspaces()
        try:
            test_dir = os.path.join(WORKSPACES_DIR, TEST_DIR)
            os.rmdir(test_dir)
        except OSError:
            pass

    def cleanup_test_workspaces(self) -> None:
        """Remove test workspace files."""
        test_dir = os.path.join(WORKSPACES_DIR, TEST_DIR)
        if os.path.exists(test_dir):
            for filename in os.listdir(test_dir):
                os.remove(os.path.join(test_dir, filename))

    def test_save_workspace_without_name(self) -> None:
        """Test saving workspace without a name (current workspace)."""
        self.canvas.create_point(100, 100, "A")
        self.canvas.create_point(200, 200, "B")
        self.canvas.create_segment(100, 100, 200, 200, "AB")

        test_current = "test_current_workspace"
        success = self.workspace_manager.save_workspace(cast(WorkspaceState, self.canvas.get_canvas_state()), test_current, TEST_DIR)
        self.assertTrue(success, "Save workspace should return True on success")
        
        workspace_path = os.path.join(WORKSPACES_DIR, TEST_DIR, f"{test_current}.json")
        self.assertTrue(os.path.exists(workspace_path))
        
        with open(workspace_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["metadata"]["name"], test_current)
            self.assertIn("Points", data["state"])
            self.assertIn("Segments", data["state"])
            self.assertEqual(len(data["state"]["Points"]), 2)
            self.assertEqual(len(data["state"]["Segments"]), 1)

    def test_save_workspace_with_name(self) -> None:
        """Test saving workspace with a specific name."""
        self.canvas.create_circle(0, 0, 100, "C1")
        
        workspace_name = "test_circle_workspace"
        success = self.workspace_manager.save_workspace(cast(WorkspaceState, self.canvas.get_canvas_state()), workspace_name, TEST_DIR)
        self.assertTrue(success, "Save workspace should return True on success")
        
        workspace_path = os.path.join(WORKSPACES_DIR, TEST_DIR, f"{workspace_name}.json")
        self.assertTrue(os.path.exists(workspace_path))
        
        with open(workspace_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["metadata"]["name"], workspace_name)
            self.assertIn("Circles", data["state"])
            self.assertEqual(len(data["state"]["Circles"]), 1)

    def test_save_workspace_failure(self) -> None:
        """Test saving workspace with invalid conditions."""
        success = self.workspace_manager.save_workspace(None, test_dir=TEST_DIR)
        self.assertFalse(success, "Save workspace should return False when state is None")
        
        success = self.workspace_manager.save_workspace(cast(WorkspaceState, self.canvas.get_canvas_state()), "test/invalid/name", TEST_DIR)
        self.assertFalse(success, "Save workspace should return False with invalid name")
        
        if os.name != 'nt':  # Skip on Windows
            test_dir = os.path.join(WORKSPACES_DIR, TEST_DIR)
            original_mode = os.stat(test_dir).st_mode
            try:
                os.chmod(test_dir, 0o444)  # Read-only
                success = self.workspace_manager.save_workspace(cast(WorkspaceState, self.canvas.get_canvas_state()), "test_readonly", TEST_DIR)
                self.assertFalse(success, "Save workspace should return False with read-only directory")
            finally:
                os.chmod(test_dir, original_mode)

    def test_load_workspace(self) -> None:
        """Test loading a workspace."""
        self.canvas.create_triangle(0, 0, 100, 0, 50, 100, "ABC")
        workspace_name = "test_triangle_workspace"
        success = self.workspace_manager.save_workspace(cast(WorkspaceState, self.canvas.get_canvas_state()), workspace_name, TEST_DIR)
        self.assertTrue(success, "Initial save should succeed")
        
        self.canvas.clear()
        self.assertEqual(len(self.canvas.get_drawables()), 0)
        
        state = self.workspace_manager.load_workspace(workspace_name, TEST_DIR)
        
        state_dict = cast(Dict[str, Any], state)
        self.assertIn("Triangles", state_dict)
        self.assertEqual(len(state_dict["Triangles"]), 1)
        triangle = state_dict["Triangles"][0]
        self.assertEqual(triangle["name"], "ABC")

    def test_load_nonexistent_workspace(self) -> None:
        """Test loading a workspace that doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.workspace_manager.load_workspace("nonexistent_workspace", TEST_DIR)

    def test_load_workspace_invalid_json(self) -> None:
        """Test loading a workspace file with invalid JSON content."""
        workspace_name = "test_invalid_json_workspace"
        workspace_path = os.path.join(WORKSPACES_DIR, TEST_DIR, f"{workspace_name}.json")
        
        with open(workspace_path, 'w') as f:
            f.write("{\"name\": \"test\", \"state\": {this_is_not_valid_json}")

        with self.assertRaises((json.JSONDecodeError, ValueError)):
            self.workspace_manager.load_workspace(workspace_name, TEST_DIR)

    def test_load_workspace_incorrect_schema(self) -> None:
        """Test loading a workspace file with valid JSON but incorrect schema."""
        workspace_name = "test_incorrect_schema_workspace"
        workspace_path = os.path.join(WORKSPACES_DIR, TEST_DIR, f"{workspace_name}.json")

        # Case 1: Valid JSON, but missing the top-level 'state' key.
        malformed_data = {
            "metadata": {"name": workspace_name, "timestamp": "sometime"},
            "unexpected_top_level_key": {"Points": []} 
        }
        with open(workspace_path, 'w') as f:
            json.dump(malformed_data, f)

        with self.assertRaisesRegex(ValueError, "Error loading workspace: 'state'"):
            self.workspace_manager.load_workspace(workspace_name, TEST_DIR)

        # Case 2: 'state' key exists, but 'Points' (a drawable type) is not a list.
        workspace_name_2 = "test_points_not_list"
        workspace_path_2 = os.path.join(WORKSPACES_DIR, TEST_DIR, f"{workspace_name_2}.json")
        malformed_data_2 = {
            "metadata": {"name": workspace_name_2},
            "state": {"Points": "this should be a list"}
        }
        with open(workspace_path_2, 'w') as f:
            json.dump(malformed_data_2, f)
        
        loaded_state_2 = self.workspace_manager.load_workspace(workspace_name_2, TEST_DIR)
        loaded_state_2_dict = cast(Dict[str, Any], loaded_state_2)
        self.assertIsInstance(loaded_state_2_dict, dict, "Loaded state should be a dictionary.")
        self.assertIn("Points", loaded_state_2_dict)
        self.assertNotIsInstance(loaded_state_2_dict["Points"], list, "Points should not be a list in this malformed case.")
        self.assertEqual(loaded_state_2_dict["Points"], "this should be a list")

    def test_list_workspaces(self) -> None:
        """Test listing all workspaces."""
        workspace_names = ["test_ws1", "test_ws2", "test_ws3"]
        for name in workspace_names:
            success = self.workspace_manager.save_workspace(cast(WorkspaceState, self.canvas.get_canvas_state()), name, TEST_DIR)
            self.assertTrue(success, f"Failed to save workspace {name}")
        
        workspaces = self.workspace_manager.list_workspaces(TEST_DIR)
        
        for name in workspace_names:
            self.assertIn(name, workspaces)

    def test_list_workspaces_with_non_workspace_files(self) -> None:
        """Test listing workspaces when non-workspace files are present."""
        self.workspace_manager.save_workspace(cast(WorkspaceState, self.canvas.get_canvas_state()), "valid_ws_for_list_test", TEST_DIR)
        
        test_dir_path = os.path.join(WORKSPACES_DIR, TEST_DIR)
        with open(os.path.join(test_dir_path, "notes.txt"), 'w') as f:
            f.write("some notes")
        with open(os.path.join(test_dir_path, "image.jpg"), 'w') as f:
            f.write("fake image data")
        with open(os.path.join(test_dir_path, "invalid_structure.json"), 'w') as f:
            json.dump({"metadata": {"name": "invalid_structure"}}, f) # Missing 'state' key
        with open(os.path.join(test_dir_path, ".hiddenfile"), 'w') as f:
            f.write("hidden")
            
        workspaces = self.workspace_manager.list_workspaces(TEST_DIR)
        self.assertEqual(len(workspaces), 1, f"Expected 1 workspace, got {len(workspaces)}: {workspaces}")
        self.assertIn("valid_ws_for_list_test", workspaces)
        self.assertNotIn("notes.txt", workspaces)
        self.assertNotIn("image.jpg", workspaces)
        self.assertNotIn("invalid_structure.json", workspaces)
        self.assertNotIn(".hiddenfile", workspaces)

    def test_list_workspaces_empty_directory(self) -> None:
        """Test listing workspaces from an empty directory."""
        # Ensure the directory is clean. setUp usually does this, but explicit call for clarity.
        self.cleanup_test_workspaces()
        workspaces = self.workspace_manager.list_workspaces(TEST_DIR)
        self.assertEqual(len(workspaces), 0)

    def test_workspace_with_computations(self) -> None:
        """Test saving and loading workspace with computations."""
        self.canvas.add_computation("2+2", 4)
        self.canvas.add_computation("sin(pi/2)", 1)
        
        workspace_name = "test_computations"
        success = self.workspace_manager.save_workspace(cast(WorkspaceState, self.canvas.get_canvas_state()), workspace_name, TEST_DIR)
        self.assertTrue(success, "Failed to save workspace with computations")
        
        self.canvas.clear()
        self.assertEqual(len(self.canvas.computations), 0)
        
        state = self.workspace_manager.load_workspace(workspace_name, TEST_DIR)
        state_dict = cast(Dict[str, Any], state)
        
        self.assertIn("computations", state_dict)
        self.assertEqual(len(state_dict["computations"]), 2)
        computations_list = cast(list, state_dict["computations"])
        expressions = [cast(Dict[str, Any], comp)["expression"] for comp in computations_list]
        results = [cast(Dict[str, Any], comp)["result"] for comp in computations_list]
        self.assertIn("2+2", expressions)
        self.assertIn("sin(pi/2)", expressions)
        self.assertIn(4, results)
        self.assertIn(1, results)

    def test_save_complex_workspace(self) -> None:
        """Test saving workspace with multiple types of objects."""
        self.canvas.create_point(0, 0, "O")
        self.canvas.create_circle(0, 0, 100, "C1")
        self.canvas.create_rectangle(-50, -50, 50, 50, "R1")
        self.canvas.draw_function("sin(x)", "f1")
        self.canvas.add_computation("area", 10000)
        
        workspace_name = "test_complex"
        success = self.workspace_manager.save_workspace(cast(WorkspaceState, self.canvas.get_canvas_state()), workspace_name, TEST_DIR)
        self.assertTrue(success, "Failed to save complex workspace")
        
        workspace_path = os.path.join(WORKSPACES_DIR, TEST_DIR, f"{workspace_name}.json")
        with open(workspace_path, 'r') as f:
            data = json.load(f)
            state = data["state"]
            self.assertIn("Points", state)
            self.assertIn("Circles", state)
            self.assertIn("Rectangles", state)
            self.assertIn("Functions", state)
            self.assertIn("computations", state)
            self.assertEqual(len(state["Points"]), 1)
            self.assertEqual(len(state["Circles"]), 1)
            self.assertEqual(len(state["Rectangles"]), 1)
            self.assertEqual(len(state["Functions"]), 1)
            self.assertEqual(len(state["computations"]), 1)

    def test_save_and_load_preserves_state_integrity(self) -> None:
        """Test that saving and then loading a workspace recreates the exact same state."""
        # 1. Create a complex canvas state
        pointA_coords = (10, 20)
        pointB_coords = (30, 40)
        pointC_coords = (50, 50) # For Circle
        pointD_coords = (5, 15)  # For Vector
        pointE_coords = (25, 35) # For Vector

        self.canvas.create_point(pointA_coords[0], pointA_coords[1], "A")
        self.canvas.create_point(pointB_coords[0], pointB_coords[1], "B")
        self.canvas.create_point(pointC_coords[0], pointC_coords[1], "C")
        self.canvas.create_point(pointD_coords[0], pointD_coords[1], "D")
        self.canvas.create_point(pointE_coords[0], pointE_coords[1], "E")

        self.canvas.create_segment(pointA_coords[0], pointA_coords[1], pointB_coords[0], pointB_coords[1], "AB")
        self.canvas.create_circle(pointC_coords[0], pointC_coords[1], 25, "CircleC") 
        self.canvas.create_vector(pointD_coords[0], pointD_coords[1], pointE_coords[0], pointE_coords[1], "DE")
        self.canvas.draw_function("x**2", "f1")
        self.canvas.add_computation("my_calc", 123.45)
        
        original_state = self.canvas.get_canvas_state()
        workspace_name = "test_integrity_workspace_canvas_methods"

        # 2. Save the workspace
        save_success = self.workspace_manager.save_workspace(cast(WorkspaceState, original_state), workspace_name, TEST_DIR)
        self.assertTrue(save_success, "Saving the integrity test workspace should succeed.")

        # 3. Load the workspace
        loaded_state = self.workspace_manager.load_workspace(workspace_name, TEST_DIR)
        
        # 4. Deeply compare the loaded state with the original state.
        self._assert_states_equal_after_sorting(original_state, loaded_state, 
                         "Loaded workspace state (from canvas methods) does not match the original state.")

    def _assert_states_equal_after_sorting(self, state1: CanvasStateDict, state2: WorkspaceState, message: str) -> None:
        """Helper to compare two canvas states after deep copying and sorting their lists."""
        import copy
        
        # Nested sort_state_lists_recursive function
        def sort_state_lists_recursive(current_state_item: Any) -> Any:
            if isinstance(current_state_item, dict):
                # Recursively process dictionary values
                for key, value in current_state_item.items():
                    current_state_item[key] = sort_state_lists_recursive(value)
            elif isinstance(current_state_item, list):
                # Attempt to sort lists.
                try:
                    # Try to sort by tuple of items if elements are dicts, ensuring consistent order
                    if all(isinstance(x, dict) for x in current_state_item):
                        current_state_item.sort(key=lambda x: tuple(sorted(x.items())))
                    else:
                        # For lists of non-dicts, attempt direct sort
                        current_state_item.sort()
                except TypeError:
                    # Fallback: if direct sort or dict item sort fails (e.g., mixed un-sortable types),
                    # try sorting by string representation.
                    try:
                        current_state_item.sort(key=str)
                    except TypeError:
                        # If all sorting attempts fail, leave the list as is.
                        pass 
            return current_state_item

        # Deep copy states to avoid modifying originals during sorting
        # Ensure that metadata is handled consistently before comparison.
        # It's often removed from the loaded state before comparison.
        
        state1_copy = json.loads(json.dumps(state1))
        state2_copy = json.loads(json.dumps(state2))

        # Remove 'metadata' key if present, as it's not part of the core state to compare
        state1_copy_dict = cast(Dict[str, Any], state1_copy)
        state2_copy_dict = cast(Dict[str, Any], state2_copy)
        if isinstance(state1_copy_dict, dict) and "metadata" in state1_copy_dict:
            del state1_copy_dict["metadata"]
        if isinstance(state2_copy_dict, dict) and "metadata" in state2_copy_dict:
            del state2_copy_dict["metadata"]

        state1_sorted = sort_state_lists_recursive(state1_copy)
        state2_sorted = sort_state_lists_recursive(state2_copy)
        
        self.assertEqual(state1_sorted, state2_sorted, message)

    def test_save_and_load_mock_state_preserves_integrity_with_vectors(self) -> None:
        """Test that saving and loading a mock dictionary state with vectors preserves its integrity."""
        # 1. Define a complex canvas state using a dictionary
        mock_canvas_state = {
            "Points": [
                {"name": "M_A", "args": {"position": {"x": 100, "y": 120}}},
                {"name": "M_B", "args": {"position": {"x": 130, "y": 140}}},
                {"name": "M_C", "args": {"position": {"x": 150, "y": 160}}},
                {"name": "M_D", "args": {"position": {"x": 105, "y": 115}}},
                {"name": "M_E", "args": {"position": {"x": 125, "y": 135}}}
            ],
            "Segments": [
                {"name": "M_AB", "args": {"p1": "M_A", "p2": "M_B"}}
            ],
            "Circles": [
                {"name": "M_CircleC", "args": {"center": "M_C", "radius": 20}}
            ],
            "Vectors": [
                {"name": "M_DE", "args": {"origin": "M_D", "tip": "M_E"}}
            ],
            "Functions": [
                {"name": "M_Func1", "args": {"function_string": "sin(x)", "left_bound": -10, "right_bound": 10}}
            ],
            "computations": [
                {"expression": "10+20", "result": 30},
                {"expression": "cos(0)", "result": 1.0} # MockCanvas computation results are floats
            ]
        }

        workspace_name = "test_integrity_mock_state_vectors"
        save_success = self.workspace_manager.save_workspace(cast(WorkspaceState, mock_canvas_state), workspace_name, TEST_DIR)
        self.assertTrue(save_success, "Saving the mock state integrity test workspace should succeed.")

        loaded_state = self.workspace_manager.load_workspace(workspace_name, TEST_DIR)
        loaded_state_dict = cast(Dict[str, Any], loaded_state)
        
        self.assertIsNotNone(loaded_state_dict, "Loaded state content should not be None.")
        self.assertIsInstance(loaded_state_dict, dict, "Loaded state should be a dictionary.")

        if "metadata" in loaded_state_dict:
            del loaded_state_dict["metadata"]

        # Quick checks for key elements
        loaded_points_list = cast(list, loaded_state_dict.get("Points", []))
        loaded_points_dict = {cast(Dict[str, Any], p)['name']: p for p in loaded_points_list}
        self.assertIn("M_E", loaded_points_dict)
        self.assertEqual(cast(Dict[str, Any], loaded_points_dict["M_E"])["args"]["position"]["x"], 125)

        loaded_vectors_list = cast(list, loaded_state_dict.get("Vectors", []))
        loaded_vectors_dict = {cast(Dict[str, Any], v)['name']: v for v in loaded_vectors_list}
        self.assertIn("M_DE", loaded_vectors_dict)
        self.assertEqual(cast(Dict[str, Any], loaded_vectors_dict["M_DE"])["args"]["origin"], "M_D")

        # Use the helper method for full comparison
        self._assert_states_equal_after_sorting(cast(CanvasStateDict, mock_canvas_state), loaded_state_dict,
                         "Loaded mock workspace state with vectors does not match the original mock state.")

    def test_save_and_load_empty_workspace(self) -> None:
        """Test saving and loading an empty workspace."""
        # 1. Canvas is already empty by default after setUp
        self.assertEqual(len(self.canvas.get_drawables()), 0)
        self.assertEqual(len(self.canvas.computations), 0)

        original_empty_state = self.canvas.get_canvas_state()
        workspace_name = "test_empty_workspace"

        # 2. Save the empty workspace
        save_success = self.workspace_manager.save_workspace(cast(WorkspaceState, original_empty_state), workspace_name, TEST_DIR)
        self.assertTrue(save_success, "Saving an empty workspace should succeed.")

        # 3. Load the empty workspace
        loaded_state = self.workspace_manager.load_workspace(workspace_name, TEST_DIR)
        loaded_state_dict = cast(Dict[str, Any], loaded_state)

        # 4. Deeply compare the loaded state with the original empty state.
        if "metadata" in loaded_state_dict:
            del loaded_state_dict["metadata"]
        
        # Note: Relies on MockCanvas.get_canvas_state() returning a consistent empty structure.
        self.assertEqual(original_empty_state, loaded_state_dict, 
                         "Loaded empty workspace state does not match the original empty state.")

    def test_delete_workspace(self) -> None:
        """Test deleting a workspace."""
        self.canvas.create_point(100, 100, "A")
        workspace_name = "test_delete_ws"
        success = self.workspace_manager.save_workspace(cast(WorkspaceState, self.canvas.get_canvas_state()), workspace_name, TEST_DIR)
        self.assertTrue(success, "Initial save should succeed")
        
        workspace_path = os.path.join(WORKSPACES_DIR, TEST_DIR, f"{workspace_name}.json")
        self.assertTrue(os.path.exists(workspace_path))
        
        success = self.workspace_manager.delete_workspace(workspace_name, TEST_DIR)
        self.assertTrue(success, "Delete workspace should return True on success")
        
        self.assertFalse(os.path.exists(workspace_path))
        
        workspaces = self.workspace_manager.list_workspaces(TEST_DIR)
        self.assertNotIn(workspace_name, workspaces)

    def test_delete_nonexistent_workspace(self) -> None:
        """Test deleting a workspace that doesn't exist."""
        success = self.workspace_manager.delete_workspace("nonexistent_workspace", TEST_DIR)
        self.assertFalse(success, "Delete workspace should return False for non-existent workspace")

    def test_delete_workspace_with_invalid_name(self) -> None:
        """Test deleting a workspace with invalid name."""
        success = self.workspace_manager.delete_workspace(cast(str, None), TEST_DIR)
        self.assertFalse(success, "Delete workspace should return False for None name")
        
        success = self.workspace_manager.delete_workspace("test/invalid/name", TEST_DIR)
        self.assertFalse(success, "Delete workspace should return False for invalid name")

if __name__ == '__main__':
    unittest.main() 