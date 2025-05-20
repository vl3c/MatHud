import unittest
import os
import json
from datetime import datetime
from ServerTests.test_mocks import MockCanvas
from static.workspace_manager import WorkspaceManager, WORKSPACES_DIR

TEST_DIR = "Test"

class TestWorkspaceManagement(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.canvas = MockCanvas(500, 500, draw_enabled=False)
        self.workspace_manager = WorkspaceManager(WORKSPACES_DIR)
        self.workspace_manager.ensure_workspaces_dir(TEST_DIR)
        self.cleanup_test_workspaces()

    def tearDown(self):
        """Clean up after each test."""
        self.cleanup_test_workspaces()
        try:
            test_dir = os.path.join(WORKSPACES_DIR, TEST_DIR)
            os.rmdir(test_dir)
        except OSError:
            pass

    def cleanup_test_workspaces(self):
        """Remove test workspace files."""
        test_dir = os.path.join(WORKSPACES_DIR, TEST_DIR)
        if os.path.exists(test_dir):
            for filename in os.listdir(test_dir):
                os.remove(os.path.join(test_dir, filename))

    def test_save_workspace_without_name(self):
        """Test saving workspace without a name (current workspace)."""
        self.canvas.create_point(100, 100, "A")
        self.canvas.create_point(200, 200, "B")
        self.canvas.create_segment(100, 100, 200, 200, "AB")

        test_current = "test_current_workspace"
        success = self.workspace_manager.save_workspace(self.canvas.get_canvas_state(), test_current, TEST_DIR)
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

    def test_save_workspace_with_name(self):
        """Test saving workspace with a specific name."""
        self.canvas.create_circle(0, 0, 100, "C1")
        
        workspace_name = "test_circle_workspace"
        success = self.workspace_manager.save_workspace(self.canvas.get_canvas_state(), workspace_name, TEST_DIR)
        self.assertTrue(success, "Save workspace should return True on success")
        
        workspace_path = os.path.join(WORKSPACES_DIR, TEST_DIR, f"{workspace_name}.json")
        self.assertTrue(os.path.exists(workspace_path))
        
        with open(workspace_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["metadata"]["name"], workspace_name)
            self.assertIn("Circles", data["state"])
            self.assertEqual(len(data["state"]["Circles"]), 1)

    def test_save_workspace_failure(self):
        """Test saving workspace with invalid conditions."""
        success = self.workspace_manager.save_workspace(None, test_dir=TEST_DIR)
        self.assertFalse(success, "Save workspace should return False when state is None")
        
        success = self.workspace_manager.save_workspace(self.canvas.get_canvas_state(), "test/invalid/name", TEST_DIR)
        self.assertFalse(success, "Save workspace should return False with invalid name")
        
        if os.name != 'nt':  # Skip on Windows
            test_dir = os.path.join(WORKSPACES_DIR, TEST_DIR)
            original_mode = os.stat(test_dir).st_mode
            try:
                os.chmod(test_dir, 0o444)  # Read-only
                success = self.workspace_manager.save_workspace(self.canvas.get_canvas_state(), "test_readonly", TEST_DIR)
                self.assertFalse(success, "Save workspace should return False with read-only directory")
            finally:
                os.chmod(test_dir, original_mode)

    def test_load_workspace(self):
        """Test loading a workspace."""
        self.canvas.create_triangle(0, 0, 100, 0, 50, 100, "ABC")
        workspace_name = "test_triangle_workspace"
        success = self.workspace_manager.save_workspace(self.canvas.get_canvas_state(), workspace_name, TEST_DIR)
        self.assertTrue(success, "Initial save should succeed")
        
        self.canvas.clear()
        self.assertEqual(len(self.canvas.get_drawables()), 0)
        
        state = self.workspace_manager.load_workspace(workspace_name, TEST_DIR)
        
        self.assertIn("Triangles", state)
        self.assertEqual(len(state["Triangles"]), 1)
        triangle = state["Triangles"][0]
        self.assertEqual(triangle["name"], "ABC")

    def test_load_nonexistent_workspace(self):
        """Test loading a workspace that doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.workspace_manager.load_workspace("nonexistent_workspace", TEST_DIR)

    def test_load_workspace_invalid_json(self):
        """Test loading a workspace file with invalid JSON content."""
        workspace_name = "test_invalid_json_workspace"
        workspace_path = os.path.join(WORKSPACES_DIR, TEST_DIR, f"{workspace_name}.json")
        
        with open(workspace_path, 'w') as f:
            f.write("{\"name\": \"test\", \"state\": {this_is_not_valid_json}")

        with self.assertRaises((json.JSONDecodeError, ValueError)):
            self.workspace_manager.load_workspace(workspace_name, TEST_DIR)

    def test_load_workspace_incorrect_schema(self):
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
        self.assertIsInstance(loaded_state_2, dict, "Loaded state should be a dictionary.")
        self.assertIn("Points", loaded_state_2)
        self.assertNotIsInstance(loaded_state_2["Points"], list, "Points should not be a list in this malformed case.")
        self.assertEqual(loaded_state_2["Points"], "this should be a list")

    def test_list_workspaces(self):
        """Test listing all workspaces."""
        workspace_names = ["test_ws1", "test_ws2", "test_ws3"]
        for name in workspace_names:
            success = self.workspace_manager.save_workspace(self.canvas.get_canvas_state(), name, TEST_DIR)
            self.assertTrue(success, f"Failed to save workspace {name}")
        
        workspaces = self.workspace_manager.list_workspaces(TEST_DIR)
        
        for name in workspace_names:
            self.assertIn(name, workspaces)

    def test_list_workspaces_with_non_workspace_files(self):
        """Test listing workspaces when non-workspace files are present."""
        self.workspace_manager.save_workspace(self.canvas.get_canvas_state(), "valid_ws_for_list_test", TEST_DIR)
        
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

    def test_list_workspaces_empty_directory(self):
        """Test listing workspaces from an empty directory."""
        # Ensure the directory is clean. setUp usually does this, but explicit call for clarity.
        self.cleanup_test_workspaces()
        workspaces = self.workspace_manager.list_workspaces(TEST_DIR)
        self.assertEqual(len(workspaces), 0)

    def test_workspace_with_computations(self):
        """Test saving and loading workspace with computations."""
        self.canvas.add_computation("2+2", 4)
        self.canvas.add_computation("sin(pi/2)", 1)
        
        workspace_name = "test_computations"
        success = self.workspace_manager.save_workspace(self.canvas.get_canvas_state(), workspace_name, TEST_DIR)
        self.assertTrue(success, "Failed to save workspace with computations")
        
        self.canvas.clear()
        self.assertEqual(len(self.canvas.computations), 0)
        
        state = self.workspace_manager.load_workspace(workspace_name, TEST_DIR)
        
        self.assertIn("computations", state)
        self.assertEqual(len(state["computations"]), 2)
        expressions = [comp["expression"] for comp in state["computations"]]
        results = [comp["result"] for comp in state["computations"]]
        self.assertIn("2+2", expressions)
        self.assertIn("sin(pi/2)", expressions)
        self.assertIn(4, results)
        self.assertIn(1, results)

    def test_save_complex_workspace(self):
        """Test saving workspace with multiple types of objects."""
        self.canvas.create_point(0, 0, "O")
        self.canvas.create_circle(0, 0, 100, "C1")
        self.canvas.create_rectangle(-50, -50, 50, 50, "R1")
        self.canvas.draw_function("sin(x)", "f1")
        self.canvas.add_computation("area", 10000)
        
        workspace_name = "test_complex"
        success = self.workspace_manager.save_workspace(self.canvas.get_canvas_state(), workspace_name, TEST_DIR)
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

    def test_save_and_load_preserves_state_integrity(self):
        """Test that saving and then loading a workspace recreates the exact same state."""
        # 1. Create a complex canvas state
        pointA_coords = (10, 20)
        pointB_coords = (30, 40)
        pointC_coords = (50, 50)

        self.canvas.create_point(pointA_coords[0], pointA_coords[1], "A")
        self.canvas.create_point(pointB_coords[0], pointB_coords[1], "B")
        self.canvas.create_point(pointC_coords[0], pointC_coords[1], "C")
        self.canvas.create_segment(pointA_coords[0], pointA_coords[1], pointB_coords[0], pointB_coords[1], "AB")
        self.canvas.create_circle(pointC_coords[0], pointC_coords[1], 25) 
        self.canvas.draw_function("x**2", "f1")
        self.canvas.add_computation("my_calc", 123.45)
        
        original_state = self.canvas.get_canvas_state()
        workspace_name = "test_integrity_workspace"

        # 2. Save the workspace
        save_success = self.workspace_manager.save_workspace(original_state, workspace_name, TEST_DIR)
        self.assertTrue(save_success, "Saving the integrity test workspace should succeed.")

        # 3. Load the workspace
        loaded_state = self.workspace_manager.load_workspace(workspace_name, TEST_DIR)
        
        # 4. Deeply compare the loaded state with the original state.
        # Note: Assumes order is preserved or lists are sorted before comparison if necessary.
        # Important: MockCanvas.get_canvas_state() structure must be stable for this to pass.
        if "metadata" in loaded_state:
            del loaded_state["metadata"]

        self.assertEqual(original_state, loaded_state, 
                         "Loaded workspace state does not match the original state.")

    def test_save_and_load_empty_workspace(self):
        """Test saving and loading an empty workspace."""
        # 1. Canvas is already empty by default after setUp
        self.assertEqual(len(self.canvas.get_drawables()), 0)
        self.assertEqual(len(self.canvas.computations), 0)

        original_empty_state = self.canvas.get_canvas_state()
        workspace_name = "test_empty_workspace"

        # 2. Save the empty workspace
        save_success = self.workspace_manager.save_workspace(original_empty_state, workspace_name, TEST_DIR)
        self.assertTrue(save_success, "Saving an empty workspace should succeed.")

        # 3. Load the empty workspace
        loaded_state = self.workspace_manager.load_workspace(workspace_name, TEST_DIR)

        # 4. Deeply compare the loaded state with the original empty state.
        if "metadata" in loaded_state:
            del loaded_state["metadata"]
        
        # Note: Relies on MockCanvas.get_canvas_state() returning a consistent empty structure.
        self.assertEqual(original_empty_state, loaded_state, 
                         "Loaded empty workspace state does not match the original empty state.")
        
        # Additionally, verify common keys are not present or are empty if state is structured that way
        # The primary check is self.assertEqual(original_empty_state, loaded_state)
        # If MockCanvas returns e.g. {"state": {"Points": [], ...}} for empty, these might need adjustment.
        loaded_state_content = loaded_state.get("state", loaded_state) # Handle if 'state' is top or nested
        self.assertNotIn("Points", loaded_state_content)
        self.assertNotIn("Segments", loaded_state_content)
        self.assertNotIn("Circles", loaded_state_content)
        self.assertNotIn("Functions", loaded_state_content)
        self.assertNotIn("computations", loaded_state_content)

    def test_delete_workspace(self):
        """Test deleting a workspace."""
        self.canvas.create_point(100, 100, "A")
        workspace_name = "test_delete_ws"
        success = self.workspace_manager.save_workspace(self.canvas.get_canvas_state(), workspace_name, TEST_DIR)
        self.assertTrue(success, "Initial save should succeed")
        
        workspace_path = os.path.join(WORKSPACES_DIR, TEST_DIR, f"{workspace_name}.json")
        self.assertTrue(os.path.exists(workspace_path))
        
        success = self.workspace_manager.delete_workspace(workspace_name, TEST_DIR)
        self.assertTrue(success, "Delete workspace should return True on success")
        
        self.assertFalse(os.path.exists(workspace_path))
        
        workspaces = self.workspace_manager.list_workspaces(TEST_DIR)
        self.assertNotIn(workspace_name, workspaces)

    def test_delete_nonexistent_workspace(self):
        """Test deleting a workspace that doesn't exist."""
        success = self.workspace_manager.delete_workspace("nonexistent_workspace", TEST_DIR)
        self.assertFalse(success, "Delete workspace should return False for non-existent workspace")

    def test_delete_workspace_with_invalid_name(self):
        """Test deleting a workspace with invalid name."""
        success = self.workspace_manager.delete_workspace(None, TEST_DIR)
        self.assertFalse(success, "Delete workspace should return False for None name")
        
        success = self.workspace_manager.delete_workspace("test/invalid/name", TEST_DIR)
        self.assertFalse(success, "Delete workspace should return False for invalid name")

if __name__ == '__main__':
    unittest.main() 