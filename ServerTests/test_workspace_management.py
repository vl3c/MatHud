import unittest
import os
import json
from datetime import datetime
from . import python_path_setup  # Import this first to set up the Python path
from ServerTests.test_mocks import MockCanvas
from workspace_manager import save_workspace, load_workspace, list_workspaces, WORKSPACES_DIR, ensure_workspaces_dir

class TestWorkspaceManagement(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.canvas = MockCanvas(500, 500, draw_enabled=False)
        # Create test workspace directory if it doesn't exist
        ensure_workspaces_dir()
        # Clean up any existing test workspaces
        self.cleanup_test_workspaces()

    def tearDown(self):
        """Clean up after each test."""
        self.cleanup_test_workspaces()

    def cleanup_test_workspaces(self):
        """Remove test workspace files."""
        if os.path.exists(WORKSPACES_DIR):
            for filename in os.listdir(WORKSPACES_DIR):
                if filename.startswith('test_') or filename == 'current_workspace.json':
                    os.remove(os.path.join(WORKSPACES_DIR, filename))

    def test_save_workspace_without_name(self):
        """Test saving workspace without a name (current workspace)."""
        # Create some test data
        self.canvas.create_point(100, 100, "A")
        self.canvas.create_point(200, 200, "B")
        self.canvas.create_segment(100, 100, 200, 200, "AB")

        # Save workspace
        success = save_workspace(self.canvas)
        self.assertTrue(success, "Save workspace should return True on success")
        
        # Check if file exists
        current_workspace_path = os.path.join(WORKSPACES_DIR, "current_workspace.json")
        self.assertTrue(os.path.exists(current_workspace_path))
        
        # Verify file contents
        with open(current_workspace_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["metadata"]["name"], "current_workspace")
            self.assertIn("Points", data["state"])
            self.assertIn("Segments", data["state"])
            self.assertEqual(len(data["state"]["Points"]), 2)
            self.assertEqual(len(data["state"]["Segments"]), 1)

    def test_save_workspace_with_name(self):
        """Test saving workspace with a specific name."""
        # Create test data
        self.canvas.create_circle(0, 0, 100, "C1")
        
        # Save workspace
        workspace_name = "test_circle_workspace"
        success = save_workspace(self.canvas, workspace_name)
        self.assertTrue(success, "Save workspace should return True on success")
        
        # Check if file exists
        workspace_path = os.path.join(WORKSPACES_DIR, f"{workspace_name}.json")
        self.assertTrue(os.path.exists(workspace_path))
        
        # Verify file contents
        with open(workspace_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["metadata"]["name"], workspace_name)
            self.assertIn("Circles", data["state"])
            self.assertEqual(len(data["state"]["Circles"]), 1)

    def test_save_workspace_failure(self):
        """Test saving workspace with invalid conditions."""
        # Test with invalid canvas
        success = save_workspace(None)
        self.assertFalse(success, "Save workspace should return False when canvas is None")
        
        # Test with invalid workspace name (containing invalid characters)
        success = save_workspace(self.canvas, "test/invalid/name")
        self.assertFalse(success, "Save workspace should return False with invalid name")
        
        # Test with read-only directory (if possible)
        if os.name != 'nt':  # Skip on Windows
            original_mode = os.stat(WORKSPACES_DIR).st_mode
            try:
                os.chmod(WORKSPACES_DIR, 0o444)  # Read-only
                success = save_workspace(self.canvas, "test_readonly")
                self.assertFalse(success, "Save workspace should return False with read-only directory")
            finally:
                os.chmod(WORKSPACES_DIR, original_mode)

    def test_load_workspace(self):
        """Test loading a workspace."""
        # Create and save test data
        self.canvas.create_triangle(0, 0, 100, 0, 50, 100, "ABC")
        workspace_name = "test_triangle_workspace"
        success = save_workspace(self.canvas, workspace_name)
        self.assertTrue(success, "Initial save should succeed")
        
        # Clear canvas
        self.canvas.clear()
        self.assertEqual(len(self.canvas.get_drawables()), 0)
        
        # Load workspace
        result = load_workspace(self.canvas, workspace_name)
        
        # Verify canvas state
        self.assertEqual(len(self.canvas.get_drawables_by_class_name("Triangle")), 1)
        triangle = self.canvas.get_drawables_by_class_name("Triangle")[0]
        self.assertEqual(triangle["name"], "ABC")

    def test_load_nonexistent_workspace(self):
        """Test loading a workspace that doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            load_workspace(self.canvas, "nonexistent_workspace")

    def test_list_workspaces(self):
        """Test listing all workspaces."""
        # Create multiple test workspaces
        workspace_names = ["test_ws1", "test_ws2", "test_ws3"]
        for name in workspace_names:
            success = save_workspace(self.canvas, name)
            self.assertTrue(success, f"Failed to save workspace {name}")
        
        # Get list of workspaces
        workspaces = list_workspaces()
        
        # Verify each test workspace is in the list
        for name in workspace_names:
            self.assertIn(name, workspaces)
        
        # Verify current_workspace is not in the list
        self.assertNotIn("current_workspace", workspaces)

    def test_workspace_with_computations(self):
        """Test saving and loading workspace with computations."""
        # Add some computations
        self.canvas.add_computation("2+2", 4)
        self.canvas.add_computation("sin(pi/2)", 1)
        
        # Save workspace
        workspace_name = "test_computations"
        success = save_workspace(self.canvas, workspace_name)
        self.assertTrue(success, "Failed to save workspace with computations")
        
        # Clear canvas
        self.canvas.clear()
        self.assertEqual(len(self.canvas.computations), 0)
        
        # Load workspace
        load_workspace(self.canvas, workspace_name)
        
        # Verify computations were restored
        self.assertEqual(len(self.canvas.computations), 2)
        expressions = [comp["expression"] for comp in self.canvas.computations]
        results = [comp["result"] for comp in self.canvas.computations]
        self.assertIn("2+2", expressions)
        self.assertIn("sin(pi/2)", expressions)
        self.assertIn(4, results)
        self.assertIn(1, results)

    def test_save_complex_workspace(self):
        """Test saving workspace with multiple types of objects."""
        # Create various objects
        self.canvas.create_point(0, 0, "O")
        self.canvas.create_circle(0, 0, 100, "C1")
        self.canvas.create_rectangle(-50, -50, 50, 50, "R1")
        self.canvas.draw_function("sin(x)", "f1")
        self.canvas.add_computation("area", 10000)
        
        # Save workspace
        workspace_name = "test_complex"
        success = save_workspace(self.canvas, workspace_name)
        self.assertTrue(success, "Failed to save complex workspace")
        
        # Verify file contents
        workspace_path = os.path.join(WORKSPACES_DIR, f"{workspace_name}.json")
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

if __name__ == '__main__':
    unittest.main() 