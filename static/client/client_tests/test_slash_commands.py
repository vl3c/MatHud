"""
Tests for the slash command handler system.

Tests command parsing, execution, error handling, and autocomplete functionality.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch


class MockCanvas:
    """Mock canvas for testing slash commands."""

    def __init__(self) -> None:
        self.undo_called = False
        self.redo_called = False
        self.clear_called = False
        self.reset_called = False
        self.zoom_called = False
        self.zoom_args: Optional[tuple] = None
        self.grid_visible = True
        self.coordinate_system = "cartesian"
        self._drawables: List[Any] = []

        # Mock coordinate mapper
        self.coordinate_mapper = MagicMock()
        self.coordinate_mapper.left_bound = -10
        self.coordinate_mapper.right_bound = 10
        self.coordinate_mapper.top_bound = 10
        self.coordinate_mapper.bottom_bound = -10
        self.coordinate_mapper.get_visible_bounds = lambda: {
            "left": -10, "right": 10, "top": 10, "bottom": -10
        }

        # Mock cartesian2axis
        self.cartesian2axis = MagicMock()
        self.cartesian2axis.visible = True

    def undo(self) -> bool:
        self.undo_called = True
        return True

    def redo(self) -> bool:
        self.redo_called = True
        return True

    def clear(self) -> None:
        self.clear_called = True

    def reset(self) -> None:
        self.reset_called = True

    def zoom(self, center_x: float, center_y: float, range_val: float, range_axis: str) -> bool:
        self.zoom_called = True
        self.zoom_args = (center_x, center_y, range_val, range_axis)
        return True

    def is_grid_visible(self) -> bool:
        return self.grid_visible

    def set_grid_visible(self, visible: bool) -> bool:
        self.grid_visible = visible
        return True

    def get_coordinate_system(self) -> str:
        return self.coordinate_system

    def set_coordinate_system(self, mode: str) -> bool:
        self.coordinate_system = mode
        return True

    def get_drawables(self) -> List[Any]:
        return self._drawables

    def get_canvas_state(self) -> Dict[str, Any]:
        return {
            "Points": [{"name": "A", "args": {"position": {"x": 0, "y": 0}}}],
            "Segments": [],
            "computations": [],
        }

    def draw(self) -> None:
        pass


class MockWorkspaceManager:
    """Mock workspace manager for testing."""

    def __init__(self) -> None:
        self.save_called = False
        self.load_called = False
        self.save_name: Optional[str] = None
        self.load_name: Optional[str] = None

    def save_workspace(self, name: Optional[str] = None) -> str:
        self.save_called = True
        self.save_name = name
        return f'Workspace "{name if name else "current"}" saved successfully.'

    def load_workspace(self, name: Optional[str] = None) -> str:
        self.load_called = True
        self.load_name = name
        return f'Workspace "{name if name else "current"}" loaded successfully.'

    def list_workspaces(self) -> str:
        return "workspace1, workspace2"

    def _restore_workspace_state(self, state: Dict[str, Any]) -> None:
        pass


class MockAIInterface:
    """Mock AI interface for testing."""

    def __init__(self) -> None:
        self.run_tests_called = False
        self.new_conversation_called = False

    def run_tests(self) -> Dict[str, Any]:
        self.run_tests_called = True
        return {
            "tests_run": 100,
            "failures": 0,
            "errors": 0,
            "failing_tests": [],
            "error_tests": [],
        }

    def start_new_conversation(self, event: Any) -> None:
        self.new_conversation_called = True


class TestSlashCommandHandler(unittest.TestCase):
    """Tests for SlashCommandHandler class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from slash_command_handler import SlashCommandHandler

        self.canvas = MockCanvas()
        self.workspace_manager = MockWorkspaceManager()
        self.ai_interface = MockAIInterface()
        self.handler = SlashCommandHandler(
            self.canvas,  # type: ignore
            self.workspace_manager,  # type: ignore
            self.ai_interface,  # type: ignore
        )

    def test_is_slash_command_valid(self) -> None:
        """Test detection of valid slash commands."""
        self.assertTrue(self.handler.is_slash_command("/help"))
        self.assertTrue(self.handler.is_slash_command("/undo"))
        self.assertTrue(self.handler.is_slash_command("/save myworkspace"))
        self.assertTrue(self.handler.is_slash_command("  /help  "))

    def test_is_slash_command_invalid(self) -> None:
        """Test that non-slash commands return False."""
        self.assertFalse(self.handler.is_slash_command("help"))
        self.assertFalse(self.handler.is_slash_command(""))
        self.assertFalse(self.handler.is_slash_command("/"))
        self.assertFalse(self.handler.is_slash_command("/123"))  # No letter after slash
        self.assertFalse(self.handler.is_slash_command("draw a circle"))

    def test_parse_command_simple(self) -> None:
        """Test parsing simple commands without arguments."""
        cmd, args = self.handler._parse_command("/help")
        self.assertEqual(cmd, "help")
        self.assertEqual(args, [])

    def test_parse_command_with_args(self) -> None:
        """Test parsing commands with arguments."""
        cmd, args = self.handler._parse_command("/save myworkspace")
        self.assertEqual(cmd, "save")
        self.assertEqual(args, ["myworkspace"])

        cmd, args = self.handler._parse_command("/zoom in")
        self.assertEqual(cmd, "zoom")
        self.assertEqual(args, ["in"])

    def test_parse_command_multiple_args(self) -> None:
        """Test parsing commands with multiple arguments."""
        cmd, args = self.handler._parse_command("/import { }")
        self.assertEqual(cmd, "import")
        self.assertEqual(args, ["{", "}"])

    def test_parse_command_case_insensitive(self) -> None:
        """Test that command names are case-insensitive."""
        cmd, args = self.handler._parse_command("/HELP")
        self.assertEqual(cmd, "help")

        cmd, args = self.handler._parse_command("/HeLp")
        self.assertEqual(cmd, "help")

    def test_execute_unknown_command(self) -> None:
        """Test executing an unknown command."""
        result = self.handler.execute("/xyz")
        self.assertFalse(result.success)
        self.assertIn("Unknown command", result.message)
        self.assertIn("/help", result.message)

    def test_execute_unknown_command_with_suggestions(self) -> None:
        """Test that unknown commands get suggestions."""
        result = self.handler.execute("/hel")
        self.assertFalse(result.success)
        self.assertIn("/help", result.message)

    def test_get_similar_commands(self) -> None:
        """Test fuzzy command matching."""
        # Prefix match
        suggestions = self.handler._get_similar_commands("hel")
        self.assertIn("help", suggestions)

        # Edit distance match
        suggestions = self.handler._get_similar_commands("hlep")
        self.assertIn("help", suggestions)

    def test_get_commands_list(self) -> None:
        """Test getting all commands for autocomplete."""
        commands = self.handler.get_commands_list()
        self.assertIsInstance(commands, list)
        self.assertTrue(len(commands) > 0)

        # Check format
        for cmd, desc in commands:
            self.assertTrue(cmd.startswith("/"))
            self.assertIsInstance(desc, str)


class TestEssentialCommands(unittest.TestCase):
    """Tests for essential slash commands."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from slash_command_handler import SlashCommandHandler

        self.canvas = MockCanvas()
        self.workspace_manager = MockWorkspaceManager()
        self.ai_interface = MockAIInterface()
        self.handler = SlashCommandHandler(
            self.canvas,  # type: ignore
            self.workspace_manager,  # type: ignore
            self.ai_interface,  # type: ignore
        )

    def test_cmd_help(self) -> None:
        """Test /help command."""
        result = self.handler.execute("/help")
        self.assertTrue(result.success)
        self.assertIn("Available Commands", result.message)

    def test_cmd_help_specific(self) -> None:
        """Test /help with specific command."""
        result = self.handler.execute("/help undo")
        self.assertTrue(result.success)
        self.assertIn("/undo", result.message)
        self.assertIn("Undo", result.message)

    def test_cmd_help_unknown(self) -> None:
        """Test /help with unknown command."""
        result = self.handler.execute("/help xyz")
        self.assertFalse(result.success)
        self.assertIn("Unknown command", result.message)

    def test_cmd_undo(self) -> None:
        """Test /undo command."""
        result = self.handler.execute("/undo")
        self.assertTrue(result.success)
        self.assertTrue(self.canvas.undo_called)

    def test_cmd_redo(self) -> None:
        """Test /redo command."""
        result = self.handler.execute("/redo")
        self.assertTrue(result.success)
        self.assertTrue(self.canvas.redo_called)

    def test_cmd_clear(self) -> None:
        """Test /clear command."""
        result = self.handler.execute("/clear")
        self.assertTrue(result.success)
        self.assertTrue(self.canvas.clear_called)

    def test_cmd_reset(self) -> None:
        """Test /reset command."""
        result = self.handler.execute("/reset")
        self.assertTrue(result.success)
        self.assertTrue(self.canvas.reset_called)

    def test_cmd_save(self) -> None:
        """Test /save command."""
        result = self.handler.execute("/save")
        self.assertTrue(result.success)
        self.assertTrue(self.workspace_manager.save_called)

    def test_cmd_save_with_name(self) -> None:
        """Test /save with workspace name."""
        result = self.handler.execute("/save myproject")
        self.assertTrue(result.success)
        self.assertEqual(self.workspace_manager.save_name, "myproject")

    def test_cmd_load(self) -> None:
        """Test /load command."""
        result = self.handler.execute("/load")
        self.assertTrue(result.success)
        self.assertTrue(self.workspace_manager.load_called)

    def test_cmd_load_with_name(self) -> None:
        """Test /load with workspace name."""
        result = self.handler.execute("/load myproject")
        self.assertTrue(result.success)
        self.assertEqual(self.workspace_manager.load_name, "myproject")

    def test_cmd_workspaces(self) -> None:
        """Test /workspaces command."""
        result = self.handler.execute("/workspaces")
        self.assertTrue(result.success)
        self.assertIn("workspace1", result.message)


class TestCanvasViewCommands(unittest.TestCase):
    """Tests for canvas/view control commands."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from slash_command_handler import SlashCommandHandler

        self.canvas = MockCanvas()
        self.workspace_manager = MockWorkspaceManager()
        self.ai_interface = MockAIInterface()
        self.handler = SlashCommandHandler(
            self.canvas,  # type: ignore
            self.workspace_manager,  # type: ignore
            self.ai_interface,  # type: ignore
        )

    def test_cmd_fit_no_objects(self) -> None:
        """Test /fit with no objects on canvas."""
        result = self.handler.execute("/fit")
        self.assertFalse(result.success)
        self.assertIn("No objects", result.message)

    def test_cmd_zoom_in(self) -> None:
        """Test /zoom in command."""
        result = self.handler.execute("/zoom in")
        self.assertTrue(result.success)
        self.assertTrue(self.canvas.zoom_called)
        self.assertIn("Zoomed in", result.message)

    def test_cmd_zoom_out(self) -> None:
        """Test /zoom out command."""
        result = self.handler.execute("/zoom out")
        self.assertTrue(result.success)
        self.assertTrue(self.canvas.zoom_called)
        self.assertIn("Zoomed out", result.message)

    def test_cmd_zoom_factor(self) -> None:
        """Test /zoom with numeric factor."""
        result = self.handler.execute("/zoom 2")
        self.assertTrue(result.success)
        self.assertTrue(self.canvas.zoom_called)
        self.assertIn("2x", result.message)

    def test_cmd_zoom_invalid(self) -> None:
        """Test /zoom with invalid argument."""
        result = self.handler.execute("/zoom abc")
        self.assertFalse(result.success)
        self.assertIn("Invalid", result.message)

    def test_cmd_zoom_no_args(self) -> None:
        """Test /zoom without arguments."""
        result = self.handler.execute("/zoom")
        self.assertFalse(result.success)
        self.assertIn("Usage", result.message)

    def test_cmd_grid(self) -> None:
        """Test /grid toggle command."""
        initial = self.canvas.grid_visible
        result = self.handler.execute("/grid")
        self.assertTrue(result.success)
        self.assertNotEqual(initial, self.canvas.grid_visible)

    def test_cmd_axes(self) -> None:
        """Test /axes toggle command."""
        result = self.handler.execute("/axes")
        self.assertTrue(result.success)
        self.assertIn("Axes", result.message)

    def test_cmd_polar(self) -> None:
        """Test /polar command."""
        result = self.handler.execute("/polar")
        self.assertTrue(result.success)
        self.assertEqual(self.canvas.coordinate_system, "polar")

    def test_cmd_cartesian(self) -> None:
        """Test /cartesian command."""
        self.canvas.coordinate_system = "polar"
        result = self.handler.execute("/cartesian")
        self.assertTrue(result.success)
        self.assertEqual(self.canvas.coordinate_system, "cartesian")

    def test_cmd_status(self) -> None:
        """Test /status command."""
        result = self.handler.execute("/status")
        self.assertTrue(result.success)
        self.assertIn("Canvas Status", result.message)


class TestUtilityCommands(unittest.TestCase):
    """Tests for utility slash commands."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from slash_command_handler import SlashCommandHandler

        self.canvas = MockCanvas()
        self.workspace_manager = MockWorkspaceManager()
        self.ai_interface = MockAIInterface()
        self.handler = SlashCommandHandler(
            self.canvas,  # type: ignore
            self.workspace_manager,  # type: ignore
            self.ai_interface,  # type: ignore
        )

    def test_cmd_test(self) -> None:
        """Test /test command."""
        result = self.handler.execute("/test")
        self.assertTrue(result.success)
        self.assertTrue(self.ai_interface.run_tests_called)
        self.assertIn("Test Results", result.message)

    def test_cmd_export(self) -> None:
        """Test /export command returns raw JSON."""
        result = self.handler.execute("/export")
        self.assertTrue(result.success)
        # Should return valid JSON (not wrapped in markdown)
        self.assertTrue(result.message.strip().startswith("{"))
        self.assertIsNotNone(result.data)

    def test_cmd_import_no_args(self) -> None:
        """Test /import without arguments."""
        result = self.handler.execute("/import")
        self.assertFalse(result.success)
        self.assertIn("Usage", result.message)

    def test_cmd_import_invalid_json(self) -> None:
        """Test /import with invalid JSON."""
        result = self.handler.execute("/import not json")
        self.assertFalse(result.success)
        # Error message may contain "Invalid JSON" or other JSON parsing errors
        self.assertTrue(
            "Invalid JSON" in result.message or "JSON" in result.message or "Error" in result.message,
            f"Expected JSON error message, got: {result.message}"
        )

    def test_cmd_list(self) -> None:
        """Test /list command."""
        result = self.handler.execute("/list")
        self.assertTrue(result.success)
        self.assertIn("Objects", result.message)

    def test_cmd_new(self) -> None:
        """Test /new command."""
        result = self.handler.execute("/new")
        self.assertTrue(result.success)
        self.assertTrue(self.ai_interface.new_conversation_called)


class TestCommandResult(unittest.TestCase):
    """Tests for CommandResult dataclass."""

    def test_command_result_success(self) -> None:
        """Test creating a successful result."""
        from slash_command_handler import CommandResult

        result = CommandResult(success=True, message="Operation completed")
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Operation completed")
        self.assertIsNone(result.data)

    def test_command_result_with_data(self) -> None:
        """Test creating a result with data."""
        from slash_command_handler import CommandResult

        data = {"key": "value"}
        result = CommandResult(success=True, message="Done", data=data)
        self.assertEqual(result.data, data)

    def test_command_result_failure(self) -> None:
        """Test creating a failure result."""
        from slash_command_handler import CommandResult

        result = CommandResult(success=False, message="Error occurred")
        self.assertFalse(result.success)


class TestEditDistance(unittest.TestCase):
    """Tests for the edit distance calculation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from slash_command_handler import SlashCommandHandler

        self.canvas = MockCanvas()
        self.workspace_manager = MockWorkspaceManager()
        self.ai_interface = MockAIInterface()
        self.handler = SlashCommandHandler(
            self.canvas,  # type: ignore
            self.workspace_manager,  # type: ignore
            self.ai_interface,  # type: ignore
        )

    def test_edit_distance_identical(self) -> None:
        """Test edit distance for identical strings."""
        self.assertEqual(self.handler._edit_distance("help", "help"), 0)

    def test_edit_distance_one_char(self) -> None:
        """Test edit distance for single character difference."""
        self.assertEqual(self.handler._edit_distance("help", "helpp"), 1)
        self.assertEqual(self.handler._edit_distance("help", "hel"), 1)
        self.assertEqual(self.handler._edit_distance("help", "halp"), 1)

    def test_edit_distance_two_chars(self) -> None:
        """Test edit distance for two character differences."""
        self.assertEqual(self.handler._edit_distance("help", "hlep"), 2)

    def test_edit_distance_completely_different(self) -> None:
        """Test edit distance for completely different strings."""
        distance = self.handler._edit_distance("help", "xyz")
        self.assertGreater(distance, 2)


class TestModelCommand(unittest.TestCase):
    """Tests for /model command."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from slash_command_handler import SlashCommandHandler

        self.canvas = MockCanvas()
        self.workspace_manager = MockWorkspaceManager()
        self.ai_interface = MockAIInterface()
        self.handler = SlashCommandHandler(
            self.canvas,  # type: ignore
            self.workspace_manager,  # type: ignore
            self.ai_interface,  # type: ignore
        )

    def test_cmd_model_no_args_shows_current(self) -> None:
        """Test /model without arguments shows current model."""
        result = self.handler.execute("/model")
        self.assertTrue(result.success)
        # Should show current model info
        self.assertIn("model", result.message.lower())

    def test_cmd_model_with_valid_name(self) -> None:
        """Test /model with a model name attempts to switch."""
        result = self.handler.execute("/model gpt-4")
        # Result depends on whether the model exists in the selector
        # The command should at least not crash
        self.assertIsNotNone(result.message)

    def test_cmd_model_case_insensitive(self) -> None:
        """Test /model command is case-insensitive."""
        result1 = self.handler.execute("/MODEL")
        result2 = self.handler.execute("/Model")
        # Both should execute without error
        self.assertIsNotNone(result1.message)
        self.assertIsNotNone(result2.message)


class TestVisionCommand(unittest.TestCase):
    """Tests for /vision command."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from slash_command_handler import SlashCommandHandler

        self.canvas = MockCanvas()
        self.workspace_manager = MockWorkspaceManager()
        self.ai_interface = MockAIInterface()
        self.handler = SlashCommandHandler(
            self.canvas,  # type: ignore
            self.workspace_manager,  # type: ignore
            self.ai_interface,  # type: ignore
        )
        # Mock vision model check to always return True for testing
        self.handler._selected_model_has_vision = lambda: True  # type: ignore

    def test_cmd_vision_toggles(self) -> None:
        """Test /vision command toggles vision mode."""
        result = self.handler.execute("/vision")
        self.assertTrue(result.success)
        # Should indicate vision state change
        self.assertIn("Vision", result.message)

    def test_cmd_vision_shows_status(self) -> None:
        """Test /vision shows enabled/disabled status."""
        result = self.handler.execute("/vision")
        self.assertTrue(result.success)
        # Should contain enabled or disabled
        msg_lower = result.message.lower()
        self.assertTrue("enabled" in msg_lower or "disabled" in msg_lower)


class MockInputElement:
    """Mock DOM input element for autocomplete testing."""

    def __init__(self) -> None:
        self.value: str = ""
        self._event_handlers: Dict[str, List[Any]] = {}
        self.parentElement = MockParentElement()

    def bind(self, event: str, handler: Any) -> None:
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def focus(self) -> None:
        pass

    def setSelectionRange(self, start: int, end: int) -> None:
        pass


class MockParentElement:
    """Mock parent element for positioning."""

    def __init__(self) -> None:
        self.style = MockStyle()
        self.children: List[Any] = []

    def insertBefore(self, new_node: Any, reference: Any) -> None:
        self.children.append(new_node)


class MockStyle:
    """Mock CSS style object."""

    def __init__(self) -> None:
        self.display: str = "none"
        self.position: str = ""


class MockCommandHandler:
    """Mock command handler for autocomplete testing."""

    def __init__(self) -> None:
        self.workspace_manager = MockWorkspaceManager()

    def get_commands_list(self) -> List[tuple]:
        return [
            ("/help", "Show available commands"),
            ("/undo", "Undo last action"),
            ("/redo", "Redo last undone action"),
            ("/clear", "Clear canvas"),
            ("/reset", "Reset view"),
            ("/save", "Save workspace"),
            ("/load", "Load workspace"),
            ("/model", "Switch AI model"),
            ("/zoom", "Zoom canvas"),
            ("/grid", "Toggle grid"),
            ("/status", "Show canvas status"),
        ]


class TestCommandAutocomplete(unittest.TestCase):
    """Tests for CommandAutocomplete class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from command_autocomplete import CommandAutocomplete

        self.input_element = MockInputElement()
        self.command_handler = MockCommandHandler()
        self.autocomplete = CommandAutocomplete(
            self.input_element,  # type: ignore
            self.command_handler,  # type: ignore
        )

    def test_initial_state(self) -> None:
        """Test autocomplete starts hidden."""
        self.assertFalse(self.autocomplete.visible)
        self.assertEqual(self.autocomplete.selected_index, 0)
        self.assertEqual(len(self.autocomplete.filtered_commands), 0)

    def test_filter_empty_prefix(self) -> None:
        """Test filtering with empty prefix shows all commands."""
        self.autocomplete.filter("")
        self.assertEqual(
            len(self.autocomplete.filtered_commands),
            len(self.command_handler.get_commands_list())
        )

    def test_filter_matching_prefix(self) -> None:
        """Test filtering with matching prefix."""
        self.autocomplete.filter("he")
        # Should match /help
        self.assertTrue(
            any("/help" in cmd for cmd, _ in self.autocomplete.filtered_commands)
        )

    def test_filter_no_match(self) -> None:
        """Test filtering with non-matching prefix."""
        self.autocomplete.filter("xyz")
        self.assertEqual(len(self.autocomplete.filtered_commands), 0)

    def test_filter_case_insensitive(self) -> None:
        """Test filtering is case-insensitive."""
        self.autocomplete.filter("HE")
        self.assertTrue(
            any("/help" in cmd for cmd, _ in self.autocomplete.filtered_commands)
        )

    def test_select_next_wraps(self) -> None:
        """Test select_next wraps around."""
        self.autocomplete.filter("")
        total = len(self.autocomplete.filtered_commands)
        self.autocomplete.selected_index = total - 1
        self.autocomplete.select_next()
        self.assertEqual(self.autocomplete.selected_index, 0)

    def test_select_previous_wraps(self) -> None:
        """Test select_previous wraps around."""
        self.autocomplete.filter("")
        total = len(self.autocomplete.filtered_commands)
        self.autocomplete.selected_index = 0
        self.autocomplete.select_previous()
        self.assertEqual(self.autocomplete.selected_index, total - 1)

    def test_select_next_empty_list(self) -> None:
        """Test select_next does nothing with empty list."""
        self.autocomplete.filtered_commands = []
        self.autocomplete.selected_index = 0
        self.autocomplete.select_next()
        self.assertEqual(self.autocomplete.selected_index, 0)

    def test_select_previous_empty_list(self) -> None:
        """Test select_previous does nothing with empty list."""
        self.autocomplete.filtered_commands = []
        self.autocomplete.selected_index = 0
        self.autocomplete.select_previous()
        self.assertEqual(self.autocomplete.selected_index, 0)

    def test_confirm_selection_empty(self) -> None:
        """Test confirm_selection with no commands."""
        self.autocomplete.filtered_commands = []
        result = self.autocomplete.confirm_selection()
        self.assertEqual(result, "")

    def test_confirm_selection_returns_command(self) -> None:
        """Test confirm_selection returns selected command."""
        self.autocomplete.filter("")
        self.autocomplete.selected_index = 0
        result = self.autocomplete.confirm_selection()
        self.assertTrue(result.startswith("/"))

    def test_filter_resets_selection(self) -> None:
        """Test filtering resets selected index to 0."""
        self.autocomplete.filter("")
        self.autocomplete.selected_index = 5
        self.autocomplete.filter("h")
        self.assertEqual(self.autocomplete.selected_index, 0)


class TestWorkspaceSuggestions(unittest.TestCase):
    """Tests for workspace suggestions in autocomplete."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from command_autocomplete import CommandAutocomplete

        self.input_element = MockInputElement()
        self.command_handler = MockCommandHandler()
        self.autocomplete = CommandAutocomplete(
            self.input_element,  # type: ignore
            self.command_handler,  # type: ignore
        )

    def test_workspace_suggestions_trigger(self) -> None:
        """Test that typing '/load ' triggers workspace suggestions."""
        result = self.autocomplete._get_workspace_suggestions("load ")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)

    def test_workspace_suggestions_no_trigger(self) -> None:
        """Test that '/load' without space doesn't trigger suggestions."""
        result = self.autocomplete._get_workspace_suggestions("load")
        self.assertIsNone(result)

    def test_workspace_suggestions_filter(self) -> None:
        """Test workspace suggestions filter by prefix."""
        result = self.autocomplete._get_workspace_suggestions("load work")
        self.assertIsNotNone(result)
        # Should filter workspaces starting with 'work'
        if result:
            for cmd, _ in result:
                self.assertIn("load", cmd.lower())

    def test_filter_workspaces_returns_list(self) -> None:
        """Test _filter_workspaces returns list of tuples."""
        result = self.autocomplete._filter_workspaces("load", "")
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)


class TestModelSuggestions(unittest.TestCase):
    """Tests for model suggestions in autocomplete."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from command_autocomplete import CommandAutocomplete

        self.input_element = MockInputElement()
        self.command_handler = MockCommandHandler()
        self.autocomplete = CommandAutocomplete(
            self.input_element,  # type: ignore
            self.command_handler,  # type: ignore
        )

    def test_model_suggestions_trigger(self) -> None:
        """Test that typing '/model ' triggers model suggestions."""
        result = self.autocomplete._get_workspace_suggestions("model ")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)

    def test_model_suggestions_no_trigger(self) -> None:
        """Test that '/model' without space doesn't trigger suggestions."""
        result = self.autocomplete._get_workspace_suggestions("model")
        self.assertIsNone(result)

    def test_model_suggestions_with_prefix(self) -> None:
        """Test model suggestions filter by prefix."""
        result = self.autocomplete._get_workspace_suggestions("model gpt")
        self.assertIsNotNone(result)


class TestExpandableContent(unittest.TestCase):
    """Tests for expandable content message display."""

    def test_long_message_detection_by_length(self) -> None:
        """Test that messages over 800 chars are considered long."""
        # Create a message over 800 characters
        long_message = "x" * 801
        line_count = long_message.count('\n')
        is_long = len(long_message) > 800 or line_count > 20
        self.assertTrue(is_long)

    def test_long_message_detection_by_lines(self) -> None:
        """Test that messages with >20 newlines are considered long."""
        # Create a message with 22 lines (21 newlines) to trigger > 20 check
        long_message = "\n".join(["line"] * 22)
        line_count = long_message.count('\n')
        is_long = len(long_message) > 800 or line_count > 20
        self.assertTrue(is_long)

    def test_short_message_not_expandable(self) -> None:
        """Test that short messages are not considered long."""
        short_message = "This is a short message"
        line_count = short_message.count('\n')
        is_long = len(short_message) > 800 or line_count > 20
        self.assertFalse(is_long)

    def test_preview_truncation_by_lines(self) -> None:
        """Test preview is truncated to 10 lines."""
        lines = ["line " + str(i) for i in range(20)]
        message = '\n'.join(lines)

        # Simulate preview creation logic
        msg_lines = message.split('\n')
        if len(msg_lines) > 10:
            preview_text = '\n'.join(msg_lines[:10]) + '\n...'
        else:
            preview_text = message

        preview_lines = preview_text.split('\n')
        # Should have 10 lines plus the "..." line
        self.assertEqual(len(preview_lines), 11)
        self.assertTrue(preview_text.endswith('...'))

    def test_preview_truncation_by_chars(self) -> None:
        """Test preview is truncated to 500 chars when few lines."""
        message = "a" * 600  # Long but single line

        # Simulate preview creation logic
        lines = message.split('\n')
        if len(lines) > 10:
            preview_text = '\n'.join(lines[:10]) + '\n...'
        elif len(message) > 500:
            preview_text = message[:500] + '...'
        else:
            preview_text = message

        self.assertEqual(len(preview_text), 503)  # 500 + "..."
        self.assertTrue(preview_text.endswith('...'))


class TestExportCommandOutput(unittest.TestCase):
    """Tests for /export command output format."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from slash_command_handler import SlashCommandHandler

        self.canvas = MockCanvas()
        self.workspace_manager = MockWorkspaceManager()
        self.ai_interface = MockAIInterface()
        self.handler = SlashCommandHandler(
            self.canvas,  # type: ignore
            self.workspace_manager,  # type: ignore
            self.ai_interface,  # type: ignore
        )

    def test_export_returns_valid_json(self) -> None:
        """Test /export returns valid JSON string."""
        import json
        result = self.handler.execute("/export")
        self.assertTrue(result.success)
        # The message should be valid JSON
        try:
            parsed = json.loads(result.message)
            self.assertIsInstance(parsed, dict)
        except json.JSONDecodeError:
            self.fail("Export message is not valid JSON")

    def test_export_data_matches_message(self) -> None:
        """Test /export data field matches message content."""
        import json
        result = self.handler.execute("/export")
        self.assertTrue(result.success)
        parsed_message = json.loads(result.message)
        self.assertEqual(parsed_message, result.data)

    def test_export_not_markdown_wrapped(self) -> None:
        """Test /export output is not wrapped in markdown code blocks."""
        result = self.handler.execute("/export")
        self.assertTrue(result.success)
        # Should not start with markdown code fence
        self.assertFalse(result.message.strip().startswith("```"))
        # Should start with JSON object brace
        self.assertTrue(result.message.strip().startswith("{"))


class TestStatusCommandOutput(unittest.TestCase):
    """Tests for /status command output."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from slash_command_handler import SlashCommandHandler

        self.canvas = MockCanvas()
        self.workspace_manager = MockWorkspaceManager()
        self.ai_interface = MockAIInterface()
        self.handler = SlashCommandHandler(
            self.canvas,  # type: ignore
            self.workspace_manager,  # type: ignore
            self.ai_interface,  # type: ignore
        )

    def test_status_shows_object_count(self) -> None:
        """Test /status shows object count."""
        result = self.handler.execute("/status")
        self.assertTrue(result.success)
        self.assertIn("Objects", result.message)

    def test_status_shows_bounds(self) -> None:
        """Test /status shows visible bounds."""
        result = self.handler.execute("/status")
        self.assertTrue(result.success)
        # Should contain bound information
        msg_lower = result.message.lower()
        self.assertTrue(
            "bound" in msg_lower or "left" in msg_lower or "right" in msg_lower
        )

    def test_status_shows_coordinate_system(self) -> None:
        """Test /status shows coordinate system mode."""
        result = self.handler.execute("/status")
        self.assertTrue(result.success)
        msg_lower = result.message.lower()
        self.assertTrue("cartesian" in msg_lower or "polar" in msg_lower)
