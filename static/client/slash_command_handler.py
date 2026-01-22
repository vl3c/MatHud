"""
MatHud Slash Command Handler

Handles slash commands typed in the chat interface for quick app management operations.
Commands execute locally without AI involvement, providing fast access to common actions.

Key Features:
    - Command parsing with argument handling
    - Fuzzy matching for unknown commands
    - Command registry with descriptions for help/autocomplete
    - Integration with canvas, workspace, and AI interface operations

Commands:
    Essential: /help, /undo, /redo, /clear, /reset, /save, /load, /workspaces
    Canvas/View: /fit, /zoom, /grid, /axes, /polar, /cartesian, /status
    Utility: /vision, /test, /export, /import, /list, /new, /model

Dependencies:
    - canvas: Canvas operations (clear, reset, undo, redo, zoom)
    - workspace_manager: Workspace persistence operations
    - ai_interface: Test execution and conversation management
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from browser import document

if TYPE_CHECKING:
    from canvas import Canvas
    from workspace_manager import WorkspaceManager
    from ai_interface import AIInterface


@dataclass
class CommandResult:
    """Result of executing a slash command.

    Attributes:
        success: Whether the command executed successfully
        message: Human-readable result message
        data: Optional additional data from the command
    """
    success: bool
    message: str
    data: Optional[Any] = None


@dataclass
class CommandInfo:
    """Information about a registered command.

    Attributes:
        name: Command name without the slash (e.g., "help")
        description: Short description of what the command does
        handler: Callable that executes the command
        usage: Optional usage string (e.g., "/save [name]")
    """
    name: str
    description: str
    handler: Callable[[List[str]], CommandResult]
    usage: str = ""


class SlashCommandHandler:
    """Handles slash commands from the chat interface.

    Parses commands starting with "/" and executes them locally without
    sending to the AI backend. Provides help, autocomplete suggestions,
    and fuzzy matching for unknown commands.

    Attributes:
        canvas: Canvas instance for drawing operations
        workspace_manager: WorkspaceManager for persistence operations
        ai_interface: AIInterface for test execution and conversations
        commands: Registry of available commands
    """

    def __init__(
        self,
        canvas: "Canvas",
        workspace_manager: "WorkspaceManager",
        ai_interface: "AIInterface",
    ) -> None:
        """Initialize the slash command handler.

        Args:
            canvas: Canvas instance for drawing operations
            workspace_manager: WorkspaceManager for persistence operations
            ai_interface: AIInterface for test execution and conversations
        """
        self.canvas: "Canvas" = canvas
        self.workspace_manager: "WorkspaceManager" = workspace_manager
        self.ai_interface: "AIInterface" = ai_interface
        self.commands: Dict[str, CommandInfo] = self._register_commands()

    def _register_commands(self) -> Dict[str, CommandInfo]:
        """Register all available slash commands.

        Returns:
            Dictionary mapping command names to CommandInfo objects.
        """
        commands: Dict[str, CommandInfo] = {}

        # Essential commands
        commands["help"] = CommandInfo(
            name="help",
            description="Show available commands or topic help",
            handler=self._cmd_help,
            usage="/help [command]",
        )
        commands["undo"] = CommandInfo(
            name="undo",
            description="Undo last action",
            handler=self._cmd_undo,
            usage="/undo",
        )
        commands["redo"] = CommandInfo(
            name="redo",
            description="Redo last undone action",
            handler=self._cmd_redo,
            usage="/redo",
        )
        commands["clear"] = CommandInfo(
            name="clear",
            description="Clear all objects from canvas",
            handler=self._cmd_clear,
            usage="/clear",
        )
        commands["reset"] = CommandInfo(
            name="reset",
            description="Reset view (zoom/pan) to default",
            handler=self._cmd_reset,
            usage="/reset",
        )
        commands["save"] = CommandInfo(
            name="save",
            description="Save workspace",
            handler=self._cmd_save,
            usage="/save [name]",
        )
        commands["load"] = CommandInfo(
            name="load",
            description="Load workspace",
            handler=self._cmd_load,
            usage="/load [name]",
        )
        commands["workspaces"] = CommandInfo(
            name="workspaces",
            description="List saved workspaces",
            handler=self._cmd_workspaces,
            usage="/workspaces",
        )

        # Canvas & View Control
        commands["fit"] = CommandInfo(
            name="fit",
            description="Fit view to show all objects",
            handler=self._cmd_fit,
            usage="/fit",
        )
        commands["zoom"] = CommandInfo(
            name="zoom",
            description="Zoom canvas in/out or to specific factor",
            handler=self._cmd_zoom,
            usage="/zoom <in|out|factor>",
        )
        commands["grid"] = CommandInfo(
            name="grid",
            description="Toggle grid visibility",
            handler=self._cmd_grid,
            usage="/grid",
        )
        commands["axes"] = CommandInfo(
            name="axes",
            description="Toggle coordinate axes",
            handler=self._cmd_axes,
            usage="/axes",
        )
        commands["polar"] = CommandInfo(
            name="polar",
            description="Switch to polar coordinate system",
            handler=self._cmd_polar,
            usage="/polar",
        )
        commands["cartesian"] = CommandInfo(
            name="cartesian",
            description="Switch to cartesian coordinate system",
            handler=self._cmd_cartesian,
            usage="/cartesian",
        )
        commands["status"] = CommandInfo(
            name="status",
            description="Show canvas info (object count, bounds)",
            handler=self._cmd_status,
            usage="/status",
        )

        # Utility commands
        commands["vision"] = CommandInfo(
            name="vision",
            description="Toggle vision mode",
            handler=self._cmd_vision,
            usage="/vision",
        )
        commands["test"] = CommandInfo(
            name="test",
            description="Run client test suite",
            handler=self._cmd_test,
            usage="/test",
        )
        commands["export"] = CommandInfo(
            name="export",
            description="Export canvas as JSON",
            handler=self._cmd_export,
            usage="/export",
        )
        commands["import"] = CommandInfo(
            name="import",
            description="Import canvas state from JSON",
            handler=self._cmd_import,
            usage="/import <json>",
        )
        commands["list"] = CommandInfo(
            name="list",
            description="List all objects on canvas",
            handler=self._cmd_list,
            usage="/list",
        )
        commands["new"] = CommandInfo(
            name="new",
            description="Start fresh (clear canvas + new conversation)",
            handler=self._cmd_new,
            usage="/new",
        )
        commands["model"] = CommandInfo(
            name="model",
            description="Switch AI model or show current",
            handler=self._cmd_model,
            usage="/model [name]",
        )

        return commands

    def is_slash_command(self, message: str) -> bool:
        """Check if a message is a slash command.

        Args:
            message: The message to check

        Returns:
            True if the message starts with "/" and is a valid command format
        """
        stripped = message.strip()
        if not stripped.startswith("/"):
            return False
        # Must have at least one character after the slash
        if len(stripped) < 2:
            return False
        # The first character after slash must be alphabetic
        return stripped[1].isalpha()

    def execute(self, message: str) -> CommandResult:
        """Execute a slash command.

        Args:
            message: The full message including the slash command

        Returns:
            CommandResult with success status and result message
        """
        command_name, args = self._parse_command(message)

        if command_name in self.commands:
            try:
                return self.commands[command_name].handler(args)
            except Exception as e:
                return CommandResult(
                    success=False,
                    message=f"Error executing /{command_name}: {str(e)}",
                )

        # Unknown command - suggest similar commands
        suggestions = self._get_similar_commands(command_name)
        if suggestions:
            suggestion_str = ", ".join(f"/{s}" for s in suggestions[:3])
            return CommandResult(
                success=False,
                message=f"Unknown command: /{command_name}. Did you mean: {suggestion_str}? Use /help for available commands.",
            )

        return CommandResult(
            success=False,
            message=f"Unknown command: /{command_name}. Use /help for available commands.",
        )

    def _parse_command(self, message: str) -> Tuple[str, List[str]]:
        """Parse a slash command into command name and arguments.

        Args:
            message: The full message including the slash

        Returns:
            Tuple of (command_name, list_of_arguments)
        """
        stripped = message.strip()
        # Remove the leading slash
        without_slash = stripped[1:]

        # Split by whitespace
        parts = without_slash.split()

        if not parts:
            return ("", [])

        command_name = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        return (command_name, args)

    def _get_similar_commands(self, input_cmd: str) -> List[str]:
        """Find commands similar to the input for suggestions.

        Uses simple prefix matching and edit distance for fuzzy matching.

        Args:
            input_cmd: The unknown command entered by the user

        Returns:
            List of similar command names, sorted by relevance
        """
        suggestions: List[Tuple[int, str]] = []
        input_lower = input_cmd.lower()

        for cmd_name in self.commands:
            # Prefix match
            if cmd_name.startswith(input_lower):
                suggestions.append((0, cmd_name))
                continue

            # Contains match
            if input_lower in cmd_name:
                suggestions.append((1, cmd_name))
                continue

            # Simple edit distance (Levenshtein-like) for close matches
            distance = self._edit_distance(input_lower, cmd_name)
            if distance <= 2:  # Allow up to 2 edits
                suggestions.append((2 + distance, cmd_name))

        # Sort by score and return just the names
        suggestions.sort(key=lambda x: x[0])
        return [name for _, name in suggestions]

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate simple edit distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Minimum number of edits to transform s1 into s2
        """
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))
            distances = new_distances

        return distances[-1]

    def get_commands_list(self) -> List[Tuple[str, str]]:
        """Get list of all commands with descriptions for autocomplete.

        Returns:
            List of tuples (command_name, description)
        """
        return [(f"/{cmd.name}", cmd.description) for cmd in self.commands.values()]

    # ============================================================
    # Command Handlers
    # ============================================================

    def _cmd_help(self, args: List[str]) -> CommandResult:
        """Show help for commands."""
        if args:
            # Help for specific command
            cmd_name = args[0].lower().lstrip("/")
            if cmd_name in self.commands:
                cmd = self.commands[cmd_name]
                message = f"**/{cmd.name}**\n{cmd.description}\nUsage: `{cmd.usage}`"
                return CommandResult(success=True, message=message)
            return CommandResult(
                success=False,
                message=f"Unknown command: /{cmd_name}. Use /help for available commands.",
            )

        # General help - list all commands
        lines = ["**Available Commands:**\n"]

        # Group commands by category
        essential = ["help", "undo", "redo", "clear", "reset", "save", "load", "workspaces"]
        canvas_view = ["fit", "zoom", "grid", "axes", "polar", "cartesian", "status"]
        utility = ["vision", "test", "export", "import", "list", "new", "model"]

        lines.append("**Essential:**")
        for cmd_name in essential:
            if cmd_name in self.commands:
                cmd = self.commands[cmd_name]
                lines.append(f"  `/{cmd.name}` - {cmd.description}")

        lines.append("\n**Canvas & View:**")
        for cmd_name in canvas_view:
            if cmd_name in self.commands:
                cmd = self.commands[cmd_name]
                lines.append(f"  `/{cmd.name}` - {cmd.description}")

        lines.append("\n**Utility:**")
        for cmd_name in utility:
            if cmd_name in self.commands:
                cmd = self.commands[cmd_name]
                lines.append(f"  `/{cmd.name}` - {cmd.description}")

        lines.append("\nType `/help <command>` for detailed help on a specific command.")

        return CommandResult(success=True, message="\n".join(lines))

    def _cmd_undo(self, args: List[str]) -> CommandResult:
        """Undo the last action."""
        result = self.canvas.undo()
        if result:
            return CommandResult(success=True, message="Undo successful.")
        return CommandResult(success=False, message="Nothing to undo.")

    def _cmd_redo(self, args: List[str]) -> CommandResult:
        """Redo the last undone action."""
        result = self.canvas.redo()
        if result:
            return CommandResult(success=True, message="Redo successful.")
        return CommandResult(success=False, message="Nothing to redo.")

    def _cmd_clear(self, args: List[str]) -> CommandResult:
        """Clear all objects from the canvas."""
        self.canvas.clear()
        return CommandResult(success=True, message="Canvas cleared.")

    def _cmd_reset(self, args: List[str]) -> CommandResult:
        """Reset the view (zoom/pan) to default."""
        self.canvas.reset()
        return CommandResult(success=True, message="View reset to default.")

    def _cmd_save(self, args: List[str]) -> CommandResult:
        """Save the current workspace."""
        name = args[0] if args else None
        result = self.workspace_manager.save_workspace(name)
        return CommandResult(success="success" in result.lower(), message=result)

    def _cmd_load(self, args: List[str]) -> CommandResult:
        """Load a workspace."""
        name = args[0] if args else None
        result = self.workspace_manager.load_workspace(name)
        return CommandResult(success="success" in result.lower(), message=result)

    def _cmd_workspaces(self, args: List[str]) -> CommandResult:
        """List all saved workspaces."""
        result = self.workspace_manager.list_workspaces()
        if result == "None":
            return CommandResult(success=True, message="No saved workspaces found.")
        return CommandResult(success=True, message=f"**Saved workspaces:** {result}")

    def _cmd_fit(self, args: List[str]) -> CommandResult:
        """Fit the view to show all objects."""
        drawables = self.canvas.get_drawables()
        if not drawables:
            return CommandResult(success=False, message="No objects on canvas to fit.")

        # Calculate bounds of all drawables
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")

        for drawable in drawables:
            try:
                if hasattr(drawable, "x"):
                    min_x = min(min_x, drawable.x)
                    max_x = max(max_x, drawable.x)
                    min_y = min(min_y, drawable.y if hasattr(drawable, "y") else 0)
                    max_y = max(max_y, drawable.y if hasattr(drawable, "y") else 0)
                if hasattr(drawable, "p1") and hasattr(drawable, "p2"):
                    min_x = min(min_x, drawable.p1.x, drawable.p2.x)
                    max_x = max(max_x, drawable.p1.x, drawable.p2.x)
                    min_y = min(min_y, drawable.p1.y, drawable.p2.y)
                    max_y = max(max_y, drawable.p1.y, drawable.p2.y)
                if hasattr(drawable, "center"):
                    center = drawable.center
                    radius = getattr(drawable, "radius", 0)
                    min_x = min(min_x, center.x - radius)
                    max_x = max(max_x, center.x + radius)
                    min_y = min(min_y, center.y - radius)
                    max_y = max(max_y, center.y + radius)
            except Exception:
                continue

        if min_x == float("inf"):
            return CommandResult(success=False, message="Could not calculate bounds.")

        # Add padding (10%)
        padding = max(max_x - min_x, max_y - min_y) * 0.1
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding

        # Calculate center and range
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        range_x = (max_x - min_x) / 2
        range_y = (max_y - min_y) / 2

        # Use the larger range to ensure all objects fit
        if range_x >= range_y:
            self.canvas.zoom(center_x, center_y, range_x, "x")
        else:
            self.canvas.zoom(center_x, center_y, range_y, "y")

        return CommandResult(success=True, message="View fitted to show all objects.")

    def _cmd_zoom(self, args: List[str]) -> CommandResult:
        """Zoom the canvas."""
        if not args:
            return CommandResult(
                success=False,
                message="Usage: /zoom <in|out|factor>\nExamples: /zoom in, /zoom out, /zoom 2",
            )

        zoom_arg = args[0].lower()

        # Get current view center from coordinate mapper
        bounds = self.canvas.coordinate_mapper.get_visible_bounds()
        center_x = (bounds["left"] + bounds["right"]) / 2
        center_y = (bounds["top"] + bounds["bottom"]) / 2
        current_range_x = (bounds["right"] - bounds["left"]) / 2

        if zoom_arg == "in":
            # Zoom in by 50%
            new_range = current_range_x / 1.5
            self.canvas.zoom(center_x, center_y, new_range, "x")
            return CommandResult(success=True, message="Zoomed in.")
        elif zoom_arg == "out":
            # Zoom out by 50%
            new_range = current_range_x * 1.5
            self.canvas.zoom(center_x, center_y, new_range, "x")
            return CommandResult(success=True, message="Zoomed out.")
        else:
            # Try to parse as a factor
            try:
                factor = float(zoom_arg)
                if factor <= 0:
                    return CommandResult(success=False, message="Zoom factor must be positive.")
                new_range = current_range_x / factor
                self.canvas.zoom(center_x, center_y, new_range, "x")
                # Format factor nicely (remove .0 for whole numbers)
                factor_str = f"{factor:g}x"
                return CommandResult(success=True, message=f"Zoomed to {factor_str}.")
            except (ValueError, Exception) as e:
                # Brython may throw different exception types for float conversion
                if "could not convert" in str(e).lower() or "invalid" in str(e).lower():
                    return CommandResult(
                        success=False,
                        message=f"Invalid zoom argument: {zoom_arg}. Use 'in', 'out', or a number.",
                    )
                raise  # Re-raise if it's a different error

    def _cmd_grid(self, args: List[str]) -> CommandResult:
        """Toggle grid visibility."""
        current = self.canvas.is_grid_visible()
        self.canvas.set_grid_visible(not current)
        status = "visible" if not current else "hidden"
        return CommandResult(success=True, message=f"Grid is now {status}.")

    def _cmd_axes(self, args: List[str]) -> CommandResult:
        """Toggle coordinate axes visibility."""
        # Use the same mechanism as /grid to support both cartesian and polar modes
        current = self.canvas.is_grid_visible()
        self.canvas.set_grid_visible(not current)
        status = "visible" if not current else "hidden"
        return CommandResult(success=True, message=f"Axes are now {status}.")

    def _cmd_polar(self, args: List[str]) -> CommandResult:
        """Switch to polar coordinate system."""
        self.canvas.set_coordinate_system("polar")
        return CommandResult(success=True, message="Switched to polar coordinate system.")

    def _cmd_cartesian(self, args: List[str]) -> CommandResult:
        """Switch to cartesian coordinate system."""
        self.canvas.set_coordinate_system("cartesian")
        return CommandResult(success=True, message="Switched to cartesian coordinate system.")

    def _cmd_status(self, args: List[str]) -> CommandResult:
        """Show canvas status information."""
        state = self.canvas.get_canvas_state()

        # Count objects by type
        counts: Dict[str, int] = {}
        for key, value in state.items():
            if isinstance(value, list) and key not in ("computations",):
                count = len(value)
                if count > 0:
                    counts[key] = count

        # Get view bounds
        visible_bounds = self.canvas.coordinate_mapper.get_visible_bounds()
        bounds = f"x: [{visible_bounds['left']:.2f}, {visible_bounds['right']:.2f}], y: [{visible_bounds['bottom']:.2f}, {visible_bounds['top']:.2f}]"

        # Get coordinate system
        coord_system = self.canvas.get_coordinate_system()

        lines = ["**Canvas Status:**"]
        lines.append(f"  Coordinate system: {coord_system}")
        lines.append(f"  View bounds: {bounds}")

        if counts:
            lines.append("\n**Objects:**")
            for obj_type, count in sorted(counts.items()):
                lines.append(f"  {obj_type}: {count}")
            total = sum(counts.values())
            lines.append(f"  **Total:** {total}")
        else:
            lines.append("\n  No objects on canvas.")

        return CommandResult(success=True, message="\n".join(lines))

    def _cmd_vision(self, args: List[str]) -> CommandResult:
        """Toggle vision mode."""
        try:
            vision_toggle = document["vision-toggle"]
            current = vision_toggle.checked
            vision_toggle.checked = not current
            status = "enabled" if not current else "disabled"
            return CommandResult(success=True, message=f"Vision mode {status}.")
        except Exception as e:
            return CommandResult(success=False, message=f"Could not toggle vision: {str(e)}")

    def _cmd_test(self, args: List[str]) -> CommandResult:
        """Run the client test suite."""
        try:
            results = self.ai_interface.run_tests()

            tests_run = results.get("tests_run", 0)
            failures = results.get("failures", 0)
            errors = results.get("errors", 0)

            # Handle both old and new result formats
            if "summary" in results:
                tests_run = results["summary"].get("tests", tests_run)
                failures = results["summary"].get("failures", failures)
                errors = results["summary"].get("errors", errors)

            lines = ["**Test Results:**"]
            lines.append(f"  Tests run: {tests_run}")
            lines.append(f"  Failures: {failures}")
            lines.append(f"  Errors: {errors}")

            if failures > 0 and "failing_tests" in results:
                lines.append("\n**Failures:**")
                for fail in results["failing_tests"][:5]:  # Limit to 5
                    lines.append(f"  - {fail.get('test', 'Unknown')}")

            if errors > 0 and "error_tests" in results:
                lines.append("\n**Errors:**")
                for err in results["error_tests"][:5]:  # Limit to 5
                    lines.append(f"  - {err.get('test', 'Unknown')}")

            success = failures == 0 and errors == 0
            return CommandResult(success=success, message="\n".join(lines), data=results)
        except Exception as e:
            return CommandResult(success=False, message=f"Error running tests: {str(e)}")

    def _cmd_export(self, args: List[str]) -> CommandResult:
        """Export canvas state as JSON."""
        state = self.canvas.get_canvas_state()
        json_str = json.dumps(state, indent=2)

        # Return raw JSON - copy will get the full JSON
        # The display will show expandable content if too long
        return CommandResult(
            success=True,
            message=json_str,
            data=state,
        )

    def _cmd_import(self, args: List[str]) -> CommandResult:
        """Import canvas state from JSON."""
        if not args:
            return CommandResult(
                success=False,
                message="Usage: /import <valid JSON canvas>\nPaste valid JSON canvas state after /import command.",
            )

        # Join all args in case JSON was split by whitespace
        json_str = " ".join(args)

        try:
            state = json.loads(json_str)
            if not isinstance(state, dict):
                return CommandResult(success=False, message="Invalid state: expected JSON object.")

            self.workspace_manager._restore_workspace_state(state)
            return CommandResult(success=True, message="Canvas state imported successfully.")
        except (json.JSONDecodeError, ValueError) as e:
            # In Brython, JSON errors may be ValueError instead of JSONDecodeError
            return CommandResult(success=False, message=f"Invalid JSON: {str(e)}")
        except Exception as e:
            return CommandResult(success=False, message=f"Error importing state: {str(e)}")

    def _cmd_list(self, args: List[str]) -> CommandResult:
        """List all objects on the canvas."""
        state = self.canvas.get_canvas_state()

        lines = ["**Objects on Canvas:**"]
        found_any = False

        for key, value in sorted(state.items()):
            if isinstance(value, list) and key not in ("computations",) and value:
                found_any = True
                lines.append(f"\n**{key}:**")
                for item in value[:20]:  # Limit to 20 per type
                    if isinstance(item, dict):
                        name = item.get("name", "Unnamed")
                        lines.append(f"  - {name}")
                    else:
                        lines.append(f"  - {item}")
                if len(value) > 20:
                    lines.append(f"  ... and {len(value) - 20} more")

        if not found_any:
            lines.append("  No objects on canvas.")

        return CommandResult(success=True, message="\n".join(lines))

    def _cmd_new(self, args: List[str]) -> CommandResult:
        """Start fresh with a new canvas and conversation."""
        # This mimics the "New Conversation" button behavior
        try:
            # The start_new_conversation method handles save, clear, and reset
            self.ai_interface.start_new_conversation(None)
            return CommandResult(success=True, message="Started new session. Canvas cleared and conversation reset.")
        except Exception as e:
            return CommandResult(success=False, message=f"Error starting new session: {str(e)}")

    def _cmd_model(self, args: List[str]) -> CommandResult:
        """Show or switch the AI model."""
        try:
            model_selector = document["ai-model-selector"]

            if not args:
                # Show current model
                current = model_selector.value
                return CommandResult(success=True, message=f"Current AI model: **{current}**")

            # Try to switch model
            new_model = args[0]

            # Get available options
            options = [opt.value for opt in model_selector.options]

            if new_model in options:
                model_selector.value = new_model
                return CommandResult(success=True, message=f"Switched to model: **{new_model}**")

            # Try case-insensitive match
            for opt in options:
                if opt.lower() == new_model.lower():
                    model_selector.value = opt
                    return CommandResult(success=True, message=f"Switched to model: **{opt}**")

            # Model not found
            options_str = ", ".join(options)
            return CommandResult(
                success=False,
                message=f"Unknown model: {new_model}. Available models: {options_str}",
            )
        except Exception as e:
            return CommandResult(success=False, message=f"Error accessing model selector: {str(e)}")
