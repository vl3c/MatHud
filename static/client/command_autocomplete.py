"""
MatHud Command Autocomplete Popup

Provides an autocomplete popup for slash commands in the chat input field.
Shows available commands with descriptions and allows keyboard navigation.

Key Features:
    - Popup appears when "/" is typed at the start of input
    - Filters commands as user types (e.g., "/he" filters to /help)
    - Keyboard navigation with Up/Down arrows
    - Enter to select, Escape to dismiss
    - Click to select command

Dependencies:
    - browser: DOM manipulation for popup UI
    - slash_command_handler: Command registry for available commands
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

from browser import document, html

if TYPE_CHECKING:
    from slash_command_handler import SlashCommandHandler


class CommandAutocomplete:
    """Autocomplete popup for slash commands.

    Displays a filterable list of available commands above the chat input
    and allows selection via keyboard or mouse.

    Attributes:
        input_element: The chat input element to attach to
        command_handler: SlashCommandHandler with command registry
        popup_element: The popup DOM element
        filtered_commands: Currently displayed commands after filtering
        selected_index: Currently highlighted command index
        visible: Whether the popup is currently shown
    """

    def __init__(
        self,
        input_element: Any,  # DOM element
        command_handler: "SlashCommandHandler",
    ) -> None:
        """Initialize the autocomplete popup.

        Args:
            input_element: The chat input DOM element
            command_handler: SlashCommandHandler with command registry
        """
        self.input_element: Any = input_element
        self.command_handler: "SlashCommandHandler" = command_handler
        self.popup_element: Optional[Any] = None
        self.filtered_commands: List[Tuple[str, str]] = []
        self.selected_index: int = 0
        self.visible: bool = False
        self._on_select_callback: Optional[Callable[[str], None]] = None

        self._create_popup_element()
        self._bind_events()

    def _create_popup_element(self) -> None:
        """Create the popup DOM element and add it to the document."""
        try:
            # Create popup container
            self.popup_element = html.DIV(Class="command-autocomplete")
            self.popup_element.id = "command-autocomplete-popup"
            self.popup_element.style.display = "none"

            # Find the chat-input-container (may be grandparent if input is in chat-input-row)
            input_container = document.querySelector(".chat-input-container")
            if input_container:
                # Position relative to input container
                input_container.style.position = "relative"
                # Append popup to container - CSS positions it above with bottom: 100%
                input_container.appendChild(self.popup_element)
        except Exception as e:
            print(f"Error creating autocomplete popup: {e}")

    def _bind_events(self) -> None:
        """Bind event listeners to the input element."""
        try:
            self.input_element.bind("input", self._on_input)
            self.input_element.bind("keydown", self._on_keydown)
            self.input_element.bind("blur", self._on_blur)
        except Exception as e:
            print(f"Error binding autocomplete events: {e}")

    def _on_input(self, event: Any) -> None:
        """Handle input events to show/filter autocomplete."""
        try:
            value = self.input_element.value

            # Show popup when "/" is at the start
            if value.startswith("/"):
                prefix = value[1:].lower()  # Text after the slash
                self.filter(prefix)
                if self.filtered_commands:
                    self.show()
                else:
                    self.hide()
            else:
                self.hide()
        except Exception as e:
            print(f"Error handling autocomplete input: {e}")

    def _on_keydown(self, event: Any) -> None:
        """Handle keyboard navigation in the popup."""
        if not self.visible:
            return

        try:
            key = event.key

            if key == "ArrowDown":
                event.preventDefault()
                self.select_next()
            elif key == "ArrowUp":
                event.preventDefault()
                self.select_previous()
            elif key == "Enter":
                if self.filtered_commands:
                    event.preventDefault()
                    self.confirm_selection()
            elif key == "Escape":
                event.preventDefault()
                self.hide()
            elif key == "Tab":
                if self.filtered_commands:
                    event.preventDefault()
                    self.confirm_selection()
        except Exception as e:
            print(f"Error handling autocomplete keydown: {e}")

    def _on_blur(self, event: Any) -> None:
        """Handle blur events to hide popup."""
        # Use setTimeout to allow click events on popup items to fire first
        try:
            from browser import window
            window.setTimeout(self._delayed_hide, 150)
        except Exception:
            self.hide()

    def _delayed_hide(self) -> None:
        """Hide the popup after a small delay (for blur handling)."""
        self.hide()

    def show(self) -> None:
        """Show the autocomplete popup."""
        if self.popup_element is None:
            return

        try:
            self.popup_element.style.display = "block"
            self.visible = True
            self._render_filtered_commands()
        except Exception as e:
            print(f"Error showing autocomplete popup: {e}")

    def hide(self) -> None:
        """Hide the autocomplete popup."""
        if self.popup_element is None:
            return

        try:
            self.popup_element.style.display = "none"
            self.visible = False
            self.selected_index = 0
        except Exception as e:
            print(f"Error hiding autocomplete popup: {e}")

    def filter(self, prefix: str) -> None:
        """Filter commands matching the given prefix.

        Args:
            prefix: The text after "/" to filter by
        """
        # Check if we're in argument mode for commands that take workspace names
        workspace_suggestions = self._get_workspace_suggestions(prefix)
        if workspace_suggestions is not None:
            self.filtered_commands = workspace_suggestions
            self.selected_index = 0
            if self.visible:
                self._render_filtered_commands()
            return

        all_commands = self.command_handler.get_commands_list()

        if not prefix:
            # Show all commands when just "/" is typed
            self.filtered_commands = all_commands
        else:
            # Filter by prefix (case-insensitive)
            prefix_lower = prefix.lower()
            self.filtered_commands = [
                (cmd, desc) for cmd, desc in all_commands
                if cmd[1:].lower().startswith(prefix_lower)  # Skip the "/" in comparison
            ]

        # Reset selection to first item
        self.selected_index = 0

        # Re-render if visible
        if self.visible:
            self._render_filtered_commands()

    def _get_workspace_suggestions(self, prefix: str) -> Optional[List[Tuple[str, str]]]:
        """Get workspace suggestions for /load command.

        Args:
            prefix: The text after "/" (e.g., "load " or "load my")

        Returns:
            List of (command, description) tuples if in workspace mode, None otherwise
        """
        prefix_lower = prefix.lower()

        # Check if typing /load with a space (entering workspace name)
        if prefix_lower.startswith("load "):
            workspace_prefix = prefix[5:]  # Text after "load "
            return self._filter_workspaces("load", workspace_prefix)

        # Check if typing /model with a space (entering model name)
        if prefix_lower.startswith("model "):
            model_prefix = prefix[6:]  # Text after "model "
            return self._filter_models(model_prefix)

        return None

    def _filter_models(self, model_prefix: str) -> List[Tuple[str, str]]:
        """Filter AI models matching the given prefix.

        Args:
            model_prefix: Partial model name to filter by

        Returns:
            List of (command_with_model, description) tuples
        """
        try:
            from browser import document

            model_selector = document["ai-model-selector"]
            models = [opt.value for opt in model_selector.options]

            # Filter by prefix
            prefix_lower = model_prefix.lower()
            filtered = [
                m for m in models
                if m.lower().startswith(prefix_lower)
            ]

            # If no matches, show all models
            if not filtered and model_prefix:
                filtered = models

            # Build suggestions
            suggestions = [
                (f"/model {m}", f"Switch to model '{m}'")
                for m in filtered
            ]

            return suggestions if suggestions else [("/model", "No matching models")]

        except Exception as e:
            print(f"Error getting model suggestions: {e}")
            return []

    def _filter_workspaces(self, command: str, workspace_prefix: str) -> List[Tuple[str, str]]:
        """Filter workspaces matching the given prefix.

        Args:
            command: The command name (e.g., "load")
            workspace_prefix: Partial workspace name to filter by

        Returns:
            List of (command_with_workspace, description) tuples
        """
        try:
            # Get workspaces from the workspace manager
            workspace_manager = self.command_handler.workspace_manager
            workspaces_str = workspace_manager.list_workspaces()

            if workspaces_str == "None" or not workspaces_str:
                return [("/" + command, "No saved workspaces found")]

            # Parse workspace list (comma-separated)
            workspaces = [ws.strip() for ws in workspaces_str.split(",")]

            # Filter by prefix
            prefix_lower = workspace_prefix.lower()
            filtered = [
                ws for ws in workspaces
                if ws.lower().startswith(prefix_lower)
            ]

            # If no matches, show all workspaces
            if not filtered and workspace_prefix:
                filtered = workspaces

            # Build suggestions
            suggestions = [
                (f"/{command} {ws}", f"Load workspace '{ws}'")
                for ws in filtered[:15]  # Limit to 15 suggestions
            ]

            return suggestions if suggestions else [("/" + command, "No matching workspaces")]

        except Exception as e:
            print(f"Error getting workspace suggestions: {e}")
            return []

    def select_next(self) -> None:
        """Move selection to the next command."""
        if not self.filtered_commands:
            return

        self.selected_index = (self.selected_index + 1) % len(self.filtered_commands)
        self._render_filtered_commands()
        self._scroll_to_selected()

    def select_previous(self) -> None:
        """Move selection to the previous command."""
        if not self.filtered_commands:
            return

        self.selected_index = (self.selected_index - 1) % len(self.filtered_commands)
        self._render_filtered_commands()
        self._scroll_to_selected()

    def confirm_selection(self) -> str:
        """Confirm the current selection and insert into input.

        Returns:
            The selected command string
        """
        if not self.filtered_commands:
            return ""

        selected_cmd, _ = self.filtered_commands[self.selected_index]

        # Insert the command into the input field
        try:
            self.input_element.value = selected_cmd + " "
            self.input_element.focus()

            # Move cursor to end
            length = len(self.input_element.value)
            self.input_element.setSelectionRange(length, length)
        except Exception as e:
            print(f"Error inserting selected command: {e}")

        self.hide()

        # Call the callback if set
        if self._on_select_callback:
            self._on_select_callback(selected_cmd)

        return selected_cmd

    def _scroll_to_selected(self) -> None:
        """Scroll the popup to ensure the selected item is visible."""
        if self.popup_element is None:
            return

        try:
            items = self.popup_element.querySelectorAll(".command-autocomplete-item")
            if self.selected_index < len(items):
                selected_item = items[self.selected_index]
                selected_item.scrollIntoView({"block": "nearest"})
        except Exception:
            pass

    def _render_filtered_commands(self) -> None:
        """Render the filtered commands in the popup."""
        if self.popup_element is None:
            return

        try:
            # Clear existing items
            self.popup_element.clear()

            # Add items for each filtered command
            for index, (cmd, desc) in enumerate(self.filtered_commands):
                item = html.DIV(Class="command-autocomplete-item")

                if index == self.selected_index:
                    item.classList.add("selected")

                # Command name span
                cmd_name = html.SPAN(cmd, Class="command-name")

                # Description span
                cmd_desc = html.SPAN(desc, Class="command-description")

                item <= cmd_name
                item <= cmd_desc

                # Bind click handler
                def make_click_handler(idx: int) -> Callable[[Any], None]:
                    def handler(event: Any) -> None:
                        event.preventDefault()
                        event.stopPropagation()
                        self.selected_index = idx
                        self.confirm_selection()
                    return handler

                item.bind("mousedown", make_click_handler(index))

                # Bind hover handler for visual feedback
                def make_hover_handler(idx: int) -> Callable[[Any], None]:
                    def handler(event: Any) -> None:
                        self.selected_index = idx
                        self._render_filtered_commands()
                    return handler

                item.bind("mouseenter", make_hover_handler(index))

                self.popup_element <= item

        except Exception as e:
            print(f"Error rendering autocomplete commands: {e}")

    def set_on_select(self, callback: Callable[[str], None]) -> None:
        """Set a callback to be called when a command is selected.

        Args:
            callback: Function to call with the selected command string
        """
        self._on_select_callback = callback

    def destroy(self) -> None:
        """Clean up the autocomplete popup and event listeners."""
        try:
            if self.popup_element:
                self.popup_element.remove()
                self.popup_element = None

            # Unbind events (Brython doesn't have a clean way to do this,
            # but we can at least clear references)
            self.input_element = None
            self.command_handler = None
            self.filtered_commands = []
            self._on_select_callback = None
        except Exception as e:
            print(f"Error destroying autocomplete: {e}")
