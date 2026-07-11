"""TTS UI manager for the AI interface.

Manages TTS read-aloud actions, markdown stripping for speech, and
the TTS settings modal dialog.

Extracted from ``AIInterface`` to reduce god-class complexity while
preserving the identical public behaviour.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from browser import document, html

from tts_controller import get_tts_controller, TTSController, TTS_VOICE_OPTIONS


class TTSUIManager:
    """Manages TTS UI interactions: read-aloud, settings modal, and text cleanup.

    Attributes:
        _tts_controller: The browser-side TTS playback controller.
        _settings_modal: The currently open settings modal DOM element, or ``None``.
        _on_system_message: Optional callback to display system messages in chat.
    """

    def __init__(self, on_system_message: Optional[Callable[[str], None]] = None) -> None:
        self._tts_controller: TTSController = get_tts_controller()
        self._settings_modal: Optional[Any] = None
        self._on_system_message = on_system_message

    # ── Read aloud ───────────────────────────────────────────────

    def handle_read_aloud(self, text: str, button_element: Any) -> None:
        """Handle TTS read aloud action.

        Args:
            text: Text to read aloud
            button_element: The menu button to update based on state
        """
        if not text or not text.strip():
            return

        # If already playing, stop instead
        if self._tts_controller.is_playing():
            self._tts_controller.stop()
            return

        # Set up state change callback to update button text
        def on_state_change(state: str) -> None:
            try:
                if state == "loading":
                    button_element.text = "Loading..."
                    button_element.classList.add("tts-loading")
                    button_element.classList.remove("tts-playing")
                elif state == "playing":
                    button_element.text = "Stop reading"
                    button_element.classList.remove("tts-loading")
                    button_element.classList.add("tts-playing")
                else:
                    button_element.text = "Read aloud"
                    button_element.classList.remove("tts-loading")
                    button_element.classList.remove("tts-playing")
            except Exception:
                pass

        # Set up error callback to show message to user
        def on_error(message: str) -> None:
            if self._on_system_message is not None:
                self._on_system_message(message)

        self._tts_controller.on_state_change = on_state_change
        self._tts_controller.on_error = on_error

        # Strip markdown formatting for cleaner TTS (basic cleanup)
        clean_text = self.strip_markdown(text)

        # Start TTS
        self._tts_controller.speak(clean_text)

    # ── Markdown stripping ───────────────────────────────────────

    def strip_markdown(self, text: str) -> str:
        """Strip markdown formatting from text for cleaner TTS output.

        Args:
            text: Text with potential markdown formatting

        Returns:
            Clean text suitable for TTS
        """
        result = text

        # Remove code blocks
        result = re.sub(r"```[\s\S]*?```", "", result)
        result = re.sub(r"`[^`]+`", "", result)

        # Remove headers
        result = re.sub(r"^#{1,6}\s+", "", result, flags=re.MULTILINE)

        # Remove bold/italic
        result = re.sub(r"\*\*([^*]+)\*\*", r"\1", result)
        result = re.sub(r"\*([^*]+)\*", r"\1", result)
        result = re.sub(r"__([^_]+)__", r"\1", result)
        result = re.sub(r"_([^_]+)_", r"\1", result)

        # Remove links, keep text
        result = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", result)

        # Remove images
        result = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", result)

        # Remove horizontal rules
        result = re.sub(r"^[-*_]{3,}$", "", result, flags=re.MULTILINE)

        # Clean up extra whitespace
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = result.strip()

        return result

    # ── Settings modal ───────────────────────────────────────────

    def show_settings_modal(self) -> None:
        """Display the TTS settings modal dialog."""
        # Remove existing modal if present
        self._close_settings_modal()

        # Create modal backdrop
        modal = html.DIV(Class="tts-settings-modal")
        modal.id = "tts-settings-modal"

        # Create modal content
        content = html.DIV(Class="tts-settings-content")

        # Header
        header = html.DIV(Class="tts-settings-header")
        title = html.H3("TTS Settings")
        close_btn = html.BUTTON("\u00d7", Class="tts-settings-close")
        close_btn.attrs["type"] = "button"
        close_btn.attrs["title"] = "Close"
        header <= title
        header <= close_btn
        content <= header

        # Voice selection
        voice_group = html.DIV(Class="tts-settings-group")
        voice_label = html.LABEL("Voice:")
        voice_select = html.SELECT(id="tts-voice-select")

        # Voice options imported from tts_controller.TTS_VOICE_OPTIONS
        # (canonical list kept in sync with TTSManager.VOICES on the server)
        voices = TTS_VOICE_OPTIONS
        current_voice = self._tts_controller.get_voice()
        for voice_id, voice_name in voices:
            option = html.OPTION(voice_name, value=voice_id)
            if voice_id == current_voice:
                option.attrs["selected"] = "selected"
            voice_select <= option

        voice_group <= voice_label
        voice_group <= voice_select
        content <= voice_group

        # Buttons
        buttons = html.DIV(Class="tts-settings-buttons")
        save_btn = html.BUTTON("Save", Class="tts-settings-save")
        save_btn.attrs["type"] = "button"
        cancel_btn = html.BUTTON("Cancel", Class="tts-settings-cancel")
        cancel_btn.attrs["type"] = "button"
        buttons <= save_btn
        buttons <= cancel_btn
        content <= buttons

        modal <= content

        # Bind events
        def on_close(ev: Any) -> None:
            self._close_settings_modal()

        def on_save(ev: Any) -> None:
            try:
                voice_value = document["tts-voice-select"].value
                self._tts_controller.set_voice(voice_value)
            except Exception as e:
                print(f"Error saving TTS settings: {e}")
            self._close_settings_modal()

        def on_backdrop_click(ev: Any) -> None:
            if ev.target == modal:
                self._close_settings_modal()

        close_btn.bind("click", on_close)
        cancel_btn.bind("click", on_close)
        save_btn.bind("click", on_save)
        modal.bind("click", on_backdrop_click)

        # Add to document
        document <= modal
        self._settings_modal = modal

    def _close_settings_modal(self) -> None:
        """Close and remove the TTS settings modal."""
        try:
            if self._settings_modal:
                self._settings_modal.remove()
                self._settings_modal = None
            # Also try by ID in case reference was lost
            existing = document.select_one("#tts-settings-modal")
            if existing:
                existing.remove()
        except Exception:
            pass
