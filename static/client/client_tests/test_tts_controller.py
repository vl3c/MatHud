"""
Tests for the TTS Controller module.

Tests the client-side TTS controller state machine, settings persistence,
and audio playback coordination.
"""

from __future__ import annotations

import unittest
from typing import Any, List, Optional

from browser import window

from tts_controller import TTSController, get_tts_controller
from .simple_mock import SimpleMock


class TestTTSControllerState(unittest.TestCase):
    """Test cases for TTS controller state machine."""

    def test_initial_state_is_idle(self) -> None:
        """Test that controller starts in idle state."""
        controller = TTSController()
        self.assertEqual(controller.state, "idle")
        self.assertTrue(controller.is_idle())
        self.assertFalse(controller.is_loading())
        self.assertFalse(controller.is_playing())

    def test_state_methods_are_consistent(self) -> None:
        """Test that state helper methods match state property."""
        controller = TTSController()

        controller._state = "idle"
        self.assertTrue(controller.is_idle())
        self.assertFalse(controller.is_loading())
        self.assertFalse(controller.is_playing())

        controller._state = "loading"
        self.assertFalse(controller.is_idle())
        self.assertTrue(controller.is_loading())
        self.assertFalse(controller.is_playing())

        controller._state = "playing"
        self.assertFalse(controller.is_idle())
        self.assertFalse(controller.is_loading())
        self.assertTrue(controller.is_playing())

    def test_state_change_callback(self) -> None:
        """Test that state change callback is invoked."""
        controller = TTSController()
        states: List[str] = []

        def on_state_change(state: str) -> None:
            states.append(state)

        controller.on_state_change = on_state_change

        controller._set_state("loading")
        controller._set_state("playing")
        controller._set_state("idle")

        self.assertEqual(states, ["loading", "playing", "idle"])

    def test_state_change_callback_not_called_for_same_state(self) -> None:
        """Test that callback is not called when state doesn't change."""
        controller = TTSController()
        call_count = [0]

        def on_state_change(state: str) -> None:
            call_count[0] += 1

        controller.on_state_change = on_state_change
        controller._set_state("loading")
        controller._set_state("loading")  # Same state

        self.assertEqual(call_count[0], 1)


class TestTTSControllerSettings(unittest.TestCase):
    """Test cases for TTS controller settings persistence."""

    def test_default_voice(self) -> None:
        """Test default voice value."""
        controller = TTSController()
        self.assertEqual(controller.DEFAULT_VOICE, "am_michael")

    def test_get_voice_returns_default_when_not_set(self) -> None:
        """Test get_voice returns default when localStorage is empty."""
        controller = TTSController()
        # Clear localStorage entry
        try:
            window.localStorage.removeItem(controller.STORAGE_KEY_VOICE)
        except Exception:
            pass

        voice = controller.get_voice()
        self.assertEqual(voice, controller.DEFAULT_VOICE)

    def test_set_and_get_voice(self) -> None:
        """Test voice setting persistence."""
        controller = TTSController()
        controller.set_voice("af_nova")

        # Get with same controller
        self.assertEqual(controller.get_voice(), "af_nova")

        # Get with new controller (simulates page reload)
        controller2 = TTSController()
        self.assertEqual(controller2.get_voice(), "af_nova")

        # Restore default
        controller.set_voice(controller.DEFAULT_VOICE)


class TestTTSControllerSingleton(unittest.TestCase):
    """Test cases for TTS controller singleton access."""

    def test_get_tts_controller_returns_same_instance(self) -> None:
        """Test that get_tts_controller returns the same instance."""
        controller1 = get_tts_controller()
        controller2 = get_tts_controller()
        self.assertIs(controller1, controller2)


class TestTTSControllerSpeak(unittest.TestCase):
    """Test cases for TTS speak functionality."""

    def test_speak_empty_text_does_nothing(self) -> None:
        """Test that speak with empty text doesn't change state."""
        controller = TTSController()
        controller.speak("")
        self.assertEqual(controller.state, "idle")

    def test_speak_whitespace_text_does_nothing(self) -> None:
        """Test that speak with whitespace text doesn't change state."""
        controller = TTSController()
        controller.speak("   \n\t  ")
        self.assertEqual(controller.state, "idle")

    def test_stop_resets_state_to_idle(self) -> None:
        """Test that stop() resets state to idle."""
        controller = TTSController()
        controller._state = "playing"
        controller.stop()
        self.assertEqual(controller.state, "idle")

    def test_stop_clears_current_text(self) -> None:
        """Test that stop() clears the current text buffer."""
        controller = TTSController()
        controller._current_text = "Some text"
        controller.stop()
        self.assertEqual(controller._current_text, "")


class TestTTSControllerErrorHandling(unittest.TestCase):
    """Test cases for TTS controller error handling."""

    def test_error_callback_is_invoked(self) -> None:
        """Test that error callback is invoked on errors."""
        controller = TTSController()
        errors: List[str] = []

        def on_error(message: str) -> None:
            errors.append(message)

        controller.on_error = on_error
        controller._notify_error("Test error")

        self.assertEqual(len(errors), 1)
        self.assertIn("Test error", errors[0])

    def test_error_callback_exception_is_handled(self) -> None:
        """Test that exceptions in error callback don't crash."""
        controller = TTSController()

        def bad_callback(message: str) -> None:
            raise RuntimeError("Callback error")

        controller.on_error = bad_callback

        # Should not raise
        controller._notify_error("Test error")
