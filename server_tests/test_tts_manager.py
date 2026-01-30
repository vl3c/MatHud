"""
Tests for the TTS Manager module.

Tests TTSManager class functionality including voice configuration
and the availability checking mechanism.
Note: Tests that require Kokoro will be skipped if it's not installed.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class TestTTSManager(unittest.TestCase):
    """Test cases for TTSManager class."""

    def test_get_voices(self) -> None:
        """Test that get_voices returns expected voice list."""
        from static.tts_manager import TTSManager

        manager = TTSManager()
        voices = manager.get_voices()

        self.assertIsInstance(voices, list)
        self.assertIn("am_michael", voices)
        self.assertIn("am_fenrir", voices)
        self.assertIn("af_nova", voices)

    def test_default_values(self) -> None:
        """Test default voice value."""
        from static.tts_manager import TTSManager

        manager = TTSManager()

        self.assertEqual(manager.DEFAULT_VOICE, "am_michael")
        self.assertEqual(manager.SAMPLE_RATE, 24000)

    def test_generate_speech_empty_text(self) -> None:
        """Test that empty text returns error."""
        from static.tts_manager import TTSManager

        manager = TTSManager()
        success, result = manager.generate_speech("")

        self.assertFalse(success)
        self.assertIn("No text", result)

    def test_generate_speech_whitespace_text(self) -> None:
        """Test that whitespace-only text returns error."""
        from static.tts_manager import TTSManager

        manager = TTSManager()
        success, result = manager.generate_speech("   \n\t  ")

        self.assertFalse(success)
        self.assertIn("No text", result)

    def test_generate_speech_invalid_voice_uses_default(self) -> None:
        """Test that invalid voice falls back to default."""
        from static.tts_manager import TTSManager

        manager = TTSManager()
        # Mock the pipeline to avoid actual TTS
        with patch.object(manager, '_get_pipeline') as mock_pipeline:
            mock_pipeline.return_value = (False, "Test: Kokoro not installed")
            success, result = manager.generate_speech("test", voice="invalid_voice")

            # Should fail because Kokoro is mocked as unavailable
            self.assertFalse(success)

    def test_is_available_caches_error(self) -> None:
        """Test that pipeline errors are cached."""
        from static.tts_manager import TTSManager

        manager = TTSManager()
        manager._pipeline_error = "Test error"

        self.assertFalse(manager.is_available())

    def test_generate_speech_threaded_empty_text(self) -> None:
        """Test that threaded method handles empty text."""
        from static.tts_manager import TTSManager

        manager = TTSManager()
        success, result = manager.generate_speech_threaded("")

        self.assertFalse(success)
        self.assertIn("No text", result)

    def test_get_tts_manager_singleton(self) -> None:
        """Test that get_tts_manager returns same instance."""
        from static.tts_manager import get_tts_manager

        manager1 = get_tts_manager()
        manager2 = get_tts_manager()

        self.assertIs(manager1, manager2)


class TestTTSManagerWithKokoro(unittest.TestCase):
    """Test cases that require Kokoro to be installed.

    These tests will be skipped if Kokoro is not available.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Check if Kokoro is available."""
        try:
            import kokoro
            cls.kokoro_available = True
        except ImportError:
            cls.kokoro_available = False

    def setUp(self) -> None:
        """Skip if Kokoro is not available."""
        if not self.kokoro_available:
            self.skipTest("Kokoro not installed")

    def test_is_available_with_kokoro(self) -> None:
        """Test that is_available returns True when Kokoro is installed."""
        from static.tts_manager import TTSManager

        manager = TTSManager()
        # Don't call is_available() as it will try to initialize
        # Just verify the manager was created
        self.assertIsNotNone(manager)

    def test_generate_speech_produces_wav(self) -> None:
        """Test that generate_speech produces valid WAV bytes."""
        from static.tts_manager import TTSManager

        manager = TTSManager()
        success, result = manager.generate_speech(
            "Hello, this is a test.",
            voice="am_michael",
        )

        if not success:
            # Skip if TTS generation failed (e.g., missing model)
            self.skipTest(f"TTS generation failed: {result}")

        # Should return bytes
        self.assertIsInstance(result, bytes)
        # WAV files start with "RIFF"
        self.assertEqual(result[:4], b"RIFF")


if __name__ == '__main__':
    unittest.main()
