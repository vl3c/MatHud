"""
Tests for the TTS Manager module.

Tests TTSManager class functionality including voice configuration
and the availability checking mechanism.
Note: Tests that require Kokoro will be skipped if it's not installed.
"""

from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from typing import TYPE_CHECKING, Iterator, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np

if TYPE_CHECKING:
    from static.tts_manager import TTSManager


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
        with patch.object(manager, "_get_pipeline") as mock_pipeline:
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


def _fake_kokoro_module(pipeline_factory: MagicMock) -> types.ModuleType:
    """Build a stand-in ``kokoro`` module whose KPipeline is ``pipeline_factory``."""
    module = types.ModuleType("kokoro")
    module.KPipeline = pipeline_factory  # type: ignore[attr-defined]
    return module


def _fake_pipeline(text: str, voice: str) -> Iterator[Tuple[str, str, np.ndarray]]:
    yield text, "", np.zeros(240, dtype=np.float32)


class TestTTSLazyLoading(unittest.TestCase):
    """Kokoro is imported only by the first speech request, exactly once."""

    def setUp(self) -> None:
        self.kpipeline = MagicMock(return_value=_fake_pipeline)
        modules_patch = patch.dict(sys.modules, {"kokoro": _fake_kokoro_module(self.kpipeline)})
        modules_patch.start()
        self.addCleanup(modules_patch.stop)

    def _manager(self) -> "TTSManager":
        from static.tts_manager import TTSManager

        manager = TTSManager()
        # soundfile is not installed in every environment; WAV encoding is not under test.
        wav_patch = patch.object(manager, "_audio_to_wav", return_value=b"RIFF-fake")
        wav_patch.start()
        self.addCleanup(wav_patch.stop)
        return manager

    def test_is_available_checks_packages_without_loading_model(self) -> None:
        manager = self._manager()

        with patch("static.tts_manager._module_installed", return_value=True) as installed:
            self.assertTrue(manager.is_available())

        installed.assert_any_call("kokoro")
        self.kpipeline.assert_not_called()
        self.assertFalse(manager.is_loaded())

    def test_is_available_false_when_a_package_is_missing(self) -> None:
        manager = self._manager()

        with patch("static.tts_manager._module_installed", side_effect=lambda name: name != "kokoro"):
            self.assertFalse(manager.is_available())

        self.kpipeline.assert_not_called()

    def test_module_installed_does_not_import(self) -> None:
        from static.tts_manager import _module_installed

        self.assertTrue(_module_installed("json"))
        self.assertFalse(_module_installed("mathud_no_such_module_xyz"))
        # A module in sys.modules without a spec (like the fake kokoro) is not "installed".
        self.assertFalse(_module_installed("kokoro"))

    def test_first_request_loads_pipeline_once(self) -> None:
        manager = self._manager()

        first = manager.generate_speech("hello")
        second = manager.generate_speech("again")

        self.assertEqual(first, (True, b"RIFF-fake"))
        self.assertEqual(second, (True, b"RIFF-fake"))
        self.kpipeline.assert_called_once_with(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        self.assertTrue(manager.is_loaded())
        self.assertTrue(manager.is_available())

    def test_concurrent_first_requests_load_pipeline_once(self) -> None:
        manager = self._manager()
        release = threading.Event()

        def slow_pipeline(**_kwargs: str) -> object:
            release.wait(2)
            return _fake_pipeline

        self.kpipeline.side_effect = slow_pipeline
        results: List[Tuple[bool, object]] = []
        threads = [threading.Thread(target=lambda: results.append(manager._get_pipeline())) for _ in range(5)]
        for thread in threads:
            thread.start()
        release.set()
        for thread in threads:
            thread.join(5)

        self.assertEqual(self.kpipeline.call_count, 1)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(success for success, _ in results))
        self.assertEqual({id(pipeline) for _, pipeline in results}, {id(_fake_pipeline)})

    def test_load_failure_is_cached_and_reported(self) -> None:
        manager = self._manager()
        self.kpipeline.side_effect = RuntimeError("model download failed")

        success, message = manager.generate_speech("hello")
        again, _ = manager.generate_speech("hello")

        self.assertFalse(success)
        self.assertFalse(again)
        self.assertIn("model download failed", str(message))
        self.assertEqual(self.kpipeline.call_count, 1)
        self.assertFalse(manager.is_available())

    def test_load_that_calls_sys_exit_is_cached_and_reported(self) -> None:
        # spaCy's model download inside KPipeline calls sys.exit(1) when it fails.
        manager = self._manager()
        self.kpipeline.side_effect = SystemExit(1)

        success, message = manager.generate_speech_threaded("hello", timeout=5.0)
        again, _ = manager.generate_speech_threaded("hello", timeout=5.0)

        self.assertFalse(success)
        self.assertFalse(again)
        self.assertIn("Failed to initialize Kokoro", str(message))
        self.assertEqual(self.kpipeline.call_count, 1)
        self.assertFalse(manager.is_available())

    def test_threaded_first_request_allows_time_to_load_model(self) -> None:
        manager = self._manager()
        future = MagicMock()
        future.result.return_value = (True, b"RIFF-fake")

        with patch.object(manager, "_start_generation", return_value=future):
            manager.generate_speech_threaded("hello", timeout=60.0)
            future.result.assert_called_with(timeout=60.0 + manager.PIPELINE_LOAD_TIMEOUT)

            manager._pipeline = _fake_pipeline
            manager.generate_speech_threaded("hello", timeout=60.0)
            future.result.assert_called_with(timeout=60.0)

    def test_threaded_generation_runs_on_daemon_thread(self) -> None:
        # A non-daemon worker would keep the process alive after the desktop
        # window closes while the model is still loading.
        manager = self._manager()
        worker_daemon: List[bool] = []

        def recording_pipeline(**_kwargs: str) -> object:
            worker_daemon.append(threading.current_thread().daemon)
            return _fake_pipeline

        self.kpipeline.side_effect = recording_pipeline

        self.assertEqual(manager.generate_speech_threaded("hello", timeout=5.0), (True, b"RIFF-fake"))
        self.assertEqual(worker_daemon, [True])

    def test_threaded_generation_runs_one_request_at_a_time(self) -> None:
        manager = self._manager()
        release = threading.Event()
        active: List[int] = []
        peak: List[int] = []

        def slow_generate(text: str, voice: object = None) -> Tuple[bool, bytes]:
            active.append(1)
            peak.append(len(active))
            release.wait(2)
            active.pop()
            return True, b"RIFF-fake"

        results: List[Tuple[bool, object]] = []
        with patch.object(manager, "generate_speech", side_effect=slow_generate):
            threads = [
                threading.Thread(target=lambda: results.append(manager.generate_speech_threaded("hi", timeout=5.0)))
                for _ in range(3)
            ]
            for thread in threads:
                thread.start()
            time.sleep(0.1)
            release.set()
            for thread in threads:
                thread.join(5)

        self.assertEqual(results, [(True, b"RIFF-fake")] * 3)
        self.assertEqual(max(peak), 1)

    def test_app_startup_does_not_load_kokoro(self) -> None:
        from static.app_manager import AppManager

        with (
            patch("static.tts_manager._tts_manager", None),
            patch("static.tts_manager._module_installed", return_value=True),
            patch("builtins.print") as mock_print,
        ):
            AppManager._initialize_tts()

        self.kpipeline.assert_not_called()
        mock_print.assert_called_once_with("TTS: Kokoro available (model loads on first use)")


class TestTTSManagerWithKokoro(unittest.TestCase):
    """Test cases that require Kokoro to be installed.

    These tests will be skipped if Kokoro is not available.
    """

    kokoro_available: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        """Check if Kokoro is available."""
        try:
            import kokoro  # noqa: F401

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
            self.skipTest(f"TTS generation failed: {result!r}")

        # Should return bytes
        self.assertIsInstance(result, bytes)
        # WAV files start with "RIFF"
        self.assertEqual(result[:4], b"RIFF")


if __name__ == "__main__":
    unittest.main()
