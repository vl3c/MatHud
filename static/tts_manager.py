"""
MatHud TTS Manager with Kokoro Integration

Provides text-to-speech functionality using the Kokoro-82M model.

Features:
    - Lazy-loaded Kokoro pipeline for efficient resource usage
    - Multiple voice options (male and female)
    - WAV format output for browser playback

Dependencies:
    - kokoro: Local TTS model
    - soundfile: Audio file I/O
    - numpy: Audio array operations

Note: Requires espeak-ng on Linux systems.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional, Tuple, Union

import numpy as np


class TTSManager:
    """Manages text-to-speech generation using Kokoro.

    Provides lazy-loaded TTS pipeline for natural voice synthesis.

    Attributes:
        VOICES: Available voice identifiers
        DEFAULT_VOICE: Default voice selection
        SAMPLE_RATE: Output audio sample rate (from Kokoro)
    """

    # Available voices
    VOICES: List[str] = [
        "am_michael",  # Male - clear, natural
        "am_fenrir",   # Male - deeper tone
        "am_onyx",     # Male - darker tone
        "am_echo",     # Male - resonant
        "af_nova",     # Female - clear
        "af_bella",    # Female - warm
    ]

    DEFAULT_VOICE: str = "am_michael"
    SAMPLE_RATE: int = 24000  # Kokoro's native sample rate

    def __init__(self) -> None:
        """Initialize TTS manager with lazy-loaded pipeline."""
        self._pipeline: Optional[object] = None
        self._pipeline_error: Optional[str] = None
        # Thread pool for non-blocking TTS generation (allows Ctrl+C to work)
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

    def _get_pipeline(self) -> Tuple[bool, Union[object, str]]:
        """Get or create the Kokoro pipeline.

        Returns:
            Tuple of (success, pipeline_or_error_message)
        """
        if self._pipeline_error is not None:
            return False, self._pipeline_error

        if self._pipeline is not None:
            return True, self._pipeline

        try:
            from kokoro import KPipeline

            # Initialize Kokoro pipeline for American English
            self._pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
            return True, self._pipeline

        except ImportError as e:
            self._pipeline_error = f"Kokoro not installed: {e}"
            return False, self._pipeline_error

        except Exception as e:
            self._pipeline_error = f"Failed to initialize Kokoro: {e}"
            return False, self._pipeline_error

    def is_available(self) -> bool:
        """Check if TTS is available (Kokoro installed and working).

        Returns:
            True if TTS can be used
        """
        success, _ = self._get_pipeline()
        return success

    def get_voices(self) -> List[str]:
        """Get list of available voice identifiers.

        Returns:
            List of voice IDs
        """
        return self.VOICES.copy()

    def generate_speech(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> Tuple[bool, Union[bytes, str]]:
        """Generate speech audio from text.

        Args:
            text: Text to convert to speech
            voice: Voice ID (default: am_michael)

        Returns:
            Tuple of (success, audio_bytes_or_error_message)
            Audio is returned as WAV format bytes
        """
        if not text or not text.strip():
            return False, "No text provided"

        # Validate voice
        voice = voice or self.DEFAULT_VOICE
        if voice not in self.VOICES:
            voice = self.DEFAULT_VOICE

        # Get pipeline
        success, result = self._get_pipeline()
        if not success:
            return False, str(result)

        pipeline = result

        try:
            # Generate audio using Kokoro
            # Kokoro returns a generator of (graphemes, phonemes, audio_chunk)
            audio_chunks: List[np.ndarray] = []

            for _, _, audio_chunk in pipeline(text, voice=voice):  # type: ignore
                if audio_chunk is not None:
                    audio_chunks.append(audio_chunk)

            if not audio_chunks:
                return False, "No audio generated"

            # Concatenate audio chunks
            audio = np.concatenate(audio_chunks)

            # Ensure float32 format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize to -1 to 1 range if needed
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val

            # Convert to WAV bytes
            wav_bytes = self._audio_to_wav(audio, self.SAMPLE_RATE)

            return True, wav_bytes

        except Exception as e:
            return False, f"TTS generation failed: {e}"

    def generate_speech_threaded(
        self,
        text: str,
        voice: Optional[str] = None,
        timeout: float = 60.0,
    ) -> Tuple[bool, Union[bytes, str]]:
        """Generate speech in a background thread (non-blocking for signal handling).

        This allows Ctrl+C to work while TTS is generating.

        Args:
            text: Text to convert to speech
            voice: Voice ID (default: am_michael)
            timeout: Maximum time to wait for generation (seconds)

        Returns:
            Tuple of (success, audio_bytes_or_error_message)
        """
        try:
            future: Future[Tuple[bool, Union[bytes, str]]] = self._executor.submit(
                self.generate_speech, text, voice
            )
            return future.result(timeout=timeout)
        except TimeoutError:
            return False, "TTS generation timed out"
        except Exception as e:
            return False, f"TTS generation failed: {e}"

    def _audio_to_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio array to WAV format bytes.

        Args:
            audio: Audio samples as numpy array (float32, -1 to 1)
            sample_rate: Sample rate in Hz

        Returns:
            WAV file as bytes
        """
        import soundfile as sf

        # Create in-memory buffer
        buffer = io.BytesIO()

        # Write WAV to buffer
        sf.write(buffer, audio, sample_rate, format='WAV', subtype='PCM_16')

        # Get bytes
        buffer.seek(0)
        return buffer.read()


# Global TTS manager instance for reuse
_tts_manager: Optional[TTSManager] = None


def get_tts_manager() -> TTSManager:
    """Get the global TTS manager instance.

    Returns:
        TTSManager instance
    """
    global _tts_manager
    if _tts_manager is None:
        _tts_manager = TTSManager()
    return _tts_manager
