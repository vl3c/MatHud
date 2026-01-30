"""
MatHud TTS Controller for Browser-Side Audio Playback

Manages text-to-speech playback in the browser, handling
the state machine for requesting, loading, and playing audio.

Features:
    - State machine: IDLE -> LOADING -> PLAYING -> IDLE
    - Audio blob management for efficient playback
    - Voice selection persistence via localStorage
    - Integration with AI message menu

Dependencies:
    - browser: Brython DOM and AJAX support
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from browser import ajax, document, window


# State type
TTSState = Literal["idle", "loading", "playing"]


class TTSController:
    """Controls TTS audio playback in the browser.

    Manages the complete lifecycle of TTS requests, from
    fetching audio from the server to playback control.

    Attributes:
        state: Current playback state
        on_state_change: Optional callback for state changes
    """

    # localStorage key for voice setting
    STORAGE_KEY_VOICE: str = "mathud.tts.voice"

    # Default value (synced with server TTSManager)
    DEFAULT_VOICE: str = "am_michael"

    def __init__(self) -> None:
        """Initialize the TTS controller."""
        self._state: TTSState = "idle"
        self._audio: Optional[Any] = None  # HTML5 Audio element
        self._audio_url: Optional[str] = None  # Blob URL for cleanup
        self._current_text: str = ""
        self._on_state_change: Optional[Callable[[TTSState], None]] = None
        self._on_error: Optional[Callable[[str], None]] = None

    @property
    def state(self) -> TTSState:
        """Get current playback state."""
        return self._state

    @property
    def on_state_change(self) -> Optional[Callable[[TTSState], None]]:
        """Get state change callback."""
        return self._on_state_change

    @on_state_change.setter
    def on_state_change(self, callback: Optional[Callable[[TTSState], None]]) -> None:
        """Set state change callback."""
        self._on_state_change = callback

    @property
    def on_error(self) -> Optional[Callable[[str], None]]:
        """Get error callback."""
        return self._on_error

    @on_error.setter
    def on_error(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set error callback."""
        self._on_error = callback

    def _set_state(self, new_state: TTSState) -> None:
        """Update state and notify listeners."""
        if self._state != new_state:
            self._state = new_state
            if self._on_state_change:
                try:
                    self._on_state_change(new_state)
                except Exception as e:
                    print(f"TTS state change callback error: {e}")

    def _notify_error(self, message: str) -> None:
        """Notify error callback if set."""
        print(f"TTS Error: {message}")
        if self._on_error:
            try:
                self._on_error(message)
            except Exception as e:
                print(f"TTS error callback error: {e}")

    def get_voice(self) -> str:
        """Get the currently selected voice from localStorage.

        Returns:
            Voice ID string
        """
        try:
            stored = window.localStorage.getItem(self.STORAGE_KEY_VOICE)
            if stored:
                return str(stored)
        except Exception:
            pass
        return self.DEFAULT_VOICE

    def set_voice(self, voice: str) -> None:
        """Save voice selection to localStorage.

        Args:
            voice: Voice ID to save
        """
        try:
            window.localStorage.setItem(self.STORAGE_KEY_VOICE, voice)
        except Exception as e:
            print(f"Failed to save TTS voice: {e}")

    def speak(self, text: str, voice: Optional[str] = None) -> None:
        """Request TTS playback for the given text.

        If already playing, stops current playback first.

        Args:
            text: Text to speak
            voice: Voice ID (uses stored preference if None)
        """
        if not text or not text.strip():
            return

        # Stop any current playback
        self.stop()

        # Use stored preference if not specified
        voice = voice or self.get_voice()

        self._current_text = text
        self._set_state("loading")

        # Make request to TTS endpoint
        try:
            req = ajax.ajax()
            req.bind('complete', self._on_request_complete)
            req.bind('error', self._on_request_error)
            req.responseType = 'blob'
            req.open('POST', '/api/tts', True)
            req.set_header('Content-Type', 'application/json')

            import json
            payload = json.dumps({
                'text': text,
                'voice': voice,
            })
            req.send(payload)

        except Exception as e:
            self._notify_error(f"Failed to request TTS: {e}")
            self._set_state("idle")

    def _on_request_complete(self, req: Any) -> None:
        """Handle TTS request completion."""
        try:
            if req.status == 200:
                # Got audio blob
                blob = req.response
                self._play_audio_blob(blob)
            else:
                # Error response - try to extract message from JSON
                error_msg = self._extract_error_message(req)
                self._notify_error(error_msg)
                self._set_state("idle")

        except Exception as e:
            self._notify_error(f"Error processing TTS response: {e}")
            self._set_state("idle")

    def _extract_error_message(self, req: Any) -> str:
        """Extract error message from failed request.

        Args:
            req: The AJAX request object

        Returns:
            Human-readable error message
        """
        status = getattr(req, 'status', 0)

        # Handle common HTTP status codes with helpful messages
        if status == 503:
            return "TTS service is not available. Please install Kokoro: pip install kokoro"
        elif status == 400:
            pass  # Try to get message from response
        elif status == 500:
            pass  # Try to get message from response

        # Try to parse JSON error response
        try:
            # Try responseText first (works better for non-blob responses)
            response_text = getattr(req, 'responseText', None) or getattr(req, 'text', None)
            if response_text:
                import json
                data = json.loads(response_text)
                if isinstance(data, dict):
                    msg = data.get('message', '')
                    if msg:
                        return f"TTS error: {msg}"
        except Exception:
            pass

        # Fallback to status code message
        return f"TTS request failed (HTTP {status})"

    def _on_request_error(self, req: Any) -> None:
        """Handle TTS request error."""
        self._notify_error("TTS request failed")
        self._set_state("idle")

    def _play_audio_blob(self, blob: Any) -> None:
        """Play audio from a blob.

        Args:
            blob: Audio blob from server response
        """
        try:
            # Create object URL for the blob
            url = window.URL.createObjectURL(blob)

            # Create Audio element
            audio = window.Audio.new(url)

            # Set up event handlers
            def on_ended(event: Any) -> None:
                self._cleanup_audio()
                self._set_state("idle")

            def on_error(event: Any) -> None:
                self._notify_error("Audio playback error")
                self._cleanup_audio()
                self._set_state("idle")

            audio.addEventListener('ended', on_ended)
            audio.addEventListener('error', on_error)

            # Store reference and play
            self._audio = audio
            self._audio_url = url
            audio.play()

            self._set_state("playing")

        except Exception as e:
            self._notify_error(f"Failed to play audio: {e}")
            self._set_state("idle")

    def _cleanup_audio(self) -> None:
        """Clean up audio resources."""
        try:
            if hasattr(self, '_audio_url') and self._audio_url:
                window.URL.revokeObjectURL(self._audio_url)
                self._audio_url = None
        except Exception:
            pass

        self._audio = None

    def stop(self) -> None:
        """Stop current playback."""
        if self._audio:
            try:
                self._audio.pause()
                self._audio.currentTime = 0
            except Exception:
                pass
            self._cleanup_audio()

        self._current_text = ""
        self._set_state("idle")

    def is_playing(self) -> bool:
        """Check if audio is currently playing.

        Returns:
            True if playing
        """
        return self._state == "playing"

    def is_loading(self) -> bool:
        """Check if TTS request is in progress.

        Returns:
            True if loading
        """
        return self._state == "loading"

    def is_idle(self) -> bool:
        """Check if controller is idle.

        Returns:
            True if idle
        """
        return self._state == "idle"


# Global TTS controller instance
_tts_controller: Optional[TTSController] = None


def get_tts_controller() -> TTSController:
    """Get the global TTS controller instance.

    Returns:
        TTSController instance
    """
    global _tts_controller
    if _tts_controller is None:
        _tts_controller = TTSController()
    return _tts_controller
