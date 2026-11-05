"""
sign_vocabulary/speech_listener.py
==================================
Asynchronous Microphone Input & Speech-to-Text Recognition Engine for Stage 5.

Features:
  - Non-blocking background listener thread (UI never stutters).
  - Robust exception handling for missing mics, permissions, or network timeouts.
  - Clear user-facing error messages for all failure modes.
  - Callback triggering upon successful recognition to auto-start sign video playback.
"""

from enum import Enum
import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False
    sr = None


class SpeechState(Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class SpeechToTextEngine:
    """
    Manages non-blocking microphone recording and speech recognition.
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        on_text_recognized: Optional[Callable[[str], None]] = None,
        timeout: float = 6.0,
        phrase_time_limit: float = 8.0,
    ):
        self.device_index = device_index
        self.on_text_recognized = on_text_recognized
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit

        self.state: SpeechState = SpeechState.IDLE
        self.last_recognized_text: str = ""
        self.last_error_message: Optional[str] = None
        self.is_available: bool = _SR_AVAILABLE

        self.recognizer: Optional[Any] = None
        if self.is_available:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True

        self._active_thread: Optional[threading.Thread] = None

    def start_listening(self) -> bool:
        """
        Triggers one-shot microphone speech recognition in a background thread.
        Returns True if started, False if already listening or engine unavailable.
        """
        if not self.is_available:
            self.state = SpeechState.ERROR
            self.last_error_message = "SpeechRecognition library not installed. Use text input [T]."
            return False

        if self.state in (SpeechState.LISTENING, SpeechState.PROCESSING):
            logger.info("Already listening or processing.")
            return False

        self.state = SpeechState.LISTENING
        self.last_error_message = None

        self._active_thread = threading.Thread(target=self._listen_worker, daemon=True)
        self._active_thread.start()
        return True

    def _listen_worker(self):
        """Worker thread that interacts with PyAudio and SpeechRecognition."""
        try:
            # 1. Test microphone device access
            try:
                mic = sr.Microphone(device_index=self.device_index)
            except Exception as e:
                self.state = SpeechState.ERROR
                self.last_error_message = f"Microphone device unavailable: {e}. Press [T] for Text Input."
                logger.error("Microphone init failed: %s", e)
                return

            with mic as source:
                # Ambient noise calibration
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                except Exception:
                    pass

                self.state = SpeechState.LISTENING
                logger.info("Microphone listening...")

                # Record audio
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.timeout,
                        phrase_time_limit=self.phrase_time_limit,
                    )
                except sr.WaitTimeoutError:
                    self.state = SpeechState.ERROR
                    self.last_error_message = "Listening timed out (no speech detected). Press [M] to retry or [T] to type."
                    logger.info("Speech recognition timed out.")
                    return

            # 2. Process audio via Google Speech Recognition API
            self.state = SpeechState.PROCESSING
            logger.info("Recognizing speech...")

            try:
                text = self.recognizer.recognize_google(audio)
                clean_text = text.strip()
                if clean_text:
                    self.last_recognized_text = clean_text
                    self.state = SpeechState.SUCCESS
                    self.last_error_message = None
                    logger.info("Speech recognized: \"%s\"", clean_text)
                    if self.on_text_recognized:
                        self.on_text_recognized(clean_text)
                else:
                    self.state = SpeechState.ERROR
                    self.last_error_message = "No words recognized in audio. Please try again."

            except sr.UnknownValueError:
                self.state = SpeechState.ERROR
                self.last_error_message = "Could not understand audio. Please speak clearly into the microphone or press [T] to type."
                logger.warning("Speech unrecognizable.")

            except sr.RequestError as e:
                self.state = SpeechState.ERROR
                self.last_error_message = f"Speech Recognition service error (Network issue). Press [T] to type."
                logger.error("Speech service error: %s", e)

        except Exception as e:
            self.state = SpeechState.ERROR
            self.last_error_message = f"Unexpected audio error: {e}. Press [T] for Text Input."
            logger.error("Speech listener error: %s", e, exc_info=True)
