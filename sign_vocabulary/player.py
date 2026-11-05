"""
sign_vocabulary/player.py
=========================
Sign Language Video & Fallback Playback Engine.

Manages:
  - Seamless sequential video playback of verified sign clips (.mp4).
  - Explicitly labelled fingerspelling fallback cards for unavailable words.
  - Interactive playback controls: Play, Pause, Resume, Replay, Step Next.
  - Sequence progress tracking and HUD overlay.
"""

from enum import Enum
import logging
from pathlib import Path
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    DEFAULT_PLAYBACK_FPS,
    FALLBACK_CARD_HOLD_SEC,
    FINGERSPELL_LETTER_HOLD_SEC,
)
from .manager import SignPlayItem

logger = logging.getLogger(__name__)


class PlaybackState(Enum):
    IDLE = "IDLE"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"


class SignVideoPlayer:
    """
    Renders sign language video sequences and fingerspelling fallbacks
    into standard frame buffers for the Left Panel display.
    """

    def __init__(
        self,
        fps: int = DEFAULT_PLAYBACK_FPS,
        letter_hold_sec: float = FINGERSPELL_LETTER_HOLD_SEC,
        fallback_hold_sec: float = FALLBACK_CARD_HOLD_SEC,
    ):
        self.fps = fps
        self.letter_hold_sec = letter_hold_sec
        self.fallback_hold_sec = fallback_hold_sec

        self.sequence: List[SignPlayItem] = []
        self.current_index: int = 0
        self.state: PlaybackState = PlaybackState.IDLE

        # Video capture state
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time: float = 0.0
        self.frame_duration: float = 1.0 / max(1, self.fps)

        # Card hold state (for fingerspelling and unavailable notices)
        self.card_start_time: float = 0.0

    def play_sequence(self, sequence: List[SignPlayItem]):
        """Starts playback of a new sequence of sign items."""
        self.stop()
        if not sequence:
            self.state = PlaybackState.IDLE
            return

        self.sequence = sequence
        self.current_index = 0
        self.state = PlaybackState.PLAYING
        self._init_current_item()

    def pause(self):
        """Pauses sign playback."""
        if self.state == PlaybackState.PLAYING:
            self.state = PlaybackState.PAUSED

    def resume(self):
        """Resumes sign playback."""
        if self.state == PlaybackState.PAUSED:
            self.state = PlaybackState.PLAYING
            self.last_frame_time = time.time()
            if self._is_timed_card():
                # Adjust card start time for paused duration
                self.card_start_time = time.time()

    def toggle_pause(self):
        """Toggles between paused and playing states."""
        if self.state == PlaybackState.PLAYING:
            self.pause()
        elif self.state == PlaybackState.PAUSED:
            self.resume()
        elif self.state == PlaybackState.FINISHED:
            self.replay()

    def replay(self):
        """Restarts the current sequence from the beginning."""
        if not self.sequence:
            return
        self.current_index = 0
        self.state = PlaybackState.PLAYING
        self._init_current_item()

    def step_next(self):
        """Jumps to the next sign item in the sequence."""
        if not self.sequence:
            return
        if self.current_index + 1 < len(self.sequence):
            self.current_index += 1
            self._init_current_item()
        else:
            self._finish_sequence()

    def stop(self):
        """Stops playback and releases video resources."""
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.state = PlaybackState.IDLE
        self.last_frame = None

    def _init_current_item(self):
        """Prepares video capture or timer for the current item."""
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.current_index >= len(self.sequence):
            self._finish_sequence()
            return

        item = self.sequence[self.current_index]
        self.last_frame_time = time.time()
        self.card_start_time = time.time()

        if item.item_type == "VERIFIED_VIDEO" and item.video_path and item.video_path.exists():
            try:
                self.cap = cv2.VideoCapture(str(item.video_path))
                if not self.cap.isOpened():
                    logger.warning("Could not open video file: %s", item.video_path)
                    self.cap = None
            except Exception as e:
                logger.error("Error opening video %s: %s", item.video_path, e)
                self.cap = None

    def _finish_sequence(self):
        """Marks sequence as completed."""
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.state = PlaybackState.FINISHED

    def _is_timed_card(self) -> bool:
        if not self.sequence or self.current_index >= len(self.sequence):
            return False
        return self.sequence[self.current_index].item_type in (
            "FALLBACK_FINGERSPELL",
            "UNAVAILABLE_NOTICE",
            "VIDEO_MISSING",
        )

    def get_current_frame(self, target_width: int = 540, target_height: int = 380) -> np.ndarray:
        """
        Advances animation timing and returns the rendered visual frame.
        """
        now = time.time()

        if self.state == PlaybackState.IDLE or not self.sequence:
            return self._render_idle_frame(target_width, target_height)

        if self.current_index >= len(self.sequence):
            self._finish_sequence()
            return self._render_finished_frame(target_width, target_height)

        item = self.sequence[self.current_index]

        # ── Handle Video Item ──
        if item.item_type == "VERIFIED_VIDEO" and self.cap is not None and self.cap.isOpened():
            if self.state == PlaybackState.PLAYING:
                # Check frame rate timing
                if now - self.last_frame_time >= self.frame_duration:
                    ret, raw_frame = self.cap.read()
                    self.last_frame_time = now
                    if ret and raw_frame is not None:
                        self.last_frame = cv2.resize(raw_frame, (target_width, target_height))
                    else:
                        # Video ended: advance to next item
                        self.step_next()
                        return self.get_current_frame(target_width, target_height)

            frame = self.last_frame if self.last_frame is not None else np.zeros((target_height, target_width, 3), dtype=np.uint8)
            return self._overlay_playback_hud(frame.copy(), item, target_width, target_height)

        # ── Handle Fingerspelling Fallback Card ──
        elif item.item_type == "FALLBACK_FINGERSPELL":
            elapsed = now - self.card_start_time
            if self.state == PlaybackState.PLAYING and elapsed >= self.letter_hold_sec:
                self.step_next()
                return self.get_current_frame(target_width, target_height)
            return self._render_fingerspell_card(item, target_width, target_height, elapsed)

        # ── Handle Unavailable / Missing Notice Card ──
        else:
            elapsed = now - self.card_start_time
            if self.state == PlaybackState.PLAYING and elapsed >= self.fallback_hold_sec:
                self.step_next()
                return self.get_current_frame(target_width, target_height)
            return self._render_notice_card(item, target_width, target_height, elapsed)

    def _render_idle_frame(self, w: int, h: int) -> np.ndarray:
        """Frame shown when waiting for speech or text input."""
        canvas = np.full((h, w, 3), (20, 20, 25), dtype=np.uint8)
        cv2.rectangle(canvas, (10, 10), (w - 10, h - 10), (45, 45, 55), 1)

        cv2.putText(canvas, "SIGN LANGUAGE PLAYBACK", (w // 2 - 160, h // 2 - 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Press 'M' to Speak or 'T' to Type", (w // 2 - 150, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Verified sign videos will appear here", (w // 2 - 145, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        return canvas

    def _render_finished_frame(self, w: int, h: int) -> np.ndarray:
        """Frame shown when entire sequence has finished."""
        canvas = np.full((h, w, 3), (18, 25, 20), dtype=np.uint8)
        cv2.rectangle(canvas, (10, 10), (w - 10, h - 10), (0, 160, 100), 2)

        cv2.putText(canvas, "SEQUENCE COMPLETE", (w // 2 - 130, h // 2 - 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 160), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Press [R] to Replay  |  [T] Type New  |  [M] Speak", (w // 2 - 180, h // 2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
        return canvas

    def _render_fingerspell_card(self, item: SignPlayItem, w: int, h: int, elapsed: float) -> np.ndarray:
        """
        Renders an explicitly labelled alphabet fingerspelling fallback card.
        Requirement 7 & 8: Never invent signs; explicitly label fingerspelling fallback.
        """
        canvas = np.full((h, w, 3), (25, 22, 28), dtype=np.uint8)

        # Warning Card Border
        cv2.rectangle(canvas, (10, 10), (w - 10, h - 10), (0, 140, 255), 2)

        # Top Explicit Fallback Banner
        cv2.rectangle(canvas, (10, 10), (w - 10, 65), (20, 35, 75), -1)
        cv2.putText(canvas, "EXPLICIT FALLBACK: FINGERSPELLING", (25, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Word: \"{item.word}\"", (w - 170, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

        # Letter Big Display (Center Left)
        letter_str = item.letter or "?"
        cv2.rectangle(canvas, (35, 90), (195, 270), (35, 35, 45), -1)
        cv2.rectangle(canvas, (35, 90), (195, 270), (0, 200, 255), 1)
        cv2.putText(canvas, letter_str, (80, 205),
                    cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 255, 220), 4, cv2.LINE_AA)
        cv2.putText(canvas, f"Letter {item.letter_index} of {item.letter_total}", (55, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Verified Reference Hand Sign Image (Center Right)
        if item.image_path and item.image_path.exists():
            try:
                ref_img = cv2.imread(str(item.image_path))
                if ref_img is not None:
                    ref_resized = cv2.resize(ref_img, (180, 180))
                    canvas[90:270, 225:405] = ref_resized
                    cv2.rectangle(canvas, (225, 90), (405, 270), (0, 255, 180), 1)
                    cv2.putText(canvas, "Ref Hand Pose", (245, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA)
            except Exception:
                pass

        # Progress info
        cv2.putText(canvas, f"Notice: '{item.word}' is not in verified video library.", (25, 305),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Spelling letter-by-letter to guarantee linguistic accuracy.", (25, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1, cv2.LINE_AA)

        # Hold countdown bar
        progress = min(1.0, elapsed / max(0.1, self.letter_hold_sec))
        bar_w = int((w - 50) * progress)
        cv2.rectangle(canvas, (25, h - 22), (25 + bar_w, h - 15), (0, 200, 255), -1)

        return self._overlay_playback_hud(canvas, item, w, h)

    def _render_notice_card(self, item: SignPlayItem, w: int, h: int, elapsed: float) -> np.ndarray:
        """Renders notice for unavailable words or missing video files."""
        canvas = np.full((h, w, 3), (20, 20, 35), dtype=np.uint8)

        is_error = (item.item_type == "VIDEO_MISSING")
        border_col = (0, 0, 255) if is_error else (0, 140, 255)
        cv2.rectangle(canvas, (10, 10), (w - 10, h - 10), border_col, 2)

        # Header
        header_text = "ERROR: SIGN VIDEO MISSING" if is_error else "WORD UNAVAILABLE IN VOCABULARY"
        cv2.rectangle(canvas, (10, 10), (w - 10, 65), (35, 20, 30) if is_error else (20, 35, 60), -1)
        cv2.putText(canvas, header_text, (25, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, border_col, 2, cv2.LINE_AA)

        # Message
        msg = item.error_message or f"Word '{item.word}' has no verified sign in library."
        cv2.putText(canvas, msg[:55], (25, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Requirement Rule: Never invent unofficial signs.", (25, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1, cv2.LINE_AA)

        # Hold bar
        progress = min(1.0, elapsed / max(0.1, self.fallback_hold_sec))
        bar_w = int((w - 50) * progress)
        cv2.rectangle(canvas, (25, h - 22), (25 + bar_w, h - 15), border_col, -1)

        return self._overlay_playback_hud(canvas, item, w, h)

    def _overlay_playback_hud(self, frame: np.ndarray, item: SignPlayItem, w: int, h: int) -> np.ndarray:
        """Draws sequence status, playback state, and interactive controls."""
        total_items = len(self.sequence)
        curr_idx = self.current_index + 1

        # Top-right sequence counter badge
        badge_text = f"Sign {curr_idx}/{total_items}"
        cv2.rectangle(frame, (w - 110, 15), (w - 15, 42), (20, 20, 20), -1)
        cv2.rectangle(frame, (w - 110, 15), (w - 15, 42), (70, 70, 70), 1)
        cv2.putText(frame, badge_text, (w - 102, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

        # Bottom control bar
        cv2.rectangle(frame, (10, h - 35), (w - 10, h - 5), (15, 15, 18), -1)

        if self.state == PlaybackState.PAUSED:
            state_str = "⏸ PAUSED"
            state_col = (0, 180, 255)
        elif self.state == PlaybackState.PLAYING:
            state_str = "▶ PLAYING"
            state_col = (0, 255, 0)
        else:
            state_str = "⏹ STOPPED"
            state_col = (160, 160, 160)

        cv2.putText(frame, state_str, (20, h - 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, state_col, 1, cv2.LINE_AA)
        cv2.putText(frame, "[P] Pause/Resume  [R] Replay  [N] Skip Next", (130, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)

        return frame
