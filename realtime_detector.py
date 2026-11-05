"""
realtime_detector.py
====================
Two-Way Real-Time Communication System:
  - LEFT PANEL: Hearing Person -> Sign Language (Stage 5)
      * Microphone speech recognition & text input alternative
      * Verified sign video playback sequence (Play, Pause, Resume, Replay)
      * Sequence of known signs for common words & phrases
      * Fallback logic: Never invents fake signs; explicitly labels fingerspelling
      * Robust error display for mic, speech engine, and video assets
  - RIGHT PANEL: Deaf Person -> Spoken English & Text (Stage 4)
      * Real-time MediaPipe hand landmark tracking
      * 88-dimensional isotropic Random Forest classifier (99.73% test acc)
      * Real-time Wrong Hand Gesture rejection (Red skeleton & warning badge)
      * Debounced hold debouncer preventing runaway duplicate letter accumulation
      * Sentence accumulator with confirmation & Text-to-Speech (TTS) voice readout

Usage:
  python realtime_detector.py
  python realtime_detector.py --camera 0 --confidence 0.60
"""

import argparse
from collections import Counter, deque
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import joblib
import numpy as np

# Configure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# MediaPipe
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    mp = None

# Text-to-Speech (TTS)
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

from model_training.config import MODELS_DIR
from model_training.feature_extractor import (
    FEATURE_SIZE,
    evaluate_gesture,
    extract_hand_features,
)
from sign_vocabulary import (
    PlaybackState,
    SignVideoPlayer,
    SpeechState,
    SpeechToTextEngine,
    VocabularyManager,
)

logger = logging.getLogger("realtime_detector")


class TextToSpeechEngine:
    """Threaded Text-to-Speech engine so audio never freezes UI frames."""

    def __init__(self):
        self.available = _TTS_AVAILABLE
        self.last_spoken_text = ""

    def speak_async(self, text: str) -> bool:
        if not text.strip():
            return False

        self.last_spoken_text = text.strip()
        print(f"🔊 Spoken sentence: \"{text.strip()}\"")

        if not self.available:
            return True

        def _speak():
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.warning("TTS error: %s", e)

        threading.Thread(target=_speak, daemon=True).start()
        return True


class TwoWaySignCommunicationSystem:
    """
    Complete Two-Panel Two-Way Communication System:
      - Left Panel: Hearing person -> Sign Language
      - Right Panel: Deaf person -> Speech & Text
    """

    def __init__(
        self,
        camera_id: int = 0,
        mic_index: Optional[int] = None,
        hold_time_sec: float = 1.0,
        confidence_threshold: float = 0.60,
        min_margin: float = 0.08,
        initial_text: Optional[str] = None,
    ):
        self.camera_id = camera_id
        self.hold_time_sec = hold_time_sec
        self.confidence_threshold = confidence_threshold
        self.min_margin = min_margin

        # ── STAGE 5: Left Panel Components (Hearing -> Sign) ──
        self.vocab_manager = VocabularyManager()
        self.sign_player = SignVideoPlayer(fps=25)

        # Threaded speech listener
        self.speech_engine = SpeechToTextEngine(
            device_index=mic_index,
            on_text_recognized=self._on_speech_recognized,
            timeout=6.0,
            phrase_time_limit=8.0,
        )

        # Left panel state (Completely separate from Right Panel)
        self.left_recognized_text = ""
        self.text_input_mode = False
        self.text_input_buffer = ""
        self.left_status_message = "Press 'M' to Speak or 'T' to Type"

        # ── Right Panel Components (Sign -> Hearing) ──
        self.tts = TextToSpeechEngine()
        self.model = None
        self.classes = []
        self.class_stats = {}
        self.model_error = None
        self.camera_error = None

        self._load_recognition_model()

        self.prediction_buffer = deque(maxlen=5)
        self.right_sentence = ""
        self.current_candidate = None
        self.candidate_start_time = 0.0
        self.candidate_committed = False   # Debounce lock preventing runaway duplicate letters
        self.last_added_time = 0.0

        # MediaPipe setup
        self.mp_hands = None
        self.mp_drawing = None
        if _MP_AVAILABLE and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.50,
                )
                self.mp_drawing = mp.solutions.drawing_utils
            except Exception as e:
                logger.warning("MediaPipe Hands initialization error: %s", e)

        # Play initial text if specified
        if initial_text:
            self._translate_and_play(initial_text)

    def _load_recognition_model(self):
        """Loads trained 88-D Landmark Random Forest model with graceful error handling."""
        classes_file = MODELS_DIR / "classes.json"
        if not classes_file.exists():
            self.model_error = "Class mapping (models/classes.json) not found. Run 'python train_model.py'."
            logger.warning(self.model_error)
            return

        try:
            classes_data = json.loads(classes_file.read_text(encoding="utf-8"))
            self.classes = classes_data["classes"]
        except Exception as e:
            self.model_error = f"Failed to parse models/classes.json: {e}"
            return

        # Check manifest
        manifest_file = MODELS_DIR / "best_model_manifest.json"
        model_path = None
        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                filename = manifest.get("model_file")
                if filename and (MODELS_DIR / filename).exists():
                    model_path = MODELS_DIR / filename
            except Exception:
                pass

        if model_path is None:
            for fname in ["best_model_rf.joblib", "model_rf.joblib"]:
                if (MODELS_DIR / fname).exists():
                    model_path = MODELS_DIR / fname
                    break

        if not model_path or not model_path.exists():
            self.model_error = "Trained model not found in models/. Run 'python train_model.py'."
            logger.warning(self.model_error)
            return

        try:
            print(f"📦 Loading gesture recognition model from: {model_path.name}")
            self.model = joblib.load(str(model_path))
            if hasattr(self.model, "n_jobs"):
                self.model.n_jobs = 1
            self.class_stats = getattr(self.model, "class_stats_", {})
            self.model_error = None
        except Exception as e:
            self.model_error = f"Failed to load model {model_path.name}: {e}"
            logger.error(self.model_error)

    def _on_speech_recognized(self, text: str):
        """Callback executed in speech thread when words are recognized."""
        self.left_recognized_text = text
        self.left_status_message = f"Speech Recognized: \"{text}\""
        self._translate_and_play(text)

    def _translate_and_play(self, text: str):
        """Parses text into verified sign videos & fallbacks, then starts playback."""
        clean = text.strip()
        if not clean:
            return
        sequence = self.vocab_manager.parse_sentence_to_sequence(clean)
        self.sign_player.play_sequence(sequence)

    def predict_right_frame(self, frame: np.ndarray, landmarks) -> Dict[str, Any]:
        """Runs 88-D feature extraction and wrong gesture evaluation for Right Panel."""
        if self.model is None:
            return {
                "pred_letter": None,
                "candidate_letter": "?",
                "confidence": 0.0,
                "margin": 0.0,
                "dist": 0.0,
                "threshold_dist": 999.0,
                "is_matched": False,
                "reason": "Model unavailable",
                "probs": np.zeros(len(self.classes)),
            }

        h, w = frame.shape[:2]
        feat = extract_hand_features(landmarks, img_w=w, img_h=h)
        return evaluate_gesture(
            feat=feat,
            model=self.model,
            class_stats=self.class_stats,
            classes=self.classes,
            confidence_threshold=self.confidence_threshold,
            min_margin=self.min_margin,
        )

    def step_gesture_accumulator(self, detected_letter: Optional[str], is_wrong_gesture: bool, now: float) -> Tuple[float, bool]:
        """
        Processes letter debounce timer with duplicate prevention.
        Returns: (progress_ratio, letter_typed_bool)
        """
        progress_ratio = 0.0
        typed_now = False

        if detected_letter and not is_wrong_gesture:
            if detected_letter == self.current_candidate:
                elapsed = now - self.candidate_start_time
                progress_ratio = min(1.0, elapsed / self.hold_time_sec)

                # Only commit ONCE per continuous hold!
                if progress_ratio >= 1.0 and not self.candidate_committed:
                    self.right_sentence += detected_letter
                    self.candidate_committed = True
                    self.last_added_time = now
                    typed_now = True
            else:
                # Gesture changed: start tracking new candidate
                self.current_candidate = detected_letter
                self.candidate_start_time = now
                self.candidate_committed = False
        else:
            # Hand removed or wrong gesture shown: reset lock and timer
            self.current_candidate = None
            self.candidate_start_time = 0.0
            self.candidate_committed = False

        return progress_ratio, typed_now

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.camera_error = f"Webcam ID {self.camera_id} could not be opened. Right panel camera disabled."
            logger.warning(self.camera_error)
            cap = None
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        window_name = "Sign Language Two-Way Communication System (Stage 5)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1380, 760)

        print("\n" + "=" * 75)
        print(" 🤟 TWO-WAY SIGN LANGUAGE COMMUNICATION SYSTEM READY")
        print("=" * 75)
        print("  LEFT PANEL  : Hearing Person -> Sign Language (Speech/Text to Video)")
        print("  RIGHT PANEL : Deaf Person -> Speech & Text (Live Camera Recognition)")
        print("-" * 75)
        print("  Controls:")
        print("    [M]            : Speak via Microphone (Speech Recognition)")
        print("    [T]            : Type Text Alternative (Interactive keyboard entry)")
        print("    [P]            : Pause / Resume Sign Playback")
        print("    [R]            : Replay Current Sign Sequence")
        print("    [N]            : Next Sign in Sequence")
        print("    [Hold Gesture] : Keep steady for 1.0s to type letter (Debounced)")
        print("    [SPACEBAR]     : Insert Space in right sentence")
        print("    [BACKSPACE]    : Delete last character")
        print("    [C]            : Clear right sentence")
        print("    [S]            : Confirm & Speak sentence aloud (Voice TTS)")
        print("    [+] / [-]      : Adjust gesture confidence threshold")
        print("    [Q / ESC]      : Exit")
        print("=" * 75 + "\n")

        panel_w = 680
        panel_h = 740

        while True:
            now = time.time()
            cam_frame = None

            if cap is not None and cap.isOpened():
                ret, raw_cam = cap.read()
                if ret and raw_cam is not None:
                    cam_frame = cv2.flip(raw_cam, 1)

            # ─────────────────────────────────────────────────────────
            # 1. RENDER LEFT PANEL: Hearing -> Sign Playback
            # ─────────────────────────────────────────────────────────
            left_panel = np.full((panel_h, panel_w, 3), (20, 20, 25), dtype=np.uint8)

            # Left Panel Header
            cv2.rectangle(left_panel, (0, 0), (panel_w, 55), (35, 25, 25), -1)
            cv2.line(left_panel, (0, 55), (panel_w, 55), (0, 180, 255), 2)
            cv2.putText(left_panel, "HEARING PERSON  -->  SIGN LANGUAGE", (20, 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

            # Mic / Input Status Badge
            if self.text_input_mode:
                badge_bg, badge_txt = (20, 80, 120), "KEYBOARD INPUT ACTIVE"
            elif self.speech_engine.state == SpeechState.LISTENING:
                badge_bg, badge_txt = (0, 60, 220), "LISTENING TO MIC..."
            elif self.speech_engine.state == SpeechState.PROCESSING:
                badge_bg, badge_txt = (0, 140, 200), "PROCESSING SPEECH..."
            else:
                badge_bg, badge_txt = (30, 45, 30), "MIC READY [M]"

            cv2.rectangle(left_panel, (panel_w - 240, 12), (panel_w - 15, 43), badge_bg, -1)
            cv2.rectangle(left_panel, (panel_w - 240, 12), (panel_w - 15, 43), (100, 100, 100), 1)
            cv2.putText(left_panel, badge_txt, (panel_w - 232, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # Text / Speech Display Box
            cv2.rectangle(left_panel, (15, 65), (panel_w - 15, 135), (28, 28, 35), -1)
            cv2.rectangle(left_panel, (15, 65), (panel_w - 15, 135), (60, 60, 75), 1)

            if self.text_input_mode:
                cv2.putText(left_panel, "TYPE INPUT [Press ENTER to translate | ESC to cancel]:", (25, 88),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
                display_input = f"> {self.text_input_buffer}|"
                cv2.putText(left_panel, display_input[:52], (25, 118),
                            cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(left_panel, "RECOGNIZED SPEECH / TEXT PROMPT:", (25, 88),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1, cv2.LINE_AA)
                active_text = f"\"{self.left_recognized_text}\"" if self.left_recognized_text else "Press 'M' to Speak or 'T' to Type a message..."
                txt_col = (255, 255, 255) if self.left_recognized_text else (130, 130, 140)
                cv2.putText(left_panel, active_text[:50], (25, 118),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, txt_col, 1, cv2.LINE_AA)

            # Sign Playback Video Frame (540 x 380 viewport)
            video_frame = self.sign_player.get_current_frame(target_width=panel_w - 30, target_height=380)
            left_panel[145 : 145 + 380, 15 : panel_w - 15] = video_frame

            # Error Message / Notice Banner (Requirement 10)
            cv2.rectangle(left_panel, (15, 535), (panel_w - 15, 585), (25, 22, 28), -1)
            err_msg = self.speech_engine.last_error_message
            if err_msg:
                cv2.rectangle(left_panel, (15, 535), (panel_w - 15, 585), (0, 0, 220), 1)
                cv2.putText(left_panel, err_msg[:68], (25, 565),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(left_panel, (15, 535), (panel_w - 15, 585), (50, 50, 60), 1)
                seq_len = len(self.sign_player.sequence)
                if seq_len > 0:
                    status_line = f"Sequence: {seq_len} sign item(s) loaded. Playing verified video/fallbacks."
                else:
                    status_line = "Ready. Spoken phrases or typed text translate to verified sign sequence."
                cv2.putText(left_panel, status_line[:68], (25, 565),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

            # Left Panel Bottom Controls
            cv2.rectangle(left_panel, (0, panel_h - 75), (panel_w, panel_h), (18, 18, 22), -1)
            cv2.line(left_panel, (0, panel_h - 75), (panel_w, panel_h - 75), (0, 180, 255), 1)
            cv2.putText(left_panel, "[M] Mic Speech  |  [T] Type Text  |  [P] Pause/Resume  |  [R] Replay",
                        (20, panel_h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(left_panel, "Rule: Verified vocabulary plays video; unknown words explicitly fingerspell.",
                        (20, panel_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1, cv2.LINE_AA)

            # ─────────────────────────────────────────────────────────
            # 2. RENDER RIGHT PANEL: Deaf Person -> Recognition & Speech
            # ─────────────────────────────────────────────────────────
            right_panel = np.full((panel_h, panel_w, 3), (20, 24, 20), dtype=np.uint8)

            # Right Panel Header
            cv2.rectangle(right_panel, (0, 0), (panel_w, 55), (20, 35, 25), -1)
            cv2.line(right_panel, (0, 55), (panel_w, 55), (0, 255, 160), 2)
            cv2.putText(right_panel, "DEAF PERSON  -->  SPEECH & TEXT", (20, 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 160), 2, cv2.LINE_AA)

            model_badge = "88-D Landmark ML (99.7% Acc)" if not self.model_error else "MODEL ERROR"
            badge_col = (0, 255, 200) if not self.model_error else (0, 0, 255)
            cv2.putText(right_panel, model_badge, (panel_w - 235, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_col, 1, cv2.LINE_AA)

            # Viewport frame setup
            view_cam = np.full((380, panel_w - 30, 3), (25, 28, 25), dtype=np.uint8)

            if self.camera_error:
                # Camera unavailable error card
                cv2.rectangle(view_cam, (10, 10), (panel_w - 40, 370), (0, 0, 180), 2)
                cv2.putText(view_cam, "CAMERA UNAVAILABLE", (panel_w // 2 - 130, 160),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 80, 255), 2, cv2.LINE_AA)
                cv2.putText(view_cam, self.camera_error[:55], (25, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(view_cam, "Left panel (Speech & Text to Sign) remains fully operational.", (25, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

            elif self.model_error:
                # Model unavailable error card
                cv2.rectangle(view_cam, (10, 10), (panel_w - 40, 370), (0, 0, 180), 2)
                cv2.putText(view_cam, "MODEL UNAVAILABLE", (panel_w // 2 - 120, 160),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 80, 255), 2, cv2.LINE_AA)
                cv2.putText(view_cam, self.model_error[:55], (25, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

            elif cam_frame is not None:
                # Process active camera feed with MediaPipe
                hand_detected = False
                last_eval: Optional[Dict[str, Any]] = None

                if self.mp_hands:
                    rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
                    results = self.mp_hands.process(rgb)

                    if results.multi_hand_landmarks:
                        hand_detected = True
                        hand = results.multi_hand_landmarks[0]
                        last_eval = self.predict_right_frame(cam_frame, hand.landmark)

                        if last_eval["is_matched"]:
                            self.prediction_buffer.append(last_eval["pred_letter"])
                            skel_col = (0, 255, 0)
                            joint_col = (0, 200, 255)
                        else:
                            self.prediction_buffer.append("WRONG_GESTURE")
                            skel_col = (0, 60, 255)
                            joint_col = (0, 0, 255)

                        if self.mp_drawing:
                            self.mp_drawing.draw_landmarks(
                                cam_frame,
                                hand,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                mp.solutions.drawing_utils.DrawingSpec(color=skel_col, thickness=2, circle_radius=3),
                                mp.solutions.drawing_utils.DrawingSpec(color=joint_col, thickness=2, circle_radius=2),
                            )
                    else:
                        self.prediction_buffer.append(None)
                else:
                    self.prediction_buffer.append(None)

                # Majority voting
                active_preds = [p for p in self.prediction_buffer if p is not None]
                detected_letter = None
                is_wrong_gesture = False
                confidence = last_eval["confidence"] if last_eval else 0.0

                if active_preds:
                    most_common, count = Counter(active_preds).most_common(1)[0]
                    if most_common == "WRONG_GESTURE" and count >= 2:
                        is_wrong_gesture = True
                    elif most_common != "WRONG_GESTURE" and count >= 3:
                        detected_letter = most_common

                # Debounced gesture accumulation
                progress_ratio, _ = self.step_gesture_accumulator(detected_letter, is_wrong_gesture, now)

                # Resize camera into Viewport
                view_cam = cv2.resize(cam_frame, (panel_w - 30, 380))

                # Heads-up overlay
                cv2.rectangle(view_cam, (0, 0), (panel_w - 30, 75), (15, 15, 18), -1)
                cv2.line(view_cam, (0, 75), (panel_w - 30, 75), (50, 50, 50), 1)

                if hand_detected and is_wrong_gesture:
                    # WRONG HAND GESTURE ALERT
                    cv2.rectangle(view_cam, (10, 8), (280, 66), (20, 20, 110), -1)
                    cv2.rectangle(view_cam, (10, 8), (280, 66), (0, 60, 255), 2)
                    cv2.putText(view_cam, "WRONG HAND GESTURE", (18, 38),
                                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 80, 255), 2, cv2.LINE_AA)
                    cand = last_eval["candidate_letter"] if last_eval else "?"
                    cv2.putText(view_cam, f"Unrecognized (Conf {confidence*100:.0f}%)", (18, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1, cv2.LINE_AA)
                    cv2.putText(view_cam, "Does not match dataset", (300, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 160, 255), 1, cv2.LINE_AA)

                elif detected_letter:
                    # VALID MATCHED SIGN GESTURE
                    cv2.putText(view_cam, f"Letter: {detected_letter}", (18, 42),
                                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(view_cam, f"Conf: {confidence*100:.1f}% | VALID", (18, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 255, 200), 1, cv2.LINE_AA)

                    # Hold Progress Bar
                    bar_x, bar_y, bar_w, bar_h = 240, 24, 180, 20
                    cv2.rectangle(view_cam, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
                    fill_w = int(bar_w * progress_ratio)
                    cv2.rectangle(view_cam, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)
                    cv2.rectangle(view_cam, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

                    pct_txt = "TYPED!" if self.candidate_committed else f"Hold {int(progress_ratio*100)}%"
                    cv2.putText(view_cam, pct_txt, (bar_x + bar_w + 10, bar_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1, cv2.LINE_AA)

                else:
                    cv2.putText(view_cam, "Letter: [Hold hand in frame]", (18, 42),
                                cv2.FONT_HERSHEY_DUPLEX, 0.65, (140, 140, 140), 1, cv2.LINE_AA)
                    h_txt = "Hand: DETECTED" if hand_detected else "Hand: NOT FOUND"
                    h_col = (0, 255, 0) if hand_detected else (0, 80, 255)
                    cv2.putText(view_cam, h_txt, (18, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.46, h_col, 1, cv2.LINE_AA)

            # Insert camera into right panel
            right_panel[145 : 145 + 380, 15 : panel_w - 15] = view_cam

            # Right Panel Accumulated Sentence Box
            cv2.rectangle(right_panel, (15, 535), (panel_w - 15, 655), (18, 22, 18), -1)
            cv2.rectangle(right_panel, (15, 535), (panel_w - 15, 655), (0, 180, 100), 1)

            cv2.putText(right_panel, "ACCUMULATED SENTENCE (From Live Sign Recognition):", (25, 560),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 140), 1, cv2.LINE_AA)
            display_sent = f"\"{self.right_sentence}\"" if self.right_sentence else "Gesturing letters accumulates words here..."
            sent_col = (255, 255, 255) if self.right_sentence else (120, 140, 120)
            cv2.putText(right_panel, display_sent[:46], (25, 605),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, sent_col, 2, cv2.LINE_AA)
            cv2.putText(right_panel, "Press [S] to Confirm & Speak aloud (Voice TTS)  |  [C] Clear",
                        (25, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 180, 160), 1, cv2.LINE_AA)

            # Right Panel Bottom Controls
            cv2.rectangle(right_panel, (0, panel_h - 75), (panel_w, panel_h), (18, 22, 18), -1)
            cv2.line(right_panel, (0, panel_h - 75), (panel_w, panel_h - 75), (0, 220, 140), 1)
            cv2.putText(right_panel, "[Hold 1.0s] Type  |  [S] Voice TTS  |  [SPACE] Space  |  [BACKSPACE] Del",
                        (20, panel_h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 160), 1, cv2.LINE_AA)
            cv2.putText(right_panel, f"Sensitivity: Min Conf {self.confidence_threshold*100:.0f}%  (Press [-] / [+] to adjust)",
                        (20, panel_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 160, 140), 1, cv2.LINE_AA)

            # ─────────────────────────────────────────────────────────
            # 3. MERGE PANELS SIDE-BY-SIDE WITH ACCENT DIVIDER
            # ─────────────────────────────────────────────────────────
            dashboard = np.hstack([left_panel, right_panel])

            # Draw vertical divider
            cv2.line(dashboard, (panel_w, 0), (panel_w, panel_h), (80, 80, 90), 2)
            cv2.line(dashboard, (panel_w - 1, 0), (panel_w - 1, panel_h), (0, 200, 255), 1)

            cv2.imshow(window_name, dashboard)

            # ─────────────────────────────────────────────────────────
            # 4. INTERACTIVE KEYBOARD DISPATCH
            # ─────────────────────────────────────────────────────────
            key = cv2.waitKey(20) & 0xFF

            if self.text_input_mode:
                # Text Input Mode: Capture typing for Left Panel
                if key in (10, 13):  # ENTER key -> Submit text
                    submitted = self.text_input_buffer.strip()
                    self.text_input_mode = False
                    self.text_input_buffer = ""
                    if submitted:
                        self.left_recognized_text = submitted
                        self.left_status_message = f"Text input submitted: \"{submitted}\""
                        self._translate_and_play(submitted)
                elif key == 27:  # ESC -> Cancel text input
                    self.text_input_mode = False
                    self.text_input_buffer = ""
                elif key in (8, 127):  # BACKSPACE
                    if self.text_input_buffer:
                        self.text_input_buffer = self.text_input_buffer[:-1]
                elif 32 <= key <= 126:  # Printable characters
                    if len(self.text_input_buffer) < 50:
                        self.text_input_buffer += chr(key)

            else:
                # Normal Mode: Command Hotkeys
                if key in (ord('q'), ord('Q'), 27):  # Quit
                    break
                elif key in (ord('m'), ord('M')):  # Trigger microphone
                    started = self.speech_engine.start_listening()
                    if started:
                        self.left_status_message = "Microphone listening... Speak clearly."
                elif key in (ord('t'), ord('T')):  # Enter text input mode
                    self.text_input_mode = True
                    self.text_input_buffer = ""
                elif key in (ord('p'), ord('P')):  # Pause / Resume playback
                    self.sign_player.toggle_pause()
                elif key in (ord('r'), ord('R')):  # Replay sign sequence
                    self.sign_player.replay()
                elif key in (ord('n'), ord('N')):  # Skip to next sign
                    self.sign_player.step_next()
                elif key == 32:  # SPACEBAR -> Space in Right Sentence
                    self.right_sentence += " "
                elif key in (8, 127):  # BACKSPACE -> Delete in Right Sentence
                    if self.right_sentence:
                        self.right_sentence = self.right_sentence[:-1]
                elif key in (ord('c'), ord('C')):  # Clear Right Sentence
                    self.right_sentence = ""
                elif key in (ord('s'), ord('S')):  # Speak Right Sentence aloud
                    if self.right_sentence.strip():
                        self.tts.speak_async(self.right_sentence)
                elif key in (ord('+'), ord('='), ord(']')):  # Sensitivity +
                    self.confidence_threshold = min(0.95, round(self.confidence_threshold + 0.05, 2))
                    print(f"🎯 Sensitivity increased: Min Conf {self.confidence_threshold*100:.0f}%")
                elif key in (ord('-'), ord('_'), ord('[')):  # Sensitivity -
                    self.confidence_threshold = max(0.40, round(self.confidence_threshold - 0.05, 2))
                    print(f"🎯 Sensitivity decreased: Min Conf {self.confidence_threshold*100:.0f}%")

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if self.mp_hands:
            self.mp_hands.close()
        self.sign_player.stop()
        print("\n👋 Two-Way Sign Communication System closed.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Two-Way Sign Language Communication System (Stage 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    p.add_argument("--mic", type=int, default=None, help="Microphone device index (default: system default)")
    p.add_argument("--hold-time", type=float, default=1.0, help="Hold duration in seconds to type a letter (default: 1.0)")
    p.add_argument("--confidence", type=float, default=0.60, help="Minimum confidence threshold (default: 0.60)")
    p.add_argument("--margin", type=float, default=0.08, help="Minimum probability margin (default: 0.08)")
    p.add_argument("--text", type=str, default=None, help="Initial text sentence to translate and play on launch")
    p.add_argument("--cli", action="store_true", help="Run in classic OpenCV dual-panel mode instead of 3-section dashboard")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cli:
        system = TwoWaySignCommunicationSystem(
            camera_id=args.camera,
            mic_index=args.mic,
            hold_time_sec=args.hold_time,
            confidence_threshold=args.confidence,
            min_margin=args.margin,
            initial_text=args.text,
        )
        system.run()
    else:
        try:
            from dashboard import TwoWayDashboardApp
            import tkinter as tk

            root = tk.Tk()
            app = TwoWayDashboardApp(root, camera_id=args.camera, mic_index=args.mic)
            if args.text:
                root.after(300, lambda: app.translate_and_play_sentence(args.text))
            root.mainloop()
        except Exception as err:
            import traceback
            import tkinter as tk
            from tkinter import messagebox
            tb = traceback.format_exc()
            try:
                err_root = tk.Tk()
                err_root.withdraw()
                messagebox.showerror("SignGesture Launch Error", f"An error occurred while starting SignGesture:\n\n{err}\n\nDetails:\n{tb}")
                err_root.destroy()
            except Exception:
                pass
            raise
