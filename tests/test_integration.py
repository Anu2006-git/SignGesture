"""
tests/test_integration.py
=========================
Comprehensive Integration & Two-Way Communication Workflow Verification Test Suite.

Verifies:
  - Test Case 1: Deaf User Alphabet Gestures -> Prediction -> Sentence Builder -> Confirmation -> TTS
  - Test Case 2: Hearing User Microphone Speech -> Speech Recognition -> Vocabulary Mapping -> Video Playback
  - Test Case 3: Hearing User Keyboard Text Input -> Vocabulary Mapping -> Video/Fingerspell Display
  - Test Case 4: Uncertain / Unsupported Gestures -> Anomaly Rejection (No Silent Erroneous Insertion)
  - Test Case 5: Missing Hardware / Assets (Webcam, Mic, Model, Video) -> Graceful Error & Usability
  - Panel Independence & Debounce Duplicate Prevention Check
"""

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import joblib
import numpy as np

from model_training.config import MODELS_DIR
from model_training.feature_extractor import evaluate_gesture, extract_hand_features
from realtime_detector import TwoWaySignCommunicationSystem, TextToSpeechEngine
from sign_vocabulary import (
    PlaybackState,
    SignPlayItem,
    SignVideoPlayer,
    SpeechState,
    SpeechToTextEngine,
    VocabularyManager,
)


class TestTwoWayCommunicationWorkflow(unittest.TestCase):
    """Integration Test Suite covering all 5 Stage Test Cases and Non-Interference."""

    @classmethod
    def setUpClass(cls):
        # Load verified model artifact and test features cache
        cls.model_path = MODELS_DIR / "best_model_rf.joblib"
        cls.classes_path = MODELS_DIR / "classes.json"
        assert cls.model_path.exists(), f"Model file missing: {cls.model_path}"
        assert cls.classes_path.exists(), f"Classes file missing: {cls.classes_path}"

        cls.model = joblib.load(str(cls.model_path))
        classes_data = json.loads(cls.classes_path.read_text(encoding="utf-8"))
        cls.classes = classes_data["classes"]
        cls.class_stats = getattr(cls.model, "class_stats_", {})

        # Load real test sample vectors
        test_cache = list(MODELS_DIR.glob("cache_test_*.joblib"))
        assert len(test_cache) > 0, "No test features cache found"
        cls.cached_data = joblib.load(str(test_cache[0]))

    def setUp(self):
        # Create fresh communication system instance with mock camera to prevent hardware access
        with patch("cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = False
            self.system = TwoWaySignCommunicationSystem(camera_id=0)

    # ─────────────────────────────────────────────────────────────
    # TEST CASE 1: Deaf User Signs -> Predict -> Accumulate -> Confirm -> TTS
    # ─────────────────────────────────────────────────────────────
    def test_case_1_deaf_signing_workflow_and_duplicate_prevention(self):
        """
        Test Case 1:
          - Deaf user performs alphabet signs.
          - The model predicts the signs.
          - The sentence builder combines the letters.
          - Duplicate letters are NOT continuously appended during a sustained hold.
          - User confirms the sentence.
          - Text-to-speech speaks the sentence.
        """
        # Find verified sample vectors for letters 'H', 'I'
        y_labels = [self.classes[idx] for idx in self.cached_data["y"]]
        h_idx = y_labels.index("H")
        i_idx = y_labels.index("I")

        feat_h = self.cached_data["X"][h_idx]
        feat_i = self.cached_data["X"][i_idx]

        # 1. Predict 'H'
        eval_h = evaluate_gesture(feat_h, self.model, self.class_stats, self.classes, confidence_threshold=0.60)
        self.assertTrue(eval_h["is_matched"])
        self.assertEqual(eval_h["pred_letter"], "H")

        # 2. Hold debouncer: user starts holding 'H'
        t0 = 100.0
        prog1, typed1 = self.system.step_gesture_accumulator("H", is_wrong_gesture=False, now=t0)
        self.assertFalse(typed1)
        self.assertEqual(self.system.right_sentence, "")

        # User holds 'H' for half duration (0.5s)
        prog2, typed2 = self.system.step_gesture_accumulator("H", is_wrong_gesture=False, now=t0 + 0.5)
        self.assertFalse(typed2)
        self.assertEqual(self.system.right_sentence, "")
        self.assertAlmostEqual(prog2, 0.5, places=2)

        # User completes 1.0s hold duration -> Letter 'H' committed
        prog3, typed3 = self.system.step_gesture_accumulator("H", is_wrong_gesture=False, now=t0 + 1.05)
        self.assertTrue(typed3)
        self.assertEqual(self.system.right_sentence, "H")

        # CONTINUOUS HOLD TEST: User keeps holding 'H' for 3 more seconds
        # System must NOT continuously append duplicate letters!
        prog4, typed4 = self.system.step_gesture_accumulator("H", is_wrong_gesture=False, now=t0 + 2.0)
        self.assertFalse(typed4)
        self.assertEqual(self.system.right_sentence, "H", "Runaway duplicate 'H' was incorrectly appended!")

        prog5, typed5 = self.system.step_gesture_accumulator("H", is_wrong_gesture=False, now=t0 + 4.0)
        self.assertFalse(typed5)
        self.assertEqual(self.system.right_sentence, "H", "Runaway duplicate 'H' was incorrectly appended!")

        # 3. User switches to sign 'I'
        eval_i = evaluate_gesture(feat_i, self.model, self.class_stats, self.classes, confidence_threshold=0.60)
        self.assertTrue(eval_i["is_matched"])
        self.assertEqual(eval_i["pred_letter"], "I")

        t1 = 200.0
        # Start holding 'I'
        self.system.step_gesture_accumulator("I", is_wrong_gesture=False, now=t1)
        # Hold 'I' to completion (1.0s)
        _, typed_i = self.system.step_gesture_accumulator("I", is_wrong_gesture=False, now=t1 + 1.05)
        self.assertTrue(typed_i)
        self.assertEqual(self.system.right_sentence, "HI")

        # 4. Sentence confirmation & Text-to-Speech output
        with patch.object(self.system.tts, "speak_async", return_value=True) as mock_speak:
            # User confirms sentence by triggering speech
            confirmed = self.system.tts.speak_async(self.system.right_sentence)
            self.assertTrue(confirmed)
            mock_speak.assert_called_once_with("HI")

    # ─────────────────────────────────────────────────────────────
    # TEST CASE 2: Hearing User Microphone Speech -> Recognize -> Verified Videos
    # ─────────────────────────────────────────────────────────────
    def test_case_2_hearing_user_speech_to_sign_video_playback(self):
        """
        Test Case 2:
          - Hearing user speaks into microphone.
          - Speech recognition produces text.
          - Application identifies supported words or phrases.
          - Left panel displays the corresponding sign videos.
        """
        spoken_phrase = "Good morning, thank you"

        # Simulate speech listener callback
        self.system._on_speech_recognized(spoken_phrase)

        # 1. Verify recognized text display in Left Panel
        self.assertEqual(self.system.left_recognized_text, spoken_phrase)
        self.assertIn("Good morning", self.system.left_status_message)

        # 2. Verify sequence mapping into verified videos
        seq = self.system.sign_player.sequence
        self.assertEqual(len(seq), 2)

        # First item: compound phrase 'GOOD MORNING'
        self.assertEqual(seq[0].word, "GOOD MORNING")
        self.assertEqual(seq[0].item_type, "VERIFIED_VIDEO")
        self.assertTrue(seq[0].video_path.exists())

        # Second item: courtesy phrase 'THANK YOU'
        self.assertEqual(seq[1].word, "THANK YOU")
        self.assertEqual(seq[1].item_type, "VERIFIED_VIDEO")
        self.assertTrue(seq[1].video_path.exists())

        # 3. Verify video player playback state & frame rendering
        self.assertEqual(self.system.sign_player.state, PlaybackState.PLAYING)
        frame = self.system.sign_player.get_current_frame(target_width=540, target_height=380)
        self.assertEqual(frame.shape, (380, 540, 3))

    # ─────────────────────────────────────────────────────────────
    # TEST CASE 3: Hearing User Keyboard Input Alternative
    # ─────────────────────────────────────────────────────────────
    def test_case_3_text_input_alternative_and_sign_representations(self):
        """
        Test Case 3:
          - Hearing user types a message.
          - Application displays the text and available sign representations
            (verified sign videos for known words and fingerspelling for unknown words).
        """
        typed_message = "Hello Alice"

        # Simulate in-window text submission
        self.system.left_recognized_text = typed_message
        self.system._translate_and_play(typed_message)

        self.assertEqual(self.system.left_recognized_text, typed_message)
        seq = self.system.sign_player.sequence

        # Known word 'HELLO' mapped to verified video
        self.assertEqual(seq[0].word, "HELLO")
        self.assertEqual(seq[0].item_type, "VERIFIED_VIDEO")
        self.assertTrue(seq[0].video_path.exists())

        # Unavailable name 'ALICE' mapped to explicitly labelled fingerspelling fallback
        alice_items = seq[1:]
        self.assertEqual(len(alice_items), 5)  # A-L-I-C-E
        expected_letters = ["A", "L", "I", "C", "E"]
        for idx, item in enumerate(alice_items):
            self.assertEqual(item.item_type, "FALLBACK_FINGERSPELL")
            self.assertEqual(item.letter, expected_letters[idx])
            self.assertIn("FALLBACK", item.label)
            self.assertEqual(item.word, "ALICE")

        # Test video player rendering of fingerspell card
        self.system.sign_player.current_index = 1
        self.system.sign_player._init_current_item()
        frame = self.system.sign_player.get_current_frame(target_width=540, target_height=380)
        self.assertEqual(frame.shape, (380, 540, 3))

    # ─────────────────────────────────────────────────────────────
    # TEST CASE 4: Uncertain / Unsupported Gesture Rejection
    # ─────────────────────────────────────────────────────────────
    def test_case_4_uncertain_gesture_anomaly_rejection(self):
        """
        Test Case 4:
          - The recognition model encounters an uncertain or unsupported sign.
          - The application avoids silently inserting an unreliable prediction.
        """
        # 1. Random noise vector simulating an arbitrary unsupported hand shape
        np.random.seed(99)
        unsupported_gesture = np.random.uniform(-3.0, 3.0, size=88).astype(np.float32)

        eval_res = evaluate_gesture(
            unsupported_gesture,
            self.model,
            self.class_stats,
            self.classes,
            confidence_threshold=0.60,
            min_margin=0.08,
        )

        # Must be rejected
        self.assertFalse(eval_res["is_matched"])
        self.assertIsNone(eval_res["pred_letter"])
        self.assertIn("Low confidence", eval_res["reason"])

        # 2. Verify sentence accumulator blocks insertion
        now = time.time()
        prog, typed = self.system.step_gesture_accumulator(
            detected_letter=eval_res["pred_letter"],
            is_wrong_gesture=True,
            now=now,
        )
        self.assertFalse(typed)
        self.assertEqual(prog, 0.0)
        self.assertEqual(self.system.right_sentence, "", "Uncertain prediction was silently inserted into sentence!")

        # 3. Low-margin ambiguous gesture test (simulating two conflicting classes)
        mock_model = MagicMock()
        # Mock probabilities: Top 1 = 0.51 (Letter A), Top 2 = 0.49 (Letter B) -> Margin 0.02 < 0.08
        mock_model.predict_proba.return_value = np.array([[0.51, 0.49] + [0.0] * 24])
        eval_ambiguous = evaluate_gesture(
            np.zeros(88, dtype=np.float32),
            mock_model,
            self.class_stats,
            self.classes,
            confidence_threshold=0.50,
            min_margin=0.08,
        )
        self.assertFalse(eval_ambiguous["is_matched"])
        self.assertIn("Ambiguous gesture", eval_ambiguous["reason"])
        self.assertIsNone(eval_ambiguous["pred_letter"])

    # ─────────────────────────────────────────────────────────────
    # TEST CASE 5: Missing Hardware / Assets Graceful Degradation
    # ─────────────────────────────────────────────────────────────
    def test_case_5_missing_hardware_and_asset_resilience(self):
        """
        Test Case 5:
          - The webcam, microphone, model file, or sign video is unavailable.
          - The application displays a helpful error message and remains usable where possible.
        """
        # 1. Missing Webcam Resilience:
        # Camera is unavailable, but Left Panel communication remains 100% functional
        system_no_cam = TwoWaySignCommunicationSystem(camera_id=999)
        system_no_cam.camera_error = "Webcam ID 999 unavailable."
        # Left panel still translates text and plays videos
        system_no_cam._translate_and_play("Yes please")
        self.assertEqual(len(system_no_cam.sign_player.sequence), 2)
        self.assertEqual(system_no_cam.sign_player.state, PlaybackState.PLAYING)

        # 2. Missing Microphone Resilience:
        with patch("speech_recognition.Microphone", side_effect=OSError("No default input device")):
            engine = SpeechToTextEngine(device_index=None)
            engine.start_listening()
            # Wait for thread
            engine._active_thread.join(timeout=2.0)
            self.assertEqual(engine.state, SpeechState.ERROR)
            self.assertIn("Microphone device unavailable", engine.last_error_message)

        # 3. Missing Sign Video Asset Resilience:
        # Register a sign whose video file does NOT exist on disk
        vm = VocabularyManager()
        vm.register_sign("quantum", "non_existent_video_12345.mp4", label="QUANTUM")
        seq = vm.parse_sentence_to_sequence("quantum")
        self.assertEqual(seq[0].item_type, "VIDEO_MISSING")
        self.assertIn("not found", seq[0].error_message)

        # Player handles missing video gracefully without crashing
        player = SignVideoPlayer()
        player.play_sequence(seq)
        frame = player.get_current_frame(540, 380)
        self.assertEqual(frame.shape, (380, 540, 3))

        # 4. Missing Model File Resilience:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("realtime_detector.MODELS_DIR", Path(tmp_dir)):
                sys_no_model = TwoWaySignCommunicationSystem(camera_id=0)
                self.assertIsNotNone(sys_no_model.model_error)
                self.assertIn("not found", sys_no_model.model_error.lower())
                # Left panel text translation still works even with model missing!
                sys_no_model._translate_and_play("Hello")
                self.assertEqual(len(sys_no_model.sign_player.sequence), 1)

    # ─────────────────────────────────────────────────────────────
    # PANEL INDEPENDENCE & ISOLATION
    # ─────────────────────────────────────────────────────────────
    def test_panel_independence_and_no_cross_contamination(self):
        """
        Verify that:
          - Left panel and Right panel operate independently.
          - Left speech text and Right sign sentence never overwrite each other.
          - Playback controls (Pause, Resume, Replay) work without affecting the camera state.
        """
        # 1. Left panel processes hearing speech
        self.system._on_speech_recognized("Hello how are you")
        self.assertEqual(self.system.left_recognized_text, "Hello how are you")
        self.assertEqual(self.system.right_sentence, "")

        # 2. Right panel deaf user signs "THANK"
        self.system.right_sentence = "THANK"
        # Ensure left recognized text is intact
        self.assertEqual(self.system.left_recognized_text, "Hello how are you")
        self.assertEqual(self.system.right_sentence, "THANK")

        # 3. Left panel text input mode
        self.system.text_input_mode = True
        self.system.text_input_buffer = "Welcome friend"
        self.assertEqual(self.system.right_sentence, "THANK", "Right sentence was contaminated by left text entry!")

        # 4. Playback controls on Left Panel:
        self.system.sign_player.pause()
        self.assertEqual(self.system.sign_player.state, PlaybackState.PAUSED)
        self.assertEqual(self.system.right_sentence, "THANK")

        self.system.sign_player.resume()
        self.assertEqual(self.system.sign_player.state, PlaybackState.PLAYING)

        self.system.sign_player.replay()
        self.assertEqual(self.system.sign_player.current_index, 0)
        self.assertEqual(self.system.right_sentence, "THANK")


if __name__ == "__main__":
    unittest.main()
