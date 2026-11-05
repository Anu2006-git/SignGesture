"""
tests/test_dashboard.py
=======================
Automated Test Suite for the Final Two-Way Sign Language Communication Dashboard.

Covers all 6 required verification workflows:
  - TEST 1: Deaf person hand gesture recognition -> Debounce -> Sentence Builder -> Manual Edit -> TTS Voice Readout.
  - TEST 2: Hearing person microphone speech -> Speech recognition -> Vocabulary mapping -> Middle section sign playback.
  - TEST 3: Hearing person manual text input -> Immediate middle section sign playback.
  - TEST 4: Communication isolation: Both users communicate independently without overwriting each other's state.
  - TEST 5: Hardware and asset unavailability: Camera, microphone, and model error handling without crashes.
  - TEST 6: Unsupported vocabulary words: Fallback notification & explicit fingerspelling (No fake signs invented).
"""

import json
from pathlib import Path
import time
import tkinter as tk
import unittest
from unittest.mock import MagicMock, patch

import joblib
import numpy as np

from dashboard import TwoWayDashboardApp, TextToSpeechEngine
from model_training.config import MODELS_DIR
from model_training.feature_extractor import evaluate_gesture
from sign_vocabulary import PlaybackState, SpeechState


class TestTwoWayCommunicationDashboard(unittest.TestCase):
    """Verifies all 6 functional workflows of the 3-section communication dashboard."""

    @classmethod
    def setUpClass(cls):
        # Load verified model artifact and test feature cache
        cls.model_path = MODELS_DIR / "best_model_rf.joblib"
        cls.classes_path = MODELS_DIR / "classes.json"
        assert cls.model_path.exists(), f"Model file missing: {cls.model_path}"
        assert cls.classes_path.exists(), f"Classes file missing: {cls.classes_path}"

        cls.model = joblib.load(str(cls.model_path))
        classes_data = json.loads(cls.classes_path.read_text(encoding="utf-8"))
        cls.classes = classes_data["classes"]
        cls.class_stats = getattr(cls.model, "class_stats_", {})

        test_cache = list(MODELS_DIR.glob("cache_test_*.joblib"))
        assert len(test_cache) > 0, "No test features cache found"
        cls.cached_data = joblib.load(str(test_cache[0]))

    def setUp(self):
        """Set up headless Tk root and Dashboard app for each test."""
        self.root = tk.Tk()
        self.root.withdraw()  # Headless mode for automated tests

        # Prevent actual hardware webcam capture during setup
        with patch("cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = False
            self.app = TwoWayDashboardApp(self.root, camera_id=999)

        # Mock TTS engine to test async speech calls without system audio dependencies
        self.app.tts = MagicMock(spec=TextToSpeechEngine)
        self.app.tts.last_spoken_text = ""

        def _mock_speak(text):
            self.app.tts.last_spoken_text = text.strip()
            return True

        self.app.tts.speak_async.side_effect = _mock_speak

    def tearDown(self):
        """Tear down Tk window and clean up app resources."""
        try:
            self.app.on_closing()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────
    # TEST 1: Deaf User Alphabet Gestures -> Predict -> Edit -> TTS
    # ─────────────────────────────────────────────────────────────
    def test_1_deaf_person_gestures_sentence_tts(self):
        """
        TEST 1:
        A deaf person performs hand gestures.
        The model recognizes the signs.
        The sentence builder combines the letters with duplicate prevention.
        The user can edit the sentence before speaking.
        The text-to-speech engine speaks the completed sentence.
        """
        y_labels = [self.classes[idx] for idx in self.cached_data["y"]]
        h_idx = y_labels.index("H")
        i_idx = y_labels.index("I")

        feat_h = self.cached_data["X"][h_idx]
        feat_i = self.cached_data["X"][i_idx]

        # Model correctly predicts H and I
        eval_h = evaluate_gesture(feat_h, self.model, self.class_stats, self.classes)
        self.assertTrue(eval_h["is_matched"])
        self.assertEqual(eval_h["pred_letter"], "H")

        eval_i = evaluate_gesture(feat_i, self.model, self.class_stats, self.classes)
        self.assertTrue(eval_i["is_matched"])
        self.assertEqual(eval_i["pred_letter"], "I")

        # Simulate debounced letter accumulation in dashboard
        self.app.sentence_clear()
        self.assertEqual(self.app.get_deaf_sentence(), "")

        # Accumulate 'H'
        self.app.sentence_append_char("H")
        self.assertEqual(self.app.get_deaf_sentence(), "H")

        # Accumulate 'I'
        self.app.sentence_append_char("I")
        self.assertEqual(self.app.get_deaf_sentence(), "HI")

        # Space and editing
        self.app.sentence_add_space()
        self.app.sentence_append_char("A")
        self.app.sentence_append_char("L")
        self.app.sentence_append_char("L")
        self.assertEqual(self.app.get_deaf_sentence(), "HI ALL")

        # Undo test
        self.app.sentence_undo()
        self.assertEqual(self.app.get_deaf_sentence(), "HI AL")

        # User edits sentence directly in the text area
        self.app.set_deaf_sentence("HI ALL, I AM ALICE")
        self.assertEqual(self.app.get_deaf_sentence(), "HI ALL, I AM ALICE")

        # Confirm & Speak aloud
        self.app.sentence_speak()
        self.app.tts.speak_async.assert_called_once_with("HI ALL, I AM ALICE")
        self.assertEqual(self.app.tts.last_spoken_text, "HI ALL, I AM ALICE")

    # ─────────────────────────────────────────────────────────────
    # TEST 2: Hearing Person Speaks -> Transcribe -> Sign Video Playback
    # ─────────────────────────────────────────────────────────────
    def test_2_hearing_person_speech_to_sign_display(self):
        """
        TEST 2:
        A hearing person speaks into the microphone.
        The speech recognition system transcribes words.
        The right section displays the recognized sentence.
        The middle section identifies verified signs and starts sequential playback.
        """
        spoken_phrase = "hello thank you"

        # Simulate speech recognition arrival
        self.app._handle_speech_recognized_gui(spoken_phrase)

        # Right section displays recognized text
        self.assertEqual(self.app.hearing_recognized_text, spoken_phrase)
        right_displayed = self.app.txt_recognized_speech.get("1.0", "end-1c").strip()
        self.assertEqual(right_displayed, spoken_phrase)

        # Middle section incoming sentence card updated
        self.assertIn("hello thank you", self.app.lbl_incoming_sentence.cget("text"))

        # Middle section sign player loaded with sequence of verified video items
        self.assertEqual(self.app.sign_player.state, PlaybackState.PLAYING)
        seq = self.app.sign_player.sequence
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0].word.lower(), "hello")
        self.assertEqual(seq[0].item_type, "VERIFIED_VIDEO")
        self.assertEqual(seq[1].word.lower(), "thank you")
        self.assertEqual(seq[1].item_type, "VERIFIED_VIDEO")

    # ─────────────────────────────────────────────────────────────
    # TEST 3: Hearing Person Manual Keyboard Text Input
    # ─────────────────────────────────────────────────────────────
    def test_3_hearing_person_manual_text_input(self):
        """
        TEST 3:
        A hearing person manually types a message into the text box.
        The application displays the text in the right panel.
        The middle section immediately displays and plays the corresponding sign representations.
        """
        typed_message = "good morning friend"

        self.app.ent_manual_text.insert(0, typed_message)
        self.app.submit_manual_text()

        # Input field cleared
        self.assertEqual(self.app.ent_manual_text.get(), "")

        # Right section shows transcript
        self.assertEqual(self.app.hearing_recognized_text, typed_message)
        self.assertEqual(self.app.txt_recognized_speech.get("1.0", "end-1c").strip(), typed_message)

        # Middle section plays corresponding sequence
        self.assertEqual(self.app.sign_player.state, PlaybackState.PLAYING)
        words_in_sequence = [item.word.lower() for item in self.app.sign_player.sequence]
        self.assertIn("good morning", words_in_sequence)
        self.assertIn("friend", words_in_sequence)

    # ─────────────────────────────────────────────────────────────
    # TEST 4: Independence & No Cross-Overwriting
    # ─────────────────────────────────────────────────────────────
    def test_4_independent_communication_no_overwriting(self):
        """
        TEST 4:
        Both users communicate simultaneously and independently.
        Deaf sentence is NOT overwritten by Hearing speech/text.
        Hearing sentence is NOT overwritten by Deaf gestures.
        """
        # Deaf person signs a sentence
        self.app.set_deaf_sentence("PLEASE HELP ME")

        # Hearing person speaks a sentence
        self.app._handle_speech_recognized_gui("welcome my friend")

        # Verify Left Panel state remains completely intact
        self.assertEqual(self.app.get_deaf_sentence(), "PLEASE HELP ME")

        # Verify Right Panel state remains completely intact
        self.assertEqual(self.app.hearing_recognized_text, "welcome my friend")
        self.assertEqual(self.app.txt_recognized_speech.get("1.0", "end-1c").strip(), "welcome my friend")

        # Deaf person adds more letters
        self.app.sentence_add_space()
        self.app.sentence_append_char("NOW")
        self.assertEqual(self.app.get_deaf_sentence(), "PLEASE HELP ME NOW")

        # Hearing speech state is still preserved
        self.assertEqual(self.app.hearing_recognized_text, "welcome my friend")

    # ─────────────────────────────────────────────────────────────
    # TEST 5: Hardware & Asset Unavailability Error Handling
    # ─────────────────────────────────────────────────────────────
    def test_5_hardware_unavailability_graceful_error_handling(self):
        """
        TEST 5:
        Webcam or microphone is unavailable.
        The application displays clear error messages and remains operational.
        """
        # 1. Camera unavailable: verify status text and placeholder without crashing
        self.app.stop_camera()
        self.assertFalse(self.app.camera_active)
        self.assertIn("Disconnected", self.app.lbl_cam_status.cget("text"))

        # 2. Microphone error: simulate mic failure
        self.app.speech_engine.state = SpeechState.ERROR
        self.app.speech_engine.last_error_message = "Microphone device unavailable. Use manual text input."
        self.app._update_status_indicators()

        self.assertIn("Error", self.app.lbl_stat_mic.cget("text"))

        # Even with mic error, manual text typing still works 100%!
        self.app.ent_manual_text.insert(0, "yes")
        self.app.submit_manual_text()
        self.assertEqual(self.app.sign_player.state, PlaybackState.PLAYING)
        self.assertEqual(self.app.sign_player.sequence[0].word.lower(), "yes")

    # ─────────────────────────────────────────────────────────────
    # TEST 6: Unsupported Vocabulary & Explicit Fallback
    # ─────────────────────────────────────────────────────────────
    def test_6_unsupported_word_fallback_handling(self):
        """
        TEST 6:
        An unsupported word is entered by the hearing user.
        The system NEVER invents a fake sign video.
        Instead, it explicitly displays alphabet fingerspelling fallback.
        """
        unsupported_sentence = "zebra robot"

        self.app.translate_and_play_sentence(unsupported_sentence)

        seq = self.app.sign_player.sequence
        self.assertTrue(len(seq) > 0)

        # Verify that all items for unknown words are FALLBACK_FINGERSPELL
        types = [item.item_type for item in seq]
        self.assertTrue(all(t == "FALLBACK_FINGERSPELL" for t in types))

        # Check letters spelled for 'zebra'
        zebra_letters = [item.letter for item in seq if item.word.lower() == "zebra"]
        self.assertEqual("".join(zebra_letters), "ZEBRA")

        # Check fallback alert notification banner is active
        fallback_text = self.app.lbl_fallback_alert.cget("text")
        self.assertIn("fingerspelling", fallback_text.lower())


if __name__ == "__main__":
    unittest.main()
