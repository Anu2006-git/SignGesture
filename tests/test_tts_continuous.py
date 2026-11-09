"""
tests/test_tts_continuous.py
============================
Validation test verifying continuous multi-alphabet Text-to-Speech voice feedback
for sequential sign language gestures ('A', 'B', 'C', 'D').
"""

import unittest
from unittest.mock import patch
from realtime_detector import TwoWaySignCommunicationSystem, TextToSpeechEngine


class TestContinuousVoiceFeedback(unittest.TestCase):
    """Verifies that voice feedback reliably speaks every sequential alphabet."""

    @patch("cv2.VideoCapture")
    def setUp(self, mock_cap):
        mock_cap.return_value.isOpened.return_value = False
        self.system = TwoWaySignCommunicationSystem(camera_id=999)
        self.system.auto_voice_deaf_enabled = True

    def test_multi_alphabet_sequential_voice_triggers(self):
        spoken_letters = []

        def capture_speak(text, interrupt=False):
            spoken_letters.append(text)
            return True

        self.system.tts.speak_async = capture_speak

        now = 100.0
        # User performs A -> B -> C -> D sequentially
        for letter in ["A", "B", "C", "D"]:
            # Frame 1: Initial detection of letter
            prog, typed = self.system.step_gesture_accumulator(letter, is_wrong_gesture=False, now=now)
            self.assertEqual(spoken_letters[-1], letter, f"Letter {letter} was not voiced on recognition!")
            self.assertFalse(typed, f"Letter {letter} should not be typed on initial detection.")

            # Frame 2: Sustained hold to commit
            now += 1.05
            prog2, typed2 = self.system.step_gesture_accumulator(letter, is_wrong_gesture=False, now=now)
            self.assertTrue(typed2, f"Letter {letter} should be committed after hold duration.")

            # Step forward to simulate hand changing to next posture
            now += 0.2

        self.assertEqual(
            spoken_letters,
            ["A", "B", "C", "D"],
            f"Expected all 4 alphabets voiced sequentially, got: {spoken_letters}",
        )
        self.assertEqual(self.system.right_sentence, "ABCD")
        print("✅ Continuous multi-alphabet voice feedback verified for ['A', 'B', 'C', 'D']!")


if __name__ == "__main__":
    unittest.main()
