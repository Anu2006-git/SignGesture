"""
tests/test_sapi_audio_worker.py
===============================
Validation test verifying that TextToSpeechEngine speaks sequential alphabets
without hanging or locking in Windows SAPI.
"""

import time
import unittest
from realtime_detector import TextToSpeechEngine


class TestSAPIAudioWorker(unittest.TestCase):
    def test_sapi_queue_sequential_speech(self):
        engine = TextToSpeechEngine(rate=160, volume=1.0)
        self.assertTrue(engine.available, "Speech engine should be available")

        # Send A, B, C, D into the queue
        for letter in ["A", "B", "C", "D"]:
            success = engine.speak_async(letter)
            self.assertTrue(success, f"Failed to enqueue {letter}")
            time.sleep(0.4)

        # Give the audio worker thread time to process all 4 items
        engine._queue.join()
        self.assertEqual(engine.last_spoken_text, "D")
        print("✅ Native SAPI audio worker processed A, B, C, D through queue successfully!")


if __name__ == "__main__":
    unittest.main()
