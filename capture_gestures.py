"""
capture_gestures.py
===================
Interactive Webcam Hand Gesture Capture Tool for Sign Language Datasets.

Features:
  - Live webcam feed with optional MediaPipe hand landmark tracking.
  - Saves clean, un-annotated frames directly into:
      dataset/<ALPHABET>/<DATASET_NAME>/
  - Single-shot mode (SPACE) and Continuous / Burst mode (C).
  - Optional Hand Bounding Box crop mode (B) with safety margin.
  - Keyboard shortcuts to switch alphabets on the fly (A-Z or [/]).
  - Seamlessly integrates with manage_dataset.py and preprocess.py.

Usage:
  python capture_gestures.py
  python capture_gestures.py --letter A --dataset dataset_1
  python capture_gestures.py --camera 0 --crop-hand
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    mp = None

DATASET_ROOT = BASE_DIR / "dataset"
ALPHABETS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


class GestureCollector:
    def __init__(
        self,
        camera_id: int = 0,
        initial_letter: str = "A",
        dataset_name: str = "dataset_1",
        crop_hand: bool = False,
        crop_margin: int = 35,
        capture_delay_sec: float = 0.25,
    ):
        self.camera_id = camera_id
        self.current_letter = initial_letter.upper()
        self.dataset_name = dataset_name
        self.crop_hand = crop_hand
        self.crop_margin = crop_margin
        self.capture_delay = capture_delay_sec

        # State tracking
        self.is_continuous = False
        self.last_capture_time = 0.0
        self.session_count = 0
        self.show_landmarks = True

        # MediaPipe Hands setup
        self.mp_hands = None
        self.mp_drawing = None
        if _MP_AVAILABLE and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            self.mp_drawing = mp.solutions.drawing_utils
        elif _MP_AVAILABLE:
            print("⚠️ Warning: MediaPipe installed does not contain 'solutions.hands'.")
            print("   Please run: pip install mediapipe==0.10.14")

    def get_target_dir(self) -> Path:
        target_dir = DATASET_ROOT / self.current_letter / self.dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def count_existing_images(self) -> int:
        target_dir = self.get_target_dir()
        if not target_dir.exists():
            return 0
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sum(1 for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts)

    def next_dataset_folder(self) -> str:
        """Find the next available dataset_N subfolder."""
        alpha_dir = DATASET_ROOT / self.current_letter
        alpha_dir.mkdir(parents=True, exist_ok=True)
        i = 1
        while (alpha_dir / f"dataset_{i}").exists() and any((alpha_dir / f"dataset_{i}").iterdir()):
            i += 1
        new_name = f"dataset_{i}"
        self.dataset_name = new_name
        return new_name

    def compute_hand_bbox(self, landmarks, frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
        """Calculate square/padded bounding box around the detected hand."""
        x_coords = [int(lm.x * frame_w) for lm in landmarks]
        y_coords = [int(lm.y * frame_h) for lm in landmarks]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add margin
        x_min = max(0, x_min - self.crop_margin)
        y_min = max(0, y_min - self.crop_margin)
        x_max = min(frame_w, x_max + self.crop_margin)
        y_max = min(frame_h, y_max + self.crop_margin)

        # Make bounding box roughly square to avoid aspect ratio distortions
        bw = x_max - x_min
        bh = y_max - y_min
        diff = abs(bw - bh)
        if bw < bh:
            x_min = max(0, x_min - diff // 2)
            x_max = min(frame_w, x_max + (diff - diff // 2))
        else:
            y_min = max(0, y_min - diff // 2)
            y_max = min(frame_h, y_max + (diff - diff // 2))

        if x_max > x_min and y_max > y_min:
            return (x_min, y_min, x_max, y_max)
        return None

    def save_image(self, clean_frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> bool:
        """Saves clean frame (or cropped region) with a unique timestamp filename."""
        target_dir = self.get_target_dir()

        img_to_save = clean_frame
        if self.crop_hand and bbox is not None:
            x1, y1, x2, y2 = bbox
            cropped = clean_frame[y1:y2, x1:x2]
            if cropped.size > 0:
                img_to_save = cropped

        timestamp = int(time.time() * 1000)
        filename = f"{self.current_letter}_{timestamp}_{self.session_count + 1:04d}.jpg"
        save_path = target_dir / filename

        success = cv2.imwrite(str(save_path), img_to_save)
        if success:
            self.session_count += 1
            self.last_capture_time = time.time()
            return True
        return False

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {self.camera_id}")
            print("Tip: Check your webcam connection or specify another index with --camera 1")
            return

        # Attempt high resolution capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        window_name = "Sign Language Gesture Collector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "=" * 65)
        print(" 🤟 SIGN LANGUAGE GESTURE COLLECTOR")
        print("=" * 65)
        print("  Controls:")
        print("    [SPACE]   : Capture 1 image")
        print("    [C]       : Toggle Continuous / Auto-capture mode")
        print("    [A - Z]   : Directly switch target alphabet")
        print("    [ / ]     : Previous / Next alphabet")
        print("    [B]       : Toggle Bounding-Box Hand Crop vs Full Frame")
        print("    [L]       : Toggle Landmark skeleton display")
        print("    [N]       : Create new dataset subfolder (dataset_2, etc.)")
        print("    [Q / ESC] : Exit")
        print("=" * 65 + "\n")

        saved_flash_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break

            # Mirror for natural user interaction
            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()  # Unmodified frame for saving
            display_frame = frame.copy()
            h, w = frame.shape[:2]

            # MediaPipe hand tracking
            hand_detected = False
            hand_bbox = None
            if self.mp_hands:
                rgb = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
                results = self.mp_hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand_detected = True
                    first_hand = results.multi_hand_landmarks[0]
                    hand_bbox = self.compute_hand_bbox(first_hand.landmark, w, h)

                    if self.show_landmarks and self.mp_drawing:
                        self.mp_drawing.draw_landmarks(
                            display_frame,
                            first_hand,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp.solutions.drawing_utils.DrawingSpec(color=(0, 100, 255), thickness=2, circle_radius=1),
                        )

                    if hand_bbox:
                        bx1, by1, bx2, by2 = hand_bbox
                        crop_color = (0, 255, 255) if self.crop_hand else (150, 150, 150)
                        cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), crop_color, 2)
                        label_text = "ROI (Saved)" if self.crop_hand else "Hand Region"
                        cv2.putText(display_frame, label_text, (bx1, max(15, by1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, crop_color, 1, cv2.LINE_AA)

            # Continuous auto-capture handling
            now = time.time()
            if self.is_continuous and (now - self.last_capture_time >= self.capture_delay):
                # If crop_hand is required, ensure hand is detected
                if not self.crop_hand or hand_detected:
                    if self.save_image(clean_frame, hand_bbox):
                        saved_flash_time = now

            # ── Drawing On-Screen HUD ──────────────────────────────
            existing_count = self.count_existing_images()

            # Top overlay panel
            cv2.rectangle(display_frame, (0, 0), (w, 80), (20, 20, 20), -1)
            cv2.line(display_frame, (0, 80), (w, 80), (70, 70, 70), 2)

            # Alphabet banner
            cv2.putText(display_frame, f"Letter: {self.current_letter}", (20, 38),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Folder: dataset/{self.current_letter}/{self.dataset_name}",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # Counts & Status
            cv2.putText(display_frame, f"In Folder: {existing_count}", (380, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Captured Now: {self.session_count}", (380, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

            # Hand status indicator
            hand_stat_color = (0, 255, 0) if hand_detected else (0, 80, 255)
            hand_stat_text = "Hand: DETECTED" if hand_detected else "Hand: NOT FOUND"
            cv2.putText(display_frame, hand_stat_text, (620, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_stat_color, 2, cv2.LINE_AA)

            mode_text = "Crop: ON (Hand ROI)" if self.crop_hand else "Crop: OFF (Full)"
            cv2.putText(display_frame, mode_text, (620, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # Continuous status badge
            if self.is_continuous:
                cv2.rectangle(display_frame, (w - 240, 15), (w - 20, 65), (0, 0, 180), -1)
                cv2.putText(display_frame, "REC (AUTO)", (w - 210, 48),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # Visual flash when image is saved
            if now - saved_flash_time < 0.12:
                cv2.rectangle(display_frame, (0, 0), (w, h), (0, 255, 0), 6)

            # Bottom controls panel
            cv2.rectangle(display_frame, (0, h - 35), (w, h), (20, 20, 20), -1)
            cv2.putText(
                display_frame,
                "SPACE: Capture | C: Auto-Burst | [ / ]: Change Letter | B: Toggle Crop | N: New Batch | Q: Quit",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA
            )

            cv2.imshow(window_name, display_frame)

            # Keyboard event handling
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # ESC or Q
                break
            elif key == 32:  # SPACE
                if not self.crop_hand or hand_detected:
                    if self.save_image(clean_frame, hand_bbox):
                        saved_flash_time = time.time()
                else:
                    print("⚠️ Hand not detected! Move hand into frame or press 'B' to disable crop.")
            elif key in (ord('c'), ord('C')):
                self.is_continuous = not self.is_continuous
                state_str = "STARTED" if self.is_continuous else "STOPPED"
                print(f"Continuous capture {state_str} for letter [{self.current_letter}]")
            elif key in (ord('b'), ord('B')):
                self.crop_hand = not self.crop_hand
                print(f"Crop hand mode: {'ENABLED' if self.crop_hand else 'DISABLED'}")
            elif key in (ord('l'), ord('L')):
                self.show_landmarks = not self.show_landmarks
            elif key in (ord('n'), ord('N')):
                new_f = self.next_dataset_folder()
                print(f"Switched to new dataset batch: {new_f}")
            elif key == ord(']'):
                curr_idx = ALPHABETS.index(self.current_letter)
                self.current_letter = ALPHABETS[(curr_idx + 1) % len(ALPHABETS)]
                print(f"Target letter changed to [{self.current_letter}]")
            elif key == ord('['):
                curr_idx = ALPHABETS.index(self.current_letter)
                self.current_letter = ALPHABETS[(curr_idx - 1) % len(ALPHABETS)]
                print(f"Target letter changed to [{self.current_letter}]")
            elif ord('a') <= key <= ord('z'):
                self.current_letter = chr(key).upper()
                print(f"Target letter changed to [{self.current_letter}]")

        cap.release()
        cv2.destroyAllWindows()
        if self.mp_hands:
            self.mp_hands.close()

        print("\n" + "=" * 65)
        print(f" Session complete! Captured {self.session_count} new images.")
        print(" Next recommended steps:")
        print("   1. Review stats:   python manage_dataset.py")
        print("   2. Preprocess:     python preprocess.py")
        print("=" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture Sign Language Gestures via Webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--letter", type=str, default="A", help="Initial alphabet to capture (A-Z)")
    parser.add_argument("--dataset", type=str, default="dataset_1", help="Target dataset subfolder name (default: dataset_1)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--crop-hand", action="store_true", help="Automatically crop bounding box around detected hand")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay in seconds between auto-captures (default: 0.25)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collector = GestureCollector(
        camera_id=args.camera,
        initial_letter=args.letter,
        dataset_name=args.dataset,
        crop_hand=args.crop_hand,
        capture_delay_sec=args.delay,
    )
    collector.run()
