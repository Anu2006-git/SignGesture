"""
preprocessing/landmark_extractor.py
=====================================
MediaPipe hand-landmark extraction for every image in the dataset.

Output per image
----------------
  • A flat float32 vector of length 63  (21 landmarks × [x, y, z])
  • Optionally normalised relative to the wrist landmark

Dual-mode design
----------------
This module produces a 63-D landmark vector that can be used:
  a) standalone — train a simple Dense/MLP model on landmarks only
  b) alongside CNN images — compare two classification approaches

If MediaPipe detects NO hand in the image:
  - LANDMARK_FALLBACK = "zeros"  → vector is all-zeros, flag set
  - LANDMARK_FALLBACK = "skip"   → entry excluded from landmark dataset
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    mp = None

from .config import (
    MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONF,
    MP_MIN_TRACKING_CONF,
    NUM_LANDMARKS,
    LANDMARK_DIMS,
    LANDMARK_VECTOR_SIZE,
    LANDMARK_NORMALIZATION,
    LANDMARK_FALLBACK,
    LANDMARK_ENABLED,
)
from .image_processor import ProcessedImage

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────
@dataclass
class LandmarkResult:
    source_path:    Path
    label:          str
    label_index:    int
    vector:         Optional[np.ndarray]   = None   # shape (63,) float32
    matrix:         Optional[np.ndarray]   = None   # shape (21, 3) float32
    hand_detected:  bool                   = False
    confidence:     float                  = 0.0
    error_message:  str                    = ""


# ─────────────────────────────────────────────────────────────
# Extractor
# ─────────────────────────────────────────────────────────────
class LandmarkExtractor:
    """
    Wraps MediaPipe Hands to extract 21 hand landmarks from images.

    Usage
    -----
    >>> extractor = LandmarkExtractor()
    >>> extractor.attach_to_processed([proc_img1, proc_img2])
    # Each ProcessedImage now has .landmark_raw populated.

    >>> result = extractor.extract_from_file(Path("dataset/A/dataset_1/img_001.jpg"))
    >>> print(result.vector.shape)   # (63,)
    """

    def __init__(self):
        if not _MP_AVAILABLE or not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            raise ImportError(
                "MediaPipe 'solutions.hands' not found. Please install the compatible version: pip install mediapipe==0.10.14"
            )
        if not LANDMARK_ENABLED:
            logger.info("Landmark extraction is disabled via config.LANDMARK_ENABLED.")

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,                # process still images
            max_num_hands=MP_MAX_NUM_HANDS,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF,
        )
        logger.info(
            "LandmarkExtractor ready — normalisation=%s, fallback=%s",
            LANDMARK_NORMALIZATION, LANDMARK_FALLBACK,
        )

    def __del__(self):
        try:
            self._hands.close()
        except Exception:
            pass

    # ── High-level: attach landmarks to ProcessedImage list ───
    def attach_to_processed(
        self, processed_images: List[ProcessedImage]
    ) -> None:
        """
        Run landmark extraction for every ProcessedImage in the list.
        Populates  .landmark_raw  and  .aug_landmarks  in-place.
        """
        if not LANDMARK_ENABLED:
            return

        for proc in processed_images:
            if not proc.is_valid:
                continue

            # ── Original image ─────────────────────────────────
            result = self.extract_from_file(proc.source_path)
            proc.landmark_raw = result.vector   # may be zeros if no hand found

            # ── Augmented images (re-extract from saved uint8) ─
            #    We pass the denormalised CNN array to MediaPipe.
            from .image_processor import ImageProcessor
            for i, aug_arr in enumerate(proc.augmented_cnn):
                if aug_arr is None:
                    proc.aug_landmarks[i] = proc.landmark_raw
                    continue
                uint8_img = ImageProcessor.denormalise(aug_arr)
                lm = self._extract_from_array(
                    uint8_img, proc.source_path, proc.label, proc.label_index
                )
                proc.aug_landmarks[i] = lm.vector

    # ── Per-file extraction ────────────────────────────────────
    def extract_from_file(
        self,
        image_path: Path,
        label: str = "",
        label_index: int = -1,
    ) -> LandmarkResult:
        """Run MediaPipe on a raw image file and return a LandmarkResult."""
        result = LandmarkResult(
            source_path=image_path,
            label=label,
            label_index=label_index,
        )

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            result.error_message = "Could not read image file."
            result.vector = self._fallback_vector()
            return result

        return self._extract_from_array(bgr, image_path, label, label_index)

    # ── Batch extraction ───────────────────────────────────────
    def extract_batch(
        self,
        image_paths: List[Path],
        labels: List[str],
        label_indices: List[int],
    ) -> List[LandmarkResult]:
        """Extract landmarks from a list of image paths."""
        results = []
        for path, lbl, idx in zip(image_paths, labels, label_indices):
            results.append(self.extract_from_file(path, lbl, idx))
        return results

    # ── Core extraction logic ─────────────────────────────────
    def _extract_from_array(
        self,
        bgr: np.ndarray,
        source_path: Path,
        label: str,
        label_index: int,
    ) -> LandmarkResult:
        result = LandmarkResult(
            source_path=source_path,
            label=label,
            label_index=label_index,
        )

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        try:
            mp_result = self._hands.process(rgb)
        except Exception as exc:  # noqa: BLE001
            result.error_message = f"MediaPipe error: {exc}"
            result.vector = self._fallback_vector()
            return result

        if not mp_result.multi_hand_landmarks:
            result.hand_detected  = False
            result.error_message  = "No hand detected by MediaPipe."
            result.vector = self._fallback_vector()
            result.matrix = result.vector.reshape(NUM_LANDMARKS, LANDMARK_DIMS)
            return result

        # ── Extract from first hand ────────────────────────────
        hand = mp_result.multi_hand_landmarks[0]
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark],
            dtype=np.float32,
        )  # shape (21, 3)

        # ── Normalise landmark coordinates ─────────────────────
        coords = self._normalise_landmarks(coords)

        result.hand_detected = True
        result.matrix        = coords
        result.vector        = coords.flatten()    # (63,)

        return result

    # ── Landmark normalisation ────────────────────────────────
    @staticmethod
    def _normalise_landmarks(coords: np.ndarray) -> np.ndarray:
        """
        Normalise landmark positions to be translation and scale invariant.

        Strategy (config.LANDMARK_NORMALIZATION):
          "wrist"  — translate so wrist=origin, then scale by max distance
          "bbox"   — normalise into hand bounding box [0,1]
          "none"   — keep MediaPipe's raw [0,1] image-relative coordinates
        """
        mode = LANDMARK_NORMALIZATION.lower()

        if mode == "wrist":
            # Translate: subtract wrist (landmark 0)
            origin  = coords[0].copy()
            shifted = coords - origin
            # Scale: divide by max L2 distance from wrist
            dists   = np.linalg.norm(shifted, axis=1)
            max_d   = dists.max()
            if max_d > 1e-6:
                shifted /= max_d
            return shifted.astype(np.float32)

        elif mode == "bbox":
            min_xy = coords[:, :2].min(axis=0)
            max_xy = coords[:, :2].max(axis=0)
            rng    = max_xy - min_xy
            rng[rng < 1e-6] = 1.0   # avoid div-by-zero
            normed = coords.copy()
            normed[:, :2] = (coords[:, :2] - min_xy) / rng
            return normed.astype(np.float32)

        else:  # "none"
            return coords.astype(np.float32)

    # ── Fallback vector ────────────────────────────────────────
    @staticmethod
    def _fallback_vector() -> np.ndarray:
        """Return a zero vector or raise, depending on config."""
        if LANDMARK_FALLBACK == "zeros":
            return np.zeros(LANDMARK_VECTOR_SIZE, dtype=np.float32)
        # "skip" — caller checks hand_detected=False and excludes entry
        return np.zeros(LANDMARK_VECTOR_SIZE, dtype=np.float32)
