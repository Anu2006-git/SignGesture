"""
preprocessing/data_saver.py
============================
Saves all preprocessing outputs to  processed_dataset/  without
ever touching the original  dataset/  directory.

Output layout
-------------
processed_dataset/
  images/
    A/
      A_img001_orig.png        ← preprocessed original (CNN size)
      A_img001_aug0.png        ← augmented copy 0
      A_img001_aug1.png
      ...
    B/ … Z/
  images_lite/
    A/
      A_img001_orig.png        ← preprocessed original (lite size)
      ...
  landmarks/
    A/
      A_img001_orig.npy        ← float32 (63,) landmark vector
      A_img001_aug0.npy
      ...
    B/ … Z/
  arrays/
    images_cnn.npz             ← combined arrays for CNN training
    images_lite.npz
    landmarks.npz              ← combined arrays for landmark training
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .config import (
    PROC_IMAGES_DIR,
    PROC_LANDMARKS_DIR,
    PROC_ARRAYS_DIR,
    LANDMARK_ENABLED,
    LANDMARK_FALLBACK,
)
from .image_processor import ProcessedImage, ImageProcessor

logger = logging.getLogger(__name__)

PROC_IMAGES_LITE_DIR = PROC_IMAGES_DIR.parent / "images_lite"


class DataSaver:
    """
    Saves ProcessedImage objects to the processed_dataset directory.

    Usage
    -----
    >>> saver = DataSaver()
    >>> stem = saver.save(proc_img, file_index=1)
    """

    def __init__(
        self,
        images_dir:     Path = PROC_IMAGES_DIR,
        images_lite_dir:Path = PROC_IMAGES_LITE_DIR,
        landmarks_dir:  Path = PROC_LANDMARKS_DIR,
    ):
        self.images_dir      = Path(images_dir)
        self.images_lite_dir = Path(images_lite_dir)
        self.landmarks_dir   = Path(landmarks_dir)

    # ── Save one ProcessedImage ────────────────────────────────
    def save(self, proc: ProcessedImage, file_index: int) -> str:
        """
        Save all arrays for a ProcessedImage.
        Returns the base stem used for all output files (for caching).
        """
        if not proc.is_valid:
            return ""

        label = proc.label
        stem  = f"{label}_{file_index:06d}"   # e.g. "A_000001"

        # ── Directories ──────────────────────────────────────
        img_dir  = self.images_dir      / label
        lite_dir = self.images_lite_dir / label
        lm_dir   = self.landmarks_dir   / label
        for d in (img_dir, lite_dir, lm_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ── Original (CNN size) ───────────────────────────────
        self._save_image(proc.array_cnn,  img_dir  / f"{stem}_orig.png")
        self._save_image(proc.array_lite, lite_dir / f"{stem}_orig.png")

        # ── Landmark for original ─────────────────────────────
        if LANDMARK_ENABLED and proc.landmark_raw is not None:
            self._save_landmark(proc.landmark_raw, lm_dir / f"{stem}_orig.npy")

        # ── Augmented copies ──────────────────────────────────
        for i, aug_cnn in enumerate(proc.augmented_cnn):
            aug_stem = f"{stem}_aug{i}"
            self._save_image(aug_cnn, img_dir / f"{aug_stem}.png")

        for i, aug_lite in enumerate(proc.augmented_lite):
            aug_stem = f"{stem}_aug{i}"
            self._save_image(aug_lite, lite_dir / f"{aug_stem}.png")

        if LANDMARK_ENABLED:
            for i, lm in enumerate(proc.aug_landmarks):
                aug_stem = f"{stem}_aug{i}"
                vec = lm if lm is not None else proc.landmark_raw
                if vec is not None:
                    self._save_landmark(vec, lm_dir / f"{aug_stem}.npy")

        return stem

    # ── Low-level save helpers ─────────────────────────────────
    @staticmethod
    def _save_image(arr: Optional[np.ndarray], path: Path) -> None:
        if arr is None:
            return
        uint8 = ImageProcessor.denormalise(arr)
        # OpenCV saves RGB as BGR — convert back
        if uint8.ndim == 3 and uint8.shape[2] == 3:
            uint8 = cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), uint8)

    @staticmethod
    def _save_landmark(vector: np.ndarray, path: Path) -> None:
        np.save(str(path), vector.astype(np.float32))
