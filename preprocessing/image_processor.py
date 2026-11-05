"""
preprocessing/image_processor.py
==================================
Core image preprocessing pipeline.

For each source image it:
  1. Reads the raw image with OpenCV.
  2. Detects and handles corrupted/unreadable images gracefully.
  3. Converts to the target colour space.
  4. Resizes to the target CNN input size (preserving aspect ratio with padding).
  5. Normalises pixel values to [0, 1].
  6. Optionally produces N augmented copies via Augmentor.
  7. Returns ProcessedImage objects ready for saving.

The original image on disk is NEVER modified.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    IMAGE_SIZE_CNN,
    IMAGE_SIZE_LITE,
    COLOR_MODE,
    NORMALIZE_SCALE,
    AUGMENTATION_ENABLED,
    AUGMENT_COPIES,
)
from .augmentor import Augmentor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Output data structure
# ─────────────────────────────────────────────────────────────
@dataclass
class ProcessedImage:
    """
    Holds all outputs for one source image (original + augmented copies).

    Fields
    ------
    source_path     : path to the original raw image
    label           : alphabet letter (A-Z)
    label_index     : integer class index (A=0 … Z=25)
    array_cnn       : float32 ndarray (H, W, C) normalised, CNN size
    array_lite      : float32 ndarray (H, W, C) normalised, lite size
    augmented_cnn   : list of augmented float32 arrays, CNN size
    augmented_lite  : list of augmented float32 arrays, lite size
    is_valid        : False if image was corrupted / unreadable
    error_message   : populated when is_valid=False
    landmark_raw    : filled later by LandmarkExtractor  (None until then)
    """
    source_path:    Path
    label:          str
    label_index:    int
    array_cnn:      Optional[np.ndarray]       = None
    array_lite:     Optional[np.ndarray]       = None
    augmented_cnn:  List[np.ndarray]           = field(default_factory=list)
    augmented_lite: List[np.ndarray]           = field(default_factory=list)
    is_valid:       bool                       = True
    error_message:  str                        = ""
    landmark_raw:   Optional[np.ndarray]       = None   # shape (63,) or (21,3)
    aug_landmarks:  List[Optional[np.ndarray]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Processor
# ─────────────────────────────────────────────────────────────
class ImageProcessor:
    """
    Reads, validates, resizes, normalises, and augments a single image.

    Usage
    -----
    >>> proc = ImageProcessor()
    >>> result = proc.process(Path("dataset/A/dataset_1/img_001.jpg"), label="A", label_index=0)
    >>> print(result.array_cnn.shape)   # (224, 224, 3)
    """

    ALPHABET_ORDER = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    def __init__(
        self,
        image_size_cnn:  Tuple[int, int]  = IMAGE_SIZE_CNN,
        image_size_lite: Tuple[int, int]  = IMAGE_SIZE_LITE,
        color_mode:      str              = COLOR_MODE,
        normalize_scale: float            = NORMALIZE_SCALE,
        augmentation:    bool             = AUGMENTATION_ENABLED,
        n_augments:      int              = AUGMENT_COPIES,
    ):
        self.image_size_cnn  = image_size_cnn   # (H, W)
        self.image_size_lite = image_size_lite
        self.color_mode      = color_mode.upper()
        self.normalize_scale = normalize_scale
        self.augmentation    = augmentation
        self.augmentor       = Augmentor(n_copies=n_augments)

    # ── Public API ─────────────────────────────────────────────
    def process(
        self,
        source_path: Path,
        label: str,
        label_index: int,
        augment: bool = None,
    ) -> ProcessedImage:
        """
        Full preprocessing pipeline for one image file.

        Parameters
        ----------
        source_path : absolute path to the raw image
        label       : 'A'…'Z'
        label_index : 0…25
        augment     : override instance-level augmentation flag

        Returns
        -------
        ProcessedImage — always returned, check .is_valid before using arrays.
        """
        apply_aug = self.augmentation if augment is None else augment

        result = ProcessedImage(
            source_path=source_path,
            label=label,
            label_index=label_index,
        )

        # ── 1. Load ────────────────────────────────────────────
        bgr = self._safe_load(source_path)
        if bgr is None:
            result.is_valid = False
            result.error_message = f"OpenCV could not read: {source_path}"
            logger.warning("Skipping unreadable image: %s", source_path)
            return result

        # ── 2. Colour conversion ───────────────────────────────
        display_img = self._convert_color(bgr)

        # ── 3. Resize with aspect-ratio-preserving letterbox ───
        cnn_img  = self._letterbox_resize(display_img, self.image_size_cnn)
        lite_img = self._letterbox_resize(display_img, self.image_size_lite)

        # ── 4. Normalise ───────────────────────────────────────
        result.array_cnn  = self._normalise(cnn_img)
        result.array_lite = self._normalise(lite_img)

        # ── 5. Augmentation ────────────────────────────────────
        if apply_aug:
            # Augment on the PRE-normalised uint8 image for accurate transforms
            aug_bgrs_cnn  = self.augmentor.augment(cnn_img.astype(np.uint8))
            aug_bgrs_lite = self.augmentor.augment(lite_img.astype(np.uint8))
            result.augmented_cnn  = [self._normalise(a) for a in aug_bgrs_cnn]
            result.augmented_lite = [self._normalise(a) for a in aug_bgrs_lite]
            # Placeholders for landmarks (filled by LandmarkExtractor later)
            result.aug_landmarks  = [None] * len(aug_bgrs_cnn)

        return result

    # ── Helpers ────────────────────────────────────────────────
    @staticmethod
    def _safe_load(path: Path) -> Optional[np.ndarray]:
        """Load image with OpenCV; return None on failure."""
        try:
            img = cv2.imread(str(path))
            return img  # None if file is corrupt
        except Exception as exc:  # noqa: BLE001
            logger.error("Exception loading %s: %s", path, exc)
            return None

    def _convert_color(self, bgr: np.ndarray) -> np.ndarray:
        """Convert BGR → target colour space."""
        if self.color_mode == "RGB":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.color_mode == "GRAY":
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            return gray[..., np.newaxis]   # keep H×W×1 shape
        # Default: keep as BGR (unusual but user-configurable)
        return bgr

    @staticmethod
    def _letterbox_resize(
        img: np.ndarray,
        target_size: Tuple[int, int],   # (H, W)
    ) -> np.ndarray:
        """
        Resize image to target_size while preserving aspect ratio.
        Pads with black borders (letterbox) to fill the remaining space.
        This keeps hand proportions consistent — no squashing.
        """
        target_h, target_w = target_size
        src_h, src_w = img.shape[:2]

        scale = min(target_w / src_w, target_h / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create black canvas
        if img.ndim == 3:
            canvas = np.zeros((target_h, target_w, img.shape[2]), dtype=np.uint8)
        else:
            canvas = np.zeros((target_h, target_w), dtype=np.uint8)

        # Centre the resized image on the canvas
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def _normalise(self, img: np.ndarray) -> np.ndarray:
        """Cast to float32 and scale to [0, 1]."""
        return img.astype(np.float32) / self.normalize_scale

    # ── Utility: denormalise for saving/display ────────────────
    @staticmethod
    def denormalise(arr: np.ndarray) -> np.ndarray:
        """Convert normalised float32 array back to uint8 for saving with cv2."""
        return (arr * 255.0).clip(0, 255).astype(np.uint8)
