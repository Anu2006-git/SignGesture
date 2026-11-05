"""
preprocessing/augmentor.py
===========================
Safe data augmentation for hand-gesture images.

Rules enforced here:
  ✅  Rotation   ±15°  (configurable)
  ✅  Brightness  ±30%
  ✅  Zoom        ±10%
  ✅  Shift       ±10% of image dimensions
  ✅  Gaussian blur (optional, probabilistic)
  ✅  Salt-and-pepper noise (optional, probabilistic)
  ✗   Horizontal flip  — NEVER applied (mirrors gestures)
  ✗   Vertical flip    — NEVER applied (flips hand upside-down)
  ✗   Large rotations  — capped to keep gesture meaning intact
"""

import random
import logging
from typing import List, Tuple

import cv2
import numpy as np

from .config import (
    AUG_ROTATION_RANGE,
    AUG_BRIGHTNESS_RANGE,
    AUG_ZOOM_RANGE,
    AUG_SHIFT_RANGE,
    AUG_BLUR_PROB,
    AUG_BLUR_KERNEL,
    AUG_NOISE_PROB,
    AUG_NOISE_INTENSITY,
    AUGMENT_COPIES,
)

logger = logging.getLogger(__name__)


class Augmentor:
    """
    Produces N augmented copies of a single BGR image.

    Each copy applies a randomly sampled combination of:
    rotation, brightness, zoom, shift, blur, and noise.

    Usage
    -----
    >>> aug = Augmentor()
    >>> copies = aug.augment(image_bgr, n=3)   # list of 3 augmented images
    """

    def __init__(
        self,
        rotation_range: float  = AUG_ROTATION_RANGE,
        brightness_range: Tuple[float, float] = AUG_BRIGHTNESS_RANGE,
        zoom_range: float      = AUG_ZOOM_RANGE,
        shift_range: float     = AUG_SHIFT_RANGE,
        blur_prob: float       = AUG_BLUR_PROB,
        blur_kernel: Tuple     = AUG_BLUR_KERNEL,
        noise_prob: float      = AUG_NOISE_PROB,
        noise_intensity: float = AUG_NOISE_INTENSITY,
        n_copies: int          = AUGMENT_COPIES,
        seed: int              = None,
    ):
        self.rotation_range   = rotation_range
        self.brightness_range = brightness_range
        self.zoom_range       = zoom_range
        self.shift_range      = shift_range
        self.blur_prob        = blur_prob
        self.blur_kernel      = blur_kernel
        self.noise_prob       = noise_prob
        self.noise_intensity  = noise_intensity
        self.n_copies         = n_copies
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ── Public API ─────────────────────────────────────────────
    def augment(self, image: np.ndarray, n: int = None) -> List[np.ndarray]:
        """
        Return a list of *n* augmented copies.
        The original image is NOT included in the list.
        """
        n = n if n is not None else self.n_copies
        return [self._apply_pipeline(image) for _ in range(n)]

    # ── Full pipeline per copy ─────────────────────────────────
    def _apply_pipeline(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        img = self._apply_rotation(img)
        img = self._apply_zoom_and_shift(img)
        img = self._apply_brightness(img)
        if random.random() < self.blur_prob:
            img = self._apply_blur(img)
        if random.random() < self.noise_prob:
            img = self._apply_noise(img)
        return img

    # ── Individual transforms ──────────────────────────────────
    def _apply_rotation(self, img: np.ndarray) -> np.ndarray:
        """Rotate within ±rotation_range degrees, fill border with border replication."""
        h, w = img.shape[:2]
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale=1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _apply_zoom_and_shift(self, img: np.ndarray) -> np.ndarray:
        """
        Apply zoom (scale) and XY shift in a single affine transform.
        Zoom-out is capped so the hand stays within the frame.
        Shift keeps the gesture roughly centred.
        """
        h, w = img.shape[:2]

        # Zoom: scale in range [1 - zoom_range, 1 + zoom_range]
        # We cap min zoom at 0.85 so the hand never disappears
        zoom_factor = random.uniform(
            max(1.0 - self.zoom_range, 0.85),
            1.0 + self.zoom_range,
        )

        # Shift in pixels
        shift_x = random.uniform(-self.shift_range, self.shift_range) * w
        shift_y = random.uniform(-self.shift_range, self.shift_range) * h

        # Combine into a single 2×3 affine matrix
        cx, cy = w / 2, h / 2
        M = np.array([
            [zoom_factor, 0,           cx * (1 - zoom_factor) + shift_x],
            [0,           zoom_factor, cy * (1 - zoom_factor) + shift_y],
        ], dtype=np.float32)

        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _apply_brightness(self, img: np.ndarray) -> np.ndarray:
        """Multiply all pixel values by a random brightness factor."""
        factor = random.uniform(*self.brightness_range)
        bright = img.astype(np.float32) * factor
        return np.clip(bright, 0, 255).astype(np.uint8)

    def _apply_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply mild Gaussian blur to simulate defocus."""
        ksize = self.blur_kernel
        # Ensure odd kernel size
        ksize = (ksize[0] | 1, ksize[1] | 1)
        return cv2.GaussianBlur(img, ksize, sigmaX=0)

    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        """Add salt-and-pepper noise to a fraction of pixels."""
        noisy = img.copy()
        num_pixels = int(self.noise_intensity * img.size)
        coords_y = np.random.randint(0, img.shape[0], num_pixels)
        coords_x = np.random.randint(0, img.shape[1], num_pixels)
        # Salt (white) on half, pepper (black) on the other half
        half = num_pixels // 2
        noisy[coords_y[:half], coords_x[:half]] = 255
        noisy[coords_y[half:], coords_x[half:]] = 0
        return noisy
