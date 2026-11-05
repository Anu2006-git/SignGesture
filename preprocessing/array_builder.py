"""
preprocessing/array_builder.py
================================
Assembles all preprocessed images and landmarks into compact NumPy
arrays ready for model training.

Outputs (saved to processed_dataset/arrays/)
--------------------------------------------
  images_cnn.npz
    X_train, y_train,       shape (N, 224, 224, 3) float32
    X_val,   y_val,         shape (M, 224, 224, 3) float32
    X_test,  y_test         shape (K, 224, 224, 3) float32

  images_lite.npz
    X_train, y_train, …     shape (N, 64, 64, 3)   float32

  landmarks.npz
    X_train, y_train, …     shape (N, 63)           float32
    label_names             list of alphabet letters

These npz files can be loaded directly into TensorFlow / PyTorch / sklearn.

Usage
-----
    from preprocessing.array_builder import ArrayBuilder
    builder = ArrayBuilder()
    builder.build()
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import (
    PROC_IMAGES_DIR,
    PROC_LANDMARKS_DIR,
    PROC_ARRAYS_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    LANDMARK_ENABLED,
    LANDMARK_FALLBACK,
    ALPHABETS,
    IMAGE_SIZE_CNN,
    IMAGE_SIZE_LITE,
    COLOR_MODE,
)

logger = logging.getLogger(__name__)

PROC_IMAGES_LITE_DIR = PROC_IMAGES_DIR.parent / "images_lite"
ALPHABETS_LIST = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


class ArrayBuilder:
    """
    Reads all saved .png images and .npy landmark files and
    packs them into train/val/test npz files.

    This step is separate from preprocessing so you can re-split
    without re-running the full pipeline.

    Usage
    -----
    >>> builder = ArrayBuilder()
    >>> builder.build()
    """

    def __init__(
        self,
        images_dir:      Path = PROC_IMAGES_DIR,
        images_lite_dir: Path = PROC_IMAGES_LITE_DIR,
        landmarks_dir:   Path = PROC_LANDMARKS_DIR,
        arrays_dir:      Path = PROC_ARRAYS_DIR,
        seed:            int  = RANDOM_SEED,
    ):
        self.images_dir      = Path(images_dir)
        self.images_lite_dir = Path(images_lite_dir)
        self.landmarks_dir   = Path(landmarks_dir)
        self.arrays_dir      = Path(arrays_dir)
        self.seed            = seed
        random.seed(seed)
        np.random.seed(seed)

    # ── Public API ─────────────────────────────────────────────
    def build(self) -> Dict[str, Dict]:
        """
        Load all processed images + landmarks, split into train/val/test,
        save as .npz files, and return the split dict.
        """
        self.arrays_dir.mkdir(parents=True, exist_ok=True)

        # ── Gather file lists per alphabet ─────────────────────
        cnn_paths,  cnn_labels  = self._collect_image_paths(self.images_dir)
        lite_paths, lite_labels = self._collect_image_paths(self.images_lite_dir)
        lm_paths,   lm_labels   = self._collect_landmark_paths()

        if not cnn_paths:
            logger.warning("No processed images found. Run preprocess.py first.")
            return {}

        # ── Load arrays ────────────────────────────────────────
        logger.info("Loading %d CNN images …", len(cnn_paths))
        X_cnn  = self._load_images(cnn_paths,  IMAGE_SIZE_CNN)

        logger.info("Loading %d lite images …", len(lite_paths))
        X_lite = self._load_images(lite_paths, IMAGE_SIZE_LITE)

        y_cnn  = np.array(cnn_labels,  dtype=np.int32)
        y_lite = np.array(lite_labels, dtype=np.int32)

        if LANDMARK_ENABLED and lm_paths:
            logger.info("Loading %d landmark vectors …", len(lm_paths))
            X_lm = self._load_landmarks(lm_paths)
            y_lm = np.array(lm_labels, dtype=np.int32)
        else:
            X_lm, y_lm = None, None

        # ── Split ──────────────────────────────────────────────
        splits_cnn  = self._split(X_cnn,  y_cnn)
        splits_lite = self._split(X_lite, y_lite)
        splits_lm   = self._split(X_lm,   y_lm) if X_lm is not None else None

        # ── Save ───────────────────────────────────────────────
        label_names = np.array(ALPHABETS_LIST)

        self._save_npz("images_cnn.npz",  splits_cnn,  label_names)
        self._save_npz("images_lite.npz", splits_lite, label_names)
        if splits_lm:
            self._save_npz("landmarks.npz", splits_lm, label_names)

        # ── Summary ────────────────────────────────────────────
        self._print_split_summary(splits_cnn, "CNN images")
        if splits_lm:
            self._print_split_summary(splits_lm, "Landmarks")

        return {"cnn": splits_cnn, "lite": splits_lite, "landmarks": splits_lm}

    # ── File collection ────────────────────────────────────────
    def _collect_image_paths(
        self, root_dir: Path
    ) -> Tuple[List[Path], List[int]]:
        paths, labels = [], []
        if not root_dir.exists():
            return paths, labels
        for idx, letter in enumerate(ALPHABETS_LIST):
            alpha_dir = root_dir / letter
            if not alpha_dir.exists():
                continue
            for img_path in sorted(alpha_dir.glob("*.png")):
                paths.append(img_path)
                labels.append(idx)
        return paths, labels

    def _collect_landmark_paths(
        self,
    ) -> Tuple[List[Path], List[int]]:
        paths, labels = [], []
        if not self.landmarks_dir.exists():
            return paths, labels
        for idx, letter in enumerate(ALPHABETS_LIST):
            lm_dir = self.landmarks_dir / letter
            if not lm_dir.exists():
                continue
            for npy_path in sorted(lm_dir.glob("*.npy")):
                paths.append(npy_path)
                labels.append(idx)
        return paths, labels

    # ── Array loading ──────────────────────────────────────────
    @staticmethod
    def _load_images(
        paths: List[Path], target_size: Tuple[int, int]
    ) -> np.ndarray:
        import cv2
        H, W = target_size
        C    = 1 if COLOR_MODE.upper() == "GRAY" else 3
        arr  = np.empty((len(paths), H, W, C), dtype=np.float32)
        for i, p in enumerate(paths):
            img = cv2.imread(str(p))
            if img is None:
                arr[i] = 0.0
                continue
            if COLOR_MODE.upper() == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif COLOR_MODE.upper() == "GRAY":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
            arr[i] = img.astype(np.float32) / 255.0
        return arr

    @staticmethod
    def _load_landmarks(paths: List[Path]) -> np.ndarray:
        vecs = [np.load(str(p)).astype(np.float32) for p in paths]
        return np.stack(vecs, axis=0)   # (N, 63)

    # ── Train / Val / Test split ───────────────────────────────
    def _split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict:
        if X is None or len(X) == 0:
            return {}

        n       = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)

        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        tr_idx  = indices[:n_train]
        val_idx = indices[n_train: n_train + n_val]
        te_idx  = indices[n_train + n_val:]

        return {
            "X_train": X[tr_idx], "y_train": y[tr_idx],
            "X_val":   X[val_idx], "y_val":   y[val_idx],
            "X_test":  X[te_idx],  "y_test":  y[te_idx],
        }

    # ── Save npz ───────────────────────────────────────────────
    def _save_npz(
        self, filename: str, splits: Dict, label_names: np.ndarray
    ) -> None:
        if not splits:
            return
        out_path = self.arrays_dir / filename
        np.savez_compressed(
            str(out_path),
            label_names=label_names,
            **splits,
        )
        logger.info("Saved → %s  (train=%d val=%d test=%d)",
                    out_path,
                    len(splits.get("X_train", [])),
                    len(splits.get("X_val",   [])),
                    len(splits.get("X_test",  [])))

    # ── Pretty print ───────────────────────────────────────────
    @staticmethod
    def _print_split_summary(splits: Dict, name: str) -> None:
        if not splits:
            return
        print(f"\n  📦  {name} splits:")
        print(f"      Train : {len(splits['X_train']):>6,} samples")
        print(f"      Val   : {len(splits['X_val']):>6,} samples")
        print(f"      Test  : {len(splits['X_test']):>6,} samples")
        print(f"      Total : {len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test']):>6,}")
