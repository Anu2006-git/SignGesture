"""
preprocessing/pipeline.py
==========================
Orchestrates the full preprocessing pipeline:

  DatasetScanner → ImageProcessor → Augmentor
                                  → LandmarkExtractor
                                  → DataSaver
                                  → CacheManager
                                  → PreprocessingReporter
                                  → ArrayBuilder  (optional)

This class is the single entry point called by preprocess.py.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    DATASET_DIR,
    ALPHABETS,
    AUGMENTATION_ENABLED,
    LANDMARK_ENABLED,
    INCREMENTAL_MODE,
)
from .image_processor    import ImageProcessor, ProcessedImage
from .augmentor          import Augmentor
from .landmark_extractor import LandmarkExtractor
from .data_saver         import DataSaver
from .cache_manager      import CacheManager
from .reporter           import PreprocessingReporter, AlphabetPreprocessStats
from .array_builder      import ArrayBuilder

logger = logging.getLogger(__name__)
ALPHABETS_LIST = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Supported image extensions (same as dataset_manager)
SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class PreprocessingPipeline:
    """
    Full preprocessing pipeline for the Sign Language dataset.

    Usage
    -----
    >>> pipeline = PreprocessingPipeline()
    >>> pipeline.run()                    # process all letters
    >>> pipeline.run(letters=["A", "B"]) # process specific letters only
    >>> pipeline.run(build_arrays=True)  # also build .npz files
    """

    def __init__(
        self,
        dataset_dir:    Path = DATASET_DIR,
        augmentation:   bool = AUGMENTATION_ENABLED,
        landmarks:      bool = LANDMARK_ENABLED,
        incremental:    bool = INCREMENTAL_MODE,
    ):
        self.dataset_dir  = Path(dataset_dir)
        self.augmentation = augmentation
        self.incremental  = incremental
        self.use_landmarks= landmarks

        self.processor  = ImageProcessor(augmentation=augmentation)
        self.saver      = DataSaver()
        self.cache      = CacheManager()
        self.reporter   = PreprocessingReporter()

        # Lazy-init: only create if needed
        self._extractor: Optional[LandmarkExtractor] = None

    # ── Public API ─────────────────────────────────────────────
    def run(
        self,
        letters: Optional[List[str]] = None,
        build_arrays: bool = False,
    ) -> None:
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        letters      : subset of A-Z to process; None = all 26
        build_arrays : if True, also call ArrayBuilder at the end
        """
        t0 = time.time()
        target_letters = [l.upper() for l in letters] if letters else ALPHABETS_LIST

        print()
        print("=" * 70)
        print("  🔧  SIGN LANGUAGE PREPROCESSING PIPELINE")
        print("=" * 70)
        print(f"  Dataset dir   : {self.dataset_dir}")
        print(f"  Augmentation  : {'ON  (×' + str(self.processor.augmentor.n_copies) + ' copies)' if self.augmentation else 'OFF'}")
        print(f"  Landmarks     : {'ON' if self.use_landmarks else 'OFF'}")
        print(f"  Incremental   : {'ON' if self.incremental else 'OFF (full reprocess)'}")
        print(f"  Letters       : {', '.join(target_letters)}")
        print("=" * 70)
        print()

        # ── Init landmark extractor once ───────────────────────
        if self.use_landmarks:
            try:
                self._extractor = LandmarkExtractor()
            except ImportError as e:
                logger.warning("MediaPipe not available: %s — skipping landmarks.", e)
                self._extractor  = None
                self.use_landmarks = False

        # ── Load cache ─────────────────────────────────────────
        self.cache.load()

        # ── Per-alphabet processing ────────────────────────────
        global_index = 0
        for letter in target_letters:
            alpha_stats = self._process_alphabet(letter, global_index)
            global_index += alpha_stats.processed
            self.reporter.record_alphabet(alpha_stats)

        # ── Save cache ─────────────────────────────────────────
        self.cache.save()

        # ── Optional: build .npz arrays ────────────────────────
        if build_arrays:
            print("\n📦  Building NumPy arrays for training …")
            builder = ArrayBuilder()
            builder.build()

        # ── Reports ────────────────────────────────────────────
        duration = time.time() - t0
        self.reporter.finalize(duration_sec=duration)

        print(f"\n✅  Preprocessing complete in {duration:.1f}s\n")

    # ── Per-alphabet ───────────────────────────────────────────
    def _process_alphabet(
        self, letter: str, global_index: int
    ) -> AlphabetPreprocessStats:
        stats = AlphabetPreprocessStats(alphabet=letter)

        alpha_dir = self.dataset_dir / letter
        if not alpha_dir.exists():
            logger.debug("No directory for '%s' — skipping.", letter)
            return stats

        # Collect all image paths recursively
        all_paths = sorted([
            p for p in alpha_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
        ])
        stats.total_source = len(all_paths)

        if not all_paths:
            logger.info("[%s] No images found.", letter)
            return stats

        label_index = ord(letter) - ord('A')
        file_counter = global_index

        print(f"  [{letter}]  {len(all_paths)} source image(s) …", end="", flush=True)

        for img_path in all_paths:
            # ── Incremental skip ───────────────────────────────
            if self.incremental and self.cache.is_processed(img_path):
                stats.skipped_cached += 1
                continue

            # ── Process ────────────────────────────────────────
            proc = self.processor.process(
                source_path=img_path,
                label=letter,
                label_index=label_index,
                augment=self.augmentation,
            )

            if not proc.is_valid:
                stats.skipped_corrupted += 1
                stats.errors.append(
                    f"{img_path.name}: {proc.error_message}"
                )
                continue

            # ── Landmark extraction ────────────────────────────
            if self.use_landmarks and self._extractor:
                self._extractor.attach_to_processed([proc])
                if proc.landmark_raw is not None:
                    if proc.landmark_raw.any():
                        stats.landmark_detected += 1
                    else:
                        stats.landmark_zeros += 1

            # ── Save ───────────────────────────────────────────
            file_counter += 1
            stem = self.saver.save(proc, file_index=file_counter)

            # ── Cache ──────────────────────────────────────────
            self.cache.mark_processed(img_path, stem)

            stats.processed += 1
            stats.augmented_total += len(proc.augmented_cnn)

        print(
            f"  done  "
            f"(new={stats.processed} cached={stats.skipped_cached} "
            f"bad={stats.skipped_corrupted})"
        )
        return stats
