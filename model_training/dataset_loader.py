"""
model_training/dataset_loader.py
=================================
Automated dataset loader, class distribution analyzer, and
leakage-preventing data splitter for Sign Language recognition.
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    ALPHABETS,
    DATASET_DIR,
    LABEL_TO_IDX,
    RANDOM_SEED,
    SUPPORTED_EXTENSIONS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


@dataclass
class ImageSample:
    """Represents a single image sample with metadata."""
    path: Path
    alphabet: str
    label_idx: int
    subfolder: str       # e.g., 'dataset_1' (recording session / person identifier)
    file_name: str


@dataclass
class DatasetDistribution:
    """Summary of class distribution across all alphabets."""
    total_images: int = 0
    per_class_counts: Dict[str, int] = field(default_factory=dict)
    active_classes: List[str] = field(default_factory=list)
    empty_classes: List[str] = field(default_factory=list)
    is_imbalanced: bool = False
    imbalance_ratio: float = 1.0
    imbalance_notes: List[str] = field(default_factory=list)


@dataclass
class DatasetSplits:
    """Container for train, validation, and test samples."""
    train: List[ImageSample] = field(default_factory=list)
    val: List[ImageSample] = field(default_factory=list)
    test: List[ImageSample] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


class DatasetLoader:
    """
    Scans dataset directories, analyzes balance, and performs
    session/subfolder-aware splitting to prevent data leakage.
    """

    def __init__(self, dataset_dir: Path = DATASET_DIR, seed: int = RANDOM_SEED):
        self.dataset_dir = Path(dataset_dir)
        self.seed = seed

    def load_and_split(self, max_samples: Optional[int] = None) -> Tuple[DatasetSplits, DatasetDistribution]:
        """
        Loads all images, checks distribution, and splits into train/val/test.
        Optional max_samples allows fast benchmarking across all classes.
        """
        random.seed(self.seed)

        # 1. Gather all samples
        samples_by_class: Dict[str, List[ImageSample]] = defaultdict(list)
        for letter in ALPHABETS:
            alpha_dir = self.dataset_dir / letter
            if not alpha_dir.exists():
                continue

            subfolders = sorted([d for d in alpha_dir.iterdir() if d.is_dir()])
            for s_dir in subfolders:
                image_files = sorted([
                    f for f in s_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
                ])
                for img_path in image_files:
                    sample = ImageSample(
                        path=img_path,
                        alphabet=letter,
                        label_idx=LABEL_TO_IDX[letter],
                        subfolder=s_dir.name,
                        file_name=img_path.name,
                    )
                    samples_by_class[letter].append(sample)

        # Optional balanced subsampling if max_samples is requested
        if max_samples is not None:
            active = [l for l, s in samples_by_class.items() if s]
            if active:
                per_class = max(4, max_samples // len(active))
                for l in active:
                    if len(samples_by_class[l]) > per_class:
                        random.shuffle(samples_by_class[l])
                        samples_by_class[l] = samples_by_class[l][:per_class]

        # 2. Analyze class distribution
        distribution = self._analyze_distribution(samples_by_class)

        # 3. Perform leakage-aware split per class
        splits = DatasetSplits()
        for letter, samples in samples_by_class.items():
            if not samples:
                continue
            tr, va, te = self._split_class_samples(samples)
            splits.train.extend(tr)
            splits.val.extend(va)
            splits.test.extend(te)

        # Shuffle each split so classes are mixed uniformly
        random.shuffle(splits.train)
        random.shuffle(splits.val)
        random.shuffle(splits.test)

        self._print_summary(distribution, splits)
        return splits, distribution

    def _analyze_distribution(
        self, samples_by_class: Dict[str, List[ImageSample]]
    ) -> DatasetDistribution:
        """Examines class balance and identifies potential imbalance issues."""
        dist = DatasetDistribution()
        counts = {letter: len(samples_by_class.get(letter, [])) for letter in ALPHABETS}
        dist.per_class_counts = counts
        dist.total_images = sum(counts.values())

        dist.active_classes = [l for l, c in counts.items() if c > 0]
        dist.empty_classes = [l for l, c in counts.items() if c == 0]

        active_counts = [c for c in counts.values() if c > 0]
        if len(active_counts) > 1:
            min_c = min(active_counts)
            max_c = max(active_counts)
            avg_c = sum(active_counts) / len(active_counts)
            dist.imbalance_ratio = max_c / max(1, min_c)

            if dist.imbalance_ratio >= 3.0:
                dist.is_imbalanced = True
                dist.imbalance_notes.append(
                    f"Severe imbalance: max class has {max_c} images, min class has {min_c} (ratio {dist.imbalance_ratio:.1f}x)"
                )
            elif dist.imbalance_ratio >= 1.8:
                dist.is_imbalanced = True
                dist.imbalance_notes.append(
                    f"Moderate imbalance: counts vary between {min_c} and {max_c} (ratio {dist.imbalance_ratio:.1f}x)"
                )

        if dist.empty_classes:
            dist.imbalance_notes.append(
                f"{len(dist.empty_classes)} alphabet(s) currently have 0 images ({', '.join(dist.empty_classes[:6])}...)"
            )

        return dist

    def _split_class_samples(
        self, samples: List[ImageSample]
    ) -> Tuple[List[ImageSample], List[ImageSample], List[ImageSample]]:
        """
        Group-aware split:
        If multiple subfolders (recording sessions) exist, keeps sessions together
        in train, val, or test to prevent temporal/session data leakage!
        If only 1 subfolder exists, performs stratified chunk splitting.
        """
        # Group by subfolder (session/person)
        by_session: Dict[str, List[ImageSample]] = defaultdict(list)
        for s in samples:
            by_session[s.subfolder].append(s)

        sessions = sorted(by_session.keys())

        train_set: List[ImageSample] = []
        val_set: List[ImageSample] = []
        test_set: List[ImageSample] = []

        if len(sessions) >= 3:
            # Multi-session: assign distinct sessions to test and val to avoid data leakage!
            random.shuffle(sessions)
            n_test_sess = max(1, int(round(len(sessions) * TEST_RATIO)))
            n_val_sess = max(1, int(round(len(sessions) * VAL_RATIO)))

            test_sessions = sessions[:n_test_sess]
            val_sessions = sessions[n_test_sess : n_test_sess + n_val_sess]
            train_sessions = sessions[n_test_sess + n_val_sess :]

            if not train_sessions:
                # If only 3 sessions total, ensure at least 1 in train
                train_sessions = [val_sessions.pop()]

            for sess in train_sessions:
                train_set.extend(by_session[sess])
            for sess in val_sessions:
                val_set.extend(by_session[sess])
            for sess in test_sessions:
                test_set.extend(by_session[sess])

        else:
            # 1 or 2 sessions: Split temporally or sequentially within each session
            # to preserve slight continuity while reserving unseen chunks for test
            for sess, sess_samples in by_session.items():
                n = len(sess_samples)
                if n < 3:
                    # Very few images: put all into train
                    train_set.extend(sess_samples)
                    continue

                shuffled = sess_samples.copy()
                random.shuffle(shuffled)

                n_test = max(1, int(round(n * TEST_RATIO)))
                n_val = max(1, int(round(n * VAL_RATIO)))
                n_train = n - n_test - n_val

                if n_train < 1:
                    n_train = 1
                    if n_val > 1:
                        n_val -= 1
                    elif n_test > 1:
                        n_test -= 1

                test_set.extend(shuffled[:n_test])
                val_set.extend(shuffled[n_test : n_test + n_val])
                train_set.extend(shuffled[n_test + n_val :])

        return train_set, val_set, test_set

    def _print_summary(self, dist: DatasetDistribution, splits: DatasetSplits) -> None:
        """Displays distribution and split statistics."""
        print("\n" + "=" * 75)
        print(" 📊  DATASET DISTRIBUTION & LEAKAGE-AWARE SPLIT SUMMARY")
        print("=" * 75)
        print(f"  Total Images Found    : {dist.total_images:,}")
        print(f"  Populated Alphabets   : {len(dist.active_classes)} / 26")
        print(f"  Empty Alphabets       : {len(dist.empty_classes)} / 26")

        if dist.is_imbalanced:
            print("  ⚠️ Class Balance Status : IMBALANCED")
            for note in dist.imbalance_notes:
                print(f"     • {note}")
        else:
            print("  ✅ Class Balance Status : BALANCED")

        print("-" * 75)
        print("  Partitioning (Leakage-Preventing):")
        print(f"    • Train Set         : {len(splits.train):>6,} images ({len(splits.train)/max(1, splits.total)*100:.1f}%)")
        print(f"    • Validation Set    : {len(splits.val):>6,} images ({len(splits.val)/max(1, splits.total)*100:.1f}%)")
        print(f"    • Test Set (Unseen) : {len(splits.test):>6,} images ({len(splits.test)/max(1, splits.total)*100:.1f}%)")
        print("=" * 75 + "\n")
