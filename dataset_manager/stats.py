"""
stats.py
========
Computes high-level statistics and per-alphabet status ratings from
scanned and validated dataset data.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from .config import (
    MIN_IMAGES_WARNING,
    MIN_IMAGES_GOOD,
    MIN_IMAGES_EXCELLENT,
    ALPHABETS,
)
from .scanner import AlphabetRecord
from .validator import AlphabetValidationSummary


# ─────────────────────────────────────────────────────────────
# Status levels
# ─────────────────────────────────────────────────────────────
class DataStatus(str, Enum):
    EMPTY       = "EMPTY"        # 0 images
    CRITICAL    = "CRITICAL"     # > 0 but all invalid or very few
    WARNING     = "WARNING"      # below MIN_IMAGES_WARNING valid images
    GOOD        = "GOOD"         # >= MIN_IMAGES_GOOD valid images
    EXCELLENT   = "EXCELLENT"    # >= MIN_IMAGES_EXCELLENT valid images


STATUS_EMOJI: Dict[DataStatus, str] = {
    DataStatus.EMPTY:     "⬜",
    DataStatus.CRITICAL:  "🔴",
    DataStatus.WARNING:   "🟡",
    DataStatus.GOOD:      "🟢",
    DataStatus.EXCELLENT: "🔵",
}

STATUS_LABEL: Dict[DataStatus, str] = {
    DataStatus.EMPTY:     "Empty — no images yet",
    DataStatus.CRITICAL:  "Critical — needs images immediately",
    DataStatus.WARNING:   "Warning — needs more images",
    DataStatus.GOOD:      "Good — sufficient for training",
    DataStatus.EXCELLENT: "Excellent — large dataset",
}


def compute_status(valid_images: int) -> DataStatus:
    if valid_images == 0:
        return DataStatus.EMPTY
    if valid_images < MIN_IMAGES_WARNING:
        return DataStatus.CRITICAL if valid_images < 10 else DataStatus.WARNING
    if valid_images >= MIN_IMAGES_EXCELLENT:
        return DataStatus.EXCELLENT
    if valid_images >= MIN_IMAGES_GOOD:
        return DataStatus.GOOD
    return DataStatus.WARNING


# ─────────────────────────────────────────────────────────────
# Per-alphabet stat block
# ─────────────────────────────────────────────────────────────
@dataclass
class AlphabetStats:
    alphabet: str
    num_datasets: int
    total_images: int
    valid_images: int
    corrupted_count: int
    duplicate_count: int
    unreadable_count: int
    status: DataStatus
    validation_clean: bool
    images_needed_for_good: int      # how many more to reach GOOD
    images_needed_for_excellent: int # how many more to reach EXCELLENT
    dataset_folder_names: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Full dataset statistics container
# ─────────────────────────────────────────────────────────────
@dataclass
class FullDatasetStats:
    per_alphabet: Dict[str, AlphabetStats] = field(default_factory=dict)

    @property
    def total_alphabets_with_data(self) -> int:
        return sum(1 for s in self.per_alphabet.values() if s.total_images > 0)

    @property
    def total_images(self) -> int:
        return sum(s.total_images for s in self.per_alphabet.values())

    @property
    def total_valid_images(self) -> int:
        return sum(s.valid_images for s in self.per_alphabet.values())

    @property
    def total_corrupted(self) -> int:
        return sum(s.corrupted_count for s in self.per_alphabet.values())

    @property
    def total_duplicates(self) -> int:
        return sum(s.duplicate_count for s in self.per_alphabet.values())

    @property
    def alphabets_empty(self) -> List[str]:
        return [k for k, v in self.per_alphabet.items()
                if v.status == DataStatus.EMPTY]

    @property
    def alphabets_critical(self) -> List[str]:
        return [k for k, v in self.per_alphabet.items()
                if v.status == DataStatus.CRITICAL]

    @property
    def alphabets_warning(self) -> List[str]:
        return [k for k, v in self.per_alphabet.items()
                if v.status == DataStatus.WARNING]

    @property
    def alphabets_good(self) -> List[str]:
        return [k for k, v in self.per_alphabet.items()
                if v.status == DataStatus.GOOD]

    @property
    def alphabets_excellent(self) -> List[str]:
        return [k for k, v in self.per_alphabet.items()
                if v.status == DataStatus.EXCELLENT]


# ─────────────────────────────────────────────────────────────
# Stats builder
# ─────────────────────────────────────────────────────────────
class DatasetStats:
    """
    Combines scanner + validator results into a clean statistics object.

    Usage
    -----
    >>> stats_builder = DatasetStats()
    >>> full_stats = stats_builder.compute(scanned_data, validation_summaries)
    """

    def compute(
        self,
        scanned_data: Dict[str, AlphabetRecord],
        validation_summaries: Dict[str, AlphabetValidationSummary],
    ) -> FullDatasetStats:
        full = FullDatasetStats()

        for letter in ALPHABETS:
            alpha = scanned_data.get(letter)
            vsummary = validation_summaries.get(letter)

            if alpha is None:
                continue

            valid_count = alpha.valid_images
            status = compute_status(valid_count)

            full.per_alphabet[letter] = AlphabetStats(
                alphabet=letter,
                num_datasets=alpha.num_datasets,
                total_images=alpha.total_images,
                valid_images=valid_count,
                corrupted_count=alpha.corrupted_count,
                duplicate_count=alpha.duplicate_count,
                unreadable_count=alpha.unreadable_count,
                status=status,
                validation_clean=(vsummary.is_clean if vsummary else True),
                images_needed_for_good=max(0, MIN_IMAGES_GOOD - valid_count),
                images_needed_for_excellent=max(0, MIN_IMAGES_EXCELLENT - valid_count),
                dataset_folder_names=[
                    f.folder_name for f in alpha.dataset_folders
                ],
            )

        return full
