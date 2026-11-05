"""
scanner.py
==========
Scans the dataset directory and collects metadata about every
alphabet's sub-datasets and images.  No model training code here.
"""

import os
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2

from .config import (
    DATASET_DIR,
    ALPHABETS,
    SUPPORTED_EXTENSIONS,
    DATASET_FOLDER_PREFIX,
    ENABLE_DUPLICATE_DETECTION,
    LOGS_DIR,
)

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "dataset_scanner.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────
@dataclass
class ImageRecord:
    """Metadata for a single image file."""
    path: Path
    alphabet: str
    dataset_folder: str        # e.g. "dataset_1"
    file_size_bytes: int = 0
    width: int = 0
    height: int = 0
    channels: int = 0
    file_hash: Optional[str] = None
    is_readable: bool = True
    is_corrupted: bool = False
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None   # path of the original
    error_message: str = ""


@dataclass
class DatasetFolderRecord:
    """Metadata for a single sub-dataset folder (e.g. dataset/A/dataset_1)."""
    alphabet: str
    folder_name: str           # "dataset_1"
    folder_path: Path
    image_records: List[ImageRecord] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return len(self.image_records)

    @property
    def valid_images(self) -> int:
        return sum(1 for r in self.image_records
                   if r.is_readable and not r.is_corrupted and not r.is_duplicate)

    @property
    def corrupted_count(self) -> int:
        return sum(1 for r in self.image_records if r.is_corrupted)

    @property
    def duplicate_count(self) -> int:
        return sum(1 for r in self.image_records if r.is_duplicate)

    @property
    def unreadable_count(self) -> int:
        return sum(1 for r in self.image_records if not r.is_readable)


@dataclass
class AlphabetRecord:
    """Aggregated metadata for all sub-datasets belonging to one alphabet."""
    alphabet: str
    alphabet_path: Path
    dataset_folders: List[DatasetFolderRecord] = field(default_factory=list)

    # ── Aggregated counts ──────────────────────────────────────
    @property
    def num_datasets(self) -> int:
        return len(self.dataset_folders)

    @property
    def total_images(self) -> int:
        return sum(f.total_images for f in self.dataset_folders)

    @property
    def valid_images(self) -> int:
        return sum(f.valid_images for f in self.dataset_folders)

    @property
    def corrupted_count(self) -> int:
        return sum(f.corrupted_count for f in self.dataset_folders)

    @property
    def duplicate_count(self) -> int:
        return sum(f.duplicate_count for f in self.dataset_folders)

    @property
    def unreadable_count(self) -> int:
        return sum(f.unreadable_count for f in self.dataset_folders)

    @property
    def all_images(self) -> List[ImageRecord]:
        images: List[ImageRecord] = []
        for folder in self.dataset_folders:
            images.extend(folder.image_records)
        return images


# ─────────────────────────────────────────────────────────────
# Scanner
# ─────────────────────────────────────────────────────────────
class DatasetScanner:
    """
    Recursively scans ``dataset/<LETTER>/dataset_N/`` and builds a
    complete in-memory index of all images.

    Usage
    -----
    >>> scanner = DatasetScanner()
    >>> data = scanner.scan()
    >>> print(data['A'].total_images)
    """

    def __init__(self, dataset_dir: Path = DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        self._hash_index: Dict[str, str] = {}   # hash -> first-seen path

    # ── Public API ─────────────────────────────────────────────
    def scan(self) -> Dict[str, AlphabetRecord]:
        """
        Scan the full dataset directory and return a dict keyed by alphabet letter.
        New datasets / images are automatically discovered — no code changes needed.
        """
        if not self.dataset_dir.exists():
            logger.warning("Dataset directory not found: %s", self.dataset_dir)
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created dataset directory: %s", self.dataset_dir)

        results: Dict[str, AlphabetRecord] = {}
        self._hash_index.clear()

        for letter in ALPHABETS:
            alpha_path = self.dataset_dir / letter
            if not alpha_path.exists():
                alpha_path.mkdir(parents=True, exist_ok=True)
                logger.debug("Created missing alphabet folder: %s", alpha_path)

            record = self._scan_alphabet(letter, alpha_path)
            results[letter] = record

        total_images = sum(r.total_images for r in results.values())
        logger.info(
            "Scan complete — %d alphabet(s) | %d total images",
            len(results), total_images,
        )
        return results

    # ── Internal helpers ────────────────────────────────────────
    def _scan_alphabet(self, letter: str, alpha_path: Path) -> AlphabetRecord:
        """Scan all dataset sub-folders for one alphabet letter."""
        record = AlphabetRecord(alphabet=letter, alphabet_path=alpha_path)

        # Collect and sort sub-folders that follow the naming convention
        sub_dirs = sorted(
            [d for d in alpha_path.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        if not sub_dirs:
            logger.debug("No sub-datasets found for alphabet '%s'", letter)
            return record

        for sub_dir in sub_dirs:
            folder_record = self._scan_dataset_folder(letter, sub_dir)
            record.dataset_folders.append(folder_record)

        return record

    def _scan_dataset_folder(
        self, letter: str, folder_path: Path
    ) -> DatasetFolderRecord:
        """Scan all image files inside a single sub-dataset folder."""
        folder_record = DatasetFolderRecord(
            alphabet=letter,
            folder_name=folder_path.name,
            folder_path=folder_path,
        )

        image_files = sorted(
            [
                f for f in folder_path.rglob("*")
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        )

        for img_path in image_files:
            img_record = self._build_image_record(letter, folder_path.name, img_path)
            folder_record.image_records.append(img_record)

        logger.debug(
            "  [%s/%s] %d image(s) found",
            letter, folder_path.name, len(folder_record.image_records),
        )
        return folder_record

    def _build_image_record(
        self, letter: str, dataset_folder: str, img_path: Path
    ) -> ImageRecord:
        """Read image metadata and flag issues."""
        record = ImageRecord(
            path=img_path,
            alphabet=letter,
            dataset_folder=dataset_folder,
        )

        # ── File size ──────────────────────────────────────────
        try:
            record.file_size_bytes = img_path.stat().st_size
        except OSError as exc:
            record.is_readable = False
            record.error_message = f"Cannot stat file: {exc}"
            return record

        # ── Try to decode with OpenCV ──────────────────────────
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                record.is_corrupted = True
                record.is_readable = False
                record.error_message = "OpenCV could not decode the image."
                return record

            h, w = img.shape[:2]
            c = img.shape[2] if len(img.shape) == 3 else 1
            record.width, record.height, record.channels = w, h, c

        except Exception as exc:  # noqa: BLE001
            record.is_corrupted = True
            record.is_readable = False
            record.error_message = str(exc)
            return record

        # ── Duplicate detection (hash-based) ───────────────────
        if ENABLE_DUPLICATE_DETECTION:
            file_hash = self._hash_file(img_path)
            record.file_hash = file_hash
            if file_hash in self._hash_index:
                record.is_duplicate = True
                record.duplicate_of = self._hash_index[file_hash]
            else:
                self._hash_index[file_hash] = str(img_path)

        return record

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Return MD5 hash of file content (fast, good enough for dedup)."""
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65_536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
