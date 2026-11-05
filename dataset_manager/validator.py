"""
validator.py
============
Deep validation of individual images and entire alphabet datasets.
Flags corrupted, unreadable, wrongly-formatted, too-small, and
out-of-range images.  Keeps all validation logic separate from
scanning and training code.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .config import (
    MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT,
    MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT,
    MIN_FILE_SIZE_BYTES, MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
)
from .scanner import AlphabetRecord, ImageRecord

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Per-image validation result
# ─────────────────────────────────────────────────────────────
@dataclass
class ImageValidationResult:
    path: Path
    is_valid: bool = True
    issues: List[str] = field(default_factory=list)

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)
        self.is_valid = False


# ─────────────────────────────────────────────────────────────
# Per-alphabet validation summary
# ─────────────────────────────────────────────────────────────
@dataclass
class AlphabetValidationSummary:
    alphabet: str
    total_checked: int = 0
    valid_count: int = 0
    corrupted_count: int = 0
    unreadable_count: int = 0
    duplicate_count: int = 0
    wrong_format_count: int = 0
    too_small_count: int = 0
    too_large_count: int = 0
    oversized_file_count: int = 0
    undersized_file_count: int = 0
    issues: List[str] = field(default_factory=list)

    @property
    def invalid_count(self) -> int:
        return self.total_checked - self.valid_count

    @property
    def is_clean(self) -> bool:
        return self.invalid_count == 0


# ─────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────
class ImageValidator:
    """
    Validates every ImageRecord produced by DatasetScanner and
    returns per-alphabet summaries.

    Usage
    -----
    >>> validator = ImageValidator()
    >>> summaries = validator.validate_all(scanned_data)
    >>> print(summaries['A'].is_clean)
    """

    def validate_all(
        self, scanned_data: Dict[str, AlphabetRecord]
    ) -> Dict[str, AlphabetValidationSummary]:
        """Validate all alphabets and return a dict of summaries."""
        summaries: Dict[str, AlphabetValidationSummary] = {}
        for letter, alpha_record in scanned_data.items():
            summaries[letter] = self.validate_alphabet(alpha_record)
        return summaries

    def validate_alphabet(
        self, alpha_record: AlphabetRecord
    ) -> AlphabetValidationSummary:
        """Validate all images in one alphabet's record."""
        summary = AlphabetValidationSummary(alphabet=alpha_record.alphabet)

        for img_record in alpha_record.all_images:
            result = self._validate_image(img_record)
            summary.total_checked += 1

            if result.is_valid:
                summary.valid_count += 1
            else:
                # Tally specific issue types
                for issue in result.issues:
                    il = issue.lower()
                    if "corrupt" in il:
                        summary.corrupted_count += 1
                    if "unreadable" in il or "cannot" in il or "opencv" in il:
                        summary.unreadable_count += 1
                    if "duplicate" in il:
                        summary.duplicate_count += 1
                    if "format" in il or "extension" in il:
                        summary.wrong_format_count += 1
                    if "too small" in il:
                        summary.too_small_count += 1
                    if "too large (dimensions)" in il:
                        summary.too_large_count += 1
                    if "file size too large" in il:
                        summary.oversized_file_count += 1
                    if "file size too small" in il:
                        summary.undersized_file_count += 1

                summary.issues.append(
                    f"{img_record.path.name}: {'; '.join(result.issues)}"
                )

        logger.debug(
            "Validation [%s]: %d valid / %d total",
            alpha_record.alphabet, summary.valid_count, summary.total_checked,
        )
        return summary

    # ── Single-image validation ────────────────────────────────
    def _validate_image(self, record: ImageRecord) -> ImageValidationResult:
        result = ImageValidationResult(path=record.path)

        # 1. Check file extension
        if record.path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            result.add_issue(
                f"Unsupported format/extension '{record.path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # 2. Check readability / corruption (already flagged by scanner)
        if not record.is_readable:
            result.add_issue(f"Unreadable — {record.error_message or 'unknown error'}")
            return result   # no point checking dimensions on an unreadable file

        if record.is_corrupted:
            result.add_issue(f"Corrupted — {record.error_message or 'OpenCV decode failed'}")
            return result

        # 3. Duplicate
        if record.is_duplicate:
            result.add_issue(f"Duplicate of {record.duplicate_of}")

        # 4. File-size bounds
        if record.file_size_bytes < MIN_FILE_SIZE_BYTES:
            result.add_issue(
                f"File size too small ({record.file_size_bytes} B < {MIN_FILE_SIZE_BYTES} B)"
            )
        if record.file_size_bytes > MAX_FILE_SIZE_BYTES:
            result.add_issue(
                f"File size too large ({record.file_size_bytes // 1024} KB > "
                f"{MAX_FILE_SIZE_BYTES // 1024} KB)"
            )

        # 5. Dimension bounds
        if record.width > 0 and record.height > 0:
            if record.width < MIN_IMAGE_WIDTH or record.height < MIN_IMAGE_HEIGHT:
                result.add_issue(
                    f"Too small (dimensions): {record.width}×{record.height} px "
                    f"(min {MIN_IMAGE_WIDTH}×{MIN_IMAGE_HEIGHT})"
                )
            if record.width > MAX_IMAGE_WIDTH or record.height > MAX_IMAGE_HEIGHT:
                result.add_issue(
                    f"Too large (dimensions): {record.width}×{record.height} px "
                    f"(max {MAX_IMAGE_WIDTH}×{MAX_IMAGE_HEIGHT})"
                )

        return result
