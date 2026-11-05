"""
preprocessing/reporter.py
==========================
Generates a comprehensive preprocessing report:
  - Per-alphabet counts (processed, augmented, landmarks detected)
  - Skipped / corrupted images
  - Augmentation statistics
  - Landmark detection rate
  - Output directory sizes

Saved to processed_dataset/reports/preprocessing_report.txt
        processed_dataset/reports/preprocessing_report.json
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    PROC_REPORTS_DIR,
    ALPHABETS,
    IMAGE_SIZE_CNN,
    IMAGE_SIZE_LITE,
    COLOR_MODE,
    AUGMENTATION_ENABLED,
    AUGMENT_COPIES,
    AUG_ROTATION_RANGE,
    AUG_BRIGHTNESS_RANGE,
    AUG_ZOOM_RANGE,
    AUG_SHIFT_RANGE,
    LANDMARK_ENABLED,
    LANDMARK_NORMALIZATION,
    LANDMARK_FALLBACK,
)

logger = logging.getLogger(__name__)
ALPHABETS_LIST = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


# ─────────────────────────────────────────────────────────────
# Per-alphabet stats collected during the pipeline
# ─────────────────────────────────────────────────────────────
@dataclass
class AlphabetPreprocessStats:
    alphabet:           str
    total_source:       int = 0   # images found in dataset/
    skipped_corrupted:  int = 0   # unreadable / corrupted
    skipped_cached:     int = 0   # already processed (incremental)
    processed:          int = 0   # newly processed this run
    augmented_total:    int = 0   # augmented copies created
    landmark_detected:  int = 0   # images where hand was found
    landmark_zeros:     int = 0   # images where no hand → zero vector
    errors:             List[str] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        base = self.processed + self.skipped_cached
        return (self.landmark_detected / base * 100) if base > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# Reporter
# ─────────────────────────────────────────────────────────────
class PreprocessingReporter:
    """
    Accumulates per-alphabet statistics and writes reports.

    Usage
    -----
    >>> reporter = PreprocessingReporter()
    >>> reporter.record_alphabet(stats)        # called after each letter
    >>> reporter.finalize(duration_sec=42.5)   # write reports
    """

    def __init__(self, reports_dir: Path = PROC_REPORTS_DIR):
        self.reports_dir: Path = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._stats: Dict[str, AlphabetPreprocessStats] = {}
        self._start_time: datetime = datetime.now()

    # ── Data collection ────────────────────────────────────────
    def record_alphabet(self, stats: AlphabetPreprocessStats) -> None:
        self._stats[stats.alphabet] = stats

    # ── Report generation ──────────────────────────────────────
    def finalize(self, duration_sec: float = 0.0) -> None:
        """Write text and JSON reports."""
        self._save_text(duration_sec)
        self._save_json(duration_sec)
        print(f"\n📄  Preprocessing reports saved → {self.reports_dir}")

    # ── Text report ────────────────────────────────────────────
    def _save_text(self, duration_sec: float) -> None:
        timestamp = self._start_time.strftime("%Y-%m-%d %H:%M:%S")
        totals = self._compute_totals()

        lines = [
            "=" * 100,
            " SIGN LANGUAGE PREPROCESSING REPORT",
            f" Generated    : {timestamp}",
            f" Duration     : {duration_sec:.1f} s",
            "=" * 100,
            "",
            "PREPROCESSING CONFIGURATION",
            "-" * 60,
            f"  CNN image size        : {IMAGE_SIZE_CNN[1]}×{IMAGE_SIZE_CNN[0]} px",
            f"  Lite image size       : {IMAGE_SIZE_LITE[1]}×{IMAGE_SIZE_LITE[0]} px",
            f"  Colour mode           : {COLOR_MODE}",
            f"  Augmentation          : {'ENABLED' if AUGMENTATION_ENABLED else 'DISABLED'}",
        ]

        if AUGMENTATION_ENABLED:
            lines += [
                f"    Copies per image    : {AUGMENT_COPIES}",
                f"    Rotation range      : ±{AUG_ROTATION_RANGE}°",
                f"    Brightness range    : {AUG_BRIGHTNESS_RANGE[0]:.0%} – {AUG_BRIGHTNESS_RANGE[1]:.0%}",
                f"    Zoom range          : ±{AUG_ZOOM_RANGE:.0%}",
                f"    Shift range         : ±{AUG_SHIFT_RANGE:.0%}",
            ]

        lines += [
            f"  Landmark extraction   : {'ENABLED' if LANDMARK_ENABLED else 'DISABLED'}",
        ]
        if LANDMARK_ENABLED:
            lines += [
                f"    Normalisation       : {LANDMARK_NORMALIZATION}",
                f"    No-hand fallback    : {LANDMARK_FALLBACK}",
            ]

        lines += [
            "",
            "OVERALL TOTALS",
            "-" * 60,
            f"  Source images found   : {totals['total_source']:,}",
            f"  Skipped (corrupted)   : {totals['skipped_corrupted']:,}",
            f"  Skipped (cached)      : {totals['skipped_cached']:,}",
            f"  Newly processed       : {totals['processed']:,}",
            f"  Augmented copies      : {totals['augmented_total']:,}",
            f"  Total output images   : {totals['processed'] + totals['augmented_total'] + totals['skipped_cached']:,}",
        ]

        if LANDMARK_ENABLED:
            detect_rate = (
                totals["landmark_detected"] / max(totals["processed"], 1) * 100
            )
            lines += [
                f"  Landmark detected     : {totals['landmark_detected']:,}  ({detect_rate:.1f}%)",
                f"  Landmark zeros        : {totals['landmark_zeros']:,}",
            ]

        lines += [
            "",
            "PER-ALPHABET BREAKDOWN",
            "-" * 110,
            f"{'#':<4}{'Letter':<8}{'Source':>8}{'Corrupted':>10}{'Cached':>8}"
            f"{'Processed':>11}{'Augmented':>11}{'LM Detected':>13}{'LM Rate':>9}",
            "-" * 110,
        ]

        for idx, letter in enumerate(ALPHABETS_LIST, 1):
            s = self._stats.get(letter)
            if s is None:
                lines.append(f"{idx:<4}{letter:<8}{'—':>8}")
                continue
            lm_rate = f"{s.detection_rate:.0f}%" if LANDMARK_ENABLED else "N/A"
            lines.append(
                f"{idx:<4}{letter:<8}{s.total_source:>8}{s.skipped_corrupted:>10}"
                f"{s.skipped_cached:>8}{s.processed:>11}{s.augmented_total:>11}"
                f"{s.landmark_detected:>13}{lm_rate:>9}"
            )

        lines += ["-" * 110]

        # ── Error summary ──────────────────────────────────────
        all_errors = [
            f"[{ltr}] {e}"
            for ltr, s in self._stats.items()
            for e in s.errors[:5]
        ]
        if all_errors:
            lines += ["", "PROCESSING ERRORS (first 5 per alphabet)", "-" * 60]
            lines.extend(f"  {e}" for e in all_errors[:30])
            if len(all_errors) > 30:
                lines.append(f"  … ({len(all_errors) - 30} more — see JSON report)")
        else:
            lines += ["", "PROCESSING ERRORS", "-" * 60, "  None ✅"]

        lines += ["", "=" * 100, "END OF REPORT", "=" * 100]

        out = self.reports_dir / "preprocessing_report.txt"
        out.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Text preprocessing report → %s", out)

    # ── JSON report ────────────────────────────────────────────
    def _save_json(self, duration_sec: float) -> None:
        totals = self._compute_totals()
        data = {
            "generated_at":    self._start_time.isoformat(),
            "duration_seconds": round(duration_sec, 2),
            "config": {
                "image_size_cnn":         list(IMAGE_SIZE_CNN),
                "image_size_lite":        list(IMAGE_SIZE_LITE),
                "color_mode":             COLOR_MODE,
                "augmentation_enabled":   AUGMENTATION_ENABLED,
                "augment_copies":         AUGMENT_COPIES,
                "landmark_enabled":       LANDMARK_ENABLED,
                "landmark_normalization": LANDMARK_NORMALIZATION,
                "landmark_fallback":      LANDMARK_FALLBACK,
            },
            "totals": totals,
            "per_alphabet": {
                ltr: {
                    "total_source":      s.total_source,
                    "skipped_corrupted": s.skipped_corrupted,
                    "skipped_cached":    s.skipped_cached,
                    "processed":         s.processed,
                    "augmented_total":   s.augmented_total,
                    "landmark_detected": s.landmark_detected,
                    "landmark_zeros":    s.landmark_zeros,
                    "detection_rate_pct": round(s.detection_rate, 1),
                    "errors":            s.errors,
                }
                for ltr, s in self._stats.items()
            },
        }
        out = self.reports_dir / "preprocessing_report.json"
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("JSON preprocessing report → %s", out)

    # ── Totals helper ──────────────────────────────────────────
    def _compute_totals(self) -> Dict:
        keys = [
            "total_source", "skipped_corrupted", "skipped_cached",
            "processed", "augmented_total", "landmark_detected", "landmark_zeros",
        ]
        return {k: sum(getattr(s, k, 0) for s in self._stats.values()) for k in keys}
