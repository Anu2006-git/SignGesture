"""
reporter.py
===========
Generates human-readable text and JSON summary reports of the dataset.
Reports are saved to the ``reports/`` directory.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from .config import REPORTS_DIR, REPORT_FILENAME, JSON_REPORT_FILENAME, ALPHABETS
from .stats import FullDatasetStats, AlphabetStats, STATUS_LABEL, STATUS_EMOJI
from .validator import AlphabetValidationSummary

logger = logging.getLogger(__name__)


class DatasetReporter:
    """
    Generates and saves dataset summary reports.

    Usage
    -----
    >>> reporter = DatasetReporter()
    >>> reporter.generate(full_stats, validation_summaries)
    """

    def __init__(self, reports_dir: Path = REPORTS_DIR):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        full_stats: FullDatasetStats,
        validation_summaries: Dict[str, AlphabetValidationSummary],
    ) -> None:
        """Generate and save both text and JSON reports."""
        self._save_text_report(full_stats, validation_summaries)
        self._save_json_report(full_stats, validation_summaries)

    # ── Text report ────────────────────────────────────────────
    def _save_text_report(
        self,
        full_stats: FullDatasetStats,
        validation_summaries: Dict[str, AlphabetValidationSummary],
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=" * 90,
            " SIGN LANGUAGE DATASET — SUMMARY REPORT",
            f" Generated : {timestamp}",
            "=" * 90,
            "",
            "OVERALL STATISTICS",
            "-" * 50,
            f"  Total alphabets with data : {full_stats.total_alphabets_with_data} / 26",
            f"  Total images (raw)        : {full_stats.total_images:,}",
            f"  Total valid images        : {full_stats.total_valid_images:,}",
            f"  Total corrupted           : {full_stats.total_corrupted:,}",
            f"  Total duplicates          : {full_stats.total_duplicates:,}",
            "",
            "PER-ALPHABET BREAKDOWN",
            "-" * 90,
            f"{'#':<4}{'Letter':<8}{'Datasets':>10}{'Total':>10}{'Valid':>10}"
            f"{'Corrupt':>10}{'Dups':>8}  {'Status':<35}{'Validation'}",
            "-" * 90,
        ]

        for idx, letter in enumerate(ALPHABETS, 1):
            s: AlphabetStats = full_stats.per_alphabet.get(letter)
            if s is None:
                continue
            vs = validation_summaries.get(letter)
            val_txt = "Clean" if (vs and vs.is_clean) else f"{vs.invalid_count if vs else 0} issue(s)"
            emoji = STATUS_EMOJI[s.status]
            status_str = f"{emoji} {s.status.value}"
            lines.append(
                f"{idx:<4}{letter:<8}{s.num_datasets:>10}{s.total_images:>10}"
                f"{s.valid_images:>10}{s.corrupted_count:>10}{s.duplicate_count:>8}"
                f"  {status_str:<35}{val_txt}"
            )

        lines += [
            "-" * 90,
            "",
            "ALPHABETS NEEDING ATTENTION",
            "-" * 50,
        ]
        for letter in full_stats.alphabets_empty:
            lines.append(f"  ⬜  {letter} — EMPTY — add images to dataset/{letter}/dataset_1/")
        for letter in full_stats.alphabets_critical:
            s = full_stats.per_alphabet[letter]
            lines.append(f"  🔴  {letter} — CRITICAL — only {s.valid_images} valid image(s)")
        for letter in full_stats.alphabets_warning:
            s = full_stats.per_alphabet[letter]
            lines.append(f"  🟡  {letter} — WARNING — {s.valid_images} images "
                         f"(need {s.images_needed_for_good} more for GOOD)")

        lines += ["", "VALIDATION ISSUES (first 20)", "-" * 50]
        issue_count = 0
        for letter in ALPHABETS:
            vs = validation_summaries.get(letter)
            if vs and not vs.is_clean:
                for issue in vs.issues[:5]:
                    lines.append(f"  [{letter}] {issue}")
                    issue_count += 1
                    if issue_count >= 20:
                        break
            if issue_count >= 20:
                lines.append("  … (see JSON report for full list)")
                break

        if issue_count == 0:
            lines.append("  No validation issues found. ✅")

        lines += ["", "=" * 90, "END OF REPORT", "=" * 90]

        out_path = self.reports_dir / REPORT_FILENAME
        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Text report saved → %s", out_path)
        print(f"\n📄  Text report saved → {out_path}")

    # ── JSON report ────────────────────────────────────────────
    def _save_json_report(
        self,
        full_stats: FullDatasetStats,
        validation_summaries: Dict[str, AlphabetValidationSummary],
    ) -> None:
        data = {
            "generated_at": datetime.now().isoformat(),
            "overall": {
                "total_alphabets_with_data": full_stats.total_alphabets_with_data,
                "total_images": full_stats.total_images,
                "total_valid_images": full_stats.total_valid_images,
                "total_corrupted": full_stats.total_corrupted,
                "total_duplicates": full_stats.total_duplicates,
                "alphabets_empty": full_stats.alphabets_empty,
                "alphabets_critical": full_stats.alphabets_critical,
                "alphabets_warning": full_stats.alphabets_warning,
                "alphabets_good": full_stats.alphabets_good,
                "alphabets_excellent": full_stats.alphabets_excellent,
            },
            "per_alphabet": {},
        }

        for letter in ALPHABETS:
            s = full_stats.per_alphabet.get(letter)
            vs = validation_summaries.get(letter)
            if s is None:
                continue
            data["per_alphabet"][letter] = {
                "num_datasets": s.num_datasets,
                "total_images": s.total_images,
                "valid_images": s.valid_images,
                "corrupted_count": s.corrupted_count,
                "duplicate_count": s.duplicate_count,
                "unreadable_count": s.unreadable_count,
                "status": s.status.value,
                "status_label": STATUS_LABEL[s.status],
                "validation_clean": s.validation_clean,
                "images_needed_for_good": s.images_needed_for_good,
                "images_needed_for_excellent": s.images_needed_for_excellent,
                "dataset_folders": s.dataset_folder_names,
                "validation_issues": (vs.issues if vs else []),
            }

        out_path = self.reports_dir / JSON_REPORT_FILENAME
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("JSON report saved → %s", out_path)
        print(f"📄  JSON report saved → {out_path}")
