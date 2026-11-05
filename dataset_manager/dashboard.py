"""
dashboard.py
============
Terminal + OpenCV-based visual dashboard for the dataset management system.

* Prints a rich table to the terminal.
* Opens an OpenCV window with a visual grid (status bars, image preview).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import (
    ALPHABETS,
    PREVIEW_GRID_COLS,
    PREVIEW_MAX_IMAGES,
    PREVIEW_THUMB_SIZE,
)
from .scanner import AlphabetRecord
from .stats import (
    FullDatasetStats,
    AlphabetStats,
    DataStatus,
    STATUS_EMOJI,
    STATUS_LABEL,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Colour palette  (BGR for OpenCV)
# ─────────────────────────────────────────────────────────────
COLORS = {
    DataStatus.EMPTY:     (180, 180, 180),   # light grey
    DataStatus.CRITICAL:  (0,   60,  220),   # red
    DataStatus.WARNING:   (0,  180,  255),   # yellow/orange
    DataStatus.GOOD:      (50,  200,   50),  # green
    DataStatus.EXCELLENT: (220, 100,   30),  # blue
}
BG_DARK   = (30,  30,  30)
BG_CARD   = (50,  50,  50)
WHITE     = (255, 255, 255)
GREY_TEXT = (180, 180, 180)
CYAN      = (255, 200,   0)


# ─────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────
class DatasetDashboard:
    """
    Renders a terminal table and an interactive OpenCV preview window.

    Usage
    -----
    >>> dashboard = DatasetDashboard()
    >>> dashboard.show_terminal(full_stats)
    >>> dashboard.show_opencv(full_stats, scanned_data)   # press Q to close
    """

    # ── Terminal dashboard ─────────────────────────────────────
    def show_terminal(self, full_stats: FullDatasetStats) -> None:
        """Print a nicely formatted statistics table to stdout."""
        self._print_header()
        self._print_table(full_stats)
        self._print_summary(full_stats)

    def _print_header(self) -> None:
        print()
        print("=" * 100)
        print(" 🤟  SIGN LANGUAGE DATASET MANAGEMENT SYSTEM  —  Dashboard")
        print("=" * 100)
        print()

    def _print_table(self, full_stats: FullDatasetStats) -> None:
        # Header row
        col_w = [4, 12, 12, 12, 10, 10, 10, 36, 22]
        header = (
            f"{'#':<{col_w[0]}}"
            f"{'Alphabet':<{col_w[1]}}"
            f"{'Datasets':>{col_w[2]}}"
            f"{'Total Imgs':>{col_w[3]}}"
            f"{'Valid':>{col_w[4]}}"
            f"{'Bad':>{col_w[5]}}"
            f"{'Dups':>{col_w[6]}}"
            f"{'Status':<{col_w[7]}}"
            f"{'Validation':<{col_w[8]}}"
        )
        sep = "-" * sum(col_w)
        print(header)
        print(sep)

        for idx, letter in enumerate(ALPHABETS, 1):
            s: AlphabetStats = full_stats.per_alphabet.get(letter)
            if s is None:
                continue

            emoji = STATUS_EMOJI[s.status]
            status_txt = f"{emoji}  {s.status.value:<10} {STATUS_LABEL[s.status]}"
            val_icon   = "✅ Clean" if s.validation_clean else f"⚠️  {s.corrupted_count} issue(s)"
            bad_count  = s.corrupted_count + s.unreadable_count

            row = (
                f"{idx:<{col_w[0]}}"
                f"{letter:<{col_w[1]}}"
                f"{s.num_datasets:>{col_w[2]}}"
                f"{s.total_images:>{col_w[3]}}"
                f"{s.valid_images:>{col_w[4]}}"
                f"{bad_count:>{col_w[5]}}"
                f"{s.duplicate_count:>{col_w[6]}}"
                f"{status_txt:<{col_w[7]}}"
                f"{val_icon:<{col_w[8]}}"
            )
            print(row)

        print(sep)

    def _print_summary(self, full_stats: FullDatasetStats) -> None:
        print()
        print("📊  DATASET SUMMARY")
        print("-" * 60)
        print(f"  Total alphabets with data : {full_stats.total_alphabets_with_data} / 26")
        print(f"  Total images (all)        : {full_stats.total_images:,}")
        print(f"  Total valid images        : {full_stats.total_valid_images:,}")
        print(f"  Total corrupted           : {full_stats.total_corrupted:,}")
        print(f"  Total duplicates          : {full_stats.total_duplicates:,}")
        print()
        print(f"  ⬜  Empty      : {', '.join(full_stats.alphabets_empty) or 'None'}")
        print(f"  🔴  Critical   : {', '.join(full_stats.alphabets_critical) or 'None'}")
        print(f"  🟡  Warning    : {', '.join(full_stats.alphabets_warning) or 'None'}")
        print(f"  🟢  Good       : {', '.join(full_stats.alphabets_good) or 'None'}")
        print(f"  🔵  Excellent  : {', '.join(full_stats.alphabets_excellent) or 'None'}")
        print()
        print("=" * 100)
        print()

    # ── OpenCV visual dashboard ────────────────────────────────
    def show_opencv(
        self,
        full_stats: FullDatasetStats,
        scanned_data: Dict[str, AlphabetRecord],
        selected_letter: Optional[str] = None,
    ) -> None:
        """
        Open an interactive OpenCV window.
        Click an alphabet button to preview its images.
        Press Q or ESC to quit.
        """
        CARD_W, CARD_H = 160, 90
        COLS = 6
        PAD  = 10
        rows = -(-len(ALPHABETS) // COLS)   # ceiling division
        grid_w = COLS * (CARD_W + PAD) + PAD
        grid_h = rows * (CARD_H + PAD) + PAD

        HEADER_H = 80
        FOOTER_H = 50
        INFO_W   = 420

        total_w = grid_w + INFO_W + PAD * 2
        total_h = HEADER_H + grid_h + FOOTER_H

        selected = selected_letter or ALPHABETS[0]
        card_rects: List = []    # (letter, x1, y1, x2, y2)

        def build_frame() -> np.ndarray:
            frame = np.full((total_h, total_w, 3), BG_DARK, dtype=np.uint8)
            nonlocal card_rects
            card_rects = []

            # ── Header ──────────────────────────────────────────
            cv2.rectangle(frame, (0, 0), (total_w, HEADER_H), (40, 40, 60), -1)
            cv2.putText(frame, "Sign Language Dataset Manager",
                        (PAD, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, CYAN, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Total: {full_stats.total_images:,} images | "
                                f"{full_stats.total_alphabets_with_data}/26 alphabets populated",
                        (PAD, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY_TEXT, 1, cv2.LINE_AA)

            # ── Alphabet grid ────────────────────────────────────
            for idx, letter in enumerate(ALPHABETS):
                col = idx % COLS
                row = idx // COLS
                x1 = PAD + col * (CARD_W + PAD)
                y1 = HEADER_H + PAD + row * (CARD_H + PAD)
                x2 = x1 + CARD_W
                y2 = y1 + CARD_H
                card_rects.append((letter, x1, y1, x2, y2))

                s = full_stats.per_alphabet.get(letter)
                status = s.status if s else DataStatus.EMPTY
                color  = COLORS[status]

                # card background
                bg = (70, 70, 100) if letter == selected else BG_CARD
                cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
                # status bar on top
                cv2.rectangle(frame, (x1, y1), (x2, y1 + 5), color, -1)
                # border highlight if selected
                if letter == selected:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # letter label
                cv2.putText(frame, letter, (x1 + 10, y1 + 38),
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, WHITE, 2, cv2.LINE_AA)

                # image count
                count_str = f"{s.valid_images if s else 0} imgs"
                cv2.putText(frame, count_str, (x1 + 8, y1 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY_TEXT, 1, cv2.LINE_AA)

                # status emoji text
                cv2.putText(frame, STATUS_EMOJI.get(status, ""), (x1 + 8, y1 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

            # ── Info panel ───────────────────────────────────────
            ix = grid_w + PAD * 2
            iy = HEADER_H + PAD
            s = full_stats.per_alphabet.get(selected)
            color = COLORS[s.status if s else DataStatus.EMPTY]

            cv2.rectangle(frame, (ix, iy), (ix + INFO_W - PAD, total_h - FOOTER_H - PAD), BG_CARD, -1)
            cv2.rectangle(frame, (ix, iy), (ix + INFO_W - PAD, iy + 6), color, -1)

            lines = [
                (f"  Alphabet : {selected}", 0.7, WHITE),
                ("", 0.5, WHITE),
                (f"  Datasets : {s.num_datasets if s else 0}", 0.55, GREY_TEXT),
                (f"  Total imgs: {s.total_images if s else 0:,}", 0.55, GREY_TEXT),
                (f"  Valid imgs: {s.valid_images if s else 0:,}", 0.55, GREY_TEXT),
                (f"  Corrupted : {s.corrupted_count if s else 0}", 0.55,
                 (0, 80, 220) if (s and s.corrupted_count > 0) else GREY_TEXT),
                (f"  Duplicates: {s.duplicate_count if s else 0}", 0.55,
                 (0, 180, 220) if (s and s.duplicate_count > 0) else GREY_TEXT),
                ("", 0.5, WHITE),
                (f"  Status : {s.status.value if s else 'N/A'}", 0.6, color),
                (f"  {STATUS_LABEL.get(s.status, '') if s else ''}", 0.45, color),
                ("", 0.5, WHITE),
                (f"  Need {s.images_needed_for_good if s else 200} more → GOOD", 0.45, GREY_TEXT),
                (f"  Need {s.images_needed_for_excellent if s else 500} more → EXCELLENT", 0.45, GREY_TEXT),
            ]
            if s:
                lines.append(("", 0.5, WHITE))
                lines.append(("  Sub-datasets:", 0.45, GREY_TEXT))
                for fn in s.dataset_folder_names[:6]:
                    lines.append((f"    • {fn}", 0.42, GREY_TEXT))
                if len(s.dataset_folder_names) > 6:
                    lines.append((f"    … +{len(s.dataset_folder_names)-6} more", 0.42, GREY_TEXT))

            ty = iy + 40
            for text, scale, col in lines:
                cv2.putText(frame, text, (ix, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
                ty += int(scale * 42 + 6)

            # ── Image preview thumbnails ──────────────────────────
            alpha_record = scanned_data.get(selected)
            if alpha_record and alpha_record.all_images:
                valid_imgs = [r for r in alpha_record.all_images
                              if r.is_readable and not r.is_corrupted][:PREVIEW_MAX_IMAGES]
                thumb_y = ty + 10
                thumb_x = ix
                TW, TH = PREVIEW_THUMB_SIZE
                for i, rec in enumerate(valid_imgs):
                    img = cv2.imread(str(rec.path))
                    if img is None:
                        continue
                    thumb = cv2.resize(img, (TW, TH))
                    px = thumb_x + (i % PREVIEW_GRID_COLS) * (TW + 4)
                    py = thumb_y + (i // PREVIEW_GRID_COLS) * (TH + 4)
                    if py + TH < total_h - FOOTER_H:
                        frame[py:py + TH, px:px + TW] = thumb

            # ── Footer ──────────────────────────────────────────
            fy = total_h - FOOTER_H
            cv2.rectangle(frame, (0, fy), (total_w, total_h), (40, 40, 60), -1)
            cv2.putText(frame, "Click alphabet card to preview | Q / ESC to quit",
                        (PAD, fy + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY_TEXT, 1, cv2.LINE_AA)

            return frame

        cv2.namedWindow("Dataset Dashboard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dataset Dashboard", total_w, total_h)

        def on_mouse(event, x, y, flags, param):
            nonlocal selected
            if event == cv2.EVENT_LBUTTONDOWN:
                for (letter, x1, y1, x2, y2) in card_rects:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        selected = letter
                        break

        cv2.setMouseCallback("Dataset Dashboard", on_mouse)

        while True:
            frame = build_frame()
            cv2.imshow("Dataset Dashboard", frame)
            key = cv2.waitKey(100) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

        cv2.destroyAllWindows()
        logger.info("Dashboard closed.")
