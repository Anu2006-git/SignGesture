"""
manage_dataset.py
=================
Main entry point for the dataset management system.

Run this script any time you:
  - Add new images to any dataset folder.
  - Add a new dataset subfolder (e.g. dataset_4).
  - Want a health check on your dataset.

Usage
-----
    python manage_dataset.py                  # terminal dashboard only
    python manage_dataset.py --visual         # + opens OpenCV window
    python manage_dataset.py --report         # generate reports only, no UI
    python manage_dataset.py --letter A       # preview a specific letter first
    python manage_dataset.py --no-duplicates  # skip duplicate detection (faster)
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Make sure the project root is on the path ──────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dataset_manager.scanner   import DatasetScanner
from dataset_manager.validator import ImageValidator
from dataset_manager.stats     import DatasetStats
from dataset_manager.dashboard import DatasetDashboard
from dataset_manager.reporter  import DatasetReporter
from dataset_manager import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sign Language Dataset Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--visual", action="store_true",
        help="Open the interactive OpenCV dashboard window.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate text and JSON reports and exit (no interactive UI).",
    )
    parser.add_argument(
        "--letter", type=str, default=None,
        help="Pre-select an alphabet letter in the visual dashboard (e.g. --letter B).",
    )
    parser.add_argument(
        "--no-duplicates", action="store_true",
        help="Skip duplicate detection for faster scans (large datasets).",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default=None,
        help="Override the dataset directory path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply runtime overrides
    if args.no_duplicates:
        config.ENABLE_DUPLICATE_DETECTION = False

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else config.DATASET_DIR

    print()
    print("🔍  Scanning dataset directory ...")
    print(f"    Path: {dataset_dir}")
    print()

    # ── Step 1: Scan ───────────────────────────────────────────
    scanner     = DatasetScanner(dataset_dir=dataset_dir)
    scanned     = scanner.scan()

    # ── Step 2: Validate ───────────────────────────────────────
    print("🔬  Validating images ...")
    validator   = ImageValidator()
    validations = validator.validate_all(scanned)

    # ── Step 3: Compute statistics ─────────────────────────────
    stats_builder = DatasetStats()
    full_stats    = stats_builder.compute(scanned, validations)

    # ── Step 4: Terminal dashboard ─────────────────────────────
    dashboard = DatasetDashboard()
    dashboard.show_terminal(full_stats)

    # ── Step 5: Reports ────────────────────────────────────────
    reporter = DatasetReporter()
    reporter.generate(full_stats, validations)

    # ── Step 6: OpenCV visual (optional) ──────────────────────
    if args.visual and not args.report:
        selected = args.letter.upper() if args.letter else None
        print("\n🖥️  Opening visual dashboard (press Q or ESC to close) ...")
        dashboard.show_opencv(full_stats, scanned, selected_letter=selected)

    print("\n✅  Dataset management run complete.\n")


if __name__ == "__main__":
    main()
