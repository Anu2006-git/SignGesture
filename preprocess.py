"""
preprocess.py
=============
Main entry point for the Sign Language image preprocessing system.

This script is completely separate from:
  • manage_dataset.py   (dataset scanning and validation)
  • model training code (not implemented yet)

It reads the ORIGINAL dataset (read-only), processes images, and
writes all results to  processed_dataset/  (separate directory).

Usage
-----
    # Process everything with landmark extraction
    python preprocess.py

    # Process without landmark extraction (faster)
    python preprocess.py --no-landmarks

    # Process specific letters only
    python preprocess.py --letters A B C

    # Force full reprocess (ignore cache)
    python preprocess.py --force

    # Process + immediately build train/val/test .npz arrays
    python preprocess.py --build-arrays

    # Disable augmentation (useful for validation sets)
    python preprocess.py --no-augmentation

    # Show what would be processed without doing anything
    python preprocess.py --dry-run

    # Clear cache and reprocess everything from scratch
    python preprocess.py --clear-cache
"""

import argparse
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Project root on path ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.pipeline      import PreprocessingPipeline
from preprocessing.cache_manager import CacheManager
from preprocessing               import config

# ── Logging setup ──────────────────────────────────────────────
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.FileHandler(
            config.LOGS_DIR / "preprocessing.log", encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sign Language Image Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--letters", nargs="+", metavar="LETTER",
        help="Process only these letters, e.g.  --letters A B C",
    )
    p.add_argument(
        "--no-landmarks", action="store_true",
        help="Skip MediaPipe landmark extraction.",
    )
    p.add_argument(
        "--no-augmentation", action="store_true",
        help="Disable data augmentation.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Ignore incremental cache; reprocess every image.",
    )
    p.add_argument(
        "--clear-cache", action="store_true",
        help="Delete the incremental cache file and exit. "
             "Next run will reprocess everything.",
    )
    p.add_argument(
        "--build-arrays", action="store_true",
        help="After preprocessing, build combined .npz train/val/test arrays.",
    )
    p.add_argument(
        "--build-arrays-only", action="store_true",
        help="Skip preprocessing; only build .npz arrays from existing processed files.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Count images that would be processed without doing anything.",
    )
    p.add_argument(
        "--dataset-dir", type=str, default=None,
        help="Override default dataset directory path.",
    )
    return p.parse_args()


def dry_run(dataset_dir: Path, letters: list) -> None:
    SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    cache = CacheManager()
    cache.load()
    print("\n🔍  DRY RUN — counting images that would be processed:\n")
    total_new   = 0
    total_cached = 0
    for letter in letters:
        alpha_dir = dataset_dir / letter
        if not alpha_dir.exists():
            continue
        paths = [p for p in alpha_dir.rglob("*")
                 if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
        new    = sum(1 for p in paths if not cache.is_processed(p))
        cached = len(paths) - new
        if paths:
            print(f"  [{letter}]  {len(paths):>5} total  |  {new:>5} new  |  {cached:>5} cached")
        total_new    += new
        total_cached += cached
    print(f"\n  Total new     : {total_new:,}")
    print(f"  Total cached  : {total_cached:,}")
    print(f"  Would create  : {total_new * (1 + config.AUGMENT_COPIES):,} processed images\n")


def main() -> None:
    args = parse_args()

    # ── Clear cache ────────────────────────────────────────────
    if args.clear_cache:
        CacheManager().clear()
        print("✅  Cache cleared. Next run will reprocess all images.")
        return

    # ── Dataset directory ──────────────────────────────────────
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else config.DATASET_DIR

    # ── Letters ────────────────────────────────────────────────
    letters = (
        [l.upper() for l in args.letters]
        if args.letters else
        [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    )

    # ── Dry run ────────────────────────────────────────────────
    if args.dry_run:
        dry_run(dataset_dir, letters)
        return

    # ── Build arrays only ──────────────────────────────────────
    if args.build_arrays_only:
        from preprocessing.array_builder import ArrayBuilder
        print("\n📦  Building NumPy arrays …")
        ArrayBuilder().build()
        return

    # ── Apply CLI overrides to config ──────────────────────────
    if args.force:
        config.INCREMENTAL_MODE = False
    if args.no_augmentation:
        config.AUGMENTATION_ENABLED = False
    if args.no_landmarks:
        config.LANDMARK_ENABLED = False

    # ── Run pipeline ───────────────────────────────────────────
    pipeline = PreprocessingPipeline(
        dataset_dir=dataset_dir,
        augmentation=config.AUGMENTATION_ENABLED,
        landmarks=config.LANDMARK_ENABLED,
        incremental=config.INCREMENTAL_MODE,
    )
    pipeline.run(letters=letters, build_arrays=args.build_arrays)


if __name__ == "__main__":
    main()
