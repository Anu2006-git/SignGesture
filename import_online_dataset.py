"""
import_online_dataset.py
========================
Helper tool to automatically import, organize, and integrate any
open-source A-Z Sign Language dataset into your project directory.

Supports:
  - Kaggle ASL Alphabet (extracted folders with 'A', 'B', ... 'Z' or 'asl_alphabet_train')
  - Roboflow ASL exports
  - Any folder containing subfolders for each letter A-Z

Usage:
  python import_online_dataset.py --source "C:/path/to/downloaded/asl_alphabet_train" --dataset-name "dataset_online"
  python import_online_dataset.py --source "C:/Downloads/asl_alphabet" --max-per-class 500
"""

import argparse
from pathlib import Path
import shutil
import sys

# Project paths
BASE_DIR = Path(__file__).resolve().parent
TARGET_DATASET_DIR = BASE_DIR / "dataset"
ALPHABETS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def import_dataset(source_path: Path, dataset_name: str = "dataset_online", max_per_class: int = None):
    source_path = Path(source_path)
    if not source_path.exists():
        print(f"❌ Error: Source directory does not exist: {source_path}")
        return

    print("\n" + "=" * 65)
    print(" 📥 IMPORTING OPEN-SOURCE A-Z DATASET")
    print("=" * 65)
    print(f"  Source Path      : {source_path}")
    print(f"  Target Subfolder : {dataset_name}")
    if max_per_class:
        print(f"  Max per Alphabet : {max_per_class} images")
    print("=" * 65 + "\n")

    total_imported = 0
    imported_classes = 0

    for letter in ALPHABETS:
        # Search for folder matching letter (case-insensitive)
        candidate_folders = [
            d for d in source_path.rglob("*")
            if d.is_dir() and d.name.strip().upper() == letter
        ]

        if not candidate_folders:
            continue

        source_letter_dir = candidate_folders[0]
        image_files = sorted([
            f for f in source_letter_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
        ])

        if not image_files:
            continue

        if max_per_class is not None:
            image_files = image_files[:max_per_class]

        # Target directory: dataset/<LETTER>/<dataset_name>/
        target_dir = TARGET_DATASET_DIR / letter / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)

        copied_count = 0
        for i, img_file in enumerate(image_files, 1):
            target_file = target_dir / f"{letter}_online_{i:05d}{img_file.suffix.lower()}"
            shutil.copy2(img_file, target_file)
            copied_count += 1

        print(f"  [{letter}] Copied {copied_count:>4} images → dataset/{letter}/{dataset_name}/")
        total_imported += copied_count
        imported_classes += 1

    print("\n" + "=" * 65)
    print(f" ✅ Import complete! Total: {total_imported:,} images across {imported_classes}/26 alphabets.")
    print(" Next steps:")
    print("   1. Verify counts:   python manage_dataset.py")
    print("   2. Retrain model:   python train_model.py")
    print("=" * 65 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Import Open-Source Sign Language Datasets")
    p.add_argument("--source", type=str, required=True, help="Path to extracted folder containing A-Z subfolders")
    p.add_argument("--dataset-name", type=str, default="dataset_online", help="Subfolder name to assign (default: dataset_online)")
    p.add_argument("--max-per-class", type=int, default=None, help="Optional limit of images to copy per letter (e.g. 300)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import_dataset(
        source_path=Path(args.source),
        dataset_name=args.dataset_name,
        max_per_class=args.max_per_class,
    )
