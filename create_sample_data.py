"""
create_sample_data.py
=====================
Creates dummy sample images for A, B, C to verify the dataset system works.
Run ONCE after setup to confirm everything is working, then replace with real images.

    python create_sample_data.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: OpenCV not installed. Run:  pip install opencv-python numpy")
    sys.exit(1)

BASE = Path(__file__).parent / "dataset"

SAMPLES = {
    "A": {"dataset_1": 5, "dataset_2": 4},
    "B": {"dataset_1": 3, "dataset_2": 3},
    "C": {"dataset_1": 2, "dataset_2": 2, "dataset_3": 1},
}

COLORS = {
    "A": (200, 80,  80),
    "B": (80,  200, 80),
    "C": (80,  80,  200),
}

total_created = 0
for letter, datasets in SAMPLES.items():
    for ds_name, count in datasets.items():
        folder = BASE / letter / ds_name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            img[:] = COLORS[letter]
            cv2.putText(img, letter,
                        (40, 170), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 255, 255), 10)
            cv2.putText(img, f"{ds_name} | img {i + 1}",
                        (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
            out = folder / f"img_{i + 1:03d}.jpg"
            cv2.imwrite(str(out), img)
            total_created += 1

print(f"✅  Created {total_created} sample images in dataset/ for letters A, B, C.")
print("    Now run:  python manage_dataset.py --visual")
