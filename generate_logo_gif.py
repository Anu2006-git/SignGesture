"""
generate_logo_gif.py
====================
Generates a dedicated 120x120 smooth looping animated 3D neon hand gesture logo badge.
"""

import math
from pathlib import Path
import json
import joblib
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = ASSETS_DIR / "hand_gesture_3d_logo.gif"

# MediaPipe Hand Connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm Base
]

# Load 'Y' / iconic ASL hand landmarks
data = joblib.load(str(BASE_DIR / "models" / "cache_train_4832.joblib"))
classes = json.load(open(str(BASE_DIR / "models" / "classes.json")))["classes"]
labels = [classes[i] for i in data["y"]]
idx = labels.index("Y")
raw_pts = data["X"][idx][:63].reshape(21, 3).copy()

# Center and normalize
raw_pts -= raw_pts[0]
raw_pts[:, 1] = -raw_pts[:, 1]  # Invert Y so up is positive
span = np.max(np.abs(raw_pts))
if span > 0:
    raw_pts /= span

SIZE = 140
NUM_FRAMES = 40
FPS = 25

frames = []

for frame_idx in range(NUM_FRAMES):
    t = frame_idx / NUM_FRAMES
    angle_rad = t * 2 * math.pi

    img = Image.new("RGBA", (SIZE, SIZE), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    cx, cy = SIZE // 2, SIZE // 2 + 6
    scale = 48

    # 1. Pulsing Energy Ring in Background
    ring_r = 54 + 3 * math.sin(angle_rad)
    ring_alpha = int(90 + 50 * math.sin(angle_rad * 2))
    draw.ellipse(
        [(cx - ring_r, cy - ring_r), (cx + ring_r, cy + ring_r)],
        outline=(20, 184, 166, ring_alpha),
        width=2
    )

    # Orbiting energy particle
    orb_angle = angle_rad * 2
    orb_x = cx + ring_r * math.cos(orb_angle)
    orb_y = cy + ring_r * math.sin(orb_angle)
    draw.ellipse([(orb_x - 3, orb_y - 3), (orb_x + 3, orb_y + 3)], fill=(56, 189, 248, 240))

    # 2. 3D Rotation of the Hand (Yaw rocking ±28°, Pitch ±12°)
    yaw = 0.50 * math.sin(angle_rad)
    pitch = 0.20 * math.cos(angle_rad)

    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)

    projected_pts = []
    for pt in raw_pts:
        x, y, z = pt[0], pt[1], pt[2]
        # Yaw
        x_rot = x * cos_y + z * sin_y
        z_temp = -x * sin_y + z * cos_y
        # Pitch
        y_rot = y * cos_p - z_temp * sin_p
        z_rot = y * sin_p + z_temp * cos_p

        fov = 3.2
        depth_factor = fov / (fov + z_rot * 0.9)
        screen_x = cx + x_rot * scale * depth_factor
        screen_y = cy - y_rot * scale * depth_factor
        projected_pts.append((screen_x, screen_y, z_rot))

    # 3. Draw 3D Hand Bones (Neon Cyan & Teal Glow)
    sorted_connections = sorted(
        HAND_CONNECTIONS,
        key=lambda pair: (projected_pts[pair[0]][2] + projected_pts[pair[1]][2]) / 2.0
    )

    for p1, p2 in sorted_connections:
        x1, y1, z1 = projected_pts[p1]
        x2, y2, z2 = projected_pts[p2]
        avg_z = (z1 + z2) / 2.0
        intensity = int(np.clip(180 + avg_z * 70, 120, 255))
        # Outer glow
        draw.line([(x1, y1), (x2, y2)], fill=(56, 189, 248, int(intensity * 0.4)), width=5)
        # Core bone
        draw.line([(x1, y1), (x2, y2)], fill=(13, 148, 136, intensity), width=2)

    # 4. Draw Joint Nodes
    for idx, (nx, ny, nz) in enumerate(projected_pts):
        if idx in [4, 8, 12, 16, 20]:
            # Fingertips
            r = 3.0 + 0.8 * math.sin(angle_rad * 3 + idx)
            draw.ellipse([(nx - r - 2, ny - r - 2), (nx + r + 2, ny + r + 2)], fill=(20, 184, 166, 80))
            draw.ellipse([(nx - r, ny - r), (nx + r, ny + r)], fill=(251, 191, 36, 250))
        elif idx == 0:
            draw.ellipse([(nx - 3.5, ny - 3.5), (nx + 3.5, ny + 3.5)], fill=(56, 189, 248, 255))
        else:
            draw.ellipse([(nx - 2, ny - 2), (nx + 2, ny + 2)], fill=(45, 212, 191, 220))

    frames.append(img)

print(f"Saving {len(frames)} frames to {OUTPUT_PATH}...")
frames[0].save(
    OUTPUT_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=int(1000 / FPS),
    loop=0,
    disposal=2
)
file_size_kb = OUTPUT_PATH.stat().st_size / 1024
print(f"Generated {OUTPUT_PATH} successfully! Size: {file_size_kb:.1f} KB")
