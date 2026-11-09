"""
generate_banner_gif.py
======================
Generates a cinematic, futuristic 3D animated motion banner for the SignGesture GitHub README.
Features:
  - 3D rotating & morphing hand skeleton using actual 21 MediaPipe landmarks.
  - Glowing nodes, neon bones, and particle waves.
  - Animated light-shimmer over 'SIGNGESTURE AI'.
  - Real-time animated audio frequency visualizer (voice feedback simulation).
  - Telemetry HUD badges & status pulses.
"""

import math
import os
from pathlib import Path
import json
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = ASSETS_DIR / "signgesture_3d_banner.gif"

# MediaPipe Hand Connections (21 landmarks)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm Base
    (0, 17)
]

# Load landmark data for key gestures
data = joblib.load(str(BASE_DIR / "models" / "cache_train_4832.joblib"))
classes = json.load(open(str(BASE_DIR / "models" / "classes.json")))["classes"]
labels = [classes[i] for i in data["y"]]

# Select gestures to morph through: S -> I -> G -> N -> Y (Iconic)
GESTURE_KEYS = ["S", "I", "G", "N", "Y"]
key_landmarks = []
for k in GESTURE_KEYS:
    idx = labels.index(k)
    pts = data["X"][idx][:63].reshape(21, 3).copy()
    # Normalize center to palm base (landmark 0) and rescale
    pts -= pts[0]
    # Invert Y so up is positive
    pts[:, 1] = -pts[:, 1]
    # Normalize scale
    span = np.max(np.abs(pts))
    if span > 0:
        pts /= span
    key_landmarks.append(pts)

# Dimensions
WIDTH = 860
HEIGHT = 280
NUM_FRAMES = 50
FPS = 25

# Load Windows TrueType fonts
def get_font(size: int, bold: bool = False):
    font_name = "segoeuib.ttf" if bold else "segoeui.ttf"
    font_path = f"C:/Windows/Fonts/{font_name}"
    if not os.path.exists(font_path):
        font_path = f"C:/Windows/Fonts/{'arialbd.ttf' if bold else 'arial.ttf'}"
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()

font_title = get_font(28, bold=True)
font_badge = get_font(10, bold=True)
font_sub = get_font(13, bold=True)
font_detail = get_font(11, bold=False)
font_tag = get_font(10, bold=True)

# Generate frames
frames = []

for frame_idx in range(NUM_FRAMES):
    # Normalized time 0.0 -> 1.0
    t = frame_idx / NUM_FRAMES
    angle_rad = t * 2 * math.pi

    # Create base dark slate background (#0A0F1D)
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(10, 15, 29))
    draw = ImageDraw.Draw(img, "RGBA")

    # 1. Subtle Perspective Grid Lines in background
    grid_y_start = 40
    for gy in range(grid_y_start, HEIGHT - 20, 24):
        alpha = int(12 + 10 * math.sin(angle_rad + gy * 0.05))
        draw.line([(20, gy), (WIDTH - 20, gy)], fill=(30, 41, 59, alpha), width=1)
    for gx in range(40, WIDTH - 20, 40):
        alpha = int(10 + 8 * math.cos(angle_rad + gx * 0.02))
        draw.line([(gx, 40), (gx, HEIGHT - 20)], fill=(30, 41, 59, alpha), width=1)

    # 2. Glowing Outer Tech Border with Corner Brackets
    border_color = (20, 184, 166, 70)  # Teal
    draw.rounded_rectangle([(10, 10), (WIDTH - 10, HEIGHT - 10)], radius=12, outline=border_color, width=1)
    
    # Glowing Corner Accents
    corner_len = 18
    corner_col = (56, 189, 248, 220)  # Bright Cyan
    # Top-Left
    draw.line([(10, 10 + corner_len), (10, 10), (10 + corner_len, 10)], fill=corner_col, width=2)
    # Top-Right
    draw.line([(WIDTH - 10 - corner_len, 10), (WIDTH - 10, 10), (WIDTH - 10, 10 + corner_len)], fill=corner_col, width=2)
    # Bottom-Left
    draw.line([(10, HEIGHT - 10 - corner_len), (10, HEIGHT - 10), (10 + corner_len, HEIGHT - 10)], fill=corner_col, width=2)
    # Bottom-Right
    draw.line([(WIDTH - 10 - corner_len, HEIGHT - 10), (WIDTH - 10, HEIGHT - 10), (WIDTH - 10, HEIGHT - 10 - corner_len)], fill=corner_col, width=2)

    # 3. 3D Hand Skeletal Tracker (Center-Left: X=145, Y=145)
    hand_cx = 150
    hand_cy = 145
    hand_scale = 95

    # Determine morphing between gestures
    num_gestures = len(GESTURE_KEYS)
    cycle_pos = (t * num_gestures) % num_gestures
    idx_from = int(cycle_pos)
    idx_to = (idx_from + 1) % num_gestures
    interp = cycle_pos - idx_from
    # Smooth step interpolation
    smooth_interp = 0.5 - 0.5 * math.cos(interp * math.pi)

    pts_from = key_landmarks[idx_from]
    pts_to = key_landmarks[idx_to]
    current_hand = (1.0 - smooth_interp) * pts_from + smooth_interp * pts_to

    # 3D Rotation (Yaw rocking ±22°, Pitch ±10°)
    yaw = 0.38 * math.sin(angle_rad)
    pitch = 0.16 * math.cos(angle_rad * 0.7)

    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)

    projected_pts = []
    for pt in current_hand:
        x, y, z = pt[0], pt[1], pt[2]

        # Rotate Yaw (around Y axis)
        x_rot = x * cos_y + z * sin_y
        z_temp = -x * sin_y + z * cos_y

        # Rotate Pitch (around X axis)
        y_rot = y * cos_p - z_temp * sin_p
        z_rot = y * sin_p + z_temp * cos_p

        # Perspective projection
        fov = 3.0
        depth_factor = fov / (fov + z_rot * 0.8)
        screen_x = hand_cx + x_rot * hand_scale * depth_factor
        screen_y = hand_cy - y_rot * hand_scale * depth_factor  # Invert Y for screen coords
        projected_pts.append((screen_x, screen_y, z_rot))

    # Background Hologram Scanner Circle around Hand
    radar_radius = 115
    radar_angle = angle_rad * 1.5
    draw.ellipse(
        [(hand_cx - radar_radius, hand_cy - radar_radius), (hand_cx + radar_radius, hand_cy + radar_radius)],
        outline=(13, 148, 136, 45),
        width=1
    )
    # Rotating radar sweep line
    rx = hand_cx + radar_radius * math.cos(radar_angle)
    ry = hand_cy + radar_radius * math.sin(radar_angle)
    draw.line([(hand_cx, hand_cy), (rx, ry)], fill=(45, 212, 191, 55), width=1)

    # Depth sort connections for correct 3D occlusion
    sorted_connections = sorted(
        HAND_CONNECTIONS,
        key=lambda pair: (projected_pts[pair[0]][2] + projected_pts[pair[1]][2]) / 2.0
    )

    # Draw 3D bones with glowing neon cyan
    for p1, p2 in sorted_connections:
        x1, y1, z1 = projected_pts[p1]
        x2, y2, z2 = projected_pts[p2]
        avg_z = (z1 + z2) / 2.0
        # Intensity based on depth
        intensity = int(np.clip(180 + avg_z * 80, 100, 255))
        bone_color = (13, 148, 136, intensity)
        glow_color = (56, 189, 248, int(intensity * 0.4))
        
        # Outer glow line
        draw.line([(x1, y1), (x2, y2)], fill=glow_color, width=4)
        # Core sharp line
        draw.line([(x1, y1), (x2, y2)], fill=bone_color, width=2)

    # Draw 3D Landmark Nodes (Joints)
    # Sort nodes by depth
    sorted_nodes = sorted(enumerate(projected_pts), key=lambda item: item[1][2])
    for orig_idx, (nx, ny, nz) in sorted_nodes:
        node_radius = 3.5 if orig_idx in [4, 8, 12, 16, 20] else 2.5
        # Fingertips have special pulse
        if orig_idx in [4, 8, 12, 16, 20]:
            pulse = 1.0 + 0.4 * math.sin(angle_rad * 3 + orig_idx)
            r = node_radius * pulse
            # Outer halo
            draw.ellipse([(nx - r - 2, ny - r - 2), (nx + r + 2, ny + r + 2)], fill=(20, 184, 166, 60))
            # Core fingertip (Bright Electric Amber/Gold)
            draw.ellipse([(nx - r, ny - r), (nx + r, ny + r)], fill=(251, 191, 36, 240))
        elif orig_idx == 0:
            # Palm base (Cyan anchor)
            draw.ellipse([(nx - 4, ny - 4), (nx + 4, ny + 4)], fill=(56, 189, 248, 250))
        else:
            # Intermediate joint (Teal node)
            draw.ellipse([(nx - node_radius, ny - node_radius), (nx + node_radius, ny + node_radius)], fill=(45, 212, 191, 230))

    # Badge below Hand showing recognized gesture
    curr_gesture_char = GESTURE_KEYS[idx_from]
    badge_x = hand_cx - 48
    badge_y = hand_cy + 102
    draw.rounded_rectangle([(badge_x, badge_y), (badge_x + 96, badge_y + 20)], radius=4, fill=(15, 23, 42, 230), outline=(20, 184, 166, 180), width=1)
    draw.text((badge_x + 10, badge_y + 3), f"GESTURE: [ {curr_gesture_char} ]", font=font_tag, fill=(56, 189, 248, 255))

    # 4. Right Side Typography & HUD (Starting at X=305)
    # ── Top Category Pill ──
    pill_x, pill_y = 310, 26
    draw.rounded_rectangle([(pill_x, pill_y), (pill_x + 225, pill_y + 22)], radius=11, fill=(19, 78, 74, 200), outline=(45, 212, 191, 160), width=1)
    # Pulsing Live LED Dot
    dot_pulse = int(180 + 75 * math.sin(angle_rad * 2))
    draw.ellipse([(pill_x + 8, pill_y + 6), (pill_x + 16, pill_y + 14)], fill=(52, 211, 153, dot_pulse))
    draw.text((pill_x + 22, pill_y + 4), "AI CORE v2.0  •  REAL-TIME 3D ASL", font=font_badge, fill=(240, 253, 250, 255))

    # ── Main Luminous Title: SIGNGESTURE AI ──
    title_text = "SIGNGESTURE AI"
    title_x, title_y = 310, 54
    
    # Animated shimmer/light beam passing across the letters
    shimmer_center = 310 + int(t * 360)
    for i, ch in enumerate(title_text):
        char_x = title_x + i * 21
        dist_to_shimmer = abs(char_x - shimmer_center)
        if dist_to_shimmer < 40:
            # Bright highlighted cyan
            col = (255, 255, 255, 255)
        else:
            # Luminous gradient
            col = (241, 245, 249, 255)
        draw.text((char_x, title_y), ch, font=font_title, fill=col)

    # ── Subtitle with Rotating Tech Highlight ──
    sub_messages = [
        "99.51% Precision (A-Z)  •  ~9ms Ultra-Fast Latency",
        "Two-Way Assistive Communication Dashboard",
        "Windows SAPI Continuous Voice Loop  •  Thread-Safe",
        "Full ASL Alphabet (A-Z) + 25 FPS Verified Video Clips"
    ]
    sub_idx = int(t * len(sub_messages)) % len(sub_messages)
    active_sub = sub_messages[sub_idx]
    draw.text((310, 102), active_sub, font=font_sub, fill=(45, 212, 191, 240))

    # ── Secondary Description Line ──
    draw.text((310, 126), "Real-Time 3D Landmark Tracking  ⇄  Speech & Text Bi-Directional Bridge", font=font_detail, fill=(148, 163, 184, 230))

    # ── 5. Real-Time Audio Frequency Equalizer (Simulating Voice Synthesis) ──
    eq_x = 310
    eq_y = 158
    eq_width = 380
    num_bars = 28
    bar_width = 8
    gap = 4

    # Label for Equalizer
    draw.text((eq_x, eq_y), "VOICE SYNTHESIS AUDIO STREAM (SAPI 16kHz):", font=font_tag, fill=(100, 116, 139, 230))

    for b in range(num_bars):
        bx = eq_x + b * (bar_width + gap)
        # Synthesize realistic voice formant heights with animation
        freq = (b / num_bars) * 3.5
        voice_wave = math.sin(angle_rad * 3.0 + freq * 2.0) * math.cos(angle_rad * 1.5 + b * 0.3)
        bar_height = int(np.clip(6 + abs(voice_wave) * 26 + math.sin(b * 0.8) * 8, 4, 38))
        by = eq_y + 18 + (38 - bar_height)

        # Gradient color for bars (Teal -> Bright Cyan -> Gold for peaks)
        if bar_height > 28:
            bar_col = (251, 191, 36, 240)  # Peak Amber
        elif bar_height > 18:
            bar_col = (56, 189, 248, 220)  # Cyan
        else:
            bar_col = (13, 148, 136, 190)  # Teal
            
        draw.rounded_rectangle([(bx, by), (bx + bar_width, eq_y + 56)], radius=2, fill=bar_col)

    # ── 6. Live Telemetry HUD Status Chips (Bottom Row) ──
    chip_y = 236
    chips = [
        ("ACCURACY: 99.51%", (16, 185, 129)),  # Green
        ("LATENCY: 9.1ms", (56, 189, 248)),    # Cyan
        ("AUDIO: SAPI COM", (14, 165, 233)),   # Sky
        ("TESTS: 17/17 PASSED", (245, 158, 11))# Amber
    ]
    curr_cx = 310
    for label, chip_col in chips:
        chip_w = len(label) * 7 + 16
        draw.rounded_rectangle([(curr_cx, chip_y), (curr_cx + chip_w, chip_y + 20)], radius=4, fill=(15, 23, 42, 220), outline=chip_col, width=1)
        draw.text((curr_cx + 8, chip_y + 4), label, font=font_tag, fill=chip_col)
        curr_cx += chip_w + 10

    frames.append(img)

# Save as optimized animated GIF
print(f"Saving {len(frames)} frames to {OUTPUT_PATH}...")
frames[0].save(
    OUTPUT_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=int(1000 / FPS),
    loop=0,
    optimize=True
)
file_size_kb = OUTPUT_PATH.stat().st_size / 1024
print(f"Generated {OUTPUT_PATH} successfully! Size: {file_size_kb:.1f} KB")
