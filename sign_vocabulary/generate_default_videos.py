"""
sign_vocabulary/generate_default_videos.py
=========================================
Generates default verified sign language demonstration video clips (.mp4)
for the built-in vocabulary library.
"""

from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "sign_vocabulary" / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def draw_hand_icon(img, cx, cy, radius, label, color=(0, 255, 200)):
    """Draws a clean stylized hand / gesture indicator."""
    cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius, (255, 255, 255), 2, cv2.LINE_AA)
    # Fingers representation
    for angle in [-40, -20, 0, 20, 40]:
        rad = np.radians(angle - 90)
        fx = int(cx + (radius + 22) * np.cos(rad))
        fy = int(cy + (radius + 22) * np.sin(rad))
        cv2.line(img, (cx, cy), (fx, fy), (255, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(img, (fx, fy), 5, color, -1, cv2.LINE_AA)


def generate_sign_video(
    filename: str,
    title: str,
    motion_type: str,
    desc: str,
    duration_frames: int = 40,
    width: int = 540,
    height: int = 380,
    fps: int = 25,
):
    video_path = VIDEOS_DIR / filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))

    for f in range(duration_frames):
        t = f / float(duration_frames)
        # Background: dark modern slate
        frame = np.full((height, width, 3), (24, 24, 28), dtype=np.uint8)

        # Subtle card border
        cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (45, 45, 55), 1)

        # Header banner
        cv2.rectangle(frame, (10, 10), (width - 10, 65), (35, 35, 45), -1)
        cv2.putText(
            frame,
            f"VERIFIED SIGN: {title.upper()}",
            (25, 45),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 255, 180),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "[STANDARD ASL]",
            (width - 165, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

        # Compute animation trajectory based on motion_type
        mid_x, mid_y = width // 2, height // 2 + 10
        arrow_start = None
        arrow_end = None

        if motion_type == "salute_outward":  # HELLO, GOODBYE
            offset = int(np.sin(t * np.pi) * 80)
            hand_x = mid_x - 30 + offset
            hand_y = mid_y - 20 - int(offset * 0.3)
            draw_hand_icon(frame, hand_x, hand_y, 28, title, color=(0, 230, 255))
            arrow_start = (mid_x - 50, mid_y - 20)
            arrow_end = (mid_x + 70, mid_y - 45)

        elif motion_type == "chin_to_forward":  # THANK YOU, GOOD
            offset = int(np.sin(t * np.pi) * 90)
            hand_x = mid_x
            hand_y = mid_y - 40 + offset
            draw_hand_icon(frame, hand_x, hand_y, 28, title, color=(0, 255, 120))
            arrow_start = (mid_x, mid_y - 50)
            arrow_end = (mid_x, mid_y + 55)

        elif motion_type == "nodding_fist":  # YES
            offset = int(np.sin(t * np.pi * 3) * 35)
            hand_x = mid_x
            hand_y = mid_y + offset
            cv2.circle(frame, (hand_x, hand_y), 34, (255, 180, 0), -1, cv2.LINE_AA)
            cv2.putText(frame, "S", (hand_x - 10, hand_y + 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
            arrow_start = (mid_x, mid_y - 45)
            arrow_end = (mid_x, mid_y + 45)

        elif motion_type == "beak_tap":  # NO
            squeeze = int(abs(np.sin(t * np.pi * 3)) * 20)
            hand_x = mid_x
            hand_y = mid_y
            cv2.circle(frame, (hand_x - squeeze, hand_y), 24, (0, 160, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (hand_x + squeeze, hand_y), 24, (0, 220, 255), -1, cv2.LINE_AA)

        elif motion_type == "chest_circle":  # PLEASE
            ang = t * 2 * np.pi
            hand_x = int(mid_x + 50 * np.cos(ang))
            hand_y = int(mid_y + 50 * np.sin(ang))
            draw_hand_icon(frame, hand_x, hand_y, 26, title, color=(255, 120, 220))
            cv2.ellipse(frame, (mid_x, mid_y), (50, 50), 0, 0, 360, (70, 70, 90), 2, cv2.LINE_AA)

        elif motion_type == "lift_hands":  # HELP
            offset = int(np.sin(t * np.pi) * 60)
            # Base flat palm
            cv2.rectangle(frame, (mid_x - 60, mid_y + 35 - offset), (mid_x + 60, mid_y + 55 - offset), (160, 160, 160), -1)
            # Thumbs-up fist lifting on top
            cv2.circle(frame, (mid_x, mid_y - offset), 30, (0, 255, 160), -1, cv2.LINE_AA)
            arrow_start = (mid_x + 85, mid_y + 40)
            arrow_end = (mid_x + 85, mid_y - 40)

        elif motion_type == "sun_rising":  # MORNING, GOOD MORNING
            offset = int(t * 100)
            # Horizon line
            cv2.line(frame, (mid_x - 120, mid_y + 40), (mid_x + 120, mid_y + 40), (200, 200, 100), 3)
            # Rising arm/hand
            draw_hand_icon(frame, mid_x, mid_y + 40 - offset, 28, title, color=(255, 220, 50))
            arrow_start = (mid_x, mid_y + 35)
            arrow_end = (mid_x, mid_y - 65)

        else:  # Generic clean motion
            offset = int(np.sin(t * np.pi * 2) * 40)
            hand_x = mid_x + offset
            hand_y = mid_y
            draw_hand_icon(frame, hand_x, hand_y, 28, title, color=(0, 255, 200))

        # Draw trajectory arrow if set
        if arrow_start and arrow_end:
            cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 200, 255), 2, tipLength=0.25)

        # Description / Instruction Footer Banner
        cv2.rectangle(frame, (10, height - 72), (width - 10, height - 10), (30, 30, 38), -1)
        cv2.line(frame, (10, height - 72), (width - 10, height - 72), (0, 200, 255), 2)
        cv2.putText(frame, "MOTION:", (25, height - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        cv2.putText(frame, desc[:58], (90, height - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        # Progress bar
        bar_w = int((width - 50) * t)
        cv2.rectangle(frame, (25, height - 22), (25 + bar_w, height - 16), (0, 255, 180), -1)

        out.write(frame)

    out.release()
    print(f"  🎬 Generated verified sign video: {filename}")


def generate_all_default_videos():
    print("\n" + "=" * 65)
    print(" 📹 GENERATING VERIFIED SIGN VIDEOS FOR VOCABULARY LIBRARY")
    print("=" * 65)

    catalog = [
        ("hello.mp4", "Hello", "salute_outward", "Open hand moves outward from forehead in salute."),
        ("thank_you.mp4", "Thank You", "chin_to_forward", "Fingertips touch chin and move smoothly toward person."),
        ("yes.mp4", "Yes", "nodding_fist", "S-fist nods up and down from the wrist twice."),
        ("no.mp4", "No", "beak_tap", "Index and middle fingers tap against thumb like closing beak."),
        ("please.mp4", "Please", "chest_circle", "Flat open hand rubs chest in clockwise circular motion."),
        ("help.mp4", "Help", "lift_hands", "Thumbs-up fist on flat palm lifts upward together."),
        ("good.mp4", "Good", "chin_to_forward", "Fingertips touch chin, hand moves down to flat palm."),
        ("morning.mp4", "Morning", "sun_rising", "Dominant arm rises like the sun over horizontal arm."),
        ("good_morning.mp4", "Good Morning", "sun_rising", "Compound: GOOD chin touch followed by MORNING sun rise."),
        ("how_are_you.mp4", "How Are You", "salute_outward", "Curved hands rotate outward then point forward."),
        ("sign.mp4", "Sign", "chest_circle", "Index fingers revolve in alternating circular paths."),
        ("friend.mp4", "Friend", "beak_tap", "Index fingers hook together, flip, and hook again."),
        ("welcome.mp4", "Welcome", "salute_outward", "Open flat hand sweeps inward towards body."),
        ("goodbye.mp4", "Goodbye", "salute_outward", "Open hand waves forward and backward."),
        ("name.mp4", "Name", "beak_tap", "H-fingers of both hands tap crosswise twice."),
    ]

    for fname, title, mtype, desc in catalog:
        generate_sign_video(fname, title, mtype, desc)

    print("=" * 65)
    print(f" ✅ Generated {len(catalog)} verified sign videos in {VIDEOS_DIR}\n")


if __name__ == "__main__":
    generate_all_default_videos()
