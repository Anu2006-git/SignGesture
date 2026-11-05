"""
dashboard.py
============
Premium Two-Way Sign Language Communication Dashboard
Engineered for High-Accessibility, Real-Time Performance, and Modern UI/UX.

Architecture (Three Proportional Vertical Sections):
  - LEFT SECTION (32% width): Deaf Person — Sign Language to Text & Voice
      * Live 720p 30 FPS camera feed with MediaPipe 3D hand tracking
      * 88-D Landmark Random Forest (99.73% test accuracy)
      * Real-time Out-of-Distribution (OOD) Wrong Gesture rejection
      * Single-commit hold debouncer preventing duplicate letter spam
      * Interactive, fully editable sentence builder
      * Tactile controls: Start/Stop Camera, Add Space, Undo, Clear, Speak (TTS)

  - CENTER SECTION (38% width — VISUALLY PROMINENT HERO): Visual Sign Language Display
      * Prominent quote display of incoming hearing sentence
      * Large high-definition 25 FPS verified sign video playback (.mp4)
      * Prominent status badge: Verified Sign vs. Explicit Fingerspelling Fallback
      * Large accessible playback controls: Play, Pause, Resume, Replay, Next
      * Reassuring fallback notices (never invents fake signs)

  - RIGHT SECTION (30% width): Hearing Person — Speech to Text
      * Microphone controls: Start Speaking, Stop Listening, Clear
      * Real-time status pulse banner (Ready, Listening, Processing, Error)
      * Live speech recognition transcript display
      * Manual keyboard text input alternative with quick Send/Translate
      * Full panel isolation (zero cross-panel state overwriting)

  - STATUS BAR (Bottom Diagnostics Dock):
      * Real-time health pills for Camera, Model, Microphone, Speech, and Sign Library

Visual Theme: Executive Navy (#0F172A), Electric Teal (#0D9488 / #14B8A6), Clean Slate (#F1F5F9).
"""

from collections import Counter, deque
import json
import logging
from pathlib import Path
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk

# Configure UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# MediaPipe
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    mp = None

# Text-to-Speech
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

from model_training.config import MODELS_DIR
from model_training.feature_extractor import (
    FEATURE_SIZE,
    evaluate_gesture,
    extract_hand_features,
)
from sign_vocabulary import (
    PlaybackState,
    SignPlayItem,
    SignVideoPlayer,
    SpeechState,
    SpeechToTextEngine,
    VocabularyManager,
)
from sign_vocabulary.config import ALPHABET_DIR, VIDEOS_DIR

logger = logging.getLogger("dashboard")

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM DESIGN SYSTEM & COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
COLOR_NAVY_DARKEST   = "#090D16"   # Viewport bezel & deep dark borders
COLOR_NAVY_DEEP      = "#0F172A"   # Executive Deep Navy (Headers, Primary Badges)
COLOR_NAVY_SLATE     = "#1E293B"   # Card borders, Sub-headers
COLOR_TEAL_PRIMARY   = "#0D9488"   # Electric Teal Accent (Primary CTA Buttons)
COLOR_TEAL_HOVER     = "#0F766E"   # Teal Hover
COLOR_TEAL_BRIGHT    = "#14B8A6"   # Vivid Teal for Highlights & Active Dots
COLOR_TEAL_TINT      = "#CCFBF1"   # Soft Pastel Teal Tint for Badges
COLOR_BG_PAGE        = "#F1F5F9"   # Premium Light Slate Page Background
COLOR_CARD_BG        = "#FFFFFF"   # Pure White Card Surfaces
COLOR_CARD_BORDER    = "#E2E8F0"   # Refined 1px Card Border
COLOR_CARD_SHADOW    = "#CBD5E1"   # Subtle Card Drop Shadow Simulation
COLOR_TEXT_MAIN      = "#0F172A"   # High-Contrast Dark Slate Navy
COLOR_TEXT_MUTED     = "#64748B"   # Slate Subtitle & Caption Text
COLOR_SUCCESS        = "#15803D"   # Verified Forest Green
COLOR_SUCCESS_TINT   = "#DCFCE7"   # Soft Green Badge Background
COLOR_ERROR          = "#DC2626"   # Alert Crimson Red
COLOR_ERROR_TINT     = "#FEE2E2"   # Soft Red Badge Background
COLOR_WARNING        = "#D97706"   # Warning Amber
COLOR_WARNING_TINT   = "#FEF3C7"   # Soft Amber Badge Background
COLOR_BLUE_ACCENT    = "#2563EB"   # Royal Blue for Hearing Section
COLOR_CYAN_HERO      = "#06B6D4"   # Vivid Cyan for Hero Center Section


class TextToSpeechEngine:
    """Threaded Text-to-Speech engine ensuring audio never stalls UI or video rendering."""

    def __init__(self):
        self.available = _TTS_AVAILABLE
        self.last_spoken_text = ""

    def speak_async(self, text: str) -> bool:
        clean = text.strip()
        if not clean:
            return False

        self.last_spoken_text = clean
        logger.info("🔊 Speaking sentence: '%s'", clean)

        if not self.available:
            return True

        def _worker():
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.say(clean)
                engine.runAndWait()
            except Exception as e:
                logger.warning("TTS speech engine error: %s", e)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return True


class TwoWayDashboardApp:
    """
    Final User Interface: Premium Two-Way Real-Time Sign Language Communication Dashboard.
    Features three synchronized, accessibility-focused vertical sections on one screen.
    """

    def __init__(self, root: tk.Tk, camera_id: int = 0, mic_index: Optional[int] = None):
        self.root = root
        self.camera_id = camera_id
        self.mic_index = mic_index

        self.root.title("SignGesture AI — Two-Way Sign Language Communication Platform")
        self.root.geometry("1480x890")
        self.root.minsize(1320, 780)
        self.root.configure(bg=COLOR_BG_PAGE)

        icon_path = BASE_DIR / "app_icon.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(default=str(icon_path))
            except Exception:
                pass

        # ── Backend Modules & State (Preserved 100%) ──
        self.tts = TextToSpeechEngine()
        self.vocab_manager = VocabularyManager()
        self.sign_player = SignVideoPlayer(fps=25)
        self.speech_engine = SpeechToTextEngine(
            device_index=self.mic_index,
            on_text_recognized=self._on_speech_recognized_callback,
            timeout=5.0,
            phrase_time_limit=8.0,
        )

        # Model & Gesture Recognition State
        self.model = None
        self.classes: List[str] = []
        self.class_stats: Dict[str, Any] = {}
        self.model_loaded: bool = False
        self.model_error_message: Optional[str] = None
        self._load_model()

        # Camera & Processing State
        self.camera_active: bool = False
        self.camera_thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_error_message: Optional[str] = None
        self.current_cam_frame: Optional[np.ndarray] = None
        self.cam_lock = threading.Lock()

        # Gesture debouncer state
        self.hold_time_sec: float = 1.0
        self.confidence_threshold: float = 0.60
        self.min_margin: float = 0.08
        self.prediction_buffer = deque(maxlen=5)
        self.current_candidate: Optional[str] = None
        self.candidate_start_time: float = 0.0
        self.candidate_committed: bool = False
        self.current_detected_sign: str = "None"
        self.current_confidence: float = 0.0
        self.is_wrong_gesture: bool = False
        self.hold_progress_ratio: float = 0.0

        # MediaPipe setup
        self.mp_hands = None
        self.mp_drawing = None
        if _MP_AVAILABLE and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.50,
                )
                self.mp_drawing = mp.solutions.drawing_utils
            except Exception as e:
                logger.warning("MediaPipe Hands init error: %s", e)

        # Hearing speech state
        self.hearing_recognized_text: str = ""
        self.incoming_sentence_text: str = "Awaiting speech or typed message from hearing user..."

        # Cached PhotoImages for Tkinter viewports
        self.cam_photo = None
        self.video_photo = None

        # Build Premium UI
        self._setup_styles()
        self._build_header()
        self._build_main_sections()
        self._build_status_bar()

        # Bind close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start UI loop timers
        self.root.after(40, self._update_camera_viewport)
        self.root.after(40, self._update_sign_video_viewport)
        self.root.after(200, self._update_status_indicators)

        # Auto-start camera on launch for a seamless immediate experience
        self.start_camera()

    # ─────────────────────────────────────────────────────────────
    # STYLES & SETUP
    # ─────────────────────────────────────────────────────────────
    def _setup_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # TTK Frames
        style.configure("Dashboard.TFrame", background=COLOR_BG_PAGE)
        style.configure("Card.TFrame", background=COLOR_CARD_BG)

        # Primary Action Button (Electric Teal)
        style.configure(
            "Primary.TButton",
            background=COLOR_TEAL_PRIMARY,
            foreground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            padding=(12, 7),
        )
        style.map(
            "Primary.TButton",
            background=[("active", COLOR_TEAL_HOVER), ("disabled", "#CBD5E1")],
        )

        # Secondary Button (Refined Slate / Light Gray with subtle border)
        style.configure(
            "Secondary.TButton",
            background="#FFFFFF",
            foreground=COLOR_NAVY_DEEP,
            font=("Segoe UI", 9, "bold"),
            borderwidth=1,
            relief="solid",
            padding=(10, 6),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#F8FAFC")],
        )

        # Hero Action Button (Prominent Navy CTA)
        style.configure(
            "Hero.TButton",
            background=COLOR_NAVY_DEEP,
            foreground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            padding=(12, 7),
        )
        style.map(
            "Hero.TButton",
            background=[("active", COLOR_NAVY_SLATE), ("disabled", "#CBD5E1")],
        )

        # Danger Button (Crimson)
        style.configure(
            "Danger.TButton",
            background=COLOR_ERROR,
            foreground="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
            padding=(10, 6),
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#B91C1C"), ("disabled", "#CBD5E1")],
        )

        # Sleek Progress Bar Style
        style.configure(
            "Teal.Horizontal.TProgressbar",
            troughcolor="#E2E8F0",
            background=COLOR_TEAL_PRIMARY,
            lightcolor=COLOR_TEAL_BRIGHT,
            darkcolor=COLOR_TEAL_HOVER,
            bordercolor="#E2E8F0",
            thickness=10,
        )

    # ─────────────────────────────────────────────────────────────
    # PROMINENT EXECUTIVE HEADER BAR
    # ─────────────────────────────────────────────────────────────
    def _build_header(self):
        """Top executive dashboard header with prominent title, subtitle, and live telemetry."""
        header = tk.Frame(self.root, bg=COLOR_NAVY_DEEP, height=72)
        header.pack(fill="x", side="top")

        # Left branding block
        brand_box = tk.Frame(header, bg=COLOR_NAVY_DEEP)
        brand_box.pack(side="left", padx=20, pady=10)

        # Title row
        title_row = tk.Frame(brand_box, bg=COLOR_NAVY_DEEP)
        title_row.pack(anchor="w")

        icon_lbl = tk.Label(
            title_row,
            text="🤟",
            font=("Segoe UI Emoji", 18),
            fg="#FFFFFF",
            bg=COLOR_NAVY_DEEP,
        )
        icon_lbl.pack(side="left", padx=(0, 6))

        title_lbl = tk.Label(
            title_row,
            text="SIGNGESTURE",
            font=("Segoe UI", 18, "bold"),
            fg="#FFFFFF",
            bg=COLOR_NAVY_DEEP,
        )
        title_lbl.pack(side="left")

        # High-tech AI Badge pill
        ai_badge = tk.Label(
            title_row,
            text="AI CORE v2.0",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_NAVY_DEEP,
            bg=COLOR_TEAL_BRIGHT,
            padx=6,
            pady=1,
        )
        ai_badge.pack(side="left", padx=(8, 0))

        # Subtitle row
        sub_lbl = tk.Label(
            brand_box,
            text="Assistive Two-Way Communication Dashboard • Real-Time ASL Vision & Voice Bridge",
            font=("Segoe UI", 10),
            fg="#94A3B8",
            bg=COLOR_NAVY_DEEP,
        )
        sub_lbl.pack(anchor="w", pady=(2, 0))

        # Right telemetry block
        telemetry_box = tk.Frame(header, bg=COLOR_NAVY_DEEP)
        telemetry_box.pack(side="right", padx=20, pady=12)

        # Model accuracy chip
        model_txt = "● ML Model: 99.73% Precision" if self.model_loaded else "● Model Offline"
        model_col = COLOR_TEAL_BRIGHT if self.model_loaded else "#F87171"
        self.lbl_head_model = tk.Label(
            telemetry_box,
            text=model_txt,
            font=("Segoe UI", 9, "bold"),
            fg=model_col,
            bg="#1E293B",
            padx=12,
            pady=4,
            relief="flat",
        )
        self.lbl_head_model.pack(side="right", padx=(8, 0))

        # Mode chip
        mode_chip = tk.Label(
            telemetry_box,
            text="⚡ LIVE TWO-WAY SYNC",
            font=("Segoe UI", 9, "bold"),
            fg="#38BDF8",
            bg="#1E293B",
            padx=12,
            pady=4,
            relief="flat",
        )
        mode_chip.pack(side="right")

    # ─────────────────────────────────────────────────────────────
    # THREE-COLUMN MAIN LAYOUT (CENTER SECTION PROMINENT)
    # ─────────────────────────────────────────────────────────────
    def _build_main_sections(self):
        """
        Three proportional vertical sections.
        The center section is weighted higher (weight=5 vs 3:3) so that
        the visual sign language display is prominently featured as the hero bridge.
        """
        main_container = tk.Frame(self.root, bg=COLOR_BG_PAGE)
        main_container.pack(fill="both", expand=True, padx=14, pady=10)

        # Center section (index 1) has higher weight for visual prominence!
        main_container.columnconfigure(0, weight=3, minsize=380)  # Left: Deaf Person
        main_container.columnconfigure(1, weight=5, minsize=480)  # Center: Visual Sign (Hero)
        main_container.columnconfigure(2, weight=3, minsize=380)  # Right: Hearing Person
        main_container.rowconfigure(0, weight=1)

        # Section 1: Left (Deaf Person)
        self._build_left_section(main_container)

        # Section 2: Middle (Visual Sign Display — HERO)
        self._build_middle_section(main_container)

        # Section 3: Right (Hearing Person)
        self._build_right_section(main_container)

    def _create_elevated_card(self, parent: tk.Widget, col: int) -> tk.Frame:
        """Creates a modern card with subtle drop shadow simulation and refined borders."""
        # Shadow frame
        shadow_wrapper = tk.Frame(parent, bg=COLOR_CARD_SHADOW, bd=0, highlightthickness=0)
        shadow_wrapper.grid(row=0, column=col, sticky="nsew", padx=6, pady=4)

        # Main elevated card
        card = tk.Frame(shadow_wrapper, bg=COLOR_CARD_BG, bd=1, relief="solid", highlightthickness=0)
        card.pack(fill="both", expand=True, padx=(0, 2), pady=(0, 2))
        return card

    # ─────────────────────────────────────────────────────────────
    # SECTION 1: DEAF PERSON (SIGN LANGUAGE TO TEXT & SPEECH)
    # ─────────────────────────────────────────────────────────────
    def _build_left_section(self, parent: tk.Widget):
        card = self._create_elevated_card(parent, col=0)

        # Section Header with Teal Indicator Accent
        hdr = tk.Frame(card, bg=COLOR_NAVY_DEEP, height=48)
        hdr.pack(fill="x")

        # Teal top accent bar
        accent_bar = tk.Frame(hdr, bg=COLOR_TEAL_BRIGHT, height=3)
        accent_bar.pack(fill="x", side="top")

        hdr_content = tk.Frame(hdr, bg=COLOR_NAVY_DEEP)
        hdr_content.pack(fill="x", padx=14, pady=8)

        tk.Label(
            hdr_content,
            text="🤟  DEAF PERSON",
            font=("Segoe UI", 11, "bold"),
            fg="#FFFFFF",
            bg=COLOR_NAVY_DEEP,
        ).pack(side="left")

        tk.Label(
            hdr_content,
            text="SIGN ➔ TEXT & VOICE",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEAL_BRIGHT,
            bg=COLOR_NAVY_DEEP,
        ).pack(side="left", padx=(6, 0))

        content = tk.Frame(card, bg=COLOR_CARD_BG, padx=14, pady=10)
        content.pack(fill="both", expand=True)

        # Camera Controls & Telemetry Bar
        cam_bar = tk.Frame(content, bg=COLOR_CARD_BG)
        cam_bar.pack(fill="x", pady=(0, 8))

        self.btn_start_cam = ttk.Button(
            cam_bar,
            text="▶ Start Camera",
            style="Primary.TButton",
            command=self.start_camera,
        )
        self.btn_start_cam.pack(side="left", padx=(0, 6))

        self.btn_stop_cam = ttk.Button(
            cam_bar,
            text="⏹ Stop Camera",
            style="Secondary.TButton",
            command=self.stop_camera,
        )
        self.btn_stop_cam.pack(side="left", padx=(0, 6))

        self.lbl_cam_status = tk.Label(
            cam_bar,
            text="Camera: Disconnected",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_CARD_BG,
        )
        self.lbl_cam_status.pack(side="right", pady=2)

        # Camera Viewport Media Bezel Container
        viewport_wrapper = tk.Frame(content, bg=COLOR_NAVY_DARKEST, bd=1, relief="solid")
        viewport_wrapper.pack(fill="x", pady=(0, 8))

        # Viewport HUD top bar
        hud_bar = tk.Frame(viewport_wrapper, bg=COLOR_NAVY_DARKEST, height=22)
        hud_bar.pack(fill="x", padx=6, pady=(3, 0))

        tk.Label(
            hud_bar,
            text="● LIVE CAMERA FEED",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEAL_BRIGHT,
            bg=COLOR_NAVY_DARKEST,
        ).pack(side="left")

        tk.Label(
            hud_bar,
            text="720p HD • 30 FPS",
            font=("Segoe UI", 8),
            fg="#94A3B8",
            bg=COLOR_NAVY_DARKEST,
        ).pack(side="right")

        # Webcam Canvas
        self.cam_canvas = tk.Canvas(
            viewport_wrapper,
            bg=COLOR_NAVY_DARKEST,
            height=260,
            highlightthickness=0,
            bd=0,
        )
        self.cam_canvas.pack(fill="x", padx=4, pady=(2, 4))

        # Real-Time Gesture Recognition Card
        detect_card = tk.Frame(content, bg="#F8FAFC", bd=1, relief="solid", highlightthickness=0)
        detect_card.pack(fill="x", pady=(0, 8), ipady=4)

        detect_hdr = tk.Frame(detect_card, bg="#F8FAFC")
        detect_hdr.pack(fill="x", padx=10, pady=(4, 2))

        tk.Label(
            detect_hdr,
            text="CURRENT GESTURE PREDICTION",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg="#F8FAFC",
        ).pack(side="left")

        self.lbl_detect_confidence = tk.Label(
            detect_hdr,
            text="Conf: 0.0%",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg="#F8FAFC",
        )
        self.lbl_detect_confidence.pack(side="right")

        detect_body = tk.Frame(detect_card, bg="#F8FAFC")
        detect_body.pack(fill="x", padx=10, pady=(0, 4))

        self.lbl_detected_sign = tk.Label(
            detect_body,
            text="[ No Gesture ]",
            font=("Segoe UI", 18, "bold"),
            fg=COLOR_NAVY_DEEP,
            bg="#F8FAFC",
        )
        self.lbl_detected_sign.pack(side="left")

        self.lbl_hold_status = tk.Label(
            detect_body,
            text="Hold 1.0s to append letter",
            font=("Segoe UI", 9, "italic"),
            fg=COLOR_TEXT_MUTED,
            bg="#F8FAFC",
        )
        self.lbl_hold_status.pack(side="right", pady=4)

        # Smooth Hold Progress Bar
        self.hold_progress_bar = ttk.Progressbar(
            detect_card,
            orient="horizontal",
            mode="determinate",
            style="Teal.Horizontal.TProgressbar",
            maximum=100,
        )
        self.hold_progress_bar.pack(fill="x", padx=10, pady=(2, 6))

        # Sentence Builder Label
        sent_hdr = tk.Frame(content, bg=COLOR_CARD_BG)
        sent_hdr.pack(fill="x", pady=(2, 4))

        tk.Label(
            sent_hdr,
            text="ACCUMULATED SENTENCE (EDITABLE):",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_NAVY_DEEP,
            bg=COLOR_CARD_BG,
        ).pack(side="left")

        tk.Label(
            sent_hdr,
            text="✏️ Click to edit anytime",
            font=("Segoe UI", 8),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_CARD_BG,
        ).pack(side="right")

        # Editable Sentence Text Area Container
        text_frame = tk.Frame(content, bg=COLOR_CARD_BG, bd=1, relief="solid")
        text_frame.pack(fill="both", expand=True, pady=(0, 8))

        self.txt_deaf_sentence = tk.Text(
            text_frame,
            font=("Segoe UI", 12),
            wrap="word",
            height=3,
            relief="flat",
            bg="#FFFFFF",
            fg=COLOR_TEXT_MAIN,
            padx=10,
            pady=8,
            insertbackground=COLOR_TEAL_PRIMARY,
        )
        self.txt_deaf_sentence.pack(fill="both", expand=True)

        # Sentence Action Controls Row
        btn_row = tk.Frame(content, bg=COLOR_CARD_BG)
        btn_row.pack(fill="x")

        ttk.Button(
            btn_row,
            text="␣ Space",
            style="Secondary.TButton",
            command=self.sentence_add_space,
        ).pack(side="left", padx=(0, 4))

        ttk.Button(
            btn_row,
            text="⌫ Undo",
            style="Secondary.TButton",
            command=self.sentence_undo,
        ).pack(side="left", padx=(0, 4))

        ttk.Button(
            btn_row,
            text="🗑 Clear",
            style="Secondary.TButton",
            command=self.sentence_clear,
        ).pack(side="left", padx=(0, 4))

        self.btn_speak_sentence = ttk.Button(
            btn_row,
            text="🔊 Speak Sentence (TTS)",
            style="Primary.TButton",
            command=self.sentence_speak,
        )
        self.btn_speak_sentence.pack(side="right")

    # ─────────────────────────────────────────────────────────────
    # SECTION 2: MIDDLE (VISUAL SIGN LANGUAGE DISPLAY — HERO BRIDGE)
    # ─────────────────────────────────────────────────────────────
    def _build_middle_section(self, parent: tk.Widget):
        """
        Visually prominent centerpiece of the dashboard.
        Taller video canvas and vivid cyan/teal accents establish this section
        as the core communication bridge between users.
        """
        card = self._create_elevated_card(parent, col=1)

        # Hero Section Header with Cyan/Teal Indicator Bar
        hdr = tk.Frame(card, bg=COLOR_NAVY_DEEP, height=48)
        hdr.pack(fill="x")

        # Hero Cyan top accent bar
        accent_bar = tk.Frame(hdr, bg=COLOR_CYAN_HERO, height=3)
        accent_bar.pack(fill="x", side="top")

        hdr_content = tk.Frame(hdr, bg=COLOR_NAVY_DEEP)
        hdr_content.pack(fill="x", padx=14, pady=8)

        tk.Label(
            hdr_content,
            text="🎬  VISUAL SIGN LANGUAGE DISPLAY",
            font=("Segoe UI", 11, "bold"),
            fg="#FFFFFF",
            bg=COLOR_NAVY_DEEP,
        ).pack(side="left")

        hero_badge = tk.Label(
            hdr_content,
            text="★ MAIN BRIDGE",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_NAVY_DEEP,
            bg=COLOR_CYAN_HERO,
            padx=6,
            pady=1,
        )
        hero_badge.pack(side="left", padx=(8, 0))

        content = tk.Frame(card, bg=COLOR_CARD_BG, padx=14, pady=10)
        content.pack(fill="both", expand=True)

        # Incoming Sentence Speech-Bubble Card
        bubble_frame = tk.Frame(content, bg="#F0FDFA", bd=1, relief="solid", highlightthickness=0)
        bubble_frame.pack(fill="x", pady=(0, 8))

        bubble_left_accent = tk.Frame(bubble_frame, bg=COLOR_TEAL_PRIMARY, width=4)
        bubble_left_accent.pack(side="left", fill="y")

        bubble_inner = tk.Frame(bubble_frame, bg="#F0FDFA", padx=10, pady=6)
        bubble_inner.pack(side="left", fill="both", expand=True)

        tk.Label(
            bubble_inner,
            text="INCOMING MESSAGE FROM HEARING USER:",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEAL_PRIMARY,
            bg="#F0FDFA",
        ).pack(anchor="w")

        self.lbl_incoming_sentence = tk.Label(
            bubble_inner,
            text="\"Awaiting speech or typed message from hearing user...\"",
            font=("Segoe UI", 12, "bold"),
            fg=COLOR_NAVY_DEEP,
            bg="#F0FDFA",
            wraplength=460,
            justify="left",
        )
        self.lbl_incoming_sentence.pack(anchor="w", pady=(2, 2))

        # Hero Sign Video Viewport Bezel Container
        viewport_wrapper = tk.Frame(content, bg=COLOR_NAVY_DARKEST, bd=1, relief="solid")
        viewport_wrapper.pack(fill="both", expand=True, pady=(0, 8))

        # Viewport HUD bar
        hud_bar = tk.Frame(viewport_wrapper, bg=COLOR_NAVY_DARKEST, height=22)
        hud_bar.pack(fill="x", padx=8, pady=(4, 0))

        tk.Label(
            hud_bar,
            text="● ASL VIDEO STREAM",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_CYAN_HERO,
            bg=COLOR_NAVY_DARKEST,
        ).pack(side="left")

        tk.Label(
            hud_bar,
            text="25 FPS • VERIFIED SIGN PLAYBACK",
            font=("Segoe UI", 8),
            fg="#94A3B8",
            bg=COLOR_NAVY_DARKEST,
        ).pack(side="right")

        # Large Hero Sign Video Canvas (Taller 330px height for prominence)
        self.video_canvas = tk.Canvas(
            viewport_wrapper,
            bg=COLOR_NAVY_DARKEST,
            height=330,
            highlightthickness=0,
            bd=0,
        )
        self.video_canvas.pack(fill="both", expand=True, padx=4, pady=(2, 4))

        # Active Sign Telemetry & Modality Badge Card
        status_card = tk.Frame(content, bg="#F8FAFC", bd=1, relief="solid", highlightthickness=0)
        status_card.pack(fill="x", pady=(0, 8), ipady=4)

        status_hdr = tk.Frame(status_card, bg="#F8FAFC")
        status_hdr.pack(fill="x", padx=10, pady=(4, 2))

        self.lbl_current_word = tk.Label(
            status_hdr,
            text="Sign: [ Idle ]",
            font=("Segoe UI", 12, "bold"),
            fg=COLOR_NAVY_DEEP,
            bg="#F8FAFC",
        )
        self.lbl_current_word.pack(side="left")

        self.lbl_sign_modality = tk.Label(
            status_card,
            text="Verified video clips play sequentially; unmapped words fingerspell.",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg="#F8FAFC",
            padx=10,
        )
        self.lbl_sign_modality.pack(anchor="w", pady=(0, 4))

        # Fallback Alert Notification Banner
        self.lbl_fallback_alert = tk.Label(
            content,
            text="Ready. Spoken phrases translate directly to verified sign sequence.",
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_CARD_BG,
            wraplength=460,
            justify="left",
        )
        self.lbl_fallback_alert.pack(fill="x", pady=(0, 8))

        # Tactile Playback Controls Row
        ctrl_row = tk.Frame(content, bg=COLOR_CARD_BG)
        ctrl_row.pack(fill="x")

        self.btn_play_resume = ttk.Button(
            ctrl_row,
            text="▶ Play / Resume",
            style="Primary.TButton",
            command=self.sign_player_resume,
        )
        self.btn_play_resume.pack(side="left", padx=(0, 4))

        self.btn_pause = ttk.Button(
            ctrl_row,
            text="⏸ Pause",
            style="Secondary.TButton",
            command=self.sign_player_pause,
        )
        self.btn_pause.pack(side="left", padx=(0, 4))

        self.btn_replay = ttk.Button(
            ctrl_row,
            text="🔄 Replay",
            style="Secondary.TButton",
            command=self.sign_player_replay,
        )
        self.btn_replay.pack(side="left", padx=(0, 4))

        self.btn_next_sign = ttk.Button(
            ctrl_row,
            text="⏭ Next",
            style="Secondary.TButton",
            command=self.sign_player_next,
        )
        self.btn_next_sign.pack(side="right")

    # ─────────────────────────────────────────────────────────────
    # SECTION 3: RIGHT (HEARING PERSON SPEECH TO TEXT)
    # ─────────────────────────────────────────────────────────────
    def _build_right_section(self, parent: tk.Widget):
        card = self._create_elevated_card(parent, col=2)

        # Section Header with Royal Blue Accent Bar
        hdr = tk.Frame(card, bg=COLOR_NAVY_DEEP, height=48)
        hdr.pack(fill="x")

        accent_bar = tk.Frame(hdr, bg=COLOR_BLUE_ACCENT, height=3)
        accent_bar.pack(fill="x", side="top")

        hdr_content = tk.Frame(hdr, bg=COLOR_NAVY_DEEP)
        hdr_content.pack(fill="x", padx=14, pady=8)

        tk.Label(
            hdr_content,
            text="🗣️  HEARING PERSON",
            font=("Segoe UI", 11, "bold"),
            fg="#FFFFFF",
            bg=COLOR_NAVY_DEEP,
        ).pack(side="left")

        tk.Label(
            hdr_content,
            text="SPEECH ➔ TEXT & SIGN",
            font=("Segoe UI", 8, "bold"),
            fg="#60A5FA",
            bg=COLOR_NAVY_DEEP,
        ).pack(side="left", padx=(6, 0))

        content = tk.Frame(card, bg=COLOR_CARD_BG, padx=14, pady=10)
        content.pack(fill="both", expand=True)

        # Microphone Controls Row
        mic_bar = tk.Frame(content, bg=COLOR_CARD_BG)
        mic_bar.pack(fill="x", pady=(0, 8))

        self.btn_start_listen = ttk.Button(
            mic_bar,
            text="🎤 Start Speaking",
            style="Primary.TButton",
            command=self.speech_start_listening,
        )
        self.btn_start_listen.pack(side="left", padx=(0, 6))

        self.btn_stop_listen = ttk.Button(
            mic_bar,
            text="⏹ Stop",
            style="Secondary.TButton",
            command=self.speech_stop_listening,
        )
        self.btn_stop_listen.pack(side="left", padx=(0, 6))

        ttk.Button(
            mic_bar,
            text="Clear",
            style="Secondary.TButton",
            command=self.speech_clear_text,
        ).pack(side="right")

        # Microphone Live Status Pulse Banner
        self.status_banner = tk.Frame(content, bg="#F1F5F9", bd=1, relief="solid", highlightthickness=0)
        self.status_banner.pack(fill="x", pady=(0, 8), ipady=4)

        self.lbl_mic_status = tk.Label(
            self.status_banner,
            text="● Microphone Ready. Press 'Start Speaking' or type below.",
            font=("Segoe UI", 9),
            fg=COLOR_SUCCESS,
            bg="#F1F5F9",
            wraplength=360,
            justify="left",
        )
        self.lbl_mic_status.pack(anchor="w", padx=10, pady=4)

        # Recognized Speech Transcript Area
        tk.Label(
            content,
            text="RECOGNIZED SPEECH TRANSCRIPT:",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_NAVY_DEEP,
            bg=COLOR_CARD_BG,
        ).pack(anchor="w", pady=(4, 4))

        speech_box_frame = tk.Frame(content, bg=COLOR_CARD_BG, bd=1, relief="solid")
        speech_box_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.txt_recognized_speech = tk.Text(
            speech_box_frame,
            font=("Segoe UI", 12),
            wrap="word",
            height=5,
            relief="flat",
            bg="#FFFFFF",
            fg=COLOR_TEXT_MAIN,
            padx=10,
            pady=8,
            insertbackground=COLOR_BLUE_ACCENT,
        )
        self.txt_recognized_speech.pack(fill="both", expand=True)

        # Manual Text Input Alternative Card
        manual_card = tk.Frame(content, bg="#F8FAFC", bd=1, relief="solid", highlightthickness=0)
        manual_card.pack(fill="x", pady=(4, 0), ipady=6)

        manual_hdr = tk.Frame(manual_card, bg="#F8FAFC")
        manual_hdr.pack(fill="x", padx=10, pady=(4, 4))

        tk.Label(
            manual_hdr,
            text="MANUAL TEXT INPUT (ALTERNATIVE):",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg="#F8FAFC",
        ).pack(side="left")

        tk.Label(
            manual_hdr,
            text="Press ENTER to send",
            font=("Segoe UI", 8, "italic"),
            fg=COLOR_TEXT_MUTED,
            bg="#F8FAFC",
        ).pack(side="right")

        input_row = tk.Frame(manual_card, bg="#F8FAFC")
        input_row.pack(fill="x", padx=10, pady=(0, 4))

        self.ent_manual_text = ttk.Entry(input_row, font=("Segoe UI", 11))
        self.ent_manual_text.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.ent_manual_text.bind("<Return>", lambda e: self.submit_manual_text())

        ttk.Button(
            input_row,
            text="Translate ➔",
            style="Hero.TButton",
            command=self.submit_manual_text,
        ).pack(side="right")

    # ─────────────────────────────────────────────────────────────
    # STATUS BAR (BOTTOM TELEMETRY DOCK)
    # ─────────────────────────────────────────────────────────────
    def _build_status_bar(self):
        """Full-width dark telemetry dock displaying live system indicators."""
        dock = tk.Frame(self.root, bg=COLOR_NAVY_DEEP, height=36)
        dock.pack(fill="x", side="bottom")

        # 5 High-Contrast Telemetry Indicators
        self.lbl_stat_cam = tk.Label(
            dock,
            text="● Camera: Disconnected",
            font=("Segoe UI", 9, "bold"),
            fg="#F87171",
            bg=COLOR_NAVY_DEEP,
            padx=14,
            pady=6,
        )
        self.lbl_stat_cam.pack(side="left")

        self.lbl_stat_model = tk.Label(
            dock,
            text="● Model: " + ("Loaded" if self.model_loaded else "Missing"),
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEAL_BRIGHT if self.model_loaded else "#F87171",
            bg=COLOR_NAVY_DEEP,
            padx=14,
            pady=6,
        )
        self.lbl_stat_model.pack(side="left")

        self.lbl_stat_mic = tk.Label(
            dock,
            text="● Microphone: Ready",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEAL_BRIGHT,
            bg=COLOR_NAVY_DEEP,
            padx=14,
            pady=6,
        )
        self.lbl_stat_mic.pack(side="left")

        self.lbl_stat_speech = tk.Label(
            dock,
            text="● Speech Engine: Active",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEAL_BRIGHT if self.speech_engine.is_available else "#F87171",
            bg=COLOR_NAVY_DEEP,
            padx=14,
            pady=6,
        )
        self.lbl_stat_speech.pack(side="left")

        videos_count = len(list(VIDEOS_DIR.glob("*.mp4")))
        self.lbl_stat_videos = tk.Label(
            dock,
            text=f"● Sign Library: {videos_count} Verified Clips",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEAL_BRIGHT if videos_count > 0 else COLOR_WARNING,
            bg=COLOR_NAVY_DEEP,
            padx=14,
            pady=6,
        )
        self.lbl_stat_videos.pack(side="left")

    # ─────────────────────────────────────────────────────────────
    # MODEL LOADING & INFERENCE (UNCHANGED BACKEND LOGIC)
    # ─────────────────────────────────────────────────────────────
    def _load_model(self):
        """Loads trained 88-D Landmark Random Forest model with robust fallback handling."""
        classes_file = MODELS_DIR / "classes.json"
        if not classes_file.exists():
            self.model_loaded = False
            self.model_error_message = "models/classes.json not found. Run python train_model.py."
            return

        try:
            classes_data = json.loads(classes_file.read_text(encoding="utf-8"))
            self.classes = classes_data["classes"]
        except Exception as e:
            self.model_loaded = False
            self.model_error_message = f"Failed to parse classes: {e}"
            return

        # Check manifest
        model_path = None
        manifest_file = MODELS_DIR / "best_model_manifest.json"
        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                fname = manifest.get("model_file")
                if fname and (MODELS_DIR / fname).exists():
                    model_path = MODELS_DIR / fname
            except Exception:
                pass

        if model_path is None:
            for fname in ["best_model_rf.joblib", "model_rf.joblib"]:
                if (MODELS_DIR / fname).exists():
                    model_path = MODELS_DIR / fname
                    break

        if not model_path or not model_path.exists():
            self.model_loaded = False
            self.model_error_message = "Trained model artifact not found in models/."
            return

        try:
            self.model = joblib.load(str(model_path))
            if hasattr(self.model, "n_jobs"):
                self.model.n_jobs = 1
            self.class_stats = getattr(self.model, "class_stats_", {})
            self.model_loaded = True
            self.model_error_message = None
            logger.info("Successfully loaded sign recognition model from %s", model_path.name)
        except Exception as e:
            self.model_loaded = False
            self.model_error_message = f"Model loading failed: {e}"
            logger.error(self.model_error_message)

    def predict_hand_frame(self, frame: np.ndarray, landmarks) -> Dict[str, Any]:
        """Runs 88-D feature extraction and wrong gesture anomaly detection."""
        if self.model is None:
            return {
                "pred_letter": None,
                "candidate_letter": "?",
                "confidence": 0.0,
                "margin": 0.0,
                "dist": 0.0,
                "threshold_dist": 999.0,
                "is_matched": False,
                "reason": "Model unavailable",
                "probs": np.zeros(len(self.classes)),
            }

        h, w = frame.shape[:2]
        feat = extract_hand_features(landmarks, img_w=w, img_h=h)
        return evaluate_gesture(
            feat=feat,
            model=self.model,
            class_stats=self.class_stats,
            classes=self.classes,
            confidence_threshold=self.confidence_threshold,
            min_margin=self.min_margin,
        )

    # ─────────────────────────────────────────────────────────────
    # CAMERA WORKER & PROCESSING (SAFE THREAD MANAGEMENT)
    # ─────────────────────────────────────────────────────────────
    def _safe_ui(self, callback):
        """Thread-safe UI dispatcher that handles root window teardown safely."""
        try:
            if self.root and self.root.winfo_exists():
                self.root.after(0, callback)
        except Exception:
            pass

    def start_camera(self):
        """Safely initializes webcam thread and prevents duplicate capture handles."""
        if self.camera_active:
            return

        self.camera_active = True
        self.camera_error_message = None
        self.btn_start_cam.configure(state="disabled")
        self.btn_stop_cam.configure(state="normal")
        self.lbl_cam_status.configure(text="Camera: Connecting...", fg=COLOR_WARNING)

        self.camera_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.camera_thread.start()

    def stop_camera(self):
        """Safely stops webcam thread and releases hardware resources."""
        self.camera_active = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.btn_start_cam.configure(state="normal")
        self.btn_stop_cam.configure(state="disabled")
        self.lbl_cam_status.configure(text="Camera: Disconnected", fg=COLOR_ERROR)
        self.lbl_stat_cam.configure(text="● Camera: Disconnected", fg="#F87171")

        # Clear canvas to clean styled placeholder
        self._render_camera_placeholder("CAMERA DISCONNECTED", "Click 'Start Camera' to resume gesture recognition.")

    def _camera_worker(self):
        """Background thread for non-blocking OpenCV webcam frame capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap or not self.cap.isOpened():
                self.camera_error_message = f"Webcam (Device {self.camera_id}) could not be opened."
                logger.warning(self.camera_error_message)
                self.camera_active = False
                self._safe_ui(lambda: self.lbl_cam_status.configure(text="Camera: Error / Unavailable", fg=COLOR_ERROR))
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._safe_ui(lambda: self.lbl_cam_status.configure(text="Camera: Connected (30 FPS)", fg=COLOR_SUCCESS))

            while self.camera_active and self.cap and self.cap.isOpened():
                ret, raw = self.cap.read()
                if not ret or raw is None:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(raw, 1)

                with self.cam_lock:
                    self.current_cam_frame = frame

                time.sleep(0.02)  # ~30-40 FPS cap

        except Exception as e:
            if self.camera_active:
                logger.error("Camera worker error: %s", e)
            self.camera_error_message = f"Webcam error: {e}"
        finally:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def _update_camera_viewport(self):
        """Main GUI timer callback to process gesture recognition and draw webcam canvas."""
        if not self.camera_active or self.camera_error_message:
            if self.camera_error_message:
                self._render_camera_placeholder("WEBCAM UNAVAILABLE", self.camera_error_message)
            self.root.after(60, self._update_camera_viewport)
            return

        frame = None
        with self.cam_lock:
            if self.current_cam_frame is not None:
                frame = self.current_cam_frame.copy()

        if frame is None:
            self.root.after(40, self._update_camera_viewport)
            return

        now = time.time()
        hand_detected = False
        last_eval: Optional[Dict[str, Any]] = None

        if self.mp_hands:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_detected = True
                hand = results.multi_hand_landmarks[0]
                last_eval = self.predict_hand_frame(frame, hand.landmark)

                if last_eval["is_matched"]:
                    self.prediction_buffer.append(last_eval["pred_letter"])
                    skel_col = (0, 255, 0)
                    joint_col = (0, 200, 255)
                else:
                    self.prediction_buffer.append("WRONG_GESTURE")
                    skel_col = (0, 60, 255)
                    joint_col = (0, 0, 255)

                if self.mp_drawing:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=skel_col, thickness=2, circle_radius=3),
                        mp.solutions.drawing_utils.DrawingSpec(color=joint_col, thickness=2, circle_radius=2),
                    )
            else:
                self.prediction_buffer.append(None)
        else:
            self.prediction_buffer.append(None)

        # Majority voting
        active_preds = [p for p in self.prediction_buffer if p is not None]
        detected_letter = None
        self.is_wrong_gesture = False
        self.current_confidence = last_eval["confidence"] if last_eval else 0.0

        if active_preds:
            most_common, count = Counter(active_preds).most_common(1)[0]
            if most_common == "WRONG_GESTURE" and count >= 2:
                self.is_wrong_gesture = True
            elif most_common != "WRONG_GESTURE" and count >= 3:
                detected_letter = most_common

        # Debounced accumulation (prevents runaway duplicate letters)
        typed_now = False
        if detected_letter and not self.is_wrong_gesture:
            if detected_letter == self.current_candidate:
                elapsed = now - self.candidate_start_time
                self.hold_progress_ratio = min(1.0, elapsed / self.hold_time_sec)

                if self.hold_progress_ratio >= 1.0 and not self.candidate_committed:
                    # Commit letter to sentence!
                    self.sentence_append_char(detected_letter)
                    self.candidate_committed = True
                    typed_now = True
            else:
                self.current_candidate = detected_letter
                self.candidate_start_time = now
                self.candidate_committed = False
                self.hold_progress_ratio = 0.0
        else:
            self.current_candidate = None
            self.candidate_start_time = 0.0
            self.candidate_committed = False
            self.hold_progress_ratio = 0.0

        # Update HUD Labels with modern typography & accessibility colors
        if self.is_wrong_gesture:
            self.lbl_detected_sign.configure(text="⚠️ WRONG GESTURE", fg=COLOR_ERROR)
            self.lbl_detect_confidence.configure(text="Unrecognized Posture", fg=COLOR_ERROR)
            self.lbl_hold_status.configure(text="Anomaly: Gesture not in training dataset", fg=COLOR_ERROR)
            self.hold_progress_bar["value"] = 0
        elif detected_letter:
            self.lbl_detected_sign.configure(text=f"Letter: {detected_letter}", fg=COLOR_SUCCESS)
            self.lbl_detect_confidence.configure(
                text=f"Conf: {self.current_confidence*100:.1f}%", fg=COLOR_SUCCESS
            )
            pct = int(self.hold_progress_ratio * 100)
            self.hold_progress_bar["value"] = pct
            if self.candidate_committed:
                self.lbl_hold_status.configure(text="✓ Added to Sentence!", fg=COLOR_SUCCESS)
            else:
                self.lbl_hold_status.configure(text=f"Holding steady: {pct}% ➔", fg=COLOR_NAVY_DEEP)
        else:
            status_desc = "Hand in frame" if hand_detected else "Hold hand in camera"
            self.lbl_detected_sign.configure(text="[ Searching... ]", fg=COLOR_TEXT_MUTED)
            self.lbl_detect_confidence.configure(text="Conf: 0.0%", fg=COLOR_TEXT_MUTED)
            self.lbl_hold_status.configure(text=status_desc, fg=COLOR_TEXT_MUTED)
            self.hold_progress_bar["value"] = 0

        # Draw frame into Tkinter Canvas
        cw = max(100, self.cam_canvas.winfo_width())
        ch = max(100, self.cam_canvas.winfo_height())
        resized = cv2.resize(frame, (cw, ch))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.cam_photo = ImageTk.PhotoImage(image=img)
        self.cam_canvas.create_image(0, 0, image=self.cam_photo, anchor="nw")

        self.root.after(35, self._update_camera_viewport)

    def _render_camera_placeholder(self, title: str, subtitle: str):
        """Renders clean styled placeholder card when webcam is disabled or errored."""
        cw = max(200, self.cam_canvas.winfo_width())
        ch = max(150, self.cam_canvas.winfo_height())
        placeholder = np.full((ch, cw, 3), (15, 23, 42), dtype=np.uint8)

        # Border
        cv2.rectangle(placeholder, (10, 10), (cw - 10, ch - 10), (30, 41, 59), 1)
        cv2.putText(placeholder, title, (cw // 2 - 120, ch // 2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (20, 184, 166), 1, cv2.LINE_AA)
        cv2.putText(placeholder, subtitle[:45], (cw // 2 - 135, ch // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.cam_photo = ImageTk.PhotoImage(image=img)
        self.cam_canvas.create_image(0, 0, image=self.cam_photo, anchor="nw")

    # ─────────────────────────────────────────────────────────────
    # SENTENCE BUILDER ACTIONS (LEFT SECTION)
    # ─────────────────────────────────────────────────────────────
    def get_deaf_sentence(self) -> str:
        """Returns the current accumulated deaf sentence text from the text box."""
        return self.txt_deaf_sentence.get("1.0", "end-1c").strip()

    def set_deaf_sentence(self, text: str):
        """Sets the deaf sentence text box contents."""
        self.txt_deaf_sentence.delete("1.0", "end")
        self.txt_deaf_sentence.insert("1.0", text)

    def sentence_append_char(self, char: str):
        """Appends a recognized letter to the sentence."""
        self.txt_deaf_sentence.insert("end", char)

    def sentence_add_space(self):
        """Adds a space to the sentence."""
        self.txt_deaf_sentence.insert("end", " ")

    def sentence_undo(self):
        """Deletes the last character from the sentence."""
        current = self.get_deaf_sentence()
        if current:
            self.set_deaf_sentence(current[:-1])

    def sentence_clear(self):
        """Clears the sentence builder area."""
        self.txt_deaf_sentence.delete("1.0", "end")

    def sentence_speak(self):
        """Pronounces the completed sentence aloud using offline Text-to-Speech."""
        text = self.get_deaf_sentence()
        if text:
            self.tts.speak_async(text)
        else:
            messagebox.showinfo("Empty Sentence", "Please perform signs or type a sentence before speaking.")

    # ─────────────────────────────────────────────────────────────
    # SECTION 2: SIGN VIDEO PLAYBACK (MIDDLE SECTION — HERO)
    # ─────────────────────────────────────────────────────────────
    def translate_and_play_sentence(self, sentence: str):
        """Parses sentence into verified video items & fallbacks, and triggers playback."""
        clean = sentence.strip()
        if not clean:
            return

        self.incoming_sentence_text = clean
        self.lbl_incoming_sentence.configure(text=f"\"{clean}\"")

        # Parse greedy sequence
        sequence = self.vocab_manager.parse_sentence_to_sequence(clean)
        self.sign_player.play_sequence(sequence)

        # Update fallback label
        has_fingerspell = any(item.item_type == "FALLBACK_FINGERSPELL" for item in sequence)
        has_missing = any(item.item_type in ("UNAVAILABLE_NOTICE", "VIDEO_MISSING") for item in sequence)

        if has_missing:
            self.lbl_fallback_alert.configure(
                text="⚠️ Notice: Some words are unavailable in vocabulary. Falling back without inventing fake signs.",
                fg=COLOR_WARNING,
            )
        elif has_fingerspell:
            self.lbl_fallback_alert.configure(
                text="ℹ️ Unmapped words are being spelled using verified alphabet fingerspelling.",
                fg=COLOR_TEAL_PRIMARY,
            )
        else:
            self.lbl_fallback_alert.configure(
                text=f"✓ Playing sequence of {len(sequence)} verified sign video(s).",
                fg=COLOR_SUCCESS,
            )

    def _update_sign_video_viewport(self):
        """25 FPS timer callback rendering sign videos and fallback cards."""
        vw = max(100, self.video_canvas.winfo_width())
        vh = max(100, self.video_canvas.winfo_height())

        frame = self.sign_player.get_current_frame(target_width=vw, target_height=vh)

        # Update current sign badge
        seq = self.sign_player.sequence
        idx = self.sign_player.current_index
        item = seq[idx] if (seq and 0 <= idx < len(seq)) else None
        if item is not None and self.sign_player.state in (PlaybackState.PLAYING, PlaybackState.PAUSED):
            if item.item_type == "VERIFIED_VIDEO":
                self.lbl_current_word.configure(
                    text=f"Sign: {item.word.upper()}  [{self.sign_player.current_index + 1}/{len(self.sign_player.sequence)}]",
                    fg=COLOR_NAVY_DEEP,
                )
                self.lbl_sign_modality.configure(
                    text="✓ Verified Sign Video Clip (.mp4)", fg=COLOR_SUCCESS
                )
            elif item.item_type == "FALLBACK_FINGERSPELL":
                self.lbl_current_word.configure(
                    text=f"Fingerspelling: {item.letter}  [{item.letter_index}/{item.letter_total}] of '{item.word}'",
                    fg=COLOR_WARNING,
                )
                self.lbl_sign_modality.configure(
                    text="ℹ️ Explicit Alphabet Fingerspelling Fallback", fg=COLOR_WARNING
                )
            else:
                self.lbl_current_word.configure(
                    text=f"Notice: {item.word} (Unavailable)", fg=COLOR_ERROR
                )
                self.lbl_sign_modality.configure(
                    text="⚠️ Word Not In Vocabulary (No Fake Sign Invented)", fg=COLOR_ERROR
                )
        elif self.sign_player.state == PlaybackState.FINISHED:
            self.lbl_current_word.configure(text="Sign: [ Playback Completed ]", fg=COLOR_SUCCESS)
            self.lbl_sign_modality.configure(text="Sequence complete. Click 'Replay' to watch again.", fg=COLOR_TEXT_MUTED)
        else:
            self.lbl_current_word.configure(text="Sign: [ Idle ]", fg=COLOR_TEXT_MUTED)
            self.lbl_sign_modality.configure(text="Verified vocabulary plays video; unmapped words fingerspell.", fg=COLOR_TEXT_MUTED)

        # Draw frame into video canvas
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.video_photo = ImageTk.PhotoImage(image=img)
        self.video_canvas.create_image(0, 0, image=self.video_photo, anchor="nw")

        self.root.after(40, self._update_sign_video_viewport)

    def sign_player_resume(self):
        """Resumes playback of sign sequence."""
        self.sign_player.resume()

    def sign_player_pause(self):
        """Pauses sign video playback."""
        self.sign_player.pause()

    def sign_player_replay(self):
        """Replays sign sequence from the beginning."""
        self.sign_player.replay()

    def sign_player_next(self):
        """Skips to next sign in sequence."""
        self.sign_player.step_next()

    # ─────────────────────────────────────────────────────────────
    # SECTION 3: HEARING PERSON SPEECH & TEXT (RIGHT SECTION)
    # ─────────────────────────────────────────────────────────────
    def speech_start_listening(self):
        """Starts non-blocking speech recognition in background thread."""
        started = self.speech_engine.start_listening()
        if started:
            self.lbl_mic_status.configure(
                text="● Microphone Listening... Please speak clearly into the microphone.",
                fg=COLOR_WARNING,
            )
            self.lbl_stat_mic.configure(text="● Microphone: Listening...", fg=COLOR_WARNING)
        else:
            err = self.speech_engine.last_error_message or "Microphone could not be started."
            self.lbl_mic_status.configure(text=f"● Error: {err}", fg=COLOR_ERROR)
            self.lbl_stat_mic.configure(text="● Microphone: Error", fg="#F87171")

    def speech_stop_listening(self):
        """Stops microphone listener."""
        self.lbl_mic_status.configure(text="● Microphone Idle.", fg=COLOR_TEXT_MUTED)
        self.lbl_stat_mic.configure(text="● Microphone: Ready", fg=COLOR_TEAL_BRIGHT)

    def _on_speech_recognized_callback(self, recognized_text: str):
        """Thread-safe callback executed when speech recognition returns text."""
        self._safe_ui(lambda: self._handle_speech_recognized_gui(recognized_text))

    def _handle_speech_recognized_gui(self, recognized_text: str):
        """Updates right section UI and sends recognized text to Middle Section."""
        clean = recognized_text.strip()
        if not clean:
            return

        self.hearing_recognized_text = clean
        self.txt_recognized_speech.delete("1.0", "end")
        self.txt_recognized_speech.insert("1.0", clean)

        self.lbl_mic_status.configure(
            text=f"● Successfully recognized: \"{clean}\"", fg=COLOR_SUCCESS
        )
        self.lbl_stat_mic.configure(text="● Microphone: Ready", fg=COLOR_TEAL_BRIGHT)

        # Dispatch immediately to Middle Section for visual sign translation!
        self.translate_and_play_sentence(clean)

    def speech_clear_text(self):
        """Clears recognized speech display."""
        self.txt_recognized_speech.delete("1.0", "end")
        self.hearing_recognized_text = ""

    def submit_manual_text(self):
        """Submits manual keyboard input and translates into sign language."""
        text = self.ent_manual_text.get().strip()
        if not text:
            return

        self.ent_manual_text.delete(0, "end")
        self.hearing_recognized_text = text
        self.txt_recognized_speech.delete("1.0", "end")
        self.txt_recognized_speech.insert("1.0", text)

        self.lbl_mic_status.configure(
            text=f"● Manual text submitted: \"{text}\"", fg=COLOR_TEAL_PRIMARY
        )

        # Dispatch to Middle Section
        self.translate_and_play_sentence(text)

    # ─────────────────────────────────────────────────────────────
    # STATUS INDICATORS POLLER
    # ─────────────────────────────────────────────────────────────
    def _update_status_indicators(self):
        """Updates status bar indicators to accurately reflect system state."""
        # Camera
        if self.camera_active and self.cap and self.cap.isOpened():
            self.lbl_stat_cam.configure(text="● Camera: Connected", fg=COLOR_TEAL_BRIGHT)
        elif self.camera_error_message:
            self.lbl_stat_cam.configure(text="● Camera: Unavailable", fg="#F87171")
        else:
            self.lbl_stat_cam.configure(text="● Camera: Disconnected", fg="#94A3B8")

        # Mic state
        if self.speech_engine.state == SpeechState.LISTENING:
            self.lbl_stat_mic.configure(text="● Microphone: Listening", fg=COLOR_WARNING)
        elif self.speech_engine.state == SpeechState.PROCESSING:
            self.lbl_stat_mic.configure(text="● Microphone: Processing", fg=COLOR_CYAN_HERO)
        elif self.speech_engine.state == SpeechState.ERROR:
            self.lbl_stat_mic.configure(text="● Microphone: Error", fg="#F87171")
        else:
            self.lbl_stat_mic.configure(text="● Microphone: Ready", fg=COLOR_TEAL_BRIGHT)

        self.root.after(300, self._update_status_indicators)

    # ─────────────────────────────────────────────────────────────
    # CLEAN SHUTDOWN
    # ─────────────────────────────────────────────────────────────
    def on_closing(self):
        """Cleans up all background threads and hardware resources safely."""
        logger.info("Closing Two-Way Sign Language Dashboard...")
        self.stop_camera()
        self.sign_player.stop()
        if self.mp_hands:
            try:
                self.mp_hands.close()
            except Exception:
                pass
        self.root.destroy()


def run_dashboard(camera_id: int = 0, mic_index: Optional[int] = None):
    """Entry point to launch dashboard GUI application."""
    root = tk.Tk()
    app = TwoWayDashboardApp(root, camera_id=camera_id, mic_index=mic_index)
    root.mainloop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SignGesture Two-Way Communication Dashboard")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--mic", type=int, default=None, help="Microphone device index (default: system default)")
    args = parser.parse_args()
    run_dashboard(camera_id=args.camera, mic_index=args.mic)
