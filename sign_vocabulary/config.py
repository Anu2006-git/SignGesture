"""
sign_vocabulary/config.py
========================
Configuration settings for Stage 5: Hearing-person-to-sign-language communication.
Defines paths for verified sign videos, vocabulary mapping, alphabet library,
and playback settings.
"""

from pathlib import Path

# Base directory: C:\SignGesture
BASE_DIR = Path(__file__).resolve().parent.parent

# Sign Vocabulary directories
VOCABULARY_DIR = BASE_DIR / "sign_vocabulary"
VIDEOS_DIR     = VOCABULARY_DIR / "videos"
ALPHABET_DIR   = VOCABULARY_DIR / "alphabet"
MAPPING_FILE   = VOCABULARY_DIR / "vocabulary.json"

# Ensure directories exist
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
ALPHABET_DIR.mkdir(parents=True, exist_ok=True)

# Playback parameters
DEFAULT_PLAYBACK_FPS = 25
FINGERSPELL_LETTER_HOLD_SEC = 1.0     # Time to show each fingerspelled letter card
FALLBACK_CARD_HOLD_SEC = 1.5          # Time to display unavailable word notice

# Audio & Speech settings
DEFAULT_MIC_INDEX = None              # None for default system mic
SPEECH_ENERGY_THRESHOLD = 300
SPEECH_RECORD_TIMEOUT = 5.0           # Seconds of silence before stopping recording
SPEECH_PHRASE_TIME_LIMIT = 8.0        # Max duration for spoken phrase
