"""
sign_vocabulary/manager.py
==========================
Vocabulary Manager for Stage 5: Hearing-person-to-sign-language communication.

Responsible for:
  - Loading and persisting the configurable sign vocabulary mapping.
  - Multi-word phrase tokenization and greedy sentence parsing.
  - Generating verified sign video playback sequences.
  - Fallback logic: If a word is unavailable, NEVER invents a fake sign;
    instead dispatches an explicitly labelled alphabet fingerspelling sequence.
  - Verification of video file existence on disk with clear error flags.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ALPHABET_DIR,
    MAPPING_FILE,
    VIDEOS_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class SignPlayItem:
    """Represents a single playable item in a sign communication sequence."""
    item_type: str        # 'VERIFIED_VIDEO' | 'FALLBACK_FINGERSPELL' | 'UNAVAILABLE_NOTICE' | 'VIDEO_MISSING'
    word: str             # The original word or phrase (e.g. 'HELLO', 'ALICE')
    label: str            # Display badge text
    video_path: Optional[Path] = None
    letter: Optional[str] = None
    letter_index: int = 1
    letter_total: int = 1
    image_path: Optional[Path] = None
    description: str = ""
    error_message: Optional[str] = None
    is_verified: bool = False


class VocabularyManager:
    """
    Manages the sign language vocabulary mapping, video directory validation,
    and sequence translation.
    """

    def __init__(self, mapping_file: Path = MAPPING_FILE, videos_dir: Path = VIDEOS_DIR, alphabet_dir: Path = ALPHABET_DIR):
        self.mapping_file = Path(mapping_file)
        self.videos_dir = Path(videos_dir)
        self.alphabet_dir = Path(alphabet_dir)

        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.alphabet_dir.mkdir(parents=True, exist_ok=True)

        self.vocabulary: Dict[str, Dict[str, Any]] = {}
        self.allow_fingerspelling_fallback: bool = True
        self.playback_fps: int = 25
        self.letter_hold_sec: float = 1.0

        self.load_mapping()

    def load_mapping(self) -> bool:
        """Loads or reloads vocabulary.json from disk."""
        if not self.mapping_file.exists():
            logger.warning("Mapping file %s does not exist.", self.mapping_file)
            return False

        try:
            data = json.loads(self.mapping_file.read_text(encoding="utf-8"))
            cfg = data.get("config", {})
            self.allow_fingerspelling_fallback = cfg.get("allow_fingerspelling_fallback", True)
            self.playback_fps = cfg.get("playback_fps", 25)
            self.letter_hold_sec = cfg.get("letter_hold_sec", 1.0)

            raw_vocab = data.get("vocabulary", {})
            # Store lowercase keys for fast case-insensitive lookup
            self.vocabulary = {k.strip().lower(): v for k, v in raw_vocab.items()}
            logger.info("Loaded %d sign vocabulary entries from %s", len(self.vocabulary), self.mapping_file.name)
            return True
        except Exception as e:
            logger.error("Failed to parse %s: %s", self.mapping_file, e)
            return False

    def save_mapping(self) -> bool:
        """Saves current vocabulary mapping to vocabulary.json."""
        try:
            data = {
                "config": {
                    "videos_dir": str(self.videos_dir.relative_to(self.mapping_file.parent) if self.videos_dir.is_relative_to(self.mapping_file.parent) else self.videos_dir),
                    "alphabet_dir": str(self.alphabet_dir.relative_to(self.mapping_file.parent) if self.alphabet_dir.is_relative_to(self.mapping_file.parent) else self.alphabet_dir),
                    "allow_fingerspelling_fallback": self.allow_fingerspelling_fallback,
                    "playback_fps": self.playback_fps,
                    "letter_hold_sec": self.letter_hold_sec,
                },
                "vocabulary": self.vocabulary,
            }
            self.mapping_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info("Saved vocabulary mapping to %s", self.mapping_file)
            return True
        except Exception as e:
            logger.error("Failed to save vocabulary: %s", e)
            return False

    def register_sign(
        self,
        word_or_phrase: str,
        video_filename: str,
        label: Optional[str] = None,
        category: str = "general",
        description: str = "",
        verified: bool = True,
    ) -> bool:
        """Adds or updates a verified sign video in the vocabulary mapping."""
        key = word_or_phrase.strip().lower()
        if not key:
            return False

        self.vocabulary[key] = {
            "video": video_filename,
            "label": label or word_or_phrase.upper(),
            "category": category,
            "verified": verified,
            "description": description,
        }
        return self.save_mapping()

    def parse_sentence_to_sequence(self, text: str) -> List[SignPlayItem]:
        """
        Translates a spoken or typed sentence into a sequence of playable sign items.

        - Supports greedy multi-word phrases (e.g. 'thank you', 'good morning').
        - Dispatches verified sign videos if present.
        - If a word is unavailable, never invents a sign; dispatches an explicitly
          labelled alphabet fingerspelling sequence.
        """
        clean_text = text.strip()
        if not clean_text:
            return []

        # Tokenize words, removing punctuation
        raw_words = re.findall(r"[A-Za-z0-9']+", clean_text.lower())
        if not raw_words:
            return []

        # Find multi-word phrase keys in vocabulary sorted by descending length (greedy match)
        phrase_keys = sorted(
            [k for k in self.vocabulary.keys() if " " in k],
            key=lambda x: len(x.split()),
            reverse=True,
        )

        sequence: List[SignPlayItem] = []
        i = 0
        n = len(raw_words)

        while i < n:
            matched_phrase = False
            # Try multi-word phrases first
            for phrase in phrase_keys:
                phrase_words = phrase.split()
                p_len = len(phrase_words)
                if i + p_len <= n and raw_words[i : i + p_len] == phrase_words:
                    vocab_entry = self.vocabulary[phrase]
                    item = self._create_item_from_vocab(phrase, vocab_entry)
                    sequence.append(item)
                    i += p_len
                    matched_phrase = True
                    break

            if matched_phrase:
                continue

            # Single word lookup
            word = raw_words[i]
            if word in self.vocabulary:
                vocab_entry = self.vocabulary[word]
                item = self._create_item_from_vocab(word, vocab_entry)
                sequence.append(item)
            else:
                # Word is NOT in vocabulary: Fallback logic
                fallback_items = self._create_fallback_items(word)
                sequence.extend(fallback_items)

            i += 1

        return sequence

    def _create_item_from_vocab(self, term: str, entry: Dict[str, Any]) -> SignPlayItem:
        """Creates a verified video item or flags a missing video error."""
        video_filename = entry.get("video", "")
        video_path = self.videos_dir / video_filename
        label = entry.get("label", term.upper())
        desc = entry.get("description", "")
        verified = entry.get("verified", True)

        if video_path.exists():
            return SignPlayItem(
                item_type="VERIFIED_VIDEO",
                word=term.upper(),
                label=f"VERIFIED SIGN: {label}",
                video_path=video_path,
                description=desc,
                is_verified=verified,
            )
        else:
            # Video file registered in mapping but missing on disk
            return SignPlayItem(
                item_type="VIDEO_MISSING",
                word=term.upper(),
                label=f"[VIDEO MISSING: '{video_filename}']",
                error_message=f"Verified sign video '{video_filename}' not found in {self.videos_dir.name}/.",
                description=desc,
                is_verified=False,
            )

    def _create_fallback_items(self, word: str) -> List[SignPlayItem]:
        """
        Creates explicitly labelled fingerspelling fallback items for an unknown word.
        Requirement 7: Never invent a sign.
        Requirement 8: Explicitly labelled fingerspelling fallback.
        """
        upper_word = word.upper()
        clean_alpha = [ch for ch in upper_word if ch.isalpha()]

        if not self.allow_fingerspelling_fallback or not clean_alpha:
            return [
                SignPlayItem(
                    item_type="UNAVAILABLE_NOTICE",
                    word=upper_word,
                    label=f"[UNAVAILABLE: '{upper_word}']",
                    error_message=f"Word '{upper_word}' is not in verified sign library.",
                    description="No verified sign available.",
                )
            ]

        # Generate letter-by-letter fingerspelling sequence
        items = []
        total_letters = len(clean_alpha)
        for idx, letter in enumerate(clean_alpha, start=1):
            letter_img = self.alphabet_dir / f"{letter}.jpg"
            items.append(
                SignPlayItem(
                    item_type="FALLBACK_FINGERSPELL",
                    word=upper_word,
                    label=f"[FALLBACK: FINGERSPELLING: {letter}]",
                    letter=letter,
                    letter_index=idx,
                    letter_total=total_letters,
                    image_path=letter_img if letter_img.exists() else None,
                    description=f"Fingerspelling letter '{letter}' ({idx}/{total_letters}) for word '{upper_word}'.",
                    is_verified=False,
                )
            )
        return items
