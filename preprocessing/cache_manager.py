"""
preprocessing/cache_manager.py
================================
Incremental processing cache.

Stores a JSON map of  { "md5_hash" : "processed_stem" }
so re-running the preprocessor only processes NEW or CHANGED images.

When you add new images:
  → new hashes are not in the cache → they are processed
  → unchanged images hit the cache  → they are skipped instantly

Set  INCREMENTAL_MODE = False  in config.py to force a full reprocess.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Set

from .config import CACHE_FILE, INCREMENTAL_MODE

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages a JSON file that tracks which source images have been
    processed already (keyed by MD5 hash of file content).

    Usage
    -----
    >>> cache = CacheManager()
    >>> cache.load()
    >>> if not cache.is_processed(img_path):
    ...     # process the image
    ...     cache.mark_processed(img_path, output_stem="A_img001")
    >>> cache.save()
    """

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file: Path     = Path(cache_file)
        self._store: Dict[str, str] = {}      # hash → output stem
        self.enabled: bool          = INCREMENTAL_MODE

    # ── Lifecycle ──────────────────────────────────────────────
    def load(self) -> None:
        """Load the cache from disk. Creates empty cache if missing."""
        if not self.enabled:
            logger.info("Incremental cache disabled — full reprocess mode.")
            self._store = {}
            return

        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_file.exists():
            try:
                self._store = json.loads(self.cache_file.read_text(encoding="utf-8"))
                logger.info("Cache loaded — %d entries.", len(self._store))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Cache file corrupted (%s). Starting fresh.", exc)
                self._store = {}
        else:
            logger.info("No cache file found — starting fresh.")
            self._store = {}

    def save(self) -> None:
        """Persist the current cache to disk."""
        if not self.enabled:
            return
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(
            json.dumps(self._store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Cache saved — %d entries.", len(self._store))

    def clear(self) -> None:
        """Wipe the cache (forces full reprocess on next run)."""
        self._store = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cache cleared.")

    # ── Query / Update ─────────────────────────────────────────
    def is_processed(self, path: Path) -> bool:
        """Return True if this exact file content has been processed before."""
        if not self.enabled:
            return False
        h = self._hash_file(path)
        return h in self._store

    def mark_processed(self, path: Path, output_stem: str) -> None:
        """Record that this file has been processed."""
        h = self._hash_file(path)
        self._store[h] = output_stem

    def get_output_stem(self, path: Path) -> Optional[str]:
        """Return the output stem for a previously processed file, or None."""
        h = self._hash_file(path)
        return self._store.get(h)

    @property
    def total_cached(self) -> int:
        return len(self._store)

    # ── Helpers ────────────────────────────────────────────────
    @staticmethod
    def _hash_file(path: Path) -> str:
        hasher = hashlib.md5()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65_536), b""):
                    hasher.update(chunk)
        except OSError:
            # If the file can't be read, use the path as fallback key
            hasher.update(str(path).encode())
        return hasher.hexdigest()
