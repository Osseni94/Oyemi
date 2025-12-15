"""
Oyemi Storage
=============
SQLite-based storage for the semantic lexicon.
Optimized for fast, read-only lookups with memory mapping.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional
from functools import lru_cache

from .exceptions import LexiconNotFoundError


class LexiconStorage:
    """
    High-performance SQLite storage for semantic codes.

    Uses:
    - Read-only mode
    - Memory-mapped I/O
    - LRU caching for hot words
    - Optimized pragmas
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the lexicon storage.

        Args:
            db_path: Path to lexicon.db. If None, uses bundled database.
        """
        if db_path is None:
            # Use bundled database
            db_path = Path(__file__).parent.parent / "data" / "lexicon.db"
        else:
            db_path = Path(db_path)

        if not db_path.exists():
            raise LexiconNotFoundError(str(db_path))

        self._db_path = str(db_path)
        self._connection: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._connection is None:
            # Open in read-only mode with URI
            uri = f"file:{self._db_path}?mode=ro"
            self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)

            # Apply performance pragmas
            cursor = self._connection.cursor()
            cursor.execute("PRAGMA journal_mode = OFF;")
            cursor.execute("PRAGMA synchronous = OFF;")
            cursor.execute("PRAGMA temp_store = MEMORY;")
            cursor.execute("PRAGMA mmap_size = 300000000;")  # 300MB memory map
            cursor.execute("PRAGMA cache_size = -64000;")    # 64MB cache
            cursor.close()

        return self._connection

    @lru_cache(maxsize=10000)
    def lookup(self, word: str) -> List[str]:
        """
        Look up all semantic codes for a word.
        Uses lemma fallback if direct lookup fails.

        Args:
            word: The word to look up (case-insensitive)

        Returns:
            List of semantic codes (may be multiple for polysemous words)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        word_lower = word.lower()

        # Direct lookup (case-insensitive)
        cursor.execute(
            "SELECT code FROM lexicon WHERE word = ? COLLATE NOCASE",
            (word_lower,)
        )
        results = [row[0] for row in cursor.fetchall()]

        # IMPROVEMENT #3: Lemma fallback if not found
        if not results:
            # Check if there's a lemma mapping for this word
            cursor.execute(
                "SELECT lemma FROM lemmas WHERE word = ?",
                (word_lower,)
            )
            lemma_row = cursor.fetchone()

            if lemma_row:
                # Look up the lemma instead
                cursor.execute(
                    "SELECT code FROM lexicon WHERE word = ? COLLATE NOCASE",
                    (lemma_row[0],)
                )
                results = [row[0] for row in cursor.fetchall()]

        cursor.close()
        return results

    def contains(self, word: str) -> bool:
        """Check if a word exists in the lexicon."""
        return len(self.lookup(word)) > 0

    def find_synonyms(
        self,
        word: str,
        limit: int = 20,
        pos_lock: bool = True,
        abstractness_lock: bool = True,
        return_weighted: bool = False
    ) -> List:
        """
        Find TRUE synonyms using synset ID (HHHH-LLLLL) with smart filtering.

        Words are synonyms if they share the same synset ID,
        meaning they come from the same WordNet synset.

        Args:
            word: The word to find synonyms for
            limit: Maximum number of synonyms to return
            pos_lock: Only return synonyms with same POS (mandatory filter)
            abstractness_lock: Don't mix abstract/concrete (recommended)
            return_weighted: If True, return list of (word, weight) tuples

        Returns:
            List of synonym words, or list of (word, weight) tuples if return_weighted
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        word_lower = word.lower()

        # Get the word's codes with full info
        cursor.execute(
            "SELECT code FROM lexicon WHERE word = ? COLLATE NOCASE",
            (word_lower,)
        )
        seed_codes = [row[0] for row in cursor.fetchall()]

        if not seed_codes:
            cursor.close()
            return []

        # Parse seed word's properties from primary code
        seed_parts = seed_codes[0].split('-')
        seed_superclass = seed_parts[0]
        seed_pos = seed_parts[2]
        seed_abstractness = seed_parts[3]

        # Collect all seed synset prefixes and superclasses
        seed_synsets = set()
        seed_superclasses = set()
        for code in seed_codes:
            parts = code.split('-')
            seed_synsets.add(parts[0] + '-' + parts[1])
            seed_superclasses.add(parts[0])

        # Find candidate synonyms
        candidates = {}  # word -> (code, weight)

        for code in seed_codes:
            parts = code.split('-')
            synset_prefix = parts[0] + '-' + parts[1]

            # Find all words with this exact synset prefix
            cursor.execute(
                "SELECT word, code FROM lexicon WHERE code LIKE ? AND word != ? COLLATE NOCASE",
                (synset_prefix + '%', word_lower)
            )

            for row in cursor.fetchall():
                syn_word = row[0]
                syn_code = row[1]
                syn_parts = syn_code.split('-')

                # Filter 1: POS lock (mandatory)
                if pos_lock and syn_parts[2] != seed_pos:
                    continue

                # Filter 2: Abstractness lock (recommended)
                if abstractness_lock and syn_parts[3] != seed_abstractness:
                    continue

                # Filter 3: Domain weight (soft filter)
                # Same superclass = 1.0, different superclass = 0.5
                weight = 1.0 if syn_parts[0] in seed_superclasses else 0.5

                # Keep best weight if word already seen
                if syn_word not in candidates or candidates[syn_word][1] < weight:
                    candidates[syn_word] = (syn_code, weight)

        cursor.close()

        if return_weighted:
            # Return (word, weight) sorted by weight desc, then length
            weighted = [(w, info[1]) for w, info in candidates.items()]
            weighted.sort(key=lambda x: (-x[1], len(x[0])))
            return weighted[:limit]
        else:
            # Sort by weight desc, then length (prefer shorter words)
            sorted_synonyms = sorted(
                candidates.keys(),
                key=lambda w: (-candidates[w][1], len(w))
            )
            return sorted_synonyms[:limit]

    def are_antonyms(self, word1: str, word2: str) -> bool:
        """
        Check if two words are antonyms according to WordNet.

        Args:
            word1: First word
            word2: Second word

        Returns:
            True if words are antonyms, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT 1 FROM antonyms WHERE word = ? AND antonym = ? LIMIT 1",
            (word1.lower(), word2.lower())
        )
        result = cursor.fetchone() is not None
        cursor.close()
        return result

    def get_antonyms(self, word: str) -> List[str]:
        """
        Get all antonyms for a word.

        Args:
            word: The word to find antonyms for

        Returns:
            List of antonym words
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT antonym FROM antonyms WHERE word = ?",
            (word.lower(),)
        )
        antonyms = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return antonyms

    def get_word_count(self) -> int:
        """Get total number of unique words in the lexicon."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT word) FROM lexicon")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def get_code_count(self) -> int:
        """Get total number of word-code mappings."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM lexicon")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def close(self):
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            self.lookup.cache_clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


# Global singleton for convenience
_default_storage: Optional[LexiconStorage] = None


def get_storage() -> LexiconStorage:
    """Get the default lexicon storage instance."""
    global _default_storage
    if _default_storage is None:
        _default_storage = LexiconStorage()
    return _default_storage
