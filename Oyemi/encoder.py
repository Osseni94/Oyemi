"""
Oyemi Encoder
=============
Runtime semantic encoder - NO WordNet dependency.
Provides deterministic word-to-code mapping via SQLite lookups.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from .storage import LexiconStorage, get_storage
from .exceptions import UnknownWordError, InvalidCodeError


@dataclass
class SemanticCode:
    """
    Parsed semantic code with component access.

    Code format: HHHH-LLLLL-P-A-V
    - HHHH: Semantic superclass (hypernym path, 4 digits)
    - LLLLL: Local synset ID (5 digits)
    - P: Part of speech (1=noun, 2=verb, 3=adj, 4=adv)
    - A: Abstractness (0=concrete, 1=mixed, 2=abstract)
    - V: Valence (0=neutral, 1=positive, 2=negative)
    """

    raw: str
    superclass: str
    synset_id: str
    pos: int
    abstractness: int
    valence: int

    POS_NAMES = {1: "noun", 2: "verb", 3: "adjective", 4: "adverb"}
    ABSTRACTNESS_NAMES = {0: "concrete", 1: "mixed", 2: "abstract"}
    VALENCE_NAMES = {0: "neutral", 1: "positive", 2: "negative"}

    @classmethod
    def parse(cls, code: str) -> "SemanticCode":
        """Parse a code string into components."""
        parts = code.split("-")
        if len(parts) != 5:
            raise InvalidCodeError(code, "expected 5 components")

        try:
            return cls(
                raw=code,
                superclass=parts[0],
                synset_id=parts[1],
                pos=int(parts[2]),
                abstractness=int(parts[3]),
                valence=int(parts[4])
            )
        except ValueError as e:
            raise InvalidCodeError(code, str(e))

    @property
    def pos_name(self) -> str:
        """Human-readable part of speech."""
        return self.POS_NAMES.get(self.pos, "unknown")

    @property
    def abstractness_name(self) -> str:
        """Human-readable abstractness level."""
        return self.ABSTRACTNESS_NAMES.get(self.abstractness, "unknown")

    @property
    def valence_name(self) -> str:
        """Human-readable valence."""
        return self.VALENCE_NAMES.get(self.valence, "unknown")

    def shares_superclass(self, other: "SemanticCode") -> bool:
        """Check if two codes share the same semantic superclass."""
        return self.superclass == other.superclass

    def __str__(self) -> str:
        return self.raw


class Encoder:
    """
    Semantic word encoder.

    Maps words to deterministic numeric codes using a pre-built lexicon.
    No WordNet or NLTK required at runtime.
    """

    def __init__(self, storage: Optional[LexiconStorage] = None):
        """
        Initialize the encoder.

        Args:
            storage: Custom storage instance. If None, uses default.
        """
        self._storage = storage or get_storage()

    def encode(self, word: str, raise_on_unknown: bool = True) -> List[str]:
        """
        Encode a word to its semantic codes.

        Args:
            word: The word to encode
            raise_on_unknown: If True, raise UnknownWordError for unknown words

        Returns:
            List of semantic code strings (may be multiple for polysemous words)

        Raises:
            UnknownWordError: If word not in lexicon and raise_on_unknown=True
        """
        codes = self._storage.lookup(word)

        if not codes and raise_on_unknown:
            raise UnknownWordError(word)

        return codes

    def encode_parsed(self, word: str, raise_on_unknown: bool = True) -> List[SemanticCode]:
        """
        Encode a word and return parsed SemanticCode objects.

        Args:
            word: The word to encode
            raise_on_unknown: If True, raise UnknownWordError for unknown words

        Returns:
            List of SemanticCode objects
        """
        codes = self.encode(word, raise_on_unknown)
        return [SemanticCode.parse(c) for c in codes]

    def encode_batch(
        self,
        words: List[str],
        raise_on_unknown: bool = False
    ) -> dict[str, List[str]]:
        """
        Encode multiple words at once.

        Args:
            words: List of words to encode
            raise_on_unknown: If True, raise on first unknown word

        Returns:
            Dict mapping each word to its list of codes
        """
        result = {}
        for word in words:
            codes = self.encode(word, raise_on_unknown=raise_on_unknown)
            result[word] = codes
        return result

    def contains(self, word: str) -> bool:
        """Check if a word exists in the lexicon."""
        return self._storage.contains(word)

    def get_primary_code(self, word: str) -> str:
        """
        Get the first/primary semantic code for a word.

        For polysemous words, returns the most common sense.

        Args:
            word: The word to encode

        Returns:
            Primary semantic code string

        Raises:
            UnknownWordError: If word not in lexicon
        """
        codes = self.encode(word)
        return codes[0]

    def get_primary_parsed(self, word: str) -> SemanticCode:
        """Get the primary code as a parsed SemanticCode object."""
        return SemanticCode.parse(self.get_primary_code(word))

    @property
    def word_count(self) -> int:
        """Number of unique words in the lexicon."""
        return self._storage.get_word_count()

    @property
    def mapping_count(self) -> int:
        """Total number of word-code mappings."""
        return self._storage.get_code_count()

    def are_antonyms(self, word1: str, word2: str) -> bool:
        """
        Check if two words are antonyms.

        Args:
            word1: First word
            word2: Second word

        Returns:
            True if words are antonyms
        """
        return self._storage.are_antonyms(word1, word2)

    def get_antonyms(self, word: str) -> List[str]:
        """
        Get all antonyms for a word.

        Args:
            word: The word to find antonyms for

        Returns:
            List of antonym words
        """
        return self._storage.get_antonyms(word)

    def find_synonyms(
        self,
        word: str,
        limit: int = 20,
        pos_lock: bool = True,
        abstractness_lock: bool = True,
        return_weighted: bool = False
    ) -> List:
        """
        Find TRUE synonyms for a word with smart filtering.

        Uses synset ID (HHHH-LLLLL) to find words from the same
        WordNet synset - these are true synonyms.

        Filters:
        1. POS lock (mandatory): Only synonyms with same part of speech
        2. Abstractness lock: Don't mix abstract ↔ concrete words
        3. Domain weight: Synonyms from same superclass weighted higher

        Args:
            word: The word to find synonyms for
            limit: Maximum synonyms to return
            pos_lock: Only return synonyms with same POS (default: True)
            abstractness_lock: Don't mix abstract/concrete (default: True)
            return_weighted: If True, return list of (word, weight) tuples

        Returns:
            List of synonym words, or list of (word, weight) tuples

        Example:
            >>> enc.find_synonyms("fear")
            ['dread', 'fright', 'fearfulness', 'awe']
            >>> enc.find_synonyms("fired")
            ['dismissed', 'discharged', 'laid-off']
            >>> enc.find_synonyms("fear", return_weighted=True)
            [('dread', 1.0), ('fright', 1.0), ('awe', 0.5)]
        """
        return self._storage.find_synonyms(
            word, limit, pos_lock, abstractness_lock, return_weighted
        )


# Convenience function for simple usage
def encode(word: str) -> List[str]:
    """
    Encode a word to semantic codes.

    Convenience function using the default encoder.

    Args:
        word: The word to encode

    Returns:
        List of semantic code strings

    Raises:
        UnknownWordError: If word not in lexicon
    """
    return Encoder().encode(word)
