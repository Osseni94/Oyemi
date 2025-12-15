"""
Oyemi - Offline Semantic Numeric Lexicon
=========================================

A deterministic, high-performance semantic encoding library.
WordNet is used ONLY at build time - runtime has ZERO NLP dependencies.

Usage:
    from Oyemi import encode, Encoder, semantic_similarity

    # Simple encoding
    codes = encode("run")
    # ["4023-0001-2-0-0", "1021-0009-1-0-0"]

    # With encoder instance
    enc = Encoder()
    codes = enc.encode("happy")
    parsed = enc.encode_parsed("happy")

    # Semantic similarity
    sim = semantic_similarity("happy", "joyful")
    # 0.85

Code Format: HHHH-LLLLL-P-A-V
    HHHH = Semantic superclass (hypernym category, 4 digits)
    LLLLL = Local synset ID (5 digits)
    P = Part of speech (1=noun, 2=verb, 3=adj, 4=adv)
    A = Abstractness (0=concrete, 1=mixed, 2=abstract)
    V = Valence (0=neutral, 1=positive, 2=negative)
"""

__version__ = "3.0.0"
__author__ = "Kaossara Osseni"

# Core encoder
from .encoder import (
    Encoder,
    SemanticCode,
    encode,
)

# Convenience function for synonyms
def find_synonyms(
    word: str,
    limit: int = 20,
    pos_lock: bool = True,
    abstractness_lock: bool = True,
    return_weighted: bool = False
):
    """
    Find TRUE synonyms for a word using synset ID matching.

    Args:
        word: The word to find synonyms for
        limit: Maximum synonyms to return
        pos_lock: Only return synonyms with same POS (default: True)
        abstractness_lock: Don't mix abstract/concrete (default: True)
        return_weighted: If True, return list of (word, weight) tuples
    """
    return Encoder().find_synonyms(word, limit, pos_lock, abstractness_lock, return_weighted)

# Distance/similarity functions
from .distance import (
    code_distance,
    word_distance,
    semantic_similarity,
    find_similar,
    cluster_by_superclass,
    DistanceResult,
)

# Storage
from .storage import (
    LexiconStorage,
    get_storage,
)

# Exceptions
from .exceptions import (
    OyemiError,
    UnknownWordError,
    LexiconNotFoundError,
    InvalidCodeError,
)

# Convenience functions for antonyms
def are_antonyms(word1: str, word2: str) -> bool:
    """Check if two words are antonyms."""
    return Encoder().are_antonyms(word1, word2)

def get_antonyms(word: str):
    """Get all antonyms for a word."""
    return Encoder().get_antonyms(word)

__all__ = [
    # Version
    "__version__",

    # Core
    "Encoder",
    "SemanticCode",
    "encode",
    "find_synonyms",
    "are_antonyms",
    "get_antonyms",

    # Distance
    "code_distance",
    "word_distance",
    "semantic_similarity",
    "find_similar",
    "cluster_by_superclass",
    "DistanceResult",

    # Storage
    "LexiconStorage",
    "get_storage",

    # Exceptions
    "OyemiError",
    "UnknownWordError",
    "LexiconNotFoundError",
    "InvalidCodeError",
]
