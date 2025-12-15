"""
Oyemi Distance
==============
Semantic distance calculations between codes.
Uses the hierarchical structure of codes for similarity measurement.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from .encoder import SemanticCode, Encoder
from .storage import get_storage


@dataclass
class DistanceResult:
    """Result of a semantic distance calculation."""

    code1: str
    code2: str
    distance: float
    similarity: float
    shared_superclass: bool
    same_pos: bool

    def __str__(self) -> str:
        return f"Distance({self.code1}, {self.code2}) = {self.distance:.3f}"


def code_distance(code1: str, code2: str) -> DistanceResult:
    """
    Calculate semantic distance between two codes.

    Distance is based on:
    1. Superclass match (major semantic category)
    2. Synset ID proximity
    3. POS match
    4. Abstractness difference
    5. Valence difference

    Returns:
        DistanceResult with distance in [0, 1] range
        0 = identical, 1 = maximally different
    """
    sc1 = SemanticCode.parse(code1)
    sc2 = SemanticCode.parse(code2)

    # Weight factors
    W_SUPERCLASS = 0.35  # Semantic category
    W_SYNSET = 0.20      # Within-category distinction
    W_POS = 0.15         # Part of speech matters
    W_ABSTRACT = 0.10    # Abstractness
    W_VALENCE = 0.20     # Sentiment/valence (important for antonyms)

    # Calculate component distances

    # 1. Superclass distance (0 if same, 1 if different)
    superclass_dist = 0.0 if sc1.superclass == sc2.superclass else 1.0

    # 2. Synset ID distance (normalized on 5-digit range)
    try:
        id1, id2 = int(sc1.synset_id), int(sc2.synset_id)
        # Normalize by max possible distance in 5-digit range
        synset_dist = abs(id1 - id2) / 99999
        synset_dist = min(synset_dist, 1.0)
    except ValueError:
        synset_dist = 1.0 if sc1.synset_id != sc2.synset_id else 0.0

    # 3. POS distance (0 if same, 1 if different)
    pos_dist = 0.0 if sc1.pos == sc2.pos else 1.0

    # 4. Abstractness distance (normalized 0-2 range)
    abstract_dist = abs(sc1.abstractness - sc2.abstractness) / 2.0

    # 5. Valence distance (normalized 0-2 range)
    valence_dist = abs(sc1.valence - sc2.valence) / 2.0

    # Weighted sum
    distance = (
        W_SUPERCLASS * superclass_dist +
        W_SYNSET * synset_dist +
        W_POS * pos_dist +
        W_ABSTRACT * abstract_dist +
        W_VALENCE * valence_dist
    )

    return DistanceResult(
        code1=code1,
        code2=code2,
        distance=distance,
        similarity=1.0 - distance,
        shared_superclass=(superclass_dist == 0.0),
        same_pos=(pos_dist == 0.0)
    )


def word_distance(
    word1: str,
    word2: str,
    encoder: Optional[Encoder] = None,
    method: str = "min",
    check_antonyms: bool = True
) -> Tuple[float, DistanceResult]:
    """
    Calculate semantic distance between two words.

    For polysemous words, uses the specified method to aggregate:
    - "min": Minimum distance (closest senses)
    - "max": Maximum distance (furthest senses)
    - "avg": Average distance across all sense pairs

    Args:
        word1: First word
        word2: Second word
        encoder: Encoder instance (uses default if None)
        method: Aggregation method ("min", "max", "avg")
        check_antonyms: If True, check for antonym relationship (default: True)

    Returns:
        Tuple of (distance, best_result)
    """
    enc = encoder or Encoder()

    # Check for antonym relationship first
    if check_antonyms:
        storage = get_storage()
        if storage.are_antonyms(word1, word2):
            # Antonyms have maximum distance (0.9 - leave some room for truly unrelated words)
            codes1 = enc.encode(word1, raise_on_unknown=False)
            codes2 = enc.encode(word2, raise_on_unknown=False)
            if codes1 and codes2:
                result = DistanceResult(
                    code1=codes1[0],
                    code2=codes2[0],
                    distance=0.9,
                    similarity=0.1,
                    shared_superclass=True,  # Antonyms often share superclass
                    same_pos=True  # Antonyms typically have same POS
                )
                return 0.9, result

    codes1 = enc.encode(word1, raise_on_unknown=False)
    codes2 = enc.encode(word2, raise_on_unknown=False)

    if not codes1 or not codes2:
        # One or both words unknown
        return 1.0, None

    # Calculate all pairwise distances
    results = []
    for c1 in codes1:
        for c2 in codes2:
            results.append(code_distance(c1, c2))

    if method == "min":
        best = min(results, key=lambda r: r.distance)
        return best.distance, best
    elif method == "max":
        worst = max(results, key=lambda r: r.distance)
        return worst.distance, worst
    elif method == "avg":
        avg_dist = sum(r.distance for r in results) / len(results)
        # Return the result closest to average
        closest = min(results, key=lambda r: abs(r.distance - avg_dist))
        return avg_dist, closest
    else:
        raise ValueError(f"Unknown method: {method}")


def semantic_similarity(word1: str, word2: str, encoder: Optional[Encoder] = None) -> float:
    """
    Calculate semantic similarity between two words.

    Convenience function returning similarity in [0, 1] range.
    1 = identical meaning, 0 = completely unrelated.

    Args:
        word1: First word
        word2: Second word
        encoder: Encoder instance (uses default if None)

    Returns:
        Similarity score in [0, 1]
    """
    distance, _ = word_distance(word1, word2, encoder, method="min")
    return 1.0 - distance


def find_similar(
    word: str,
    candidates: List[str],
    encoder: Optional[Encoder] = None,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find most semantically similar words from a list of candidates.

    Args:
        word: Target word
        candidates: List of candidate words to compare
        encoder: Encoder instance (uses default if None)
        top_k: Number of top results to return

    Returns:
        List of (word, similarity) tuples, sorted by similarity descending
    """
    enc = encoder or Encoder()

    similarities = []
    for candidate in candidates:
        if candidate.lower() == word.lower():
            continue
        try:
            sim = semantic_similarity(word, candidate, enc)
            similarities.append((candidate, sim))
        except Exception:
            continue

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def cluster_by_superclass(words: List[str], encoder: Optional[Encoder] = None) -> dict:
    """
    Cluster words by their semantic superclass.

    Args:
        words: List of words to cluster
        encoder: Encoder instance (uses default if None)

    Returns:
        Dict mapping superclass codes to lists of words
    """
    enc = encoder or Encoder()
    clusters = {}

    for word in words:
        try:
            codes = enc.encode_parsed(word, raise_on_unknown=False)
            if codes:
                # Use primary sense
                superclass = codes[0].superclass
                if superclass not in clusters:
                    clusters[superclass] = []
                clusters[superclass].append(word)
        except Exception:
            continue

    return clusters


def get_synset_key(code: str) -> str:
    """Extract synset key (HHHH-LLLLL) from full code."""
    parts = code.split('-')
    return f"{parts[0]}-{parts[1]}"
