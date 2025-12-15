<img src="Oyemi_Logo_white_BG.png" alt="Oyemi Logo" width="300" align="right"/>

# Oyemi

<a href="https://github.com/Osseni94/Oyemi/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
<a href="https://pypi.org/project/oyemi/"><img src="https://img.shields.io/pypi/v/oyemi.svg" alt="PyPI"></a>
<a href="https://pypi.org/project/oyemi/"><img src="https://img.shields.io/pypi/pyversions/oyemi.svg" alt="Python"></a>

**Offline Semantic Numeric Lexicon**

Oyemi is a deterministic, high-performance semantic encoding library. It maps words to numeric codes that capture semantic meaning, enabling fast similarity calculations and synonym discovery without neural networks.



## Key Features

- **Zero Runtime Dependencies** - No WordNet, NLTK, or ML models needed at runtime
- **Deterministic Codes** - Same word always produces same codes
- **Fast Lookups** - SQLite with memory mapping (~0.01ms per lookup)
- **TRUE Synonym Finder** - Find synonyms using WordNet synset matching
- **Antonym Detection** - Identify antonyms (happy/unhappy, good/bad) with low similarity
- **Semantic Distance** - Calculate word similarity using codes
- **Sentiment/Valence** - Built-in positive/negative classification (SentiWordNet + antonym inference)
- **Lemma Fallback** - Automatically handles word variations (running -> run)
- **Polysemy Support** - Multiple codes for words with multiple meanings

## Installation

```bash
pip install oyemi
```

## Quick Start

```python
from Oyemi import encode, semantic_similarity, find_synonyms, Encoder

# Simple encoding
codes = encode("happy")
# ['0122-00042-3-2-1']

# Check similarity
sim = semantic_similarity("happy", "joyful")
# 0.85

# Find TRUE synonyms
synonyms = find_synonyms("fear")
# ['awe', 'dread', 'fright', 'concern']

# Full encoder with parsed codes
enc = Encoder()
parsed = enc.encode_parsed("worried")
for code in parsed:
    print(f"{code.raw}: {code.pos_name}, {code.valence_name}")
    # 3999-04518-3-1-2: adjective, negative
```

## Code Format

Codes follow the format: `HHHH-LLLLL-P-A-V`

| Component | Meaning | Values |
|-----------|---------|--------|
| HHHH | Semantic superclass | 0001-9999 (100+ categories) |
| LLLLL | Synset ID | 00001-99999 |
| P | Part of speech | 1=noun, 2=verb, 3=adj, 4=adv |
| A | Abstractness | 0=concrete, 1=mixed, 2=abstract |
| V | Valence | 0=neutral, 1=positive, 2=negative |

### Superclass Categories

Oyemi includes 100+ semantic categories for precise classification:

| Range | Domain | Examples |
|-------|--------|----------|
| 0100-0199 | Emotions | fear, joy, anger, sadness |
| 0200-0299 | Work/Business | job, salary, manager, career |
| 0300-0399 | Communication | speak, write, message |
| 0400-0499 | Cognition | think, know, believe |
| 0500-0599 | Social | family, friend, group |
| 1000-1999 | Physical/Concrete | object, place, body |
| 2000-2999 | Actions | move, create, change |
| 3000-3999 | Properties | size, color, quality |

## API Reference

### Encoding

```python
from Oyemi import encode, Encoder

# Simple function
codes = encode("word")  # Returns List[str]

# Full encoder
enc = Encoder()
codes = enc.encode("word")              # List[str]
parsed = enc.encode_parsed("word")      # List[SemanticCode]
primary = enc.get_primary_code("word")  # str
exists = enc.contains("word")           # bool
batch = enc.encode_batch(["a", "b"])    # Dict[str, List[str]]
```

### Synonym Discovery

```python
from Oyemi import find_synonyms, Encoder

# Simple usage
synonyms = find_synonyms("fear")
# ['awe', 'dread', 'fright', 'concern']

# With filters (default: all enabled)
enc = Encoder()
synonyms = enc.find_synonyms(
    "fear",
    limit=10,
    pos_lock=True,           # Only same part-of-speech
    abstractness_lock=True,  # Don't mix abstract/concrete
    return_weighted=False    # Return list of words
)

# Get weighted synonyms (for ranking)
weighted = enc.find_synonyms("fear", return_weighted=True)
# [('dread', 1.0), ('fright', 1.0), ('awe', 0.5)]
# Weight 1.0 = same superclass, 0.5 = different superclass
```

**How it works:** Words with the same `HHHH-LLLLL` (superclass + synset ID) are TRUE synonyms - they come from the same WordNet synset.

### Similarity

```python
from Oyemi import semantic_similarity, word_distance, find_similar

# Similarity (0-1, higher = more similar)
sim = semantic_similarity("cat", "dog")

# Distance with details
dist, result = word_distance("cat", "dog")
print(result.shared_superclass)  # True
print(result.same_pos)           # True

# Find similar words from candidates
similar = find_similar("happy", ["sad", "joyful", "angry", "content"])
# [("joyful", 0.85), ("content", 0.72), ...]
```

### Clustering

```python
from Oyemi import cluster_by_superclass

words = ["dog", "cat", "run", "walk", "happy", "sad"]
clusters = cluster_by_superclass(words)
# {'0011': ['dog', 'cat'], '2002': ['run', 'walk'], ...}
```

### Sentiment/Valence

```python
from Oyemi import Encoder

enc = Encoder()

# Check word valence
for word in ["happy", "sad", "worried", "excellent", "terrible"]:
    parsed = enc.encode_parsed(word)
    if parsed:
        valence = parsed[0].valence_name
        print(f"{word}: {valence}")

# Output:
# happy: positive
# sad: negative
# worried: negative
# excellent: positive
# terrible: negative
```

### SemanticCode Object

```python
from Oyemi import Encoder

enc = Encoder()
codes = enc.encode_parsed("run")

for code in codes:
    print(code.raw)              # "2001-00042-2-1-0"
    print(code.superclass)       # "2001"
    print(code.synset_id)        # "00042"
    print(code.pos)              # 2
    print(code.pos_name)         # "verb"
    print(code.abstractness)     # 1
    print(code.abstractness_name)# "mixed"
    print(code.valence)          # 0
    print(code.valence_name)     # "neutral"
```

### Exceptions

```python
from Oyemi import (
    OyemiError,           # Base exception
    UnknownWordError,     # Word not in lexicon
    LexiconNotFoundError, # Database file missing
    InvalidCodeError,     # Malformed code string
)

try:
    codes = encode("xyznotaword")
except UnknownWordError as e:
    print(f"Unknown: {e.word}")
```

## Building the Lexicon

The lexicon database is pre-built and included. To rebuild from WordNet:

```bash
# Install build dependencies
pip install oyemi[build]

# Download WordNet and SentiWordNet
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('sentiwordnet')"

# Build lexicon
python tools/build_lexicon.py

# Validate
python tools/validate_lexicon.py
```

## Use Cases

- **Taxonomy Expansion** - Expand keyword lists with true synonyms
- **Fast Text Similarity** - Compare documents without embeddings
- **Sentiment Analysis** - Quick valence detection
- **Semantic Clustering** - Group words by meaning
- **Feature Engineering** - Convert words to numeric features
- **Offline NLP** - No API calls or model downloads
- **Deterministic Pipelines** - Reproducible results

### Example: Expand Sentiment Keywords

```python
from Oyemi import Encoder

enc = Encoder()

# Original keywords
negative_words = ["fear", "worried", "anxious", "stressed"]

# Expand with synonyms
expanded = set(negative_words)
for word in negative_words:
    synonyms = enc.find_synonyms(word, limit=5)
    expanded.update(synonyms)

print(f"Expanded: {len(negative_words)} -> {len(expanded)} words")
# Expanded: 4 -> 20 words
```

## Performance

| Operation | Time |
|-----------|------|
| Single lookup | ~0.01ms |
| Batch (1000 words) | ~5ms |
| Similarity | ~0.02ms |
| Find synonyms | ~0.1ms |

## Versioning

- Codes **never change** once released (semantic stability)
- New words get new codes in minor versions
- Schema changes require major version bump

## Author

**Kaossara Osseni**

## License

MIT License
