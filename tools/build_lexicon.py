"""
Oyemi Lexicon Builder v2.0
==========================
BUILD-TIME ONLY - Uses WordNet + SentiWordNet to generate the lexicon database.

IMPROVEMENTS:
1. SentiWordNet for accurate valence detection
2. Expanded superclass hierarchy (100+ categories)
3. Lemmatization support for word variants

Author: Kaossara Osseni
"""

import sqlite3
import sys
import io
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from collections import defaultdict
from typing import Dict, List, Tuple, Set

try:
    from nltk.corpus import wordnet as wn
    from nltk.corpus import sentiwordnet as swn
    from nltk.stem import WordNetLemmatizer
    from tqdm import tqdm
except ImportError:
    print("ERROR: Build-time dependencies not installed.")
    print("Run: pip install nltk tqdm")
    print("Then: python -c \"import nltk; nltk.download('wordnet'); nltk.download('sentiwordnet'); nltk.download('omw-1.4')\"")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "lexicon.db"

# Part of speech mapping
POS_MAP = {
    'n': 1,  # noun
    'v': 2,  # verb
    'a': 3,  # adjective
    's': 3,  # satellite adjective (treat as adjective)
    'r': 4,  # adverb
}

# =============================================================================
# IMPROVEMENT #2: EXPANDED SUPERCLASS HIERARCHY (100+ categories)
# =============================================================================

SUPERCLASS_ROOTS = {
    # =========================================================================
    # NOUNS - PHYSICAL ENTITIES (0001-0099)
    # =========================================================================
    'entity.n.01': '0001',
    'physical_entity.n.01': '0002',
    'thing.n.12': '0003',
    'object.n.01': '0004',
    'whole.n.02': '0005',
    'part.n.01': '0006',
    'artifact.n.01': '0007',
    'natural_object.n.01': '0008',

    # Living things
    'living_thing.n.01': '0010',
    'organism.n.01': '0011',
    'person.n.01': '0012',
    'human.n.01': '0013',
    'animal.n.01': '0014',
    'mammal.n.01': '0015',
    'bird.n.01': '0016',
    'fish.n.01': '0017',
    'insect.n.01': '0018',
    'plant.n.02': '0019',
    'tree.n.01': '0020',
    'flower.n.01': '0021',

    # Body & Health
    'body.n.01': '0025',
    'body_part.n.01': '0026',
    'organ.n.01': '0027',
    'disease.n.01': '0028',
    'illness.n.01': '0029',
    'symptom.n.01': '0030',

    # Physical substances
    'substance.n.01': '0035',
    'matter.n.03': '0036',
    'material.n.01': '0037',
    'food.n.01': '0038',
    'liquid.n.01': '0039',
    'solid.n.01': '0040',

    # Places & Structures
    'location.n.01': '0045',
    'region.n.01': '0046',
    'area.n.01': '0047',
    'place.n.02': '0048',
    'structure.n.01': '0050',
    'building.n.01': '0051',
    'room.n.01': '0052',
    'facility.n.01': '0053',

    # Objects & Tools
    'container.n.01': '0055',
    'vehicle.n.01': '0056',
    'clothing.n.01': '0057',
    'furnishing.n.01': '0058',
    'device.n.01': '0060',
    'tool.n.01': '0061',
    'machine.n.01': '0062',
    'instrument.n.01': '0063',
    'equipment.n.01': '0064',
    'computer.n.01': '0065',
    'weapon.n.01': '0066',

    # =========================================================================
    # NOUNS - ABSTRACT ENTITIES (0100-0199)
    # =========================================================================
    'abstraction.n.06': '0100',
    'abstract_entity.n.01': '0101',

    # Mental & Psychological
    'psychological_feature.n.01': '0105',
    'cognition.n.01': '0106',
    'knowledge.n.01': '0107',
    'belief.n.01': '0108',
    'idea.n.01': '0109',
    'concept.n.01': '0110',
    'thought.n.01': '0111',
    'plan.n.01': '0112',
    'intention.n.01': '0113',

    # Emotions & Feelings (IMPORTANT FOR SENTIMENT)
    'feeling.n.01': '0120',
    'emotion.n.01': '0121',
    'fear.n.01': '0122',
    'anger.n.01': '0123',
    'sadness.n.01': '0124',
    'happiness.n.01': '0125',
    'joy.n.01': '0126',
    'love.n.01': '0127',
    'hate.n.01': '0128',
    'anxiety.n.02': '0129',
    'worry.n.01': '0130',
    'hope.n.01': '0131',
    'despair.n.01': '0132',
    'surprise.n.01': '0133',
    'disgust.n.01': '0134',

    # States & Conditions
    'state.n.02': '0140',
    'condition.n.01': '0141',
    'situation.n.01': '0142',
    'status.n.01': '0143',
    'health.n.01': '0144',
    'stress.n.01': '0145',

    # Communication
    'communication.n.02': '0150',
    'message.n.02': '0151',
    'language.n.01': '0152',
    'word.n.01': '0153',
    'document.n.01': '0154',
    'speech.n.02': '0155',

    # Time & Events
    'time.n.05': '0160',
    'time_period.n.01': '0161',
    'event.n.01': '0162',
    'act.n.02': '0163',
    'activity.n.01': '0164',
    'process.n.06': '0165',
    'change.n.01': '0166',

    # Social & Groups
    'group.n.01': '0170',
    'social_group.n.01': '0171',
    'organization.n.01': '0172',
    'institution.n.01': '0173',
    'company.n.01': '0174',
    'team.n.01': '0175',
    'family.n.01': '0176',

    # Relations & Attributes
    'relation.n.01': '0180',
    'attribute.n.02': '0181',
    'quality.n.01': '0182',
    'quantity.n.01': '0183',
    'measure.n.02': '0184',
    'number.n.02': '0185',

    # =========================================================================
    # NOUNS - WORK & EMPLOYMENT (0200-0249) - NEW CATEGORY
    # =========================================================================
    'occupation.n.01': '0200',
    'job.n.02': '0201',
    'profession.n.01': '0202',
    'employment.n.01': '0203',
    'work.n.01': '0204',
    'career.n.01': '0205',
    'worker.n.01': '0206',
    'employee.n.01': '0207',
    'employer.n.01': '0208',

    # Management & Leadership
    'management.n.01': '0210',
    'manager.n.01': '0211',
    'supervisor.n.01': '0212',
    'executive.n.01': '0213',
    'leader.n.01': '0214',
    'director.n.01': '0215',
    'boss.n.01': '0216',

    # Compensation & Benefits
    'wage.n.01': '0220',
    'salary.n.01': '0221',
    'income.n.01': '0222',
    'payment.n.01': '0223',
    'bonus.n.01': '0224',
    'benefit.n.01': '0225',
    'compensation.n.01': '0226',

    # Employment Actions (LAYOFFS)
    'dismissal.n.01': '0230',
    'discharge.n.01': '0231',
    'termination.n.01': '0232',
    'layoff.n.01': '0233',
    'firing.n.01': '0234',
    'resignation.n.01': '0235',
    'retirement.n.01': '0236',
    'downsizing.n.01': '0237',

    # =========================================================================
    # NOUNS - MONEY & ECONOMICS (0250-0299)
    # =========================================================================
    'money.n.01': '0250',
    'currency.n.01': '0251',
    'wealth.n.01': '0252',
    'asset.n.01': '0253',
    'debt.n.01': '0254',
    'cost.n.01': '0255',
    'price.n.01': '0256',
    'profit.n.01': '0257',
    'loss.n.01': '0258',
    'budget.n.01': '0259',
    'investment.n.01': '0260',
    'stock.n.01': '0261',

    # =========================================================================
    # VERBS (2000-2999)
    # =========================================================================
    # State & Being
    'be.v.01': '2000',
    'have.v.01': '2001',
    'exist.v.01': '2002',

    # Change
    'change.v.01': '2010',
    'become.v.01': '2011',
    'increase.v.01': '2012',
    'decrease.v.01': '2013',
    'grow.v.01': '2014',
    'reduce.v.01': '2015',

    # Motion
    'move.v.02': '2020',
    'travel.v.01': '2021',
    'go.v.01': '2022',
    'come.v.01': '2023',
    'run.v.01': '2024',
    'walk.v.01': '2025',
    'leave.v.01': '2026',
    'arrive.v.01': '2027',

    # Transfer
    'transfer.v.05': '2030',
    'give.v.01': '2031',
    'take.v.01': '2032',
    'get.v.01': '2033',
    'receive.v.01': '2034',
    'send.v.01': '2035',

    # Action
    'act.v.01': '2040',
    'do.v.01': '2041',
    'make.v.03': '2042',
    'create.v.02': '2043',
    'produce.v.01': '2044',
    'build.v.01': '2045',
    'destroy.v.02': '2046',
    'break.v.01': '2047',
    'fix.v.01': '2048',

    # Communication
    'communicate.v.02': '2050',
    'say.v.01': '2051',
    'tell.v.01': '2052',
    'speak.v.01': '2053',
    'write.v.01': '2054',
    'read.v.01': '2055',
    'ask.v.01': '2056',
    'answer.v.01': '2057',

    # Cognition
    'think.v.03': '2060',
    'know.v.01': '2061',
    'believe.v.01': '2062',
    'understand.v.01': '2063',
    'learn.v.01': '2064',
    'remember.v.01': '2065',
    'forget.v.01': '2066',
    'decide.v.01': '2067',
    'plan.v.01': '2068',

    # Perception
    'perceive.v.02': '2070',
    'see.v.01': '2071',
    'hear.v.01': '2072',
    'feel.v.01': '2073',
    'notice.v.01': '2074',

    # Emotion
    'feel.v.03': '2080',
    'like.v.02': '2081',
    'love.v.01': '2082',
    'hate.v.01': '2083',
    'fear.v.01': '2084',
    'worry.v.01': '2085',
    'hope.v.01': '2086',
    'want.v.02': '2087',
    'need.v.01': '2088',

    # Social
    'interact.v.01': '2090',
    'meet.v.01': '2091',
    'help.v.01': '2092',
    'support.v.01': '2093',
    'work.v.02': '2094',
    'manage.v.01': '2095',
    'lead.v.01': '2096',

    # Employment Actions (IMPORTANT)
    'employ.v.01': '2100',
    'hire.v.01': '2101',
    'fire.v.02': '2102',
    'dismiss.v.02': '2103',
    'terminate.v.04': '2104',
    'resign.v.01': '2105',
    'retire.v.01': '2106',
    'quit.v.01': '2107',
    'layoff.v.01': '2108',

    # =========================================================================
    # ADJECTIVES (3000-3999)
    # =========================================================================
    # Quality
    'good.a.01': '3000',
    'bad.a.01': '3001',
    'great.a.01': '3002',
    'small.a.01': '3003',
    'big.a.01': '3004',
    'new.a.01': '3005',
    'old.a.01': '3006',

    # Emotion Adjectives (SENTIMENT IMPORTANT)
    'happy.a.01': '3010',
    'sad.a.01': '3011',
    'angry.a.01': '3012',
    'afraid.a.01': '3013',
    'worried.a.01': '3014',
    'anxious.a.01': '3015',
    'nervous.a.01': '3016',
    'stressed.a.01': '3017',
    'depressed.a.01': '3018',
    'hopeful.a.01': '3019',
    'frustrated.a.01': '3020',
    'disappointed.a.01': '3021',
    'satisfied.a.01': '3022',
    'content.a.01': '3023',

    # Work-related Adjectives
    'employed.a.01': '3030',
    'unemployed.a.01': '3031',
    'busy.a.01': '3032',
    'productive.a.01': '3033',
    'efficient.a.01': '3034',
    'incompetent.a.01': '3035',
    'professional.a.01': '3036',
    'qualified.a.01': '3037',
    'experienced.a.01': '3038',

    # =========================================================================
    # ADVERBS (4000-4999)
    # =========================================================================
    'very.r.01': '4000',
    'really.r.01': '4001',
    'quickly.r.01': '4002',
    'slowly.r.01': '4003',
    'well.r.01': '4004',
    'badly.r.01': '4005',
    'never.r.01': '4006',
    'always.r.01': '4007',
    'often.r.01': '4008',
    'sometimes.r.01': '4009',
}

# =============================================================================
# IMPROVEMENT #1: SENTIWORDNET VALENCE DETECTION
# =============================================================================

def get_valence_sentiwordnet(synset) -> int:
    """
    Get valence using SentiWordNet scores.
    Returns: 0=neutral, 1=positive, 2=negative
    """
    try:
        senti = swn.senti_synset(synset.name())
        pos_score = senti.pos_score()
        neg_score = senti.neg_score()

        # Strong negative
        if neg_score >= 0.25:
            return 2
        # Strong positive
        elif pos_score >= 0.25:
            return 1
        # Weak signals - check which is stronger
        elif neg_score > pos_score and neg_score >= 0.1:
            return 2
        elif pos_score > neg_score and pos_score >= 0.1:
            return 1
        else:
            return 0  # neutral

    except Exception:
        return 0  # neutral if lookup fails


# =============================================================================
# ABSTRACTNESS DETECTION (improved)
# =============================================================================

ABSTRACT_HYPERNYMS = {
    'abstraction.n.06', 'psychological_feature.n.01', 'cognition.n.01',
    'attribute.n.02', 'relation.n.01', 'communication.n.02', 'measure.n.02',
    'state.n.02', 'event.n.01', 'group.n.01', 'feeling.n.01', 'emotion.n.01'
}

CONCRETE_HYPERNYMS = {
    'physical_entity.n.01', 'object.n.01', 'artifact.n.01', 'organism.n.01',
    'substance.n.01', 'body_part.n.01', 'natural_object.n.01', 'food.n.01',
    'location.n.01', 'structure.n.01', 'person.n.01', 'animal.n.01'
}


def get_abstractness(synset) -> int:
    """
    Determine abstractness: 0=concrete, 1=mixed, 2=abstract
    """
    try:
        hypernym_names = set()
        for path in synset.hypernym_paths():
            for h in path:
                hypernym_names.add(h.name())

        is_abstract = bool(hypernym_names & ABSTRACT_HYPERNYMS)
        is_concrete = bool(hypernym_names & CONCRETE_HYPERNYMS)

        if is_abstract and not is_concrete:
            return 2
        elif is_concrete and not is_abstract:
            return 0
        else:
            return 1  # mixed or unknown

    except Exception:
        return 1  # default to mixed


# =============================================================================
# SUPERCLASS DETECTION (improved with hierarchy walking)
# =============================================================================

def get_superclass_code(synset) -> str:
    """
    Determine the HHHH superclass code using expanded hierarchy.
    Walks the hypernym path to find the best matching category.
    """
    try:
        paths = synset.hypernym_paths()
        if not paths:
            return _get_fallback_superclass(synset)

        # Walk from synset up to root, find first matching superclass
        for path in paths:
            # Reverse to go from most specific to most general
            for ancestor in reversed(path):
                name = ancestor.name()
                if name in SUPERCLASS_ROOTS:
                    return SUPERCLASS_ROOTS[name]

        return _get_fallback_superclass(synset)

    except Exception:
        return _get_fallback_superclass(synset)


def _get_fallback_superclass(synset) -> str:
    """Fallback superclass based on POS."""
    pos = synset.pos()
    if pos == 'n':
        return '0999'  # Unknown noun
    elif pos == 'v':
        return '2999'  # Unknown verb
    elif pos in ('a', 's'):
        return '3999'  # Unknown adjective
    else:
        return '4999'  # Unknown adverb


# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build_lexicon():
    """
    Build the improved lexicon database from WordNet + SentiWordNet.
    """
    print("=" * 70)
    print("OYEMI LEXICON BUILDER v2.0")
    print("With SentiWordNet + Expanded Hierarchy + Lemmatization")
    print("Author: Kaossara Osseni")
    print("=" * 70)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Delete existing database
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
        print(f"\nDeleted existing: {OUTPUT_PATH}")

    # Create database
    conn = sqlite3.connect(str(OUTPUT_PATH))
    cursor = conn.cursor()

    # Create schema with lemma support
    cursor.execute("""
        CREATE TABLE lexicon (
            word TEXT NOT NULL,
            code TEXT NOT NULL,
            PRIMARY KEY (word, code)
        )
    """)

    # IMPROVEMENT #3: Add lemma table for fallback lookups
    cursor.execute("""
        CREATE TABLE lemmas (
            word TEXT NOT NULL,
            lemma TEXT NOT NULL,
            PRIMARY KEY (word)
        )
    """)

    cursor.execute("CREATE INDEX idx_word ON lexicon(word)")
    cursor.execute("CREATE INDEX idx_lemma ON lemmas(lemma)")

    print("\n[1/4] Processing WordNet synsets...")

    # Track synset IDs per superclass
    superclass_counters: Dict[str, int] = defaultdict(int)

    # Track all entries
    entries: List[Tuple[str, str]] = []

    # Initialize lemmatizer for improvement #3
    lemmatizer = WordNetLemmatizer()
    lemma_mappings: Dict[str, str] = {}

    # Iterate all synsets
    all_synsets = list(wn.all_synsets())
    print(f"   Total synsets: {len(all_synsets)}")

    # Stats
    valence_stats = {'positive': 0, 'negative': 0, 'neutral': 0}
    superclass_stats = defaultdict(int)

    for synset in tqdm(all_synsets, desc="   Processing"):
        # Get superclass code (HHHH)
        superclass = get_superclass_code(synset)
        superclass_stats[superclass] += 1

        # Increment counter for this superclass (LLLLL - 5 digits)
        superclass_counters[superclass] += 1
        local_id = f"{superclass_counters[superclass]:05d}"

        # Get POS code
        pos = POS_MAP.get(synset.pos(), 1)

        # Get abstractness
        abstractness = get_abstractness(synset)

        # IMPROVEMENT #1: Get valence from SentiWordNet
        valence = get_valence_sentiwordnet(synset)

        # Track stats
        valence_names = {0: 'neutral', 1: 'positive', 2: 'negative'}
        valence_stats[valence_names[valence]] += 1

        # Process each lemma (word form)
        for lemma in synset.lemmas():
            word = lemma.name().lower().replace('_', ' ')

            # Skip words with special characters (except hyphen and space)
            clean_word = word.replace(' ', '').replace('-', '')
            if not clean_word.isalpha():
                continue

            # Build code: HHHH-LLLLL-P-A-V
            code = f"{superclass}-{local_id}-{pos}-{abstractness}-{valence}"
            entries.append((word, code))

            # IMPROVEMENT #3: Store lemma mapping for variants
            if ' ' not in word and '-' not in word:
                base_lemma = lemmatizer.lemmatize(word)
                if base_lemma != word:
                    lemma_mappings[word] = base_lemma

    print(f"\n[2/4] Inserting {len(entries):,} word entries...")
    cursor.executemany(
        "INSERT OR IGNORE INTO lexicon (word, code) VALUES (?, ?)",
        entries
    )

    print(f"\n[3/4] Inserting {len(lemma_mappings):,} lemma mappings...")
    cursor.executemany(
        "INSERT OR IGNORE INTO lemmas (word, lemma) VALUES (?, ?)",
        list(lemma_mappings.items())
    )

    conn.commit()

    # Get statistics
    cursor.execute("SELECT COUNT(DISTINCT word) FROM lexicon")
    word_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM lexicon")
    mapping_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT code) FROM lexicon")
    code_count = cursor.fetchone()[0]

    conn.close()

    print(f"\n[4/4] Build complete!")

    print("\n" + "=" * 70)
    print("BUILD SUMMARY")
    print("=" * 70)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Unique words: {word_count:,}")
    print(f"Unique codes: {code_count:,}")
    print(f"Total mappings: {mapping_count:,}")
    print(f"Lemma mappings: {len(lemma_mappings):,}")
    print(f"Avg codes/word: {mapping_count / word_count:.2f}")

    print(f"\nValence Distribution (SentiWordNet):")
    for v, count in sorted(valence_stats.items(), key=lambda x: x[1], reverse=True):
        pct = count / sum(valence_stats.values()) * 100
        bar = "█" * int(pct / 2)
        print(f"   {v:10} {count:6,} ({pct:5.1f}%) {bar}")

    print(f"\nTop 15 Superclasses:")
    sorted_superclasses = sorted(superclass_stats.items(), key=lambda x: x[1], reverse=True)[:15]
    for sc, count in sorted_superclasses:
        print(f"   {sc}: {count:,}")

    # Sample entries
    print("\nSample entries:")
    conn = sqlite3.connect(str(OUTPUT_PATH))
    cursor = conn.cursor()

    test_words = ['layoff', 'fired', 'happy', 'sad', 'worried', 'fear', 'angry',
                  'manager', 'salary', 'stress', 'anxiety', 'love', 'hate']

    for word in test_words:
        cursor.execute("SELECT code FROM lexicon WHERE word = ?", (word,))
        rows = cursor.fetchall()
        if rows:
            code = rows[0][0]
            parts = code.split('-')
            val_name = {0: 'neu', 1: 'pos', 2: 'neg'}[int(parts[4])]
            print(f"   {word:12} {code} [{val_name}]")
        else:
            print(f"   {word:12} NOT FOUND")

    conn.close()

    print("\n" + "=" * 70)
    print("Lexicon v2.0 ready!")
    print("Improvements: SentiWordNet valence + Expanded hierarchy + Lemmas")
    print("=" * 70)


if __name__ == "__main__":
    build_lexicon()
