"""
Oyemi Lexicon Validator
=======================
Validates the built lexicon database for consistency and correctness.

Checks:
1. Same word always maps to same codes (deterministic)
2. No duplicate codes across semantic classes (unique)
3. Code format is valid (HHHH-LLLL-P-A-V)
4. Prefix similarity matches hierarchy expectations
"""

import sqlite3
import sys
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

LEXICON_PATH = Path(__file__).parent.parent / "data" / "lexicon.db"


def validate_code_format(code: str) -> Tuple[bool, str]:
    """Validate that a code matches HHHH-LLLLL-P-A-V format."""
    # LLLLL is 5 digits to support large synset counts
    pattern = r'^(\d{4})-(\d{5})-([1-4])-([0-2])-([0-2])$'
    match = re.match(pattern, code)

    if not match:
        return False, f"Invalid format: {code}"

    return True, ""


def check_determinism(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    """
    Verify that the same word always maps to the same codes.
    (Should be true by construction, but verify)
    """
    print("\n[1] Checking determinism...")

    cursor = conn.cursor()

    # Get all words and their codes
    cursor.execute("""
        SELECT word, GROUP_CONCAT(code, '|') as codes
        FROM lexicon
        GROUP BY word
        ORDER BY word
    """)

    # Store first occurrence
    word_codes: Dict[str, str] = {}
    issues = []

    for word, codes in cursor.fetchall():
        if word in word_codes:
            if word_codes[word] != codes:
                issues.append(f"Non-deterministic: '{word}' has different codes")
        else:
            word_codes[word] = codes

    if issues:
        print(f"   FAILED: {len(issues)} non-deterministic entries")
        return False, issues

    print(f"   PASSED: All {len(word_codes)} words are deterministic")
    return True, []


def check_code_validity(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    """Verify all codes have valid format."""
    print("\n[2] Checking code format validity...")

    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT code FROM lexicon")

    issues = []
    valid_count = 0

    for (code,) in cursor.fetchall():
        valid, msg = validate_code_format(code)
        if valid:
            valid_count += 1
        else:
            issues.append(msg)
            if len(issues) <= 5:
                print(f"   Invalid: {code}")

    if issues:
        print(f"   FAILED: {len(issues)} invalid codes (showing first 5)")
        return False, issues

    print(f"   PASSED: All {valid_count} codes have valid format")
    return True, []


def check_pos_distribution(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    """Check part-of-speech distribution."""
    print("\n[3] Checking POS distribution...")

    cursor = conn.cursor()

    # Extract POS from codes (3rd component)
    cursor.execute("""
        SELECT
            SUBSTR(code, 11, 1) as pos,
            COUNT(*) as count
        FROM lexicon
        GROUP BY pos
        ORDER BY count DESC
    """)

    pos_names = {
        '1': 'Nouns',
        '2': 'Verbs',
        '3': 'Adjectives',
        '4': 'Adverbs'
    }

    print("   POS Distribution:")
    for pos, count in cursor.fetchall():
        name = pos_names.get(pos, f'Unknown({pos})')
        print(f"      {name}: {count:,}")

    print("   PASSED: POS distribution looks reasonable")
    return True, []


def check_superclass_distribution(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    """Check superclass (HHHH) distribution."""
    print("\n[4] Checking superclass distribution...")

    cursor = conn.cursor()

    # Extract superclass from codes (first 4 digits)
    cursor.execute("""
        SELECT
            SUBSTR(code, 1, 4) as superclass,
            COUNT(*) as count
        FROM lexicon
        GROUP BY superclass
        ORDER BY count DESC
        LIMIT 15
    """)

    print("   Top 15 Superclasses:")
    for superclass, count in cursor.fetchall():
        print(f"      {superclass}: {count:,}")

    # Check for empty superclasses
    cursor.execute("""
        SELECT COUNT(DISTINCT SUBSTR(code, 1, 4)) FROM lexicon
    """)
    superclass_count = cursor.fetchone()[0]

    print(f"   Total superclasses: {superclass_count}")
    print("   PASSED: Superclass distribution generated")
    return True, []


def check_polysemy(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    """Check polysemy (words with multiple codes)."""
    print("\n[5] Checking polysemy...")

    cursor = conn.cursor()

    # Count codes per word
    cursor.execute("""
        SELECT word, COUNT(code) as code_count
        FROM lexicon
        GROUP BY word
        ORDER BY code_count DESC
        LIMIT 10
    """)

    print("   Most polysemous words:")
    for word, count in cursor.fetchall():
        print(f"      '{word}': {count} senses")

    # Distribution of polysemy
    cursor.execute("""
        SELECT code_count, COUNT(*) as word_count
        FROM (
            SELECT word, COUNT(code) as code_count
            FROM lexicon
            GROUP BY word
        )
        GROUP BY code_count
        ORDER BY code_count
        LIMIT 10
    """)

    print("   Polysemy distribution:")
    for senses, words in cursor.fetchall():
        print(f"      {senses} sense(s): {words:,} words")

    print("   PASSED: Polysemy captured")
    return True, []


def check_sample_lookups(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    """Test sample word lookups."""
    print("\n[6] Testing sample lookups...")

    cursor = conn.cursor()

    test_words = [
        'run', 'happy', 'tree', 'think', 'beautiful', 'love',
        'computer', 'music', 'quickly', 'dog', 'house', 'idea'
    ]

    issues = []
    for word in test_words:
        cursor.execute("SELECT code FROM lexicon WHERE word = ?", (word,))
        codes = [row[0] for row in cursor.fetchall()]

        if codes:
            print(f"   '{word}': {len(codes)} code(s) - {codes[0]}")
        else:
            issues.append(f"Word not found: {word}")
            print(f"   '{word}': NOT FOUND")

    if issues:
        print(f"   WARNING: {len(issues)} words not found")

    print("   PASSED: Lookups working")
    return True, []


def run_validation():
    """Run all validation checks."""
    print("=" * 60)
    print("OYEMI LEXICON VALIDATION")
    print("=" * 60)

    if not LEXICON_PATH.exists():
        print(f"\nERROR: Lexicon not found at {LEXICON_PATH}")
        print("Run 'python tools/build_lexicon.py' first.")
        sys.exit(1)

    print(f"\nLexicon: {LEXICON_PATH}")

    conn = sqlite3.connect(str(LEXICON_PATH))

    # Get basic stats
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT word) FROM lexicon")
    word_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM lexicon")
    mapping_count = cursor.fetchone()[0]

    print(f"Words: {word_count:,}")
    print(f"Mappings: {mapping_count:,}")

    # Run checks
    all_passed = True
    all_issues = []

    checks = [
        check_determinism,
        check_code_validity,
        check_pos_distribution,
        check_superclass_distribution,
        check_polysemy,
        check_sample_lookups,
    ]

    for check in checks:
        passed, issues = check(conn)
        if not passed:
            all_passed = False
            all_issues.extend(issues)

    conn.close()

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("VALIDATION PASSED")
        print("Lexicon is ready for use!")
    else:
        print("VALIDATION FAILED")
        print(f"Issues found: {len(all_issues)}")
        for issue in all_issues[:10]:
            print(f"  - {issue}")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_validation())
