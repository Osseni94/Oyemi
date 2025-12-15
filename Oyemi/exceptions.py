"""
Oyemi Exceptions
================
Custom exceptions for the Oyemi semantic encoding library.
"""


class OyemiError(Exception):
    """Base exception for all Oyemi errors."""
    pass


class UnknownWordError(OyemiError):
    """Raised when a word is not found in the lexicon."""

    def __init__(self, word: str):
        self.word = word
        super().__init__(f"Word not found in lexicon: '{word}'")


class LexiconNotFoundError(OyemiError):
    """Raised when the lexicon database file is not found."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Lexicon database not found: {path}")


class InvalidCodeError(OyemiError):
    """Raised when an invalid semantic code is encountered."""

    def __init__(self, code: str, reason: str = ""):
        self.code = code
        msg = f"Invalid semantic code: '{code}'"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)
