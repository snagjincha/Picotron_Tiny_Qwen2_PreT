"""Deterministic document cleaning for streamed web text."""

from __future__ import annotations

import html
import re
import unicodedata

_HTML_TAG = re.compile(r"<[^>]+>")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_LONG_REPETITION = re.compile(r"(.)\1{99,}")


def clean_document(text: str, min_characters: int) -> str | None:
    """Normalize a document and reject empty, short, or degenerate records."""
    normalized = unicodedata.normalize("NFKC", html.unescape(text))
    normalized = _HTML_TAG.sub(" ", normalized)
    normalized = _CONTROL.sub(" ", normalized)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    if len(normalized) < min_characters or _LONG_REPETITION.search(normalized):
        return None
    return normalized
