"""Query normalisation and terminology expansion (FR-03 paraphrases). OWNER: Member 3."""

from __future__ import annotations

from pathlib import Path


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords. Shared by BM25 index and queries."""
    raise NotImplementedError("Member 3: implement tokenize")


def load_glossary(path: Path) -> dict[str, list[str]]:
    """Load config/glossary.yaml -> {canonical: [synonyms]}."""
    raise NotImplementedError("Member 3: implement load_glossary")


def expand_query(question: str, glossary: dict[str, list[str]]) -> str:
    """Append canonical terms/synonyms found in the question, e.g.
    'can't log in' -> 'can't log in sign in authenticate'. Used for BM25 only."""
    raise NotImplementedError("Member 3: implement expand_query")
