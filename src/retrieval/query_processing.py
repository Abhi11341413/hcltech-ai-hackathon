"""Query normalisation and terminology expansion (FR-03 paraphrases). OWNER: Member 3."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.errors import ConfigError

# Small, conservative English stopword list. Quantity words ("many", "most", "limit") and
# negations ("not", "no") are deliberately KEPT: they carry meaning in policy questions.
STOPWORDS: frozenset[str] = frozenset("""
a an the and or but if then else of to in on at by for from with without into onto over under
about as is are was were be been being am do does did doing done have has had having
i me my mine we our ours you your yours he him his she her it its they them their this that
these those there here what which who whom whose when where why how can could should would
will shall may might must too very just so than also only own same such s t
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on any non-alphanumeric character, drop stopwords.

    Splitting on '-' turns 'ERR-429' into ['err', '429'], so the queries "ERR-429",
    "err 429" and "error 429" all share the token '429'. No stemming: it keeps the
    behaviour explainable and the glossary covers the important variants.
    """
    return [tok for tok in _TOKEN.findall(text.lower()) if tok not in STOPWORDS]


def load_glossary(path: Path) -> dict[str, list[str]]:
    """Load config/glossary.yaml -> {canonical: [synonyms]} (all lower-cased).

    Raises:
        ConfigError: file missing or not a mapping of str -> list[str].
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Glossary not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Glossary must be a mapping of canonical term -> list of synonyms")
    glossary: dict[str, list[str]] = {}
    for canonical, synonyms in raw.items():
        if not isinstance(synonyms, list):
            raise ConfigError(f"Glossary entry '{canonical}' must be a list")
        glossary[str(canonical).lower().strip()] = [str(s).lower().strip() for s in synonyms]
    return glossary


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


def expand_query(question: str, glossary: dict[str, list[str]]) -> str:
    """Append canonical terms/synonyms found in the question, e.g.
    'can't log in' -> "can't log in sign in login logon ...". Used for BM25 only
    (the dense retriever already handles paraphrases and would be diluted by the extra words).

    A glossary group fires when the question contains the canonical term or any synonym as a
    whole phrase; then every term of the group not already in the question is appended once.
    """
    lowered = question.lower()
    additions: list[str] = []
    for canonical, synonyms in glossary.items():
        group = [canonical, *synonyms]
        if any(_contains_phrase(lowered, term) for term in group):
            additions.extend(t for t in group if not _contains_phrase(lowered, t) and t not in additions)
    return f"{question} {' '.join(additions)}" if additions else question
