"""Query normalisation and terminology expansion (FR-03 paraphrases). OWNER: Member 3.

* ``tokenize`` is shared by the BM25 index and queries, so both sides are normalised
  identically (lower-case, punctuation removed, stopwords dropped).
* ``expand_query`` adds enterprise synonyms from ``config/glossary.yaml`` so that
  "I can't log in" also matches documents that say "sign in". Expansion is used for
  BM25 only. Dense embeddings already capture paraphrases, and extra words would
  blur the embedding of the original question.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.errors import ConfigError

# Small English stopword list. Kept short on purpose: words like "not", "no",
# "limit", "per" carry meaning in documentation questions and must NOT be removed.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from had has have
    having how i if in into is it its just me my of on or our so than that the their them then
    there these they this those to too very was we were what when where which who whom why will
    with would you your yours should shall may might must also any some
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case, split into word tokens, drop stopwords.

    Keeps codes like ``err-403``, ``v2.0`` and ``429`` as single tokens so exact
    error-code and version searches work.

    >>> tokenize("HTTP 429, Too Many!")
    ['http', '429', 'many']
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def load_glossary(path: Path) -> dict[str, list[str]]:
    """Load ``{canonical term: [synonyms]}`` from YAML. A missing file gives an empty glossary.

    Raises:
        ConfigError: the file exists but is not a mapping of strings to string lists.
    """
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid glossary YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Glossary {path} must be a mapping of term -> list of synonyms")
    glossary: dict[str, list[str]] = {}
    for term, synonyms in raw.items():
        if not isinstance(synonyms, list):
            raise ConfigError(f"Glossary entry '{term}' must map to a list")
        glossary[str(term).lower()] = [str(s).lower() for s in synonyms]
    return glossary


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def expand_query(question: str, glossary: dict[str, list[str]]) -> str:
    """Append the canonical term and all synonyms of every glossary group the question mentions.

    A group matches when the question contains the canonical term or any synonym as a
    whole phrase (case-insensitive). The original question always comes first.

    >>> expand_query("I forgot password", {"password reset": ["forgot password"]})
    'I forgot password password reset forgot password'
    """
    lowered = question.lower()
    additions: list[str] = []
    for canonical, synonyms in glossary.items():
        group = [canonical, *synonyms]
        if any(_contains_phrase(lowered, phrase) for phrase in group):
            additions.extend(group)
    if not additions:
        return question
    return f"{question} {' '.join(additions)}"
