"""Small, dependency-free text helpers shared by the reasoning modules. OWNER: Member 1.

Kept deliberately simple (regex + a stop-word list + crude suffix stripping) so every
decision made by conflict detection and ambiguity detection can be explained line by line.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Function words plus "question scaffolding" words that carry no topic on their own.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then else of to in on at by for from with without into onto over under
    about after before between during per via as than so such too very can could should would will
    shall may might must do does did doing done have has had having be is are was were been being am
    i me my mine we us our you your yours he she him her his they them their it its this that these
    those there here what which who whom whose when where why how much many long often some any all
    each every no not nor only own same other more most few both either neither just also again
    get got make made use used need want go going let please tell know way thing things
    up down out off yes ok q a
    """.split()
)

# Pronouns that point at something the question never names ("How do I connect it?").
VAGUE_REFERENTS: frozenset[str] = frozenset({"it", "this", "that", "they", "them", "these", "those", "one"})

NEGATIONS: frozenset[str] = frozenset({"not", "no", "never", "cannot", "can't", "cant", "isn't", "aren't",
                                       "doesn't", "don't", "won't", "unsupported", "without"})

_SUFFIXES = ("ations", "ation", "ings", "ing", "able", "ied", "ies", "ed", "s")


def stem(word: str) -> str:
    """Crude suffix stripping so 'refunded', 'refunds', 'refundable' all become 'refund'
    and 'minute'/'minutes', 'expire'/'expired' collapse to the same stem.

    Input: one lower-case word. Output: the stem (never shorter than 3 characters).
    """
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + ("y" if suffix in {"ied", "ies"} else "")
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def words(text: str) -> list[str]:
    """Lower-case word tokens (keeps hyphenated codes such as 'err-403')."""
    return _WORD_RE.findall(text.lower())


def content_terms(text: str) -> set[str]:
    """Stemmed, non-stop-word, non-numeric terms of ``text``."""
    return {stem(w) for w in words(text) if w not in STOPWORDS and not w.isdigit() and len(w) > 1}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation or line breaks; drops empty pieces."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()]


def last_heading(section: str) -> str:
    """'API Usage Limits > Rate limits' -> 'Rate limits'."""
    return section.split(" > ")[-1].strip() if section else ""
