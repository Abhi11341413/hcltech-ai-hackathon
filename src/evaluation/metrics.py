"""Retrieval and answer-quality metrics (FR-08). OWNER: Member 1.

Relevance is judged at doc+section level (``RelevanceLabel``) so labels stay valid
when chunking changes. A retrieved passage is relevant if its doc_id matches and
(label.section is None or the chunk's section starts with label.section).

All functions are pure and deterministic - easy to unit test by hand.

Conventions:
* recall@k counts LABELS covered (one label = one doc/section that should be found).
* Each label is credited at most once in nDCG, so several chunks of the same section do not
  inflate the gain (otherwise nDCG could exceed 1).
"""

from __future__ import annotations

import math

from src.errors import ValidationFailedError
from src.schemas import AnswerStatus, Citation, RelevanceLabel, RetrievedPassage


def _matches(doc_id: str, section: str, label: RelevanceLabel) -> bool:
    return doc_id == label.doc_id and (label.section is None or section.startswith(label.section))


def _best_label(doc_id: str, section: str, labels: list[RelevanceLabel]) -> int | None:
    """Index of the highest-grade label matching (doc_id, section), or None."""
    best: int | None = None
    for i, label in enumerate(labels):
        if _matches(doc_id, section, label) and (best is None or label.grade > labels[best].grade):
            best = i
    return best


def is_relevant(passage: RetrievedPassage, labels: list[RelevanceLabel]) -> int:
    """Return the relevance grade (0 if not relevant)."""
    i = _best_label(passage.chunk.doc_id, passage.chunk.section, labels)
    return 0 if i is None else labels[i].grade


def _ranked(passages: list[RetrievedPassage]) -> list[RetrievedPassage]:
    return sorted(passages, key=lambda p: p.rank)


def recall_at_k(passages: list[RetrievedPassage], labels: list[RelevanceLabel], k: int) -> float:
    """Fraction of labels covered by the top-k passages (0.0 when there are no labels)."""
    if not labels:
        return 0.0
    top = _ranked(passages)[:k]
    covered = sum(1 for label in labels if any(_matches(p.chunk.doc_id, p.chunk.section, label) for p in top))
    return covered / len(labels)


def precision_at_k(passages: list[RetrievedPassage], labels: list[RelevanceLabel], k: int) -> float:
    """Fraction of the top-k passages that are relevant (denominator = number actually returned, max k)."""
    top = _ranked(passages)[:k]
    if not top:
        return 0.0
    return sum(1 for p in top if is_relevant(p, labels) > 0) / len(top)


def reciprocal_rank(passages: list[RetrievedPassage], labels: list[RelevanceLabel]) -> float:
    """1 / rank of the first relevant passage (0.0 if none)."""
    for position, p in enumerate(_ranked(passages), start=1):
        if is_relevant(p, labels) > 0:
            return 1.0 / position
    return 0.0


def ndcg_at_k(passages: list[RetrievedPassage], labels: list[RelevanceLabel], k: int) -> float:
    """Graded nDCG@k using RelevanceLabel.grade.

    DCG = sum_i (2^grade_i - 1) / log2(i + 1) over the top-k positions, each label credited once.
    IDCG = same formula over the label grades sorted descending.
    """
    if not labels:
        return 0.0
    credited: set[int] = set()
    dcg = 0.0
    for position, p in enumerate(_ranked(passages)[:k], start=1):
        candidates = [i for i, label in enumerate(labels)
                      if i not in credited and _matches(p.chunk.doc_id, p.chunk.section, label)]
        if candidates:
            best = max(candidates, key=lambda i: labels[i].grade)
            credited.add(best)
            dcg += (2 ** labels[best].grade - 1) / math.log2(position + 1)
    ideal = sorted((label.grade for label in labels), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def citation_precision(citations: list[Citation], labels: list[RelevanceLabel]) -> float:
    """Fraction of citations that point to a gold-relevant doc/section (0.0 if no citations)."""
    if not citations:
        return 0.0
    return sum(1 for c in citations if _best_label(c.doc_id, c.section, labels) is not None) / len(citations)


def status_accuracy(expected: list[AnswerStatus], predicted: list[AnswerStatus]) -> float:
    """Exact-match accuracy of answer status (tests fallback + conflict behaviour).

    Raises:
        ValidationFailedError: the two lists have different lengths.
    """
    if len(expected) != len(predicted):
        raise ValidationFailedError("expected and predicted must have the same length")
    if not expected:
        return 0.0
    return sum(1 for e, p in zip(expected, predicted, strict=True) if e == p) / len(expected)
