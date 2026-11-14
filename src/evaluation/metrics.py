"""Retrieval and answer-quality metrics (FR-08). OWNER: Member 1.

Relevance is judged at doc+section level (``RelevanceLabel``) so labels stay valid
when chunking changes. A retrieved passage is relevant if its doc_id matches and
(label.section is None or the chunk's section starts with label.section).

All functions are pure and deterministic - easy to unit test by hand.
"""

from __future__ import annotations

from src.schemas import AnswerStatus, Citation, RelevanceLabel, RetrievedPassage


def is_relevant(passage: RetrievedPassage, labels: list[RelevanceLabel]) -> int:
    """Return the relevance grade (0 if not relevant)."""
    raise NotImplementedError("Member 1: implement is_relevant")


def recall_at_k(passages: list[RetrievedPassage], labels: list[RelevanceLabel], k: int) -> float:
    """Fraction of labels covered by the top-k passages."""
    raise NotImplementedError("Member 1: implement recall_at_k")


def precision_at_k(passages: list[RetrievedPassage], labels: list[RelevanceLabel], k: int) -> float:
    raise NotImplementedError("Member 1: implement precision_at_k")


def reciprocal_rank(passages: list[RetrievedPassage], labels: list[RelevanceLabel]) -> float:
    """1 / rank of the first relevant passage (0.0 if none)."""
    raise NotImplementedError("Member 1: implement reciprocal_rank")


def ndcg_at_k(passages: list[RetrievedPassage], labels: list[RelevanceLabel], k: int) -> float:
    """Graded nDCG@k using RelevanceLabel.grade."""
    raise NotImplementedError("Member 1: implement ndcg_at_k")


def citation_precision(citations: list[Citation], labels: list[RelevanceLabel]) -> float:
    """Fraction of citations that point to a gold-relevant doc/section."""
    raise NotImplementedError("Member 1: implement citation_precision")


def status_accuracy(expected: list[AnswerStatus], predicted: list[AnswerStatus]) -> float:
    """Exact-match accuracy of answer status (tests fallback + conflict behaviour)."""
    raise NotImplementedError("Member 1: implement status_accuracy")
