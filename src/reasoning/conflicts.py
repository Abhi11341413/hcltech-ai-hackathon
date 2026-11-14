"""Conflict and version-sensitivity detection (FR-06). OWNER: Member 1.

Deterministic first, explainable to the panel:

1. Topic grouping: only compare passages that are about the same thing
   (both have score >= threshold AND share key terms / high embedding similarity).
2. VERSION conflict: same topic, different ``Chunk.version`` values.
3. DATE conflict: same topic, different ``effective_date`` and different facts.
4. VALUE conflict: same topic and same unit ("requests per minute", "days", "$")
   but different numbers -> extract with regex ``(\\d+(?:\\.\\d+)?)\\s*(unit)``.
5. CONTRADICTION: negation mismatch ("is supported" vs "is not supported").
6. ``GeneratedAnswer.llm_reported_conflict`` is a secondary signal only.

Never silently pick one side: the pipeline surfaces every Conflict to the user.
"""

from __future__ import annotations

from src.schemas import Conflict, GeneratedAnswer, RetrievedPassage


def extract_quantities(text: str) -> list[tuple[float, str]]:
    """Return (value, normalised_unit) pairs, e.g. [(300.0, 'requests/minute')]."""
    raise NotImplementedError("Member 1: implement extract_quantities")


def same_topic(a: RetrievedPassage, b: RetrievedPassage) -> bool:
    """Heuristic: do two passages talk about the same subject?"""
    raise NotImplementedError("Member 1: implement same_topic")


def detect_conflicts(question: str, passages: list[RetrievedPassage],
                     generated: GeneratedAnswer | None = None,
                     min_score: float = 0.3) -> list[Conflict]:
    """Return all detected conflicts among relevant passages (empty list if none)."""
    raise NotImplementedError("Member 1: implement detect_conflicts")
