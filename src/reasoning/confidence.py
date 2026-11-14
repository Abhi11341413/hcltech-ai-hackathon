"""Confidence scoring and fallback decision (FR-07). OWNER: Member 1.

Confidence = weighted combination of transparent signals, e.g.
    0.5 * top retrieval score
  + 0.3 * groundedness score (from Member 4)
  + 0.2 * score margin / agreement between top passages
  - penalty if conflicts present
Weights live in config, are documented, and are tuned on the eval set.

Decision table (status):
  no passages or top score < min_top_score_for_answer        -> NOT_FOUND
  generated.insufficient_evidence                             -> NOT_FOUND
  question is under-specified (ambiguous) / evidence spread
     across unrelated topics                                  -> NEEDS_CLARIFICATION
  conflicts non-empty                                         -> CONFLICT
  groundedness < min_groundedness                             -> NOT_FOUND (refuse to bluff)
  otherwise                                                   -> ANSWERED
"""

from __future__ import annotations

from src.config import ReasoningSettings
from src.schemas import (
    AnswerStatus,
    ConfidenceAssessment,
    Conflict,
    GeneratedAnswer,
    RetrievedPassage,
)

NOT_FOUND_MESSAGE = (
    "I could not find this in the supplied documentation. "
    "Try rephrasing, or tell me the product, version or area you are asking about."
)


def is_ambiguous(question: str, passages: list[RetrievedPassage]) -> bool:
    """True for under-specified questions ('What is the limit?') whose top passages
    span several unrelated documents/sections with similar scores."""
    raise NotImplementedError("Member 1: implement is_ambiguous")


def build_clarifying_question(question: str, passages: list[RetrievedPassage]) -> str:
    """e.g. 'Do you mean the API rate limit, the storage limit or the user-seat limit?'
    built from the distinct section titles of the top passages."""
    raise NotImplementedError("Member 1: implement build_clarifying_question")


def assess_confidence(passages: list[RetrievedPassage], generated: GeneratedAnswer | None,
                      conflicts: list[Conflict], settings: ReasoningSettings) -> ConfidenceAssessment:
    raise NotImplementedError("Member 1: implement assess_confidence")


def decide_status(question: str, passages: list[RetrievedPassage], generated: GeneratedAnswer | None,
                  conflicts: list[Conflict], settings: ReasoningSettings) -> AnswerStatus:
    raise NotImplementedError("Member 1: implement decide_status")
