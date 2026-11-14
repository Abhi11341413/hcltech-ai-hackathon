"""Confidence scoring and fallback decision (FR-07). OWNER: Member 1.

Confidence = weighted combination of transparent signals:
    0.5 * top retrieval score               (is there one clearly relevant passage?)
  + 0.3 * groundedness score (Member 4)     (are the answer's claims supported?)
  + 0.2 * evidence support                  (share of the top-3 passages above the answer threshold)
  - 0.15 if conflicts are present           (the user has to choose between sources)
Weights are module constants (below) so they are visible in one place and can be tuned on the
eval set without touching the shared config contract.

Decision table (status), evaluated top to bottom:
  no passages                                                 -> NOT_FOUND
  question is under-specified AND the top passages are spread
     across different sections/documents                      -> NEEDS_CLARIFICATION
  top score < min_top_score_for_answer                        -> NOT_FOUND
  generated.insufficient_evidence                             -> NOT_FOUND
  conflicts non-empty                                         -> CONFLICT
  no generated answer                                         -> NOT_FOUND
  groundedness < min_groundedness                             -> NOT_FOUND (refuse to bluff)
  otherwise                                                   -> ANSWERED

The ambiguity check runs before the score threshold on purpose: vague questions ("How do I
connect it?") have few terms, so they often score low, and asking the user what they mean is
more useful than "not found".
"""

from __future__ import annotations

from src.config import ReasoningSettings
from src.reasoning.text_utils import VAGUE_REFERENTS, content_terms, last_heading, words
from src.schemas import (
    AnswerStatus,
    ConfidenceAssessment,
    ConfidenceLevel,
    Conflict,
    GeneratedAnswer,
    RetrievedPassage,
)

NOT_FOUND_MESSAGE = (
    "I could not find this in the supplied documentation. "
    "Try rephrasing, or tell me the product, version or area you are asking about."
)

W_TOP_SCORE = 0.5
W_GROUNDEDNESS = 0.3
W_SUPPORT = 0.2
CONFLICT_PENALTY = 0.15
SUPPORT_WINDOW = 3                 # passages considered for "evidence support"
AMBIGUITY_MAX_CONTENT_TERMS = 1    # "What is the limit?" has one content term ('limit')
AMBIGUITY_SCORE_BAND = 0.25        # passages within this distance of the top score count as competitors
MAX_CLARIFY_OPTIONS = 4


def _topic_label(passage: RetrievedPassage) -> str:
    heading = last_heading(passage.chunk.section)
    return heading or passage.chunk.doc_title


def _competing_topics(passages: list[RetrievedPassage]) -> list[str]:
    """Distinct topic labels of passages scoring close to the top passage, best first."""
    if not passages:
        return []
    ranked = sorted(passages, key=lambda p: p.score, reverse=True)
    top = ranked[0].score
    seen: dict[str, str] = {}
    for p in ranked:
        if top - p.score <= AMBIGUITY_SCORE_BAND:
            label = _topic_label(p)
            seen.setdefault(label.lower(), label)
    return list(seen.values())


def is_under_specified(question: str) -> bool:
    """True when the question names (almost) no subject: at most one content term, or a vague
    pronoun ('it', 'this') with at most two content terms."""
    terms = content_terms(question)
    if len(terms) <= AMBIGUITY_MAX_CONTENT_TERMS:
        return True
    return bool(set(words(question)) & VAGUE_REFERENTS) and len(terms) <= 2 and len(question.split()) <= 6


def is_ambiguous(question: str, passages: list[RetrievedPassage]) -> bool:
    """True for under-specified questions ('What is the limit?') whose top passages
    span several unrelated documents/sections with similar scores.

    Inputs: question text and retrieved passages. Output: bool. Never raises.
    """
    return is_under_specified(question) and len(_competing_topics(passages)) >= 2


def build_clarifying_question(question: str, passages: list[RetrievedPassage]) -> str:
    """e.g. 'Do you mean the API rate limit, the storage limit or the user-seat limit?'
    built from the distinct section titles of the top passages."""
    topics = _competing_topics(passages)[:MAX_CLARIFY_OPTIONS]
    if not topics:
        return "Could you add more detail, such as the feature, plan or product version you mean?"
    options = topics[0] if len(topics) == 1 else ", ".join(topics[:-1]) + " or " + topics[-1]
    return (f"Your question could refer to several topics in the documentation. Do you mean: {options}? "
            "Please add the feature, plan or product version you are asking about.")


def groundedness_of(generated: GeneratedAnswer | None) -> float:
    """Groundedness in [0, 1]: Member 4's score if set, else the share of claims marked
    supported; an answer with no claims counts as grounded only if it has citations."""
    if generated is None:
        return 0.0
    if generated.groundedness_score is not None:
        return generated.groundedness_score
    if generated.claims:
        return sum(1 for c in generated.claims if c.supported) / len(generated.claims)
    return 1.0 if generated.citations else 0.0


def assess_confidence(passages: list[RetrievedPassage], generated: GeneratedAnswer | None,
                      conflicts: list[Conflict], settings: ReasoningSettings) -> ConfidenceAssessment:
    """Combine retrieval, groundedness and conflict signals into a score, level and reasons.

    Inputs: retrieved passages, the generated answer (None before/without generation), detected
    conflicts and the reasoning thresholds. Output: ``ConfidenceAssessment``. Never raises.
    """
    reasons: list[str] = []
    top = max((p.score for p in passages), default=0.0)
    window = sorted(passages, key=lambda p: p.score, reverse=True)[:SUPPORT_WINDOW]
    support = (sum(1 for p in window if p.score >= settings.min_top_score_for_answer) / SUPPORT_WINDOW
               if window else 0.0)
    grounded = groundedness_of(generated)

    if not passages:
        reasons.append("no passages retrieved")
    elif top >= settings.high_confidence_threshold:
        reasons.append(f"strong top retrieval score ({top:.2f})")
    elif top >= settings.min_top_score_for_answer:
        reasons.append(f"moderate top retrieval score ({top:.2f})")
    else:
        reasons.append(f"weak top retrieval score ({top:.2f} < {settings.min_top_score_for_answer})")

    if generated is None:
        reasons.append("no answer generated")
    elif generated.insufficient_evidence:
        reasons.append("answer model reported insufficient evidence")
    elif grounded >= settings.min_groundedness:
        reasons.append(f"claims supported by cited passages (groundedness {grounded:.2f})")
    else:
        reasons.append(f"some claims unsupported (groundedness {grounded:.2f} < {settings.min_groundedness})")

    reasons.append(f"{round(support * SUPPORT_WINDOW)} of top {SUPPORT_WINDOW} passages above answer threshold")

    score = W_TOP_SCORE * top + W_GROUNDEDNESS * grounded + W_SUPPORT * support
    if generated is not None and generated.insufficient_evidence:
        score = min(score, settings.medium_confidence_threshold - 0.01)
    if conflicts:
        score -= CONFLICT_PENALTY
        reasons.append(f"{len(conflicts)} conflict(s) between sources")
    score = round(min(1.0, max(0.0, score)), 3)

    if score >= settings.high_confidence_threshold:
        level = ConfidenceLevel.HIGH
    elif score >= settings.medium_confidence_threshold:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW
    return ConfidenceAssessment(score=score, level=level, reasons=reasons)


def pre_generation_status(question: str, passages: list[RetrievedPassage],
                          settings: ReasoningSettings) -> AnswerStatus | None:
    """Early exit before calling the LLM: NOT_FOUND / NEEDS_CLARIFICATION, or None to continue.

    Saves an LLM call (latency + cost) when retrieval alone already decides the outcome.
    """
    if not passages:
        return AnswerStatus.NOT_FOUND
    if is_ambiguous(question, passages):
        return AnswerStatus.NEEDS_CLARIFICATION
    if max(p.score for p in passages) < settings.min_top_score_for_answer:
        return AnswerStatus.NOT_FOUND
    return None


def decide_status(question: str, passages: list[RetrievedPassage], generated: GeneratedAnswer | None,
                  conflicts: list[Conflict], settings: ReasoningSettings) -> AnswerStatus:
    """Apply the decision table in the module docstring. Never raises."""
    early = pre_generation_status(question, passages, settings)
    if early is not None:
        return early
    if generated is not None and generated.insufficient_evidence:
        return AnswerStatus.NOT_FOUND
    if conflicts:
        return AnswerStatus.CONFLICT
    if generated is None:
        return AnswerStatus.NOT_FOUND
    if groundedness_of(generated) < settings.min_groundedness:
        return AnswerStatus.NOT_FOUND
    return AnswerStatus.ANSWERED
