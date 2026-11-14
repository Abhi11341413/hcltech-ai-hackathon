"""Member 1 - conflict detection, confidence/fallback and metrics tests."""

from __future__ import annotations

from datetime import date

import pytest

from src import mocks
from src.errors import ValidationFailedError
from src.evaluation import metrics
from src.reasoning import confidence, conflicts
from src.schemas import (
    AnswerStatus,
    Chunk,
    ConfidenceLevel,
    ConflictType,
    GeneratedAnswer,
    RelevanceLabel,
    RetrievedPassage,
    RetrieverName,
)


def make_passage(chunk_id: str, section: str, text: str, *, title: str | None = None, version: str | None = None,
                 effective: date | None = None, score: float = 0.8, rank: int = 1) -> RetrievedPassage:
    doc_id = chunk_id.split("::")[0]
    chunk = Chunk(chunk_id=chunk_id, doc_id=doc_id, doc_title=title or doc_id, source=f"{doc_id}.md",
                  section=section, text=text, version=version, effective_date=effective,
                  ordinal=int(chunk_id.rsplit("::", 1)[1]))
    return RetrievedPassage(chunk=chunk, score=score, rank=rank, retriever=RetrieverName.HYBRID)


# Passages built from the real synthetic corpus (data/sample_docs) for conflicts C1-C4.
RELEASE_V3 = make_passage(
    "release-notes-v3::0001", "Orbitly Workspace Release Notes (v3.0) > Changed limits",
    "The API rate limit for Standard plan accounts is increased from 300 to 500 requests per minute. "
    "The Enterprise plan rate limit remains 1,000 requests per minute. "
    "The default session timeout is extended from 8 hours to 12 hours. "
    "The maximum file upload size is increased from 100 MB to 250 MB.",
    version="3.0", effective=date(2026, 10, 1))
LIMITS_V2 = make_passage(
    "api-usage-limits-v2::0001", "API Usage Limits (v2.0) > Rate limits",
    "Standard plan accounts may send up to 300 requests per minute. "
    "Enterprise plan accounts may send up to 1,000 requests per minute.",
    version="2.0", effective=date(2026, 7, 1))
BILLING = make_passage("billing-policy-2026::0002", "Billing Policy (2026) > Refunds",
                       "Annual plans can be refunded within 30 days of purchase. Monthly plans are not refundable.",
                       effective=date(2026, 1, 1))
FAQ_BILLING = make_passage("support-faq-v1::0001", "BILLING",
                           "Q: Can I get a refund? A: Refunds are available within 14 days of purchase for annual plans.",
                           version="1.x", effective=date(2025, 3, 15))
TROUBLESHOOTING = make_passage("troubleshooting-guide::0005", "Troubleshooting Guide > Uploads fail",
                               "Uploads fail when a file exceeds the maximum upload size of 100 MB.",
                               effective=date(2026, 9, 12))


# ----------------------------------------------------------------------------- quantities
def test_extract_quantities():
    assert (300.0, "requests/minute") in conflicts.extract_quantities("up to 300 requests per minute")


@pytest.mark.parametrize(("text", "expected"), [
    ("limited to 1,000 requests per minute", [(1000.0, "requests/minute")]),
    ("increased from 100 MB to 250 MB", [(250.0, "MB")]),
    ("extended from 8 hours to 12 hours", [(12.0, "hours")]),
    ("within 30 days of purchase", [(30.0, "days")]),
    ("99.95% monthly uptime", [(99.95, "%")]),
    ("costs $49 per seat", [(49.0, "USD")]),
    ("HTTP 429 is returned", []),
])
def test_extract_quantities_variants(text, expected):
    assert conflicts.extract_quantities(text) == expected


# ----------------------------------------------------------------------------- conflicts
def test_version_conflict_detected(passages):
    found = conflicts.detect_conflicts("rate limit?", passages)
    assert found and found[0].conflict_type in {ConflictType.VERSION, ConflictType.VALUE}
    assert set(found[0].chunk_ids) == {"api-usage-limits-v2::0001", "faq-legacy::0004"}


def test_no_conflict_for_single_passage(passages):
    assert conflicts.detect_conflicts("password?", passages[2:]) == []


def test_three_way_rate_limit_conflict_is_one_group():
    faq = make_passage("support-faq-v1::0000", "LIMITS",
                       "Q: What is the API rate limit? A: Each account is limited to 100 requests per minute.",
                       version="1.x", effective=date(2025, 3, 15))
    found = conflicts.detect_conflicts("How many API requests can I make per minute?", [RELEASE_V3, LIMITS_V2, faq])
    assert len(found) == 1
    assert found[0].conflict_type == ConflictType.VERSION
    assert set(found[0].chunk_ids) == {RELEASE_V3.chunk.chunk_id, LIMITS_V2.chunk.chunk_id, faq.chunk.chunk_id}
    assert "500" in found[0].values[RELEASE_V3.chunk.chunk_id]


def test_shared_value_is_not_a_conflict():
    """Both docs say Enterprise = 1,000/min; that agreement alone must not be flagged."""
    other = make_passage("other-doc::0001", "Rate limits",
                         "Enterprise plan accounts may send up to 1,000 requests per minute.", version="9.9")
    assert conflicts.detect_conflicts("Enterprise requests per minute?", [LIMITS_V2, other]) == []


def test_date_conflict_refund_window():
    found = conflicts.detect_conflicts("What is the refund window?", [BILLING, FAQ_BILLING])
    assert len(found) == 1 and found[0].conflict_type == ConflictType.DATE
    assert "30" in found[0].values[BILLING.chunk.chunk_id] and "14" in found[0].values[FAQ_BILLING.chunk.chunk_id]


def test_newer_by_date_older_by_version_is_still_shown():
    """C4: troubleshooting guide is newer by date but states the old limit -> must be surfaced."""
    found = conflicts.detect_conflicts("What is the maximum file upload size?", [RELEASE_V3, TROUBLESHOOTING])
    assert len(found) == 1 and set(found[0].chunk_ids) == {RELEASE_V3.chunk.chunk_id, TROUBLESHOOTING.chunk.chunk_id}


def test_unrelated_numbers_are_not_conflicts():
    password = make_passage("account-access::0002", "Account Access > Password reset",
                            "A link valid for 30 minutes is emailed to you.")
    sessions = make_passage("security-faq-v2::0002", "SESSIONS",
                            "Admins can shorten the session to as little as 15 minutes.", version="2.x")
    assert conflicts.detect_conflicts("How do I reset my password?", [password, sessions]) == []


def test_low_score_passages_are_ignored():
    low = FAQ_BILLING.model_copy(update={"score": 0.1})
    assert conflicts.detect_conflicts("What is the refund window?", [BILLING, low]) == []


def test_same_document_chunks_never_conflict():
    a = make_passage("doc-a::0001", "Refunds", "Annual plans can be refunded within 30 days of purchase.")
    b = make_passage("doc-a::0002", "Refunds", "Annual plans can be refunded within 14 days of purchase.")
    assert conflicts.detect_conflicts("refund window?", [a, b]) == []


def test_negation_contradiction():
    a = make_passage("doc-a::0001", "Jira", "The Jira integration supports Jira Server deployments.")
    b = make_passage("doc-b::0001", "Jira", "The Jira integration does not support Jira Server deployments.")
    found = conflicts.detect_conflicts("Is Jira Server supported by the Jira integration?", [a, b])
    assert len(found) == 1 and found[0].conflict_type == ConflictType.CONTRADICTION


def test_llm_flag_is_secondary_signal(generated_answer):
    a = make_passage("api-usage-limits-v2::0001", "Overview", "Text without numbers about the API.")
    b = make_passage("faq-legacy::0004", "Overview", "Other text without numbers about the API.")
    found = conflicts.detect_conflicts("api?", [a, b], generated_answer)
    assert len(found) == 1 and set(found[0].chunk_ids) == {a.chunk.chunk_id, b.chunk.chunk_id}
    assert conflicts.detect_conflicts("api?", [a, b], None) == []


def test_same_topic():
    assert conflicts.same_topic(BILLING, FAQ_BILLING)
    assert not conflicts.same_topic(BILLING, TROUBLESHOOTING)


# ----------------------------------------------------------------------------- confidence / status
def test_not_found_when_no_passages(settings):
    assert confidence.decide_status("q?", [], None, [], settings.reasoning) == AnswerStatus.NOT_FOUND


def test_conflict_status(settings, passages, generated_answer):
    s = confidence.decide_status("rate limit?", passages, generated_answer, mocks.mock_conflicts(), settings.reasoning)
    assert s == AnswerStatus.CONFLICT


def test_answered_status(settings, passages, generated_answer):
    s = confidence.decide_status("rate limit?", passages, generated_answer, [], settings.reasoning)
    assert s == AnswerStatus.ANSWERED


def test_not_found_when_top_score_low(settings, passages, generated_answer):
    weak = [p.model_copy(update={"score": 0.2}) for p in passages]
    assert confidence.decide_status("rate limit?", weak, generated_answer, [], settings.reasoning) \
        == AnswerStatus.NOT_FOUND


def test_not_found_when_llm_says_insufficient(settings, passages, generated_answer):
    gen = generated_answer.model_copy(update={"insufficient_evidence": True})
    assert confidence.decide_status("rate limit?", passages, gen, [], settings.reasoning) == AnswerStatus.NOT_FOUND


def test_not_found_when_groundedness_low(settings, passages, generated_answer):
    gen = generated_answer.model_copy(update={"groundedness_score": 0.3})
    assert confidence.decide_status("rate limit?", passages, gen, [], settings.reasoning) == AnswerStatus.NOT_FOUND


@pytest.mark.parametrize("question", ["What is the limit?", "How long is it kept?", "How do I connect it?"])
def test_ambiguous_questions_ask_for_clarification(settings, question):
    spread = [make_passage("api-usage-limits-v2::0001", "API Usage Limits > Rate limits", "300 requests per minute.",
                           score=0.6, rank=1),
              make_passage("onboarding-quickstart::0003", "Onboarding Quickstart > Seat limits", "50 paid seats.",
                           score=0.55, rank=2)]
    assert confidence.is_ambiguous(question, spread)
    assert confidence.decide_status(question, spread, None, [], settings.reasoning) \
        == AnswerStatus.NEEDS_CLARIFICATION
    clarify = confidence.build_clarifying_question(question, spread)
    assert "Rate limits" in clarify and "Seat limits" in clarify


@pytest.mark.parametrize("question", ["How do I reset my password?", "Who is the CEO of Orbitly?",
                                      "Does it run on a Mars rover?", "What is the refund window?"])
def test_specific_questions_are_not_ambiguous(question, passages):
    assert not confidence.is_ambiguous(question, passages)


def test_ambiguous_needs_competing_topics():
    one_topic = [make_passage("doc-a::0001", "Rate limits", "300 requests per minute.", score=0.7)]
    assert not confidence.is_ambiguous("What is the limit?", one_topic)


def test_confidence_mock_scenario(settings, passages, generated_answer):
    c = confidence.assess_confidence(passages, generated_answer, mocks.mock_conflicts(), settings.reasoning)
    # 0.5*0.82 + 0.3*0.92 + 0.2*(2/3) - 0.15 = 0.669
    assert c.score == pytest.approx(0.669, abs=1e-3)
    assert c.level == ConfidenceLevel.MEDIUM
    assert any("conflict" in r for r in c.reasons)


def test_confidence_bounds_and_levels(settings, passages, generated_answer):
    high = confidence.assess_confidence(passages, generated_answer, [], settings.reasoning)
    none = confidence.assess_confidence([], None, [], settings.reasoning)
    assert high.level == ConfidenceLevel.HIGH and high.score > 0.75
    assert none.score == 0.0 and none.level == ConfidenceLevel.LOW


def test_groundedness_falls_back_to_claims():
    gen = GeneratedAnswer(answer_text="x", model="m", claims=[
        {"text": "a", "supported": True}, {"text": "b", "supported": False}])
    assert confidence.groundedness_of(gen) == 0.5


# ----------------------------------------------------------------------------- metrics
def test_retrieval_metrics(passages):
    labels = [RelevanceLabel(doc_id="faq-legacy")]
    assert metrics.reciprocal_rank(passages, labels) == 0.5
    assert metrics.recall_at_k(passages, labels, k=1) == 0.0
    assert metrics.recall_at_k(passages, labels, k=2) == 1.0
    assert 0.0 < metrics.ndcg_at_k(passages, labels, k=3) < 1.0


def test_section_prefix_relevance(passages):
    labels = [RelevanceLabel(doc_id="account-access", section="Account Access", grade=3)]
    assert metrics.is_relevant(passages[2], labels) == 3
    assert metrics.is_relevant(passages[0], labels) == 0
    wrong_section = [RelevanceLabel(doc_id="account-access", section="Account Access > Two-factor")]
    assert metrics.is_relevant(passages[2], wrong_section) == 0


def test_ndcg_perfect_order_is_one(passages):
    labels = [RelevanceLabel(doc_id="api-usage-limits-v2", grade=3), RelevanceLabel(doc_id="faq-legacy", grade=2)]
    assert metrics.ndcg_at_k(passages, labels, k=3) == pytest.approx(1.0)
    assert metrics.precision_at_k(passages, labels, k=3) == pytest.approx(2 / 3)


def test_empty_labels_give_zero(passages):
    assert metrics.recall_at_k(passages, [], 5) == 0.0
    assert metrics.ndcg_at_k(passages, [], 5) == 0.0
    assert metrics.reciprocal_rank(passages, []) == 0.0


def test_citation_precision():
    cits = mocks.mock_citations()
    assert metrics.citation_precision(cits, [RelevanceLabel(doc_id="faq-legacy")]) == 0.5
    assert metrics.citation_precision([], [RelevanceLabel(doc_id="faq-legacy")]) == 0.0


def test_status_accuracy():
    exp = [AnswerStatus.ANSWERED, AnswerStatus.NOT_FOUND]
    assert metrics.status_accuracy(exp, [AnswerStatus.ANSWERED, AnswerStatus.ANSWERED]) == 0.5
    with pytest.raises(ValidationFailedError):
        metrics.status_accuracy(exp, [AnswerStatus.ANSWERED])
