"""Member 1 - conflict detection, confidence/fallback and metrics tests.
Remove the xfail marker once a function is implemented."""

from __future__ import annotations

import pytest

from src.evaluation import metrics
from src.reasoning import confidence, conflicts
from src.schemas import AnswerStatus, ConflictType, RelevanceLabel

todo = pytest.mark.xfail(raises=NotImplementedError, reason="Member 1: not implemented yet", strict=False)


@todo
def test_extract_quantities():
    assert (300.0, "requests/minute") in conflicts.extract_quantities("up to 300 requests per minute")


@todo
def test_version_conflict_detected(passages):
    found = conflicts.detect_conflicts("rate limit?", passages)
    assert found and found[0].conflict_type in {ConflictType.VERSION, ConflictType.VALUE}
    assert set(found[0].chunk_ids) == {"api-usage-limits-v2::0001", "faq-legacy::0004"}


@todo
def test_no_conflict_for_single_passage(passages):
    assert conflicts.detect_conflicts("password?", passages[2:]) == []


@todo
def test_not_found_when_no_passages(settings):
    assert confidence.decide_status("q?", [], None, [], settings.reasoning) == AnswerStatus.NOT_FOUND


@todo
def test_conflict_status(settings, passages, generated_answer):
    from src import mocks
    s = confidence.decide_status("rate limit?", passages, generated_answer, mocks.mock_conflicts(), settings.reasoning)
    assert s == AnswerStatus.CONFLICT


@todo
def test_retrieval_metrics(passages):
    labels = [RelevanceLabel(doc_id="faq-legacy")]
    assert metrics.reciprocal_rank(passages, labels) == 0.5
    assert metrics.recall_at_k(passages, labels, k=1) == 0.0
    assert metrics.recall_at_k(passages, labels, k=2) == 1.0
    assert 0.0 < metrics.ndcg_at_k(passages, labels, k=3) < 1.0


@todo
def test_status_accuracy():
    exp = [AnswerStatus.ANSWERED, AnswerStatus.NOT_FOUND]
    assert metrics.status_accuracy(exp, [AnswerStatus.ANSWERED, AnswerStatus.ANSWERED]) == 0.5
