"""Member 3 - retrieval tests. Remove the xfail marker once a function is implemented."""

from __future__ import annotations

import pytest

from src.retrieval import hybrid, query_processing

todo = pytest.mark.xfail(raises=NotImplementedError, reason="Member 3: not implemented yet", strict=False)


@todo
def test_rrf_rewards_agreement():
    scores = hybrid.reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert scores["a"] == pytest.approx(scores["b"])
    assert scores["a"] > scores["c"] > 0 and "d" in scores


@todo
def test_expand_query_adds_synonyms():
    glossary = {"password reset": ["forgot password"]}
    assert "password reset" in query_processing.expand_query("I forgot password", glossary)


@todo
def test_tokenize_lowercases_and_drops_punctuation():
    assert query_processing.tokenize("HTTP 429, Too Many!") == ["http", "429", "many"]


@todo
def test_hybrid_retrieve_returns_ranked_normalised_passages(settings, chunks):
    r = hybrid.HybridRetriever(settings.retrieval)
    r.build(chunks)
    out = r.retrieve("How many requests per minute are allowed?", top_k=2)
    assert [p.rank for p in out] == [1, 2]
    assert all(0.0 <= p.score <= 1.0 for p in out)
    assert out[0].chunk.doc_id in {"api-usage-limits-v2", "faq-legacy"}


@todo
def test_retrieve_before_build_raises(settings):
    from src.errors import IndexNotBuiltError
    with pytest.raises(IndexNotBuiltError):
        hybrid.HybridRetriever(settings.retrieval).retrieve("anything")
