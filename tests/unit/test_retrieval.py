"""Member 3 - retrieval tests.

Unit tests use ``FakeEncoder`` (hashed bag-of-words vectors) so they are fast and
offline. ``test_real_model_handles_paraphrase`` uses the real MiniLM model and is
skipped if sentence-transformers or the model download is unavailable.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.errors import ConfigError, IndexNotBuiltError, RetrievalError
from src.retrieval import hybrid, query_processing
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.schemas import Chunk, RetrieverName


class FakeEncoder:
    """Deterministic stand-in for a sentence embedding model."""

    model_name = "fake-hash-encoder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in query_processing.tokenize(text):
                out[row, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


class BrokenEncoder:
    model_name = "broken"

    def encode(self, texts: list[str]) -> np.ndarray:
        raise RetrievalError("model download failed")


class ReverseReranker:
    """Fake reranker that prefers the LAST candidate, to prove reranking reorders."""

    model_name = "fake-reranker"

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        n = len(chunks)
        return [(i + 1) / (n + 1) for i in range(n)]


def make_retriever(settings, chunks, encoder=None, **kwargs) -> hybrid.HybridRetriever:
    r = hybrid.HybridRetriever(settings.retrieval, embedder=encoder or FakeEncoder(), glossary={}, **kwargs)
    r.build(chunks)
    return r


# ----------------------------------------------------------- query processing
def test_tokenize_lowercases_and_drops_punctuation():
    assert query_processing.tokenize("HTTP 429, Too Many!") == ["http", "429", "many"]


def test_tokenize_keeps_codes_and_versions():
    assert query_processing.tokenize("ERR-403 in v2.0") == ["err-403", "v2.0"]


def test_tokenize_keeps_meaningful_short_words():
    assert "not" in query_processing.tokenize("Jira Server is not supported")


def test_expand_query_adds_synonyms():
    glossary = {"password reset": ["forgot password"]}
    assert "password reset" in query_processing.expand_query("I forgot password", glossary)


def test_expand_query_matches_whole_phrases_only():
    glossary = {"sso": ["single sign-on"]}
    assert query_processing.expand_query("lasso tool", glossary) == "lasso tool"
    assert "single sign-on" in query_processing.expand_query("Configure SSO", glossary)


def test_load_glossary_from_repo_config(settings):
    from src.config import PROJECT_ROOT
    glossary = query_processing.load_glossary(PROJECT_ROOT / settings.retrieval.glossary_path)
    assert "sso" in glossary and "single sign-on" in glossary["sso"]


def test_load_glossary_rejects_bad_shape(tmp_path):
    bad = tmp_path / "g.yaml"
    bad.write_text("sso: single sign-on\n")
    with pytest.raises(ConfigError):
        query_processing.load_glossary(bad)
    assert query_processing.load_glossary(tmp_path / "missing.yaml") == {}


# ------------------------------------------------------------------- fusion
def test_rrf_rewards_agreement():
    scores = hybrid.reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert scores["a"] == pytest.approx(scores["b"])
    assert scores["a"] > scores["c"] > 0 and "d" in scores


def test_rrf_value():
    assert hybrid.reciprocal_rank_fusion([["x"]], k=60)["x"] == pytest.approx(1 / 61)


# --------------------------------------------------------------------- BM25
def test_bm25_finds_exact_term(chunks):
    idx = BM25Index()
    idx.build(chunks)
    top_chunk, score = idx.search("HTTP 429", top_n=3)[0]
    assert top_chunk.chunk_id == "api-usage-limits-v2::0001" and score > 0


def test_bm25_excludes_zero_score_and_requires_build(chunks):
    idx = BM25Index()
    with pytest.raises(IndexNotBuiltError):
        idx.search("anything", 3)
    idx.build(chunks)
    assert idx.search("kubernetes helm", 3) == []


def test_bm25_term_in_half_of_docs_still_scores():
    """Regression: rank_bm25's default IDF is 0 when a term is in exactly N/2 docs."""
    a = Chunk(chunk_id="a::0000", doc_id="a", doc_title="A", source="s", section="S", text="refund policy", ordinal=0)
    b = Chunk(chunk_id="b::0000", doc_id="b", doc_title="B", source="s", section="S", text="invoice dates", ordinal=0)
    idx = BM25Index()
    idx.build([a, b])
    hits = idx.search("refund", 2)
    assert [c.doc_id for c, _ in hits] == ["a"] and hits[0][1] > 0


# -------------------------------------------------------------------- dense
def test_dense_save_load_roundtrip(tmp_path, chunks):
    idx = DenseIndex(FakeEncoder())
    idx.build(chunks)
    idx.save(tmp_path)
    loaded = DenseIndex(FakeEncoder())
    loaded.load(tmp_path)
    assert loaded.num_chunks == 3
    assert loaded.search("password reset", 1)[0][0].chunk_id == idx.search("password reset", 1)[0][0].chunk_id


def test_dense_load_rejects_other_model(tmp_path, chunks):
    idx = DenseIndex(FakeEncoder())
    idx.build(chunks)
    idx.save(tmp_path)
    other = FakeEncoder()
    other.model_name = "different-model"
    with pytest.raises(RetrievalError):
        DenseIndex(other).load(tmp_path)


# ------------------------------------------------------------------- hybrid
def test_hybrid_retrieve_returns_ranked_normalised_passages(settings, chunks):
    r = make_retriever(settings, chunks)
    out = r.retrieve("How many requests per minute are allowed?", top_k=2)
    assert [p.rank for p in out] == [1, 2]
    assert all(0.0 <= p.score <= 1.0 for p in out)
    assert all(p.retriever == RetrieverName.HYBRID for p in out)
    assert {p.chunk.doc_id for p in out} == {"api-usage-limits-v2", "faq-legacy"}
    assert "rrf" in out[0].score_breakdown


def test_retrieve_before_build_raises(settings):
    with pytest.raises(IndexNotBuiltError):
        hybrid.HybridRetriever(settings.retrieval, embedder=FakeEncoder(), glossary={}).retrieve("anything")


def test_doc_id_filter(settings, chunks):
    out = make_retriever(settings, chunks).retrieve("requests per minute", top_k=5, doc_ids=["faq-legacy"])
    assert out and all(p.chunk.doc_id == "faq-legacy" for p in out)


def test_single_modes_for_ablation(settings, chunks):
    r = make_retriever(settings, chunks)
    assert r.retrieve("password reset", mode="bm25")[0].retriever == RetrieverName.BM25
    assert r.retrieve("password reset", mode="dense")[0].retriever == RetrieverName.DENSE


def test_falls_back_to_bm25_when_model_unavailable(settings, chunks):
    r = make_retriever(settings, chunks, encoder=BrokenEncoder())
    assert not r.has_dense
    out = r.retrieve("password reset")
    assert out[0].retriever == RetrieverName.BM25
    assert out[0].chunk.doc_id == "account-access" and 0.0 < out[0].score < 1.0
    with pytest.raises(RetrievalError):
        r.retrieve("password reset", mode="dense")


def test_reranker_reorders_candidates(settings, chunks):
    r = make_retriever(settings, chunks, reranker=ReverseReranker())
    out = r.retrieve("requests per minute", top_k=3)
    assert out[0].retriever == RetrieverName.RERANK
    assert out[0].score == pytest.approx(out[0].score_breakdown["rerank"], abs=1e-4)
    assert [p.score for p in out] == sorted((p.score for p in out), reverse=True)


def test_query_expansion_helps_bm25(settings):
    chunk = Chunk(chunk_id="d::0000", doc_id="d", doc_title="Guide", source="s", section="Signing in",
                  text="Use your work email on the sign in page.", ordinal=0)
    other = Chunk(chunk_id="e::0000", doc_id="e", doc_title="Billing", source="s", section="Invoices",
                  text="Invoices are issued monthly.", ordinal=0)
    r = hybrid.HybridRetriever(settings.retrieval, embedder=BrokenEncoder(),
                               glossary={"sign in": ["log in", "login"]})
    r.build([chunk, other])
    assert r.retrieve("How do I login?")[0].chunk.doc_id == "d"


# --------------------------------------------------------------- real model
def test_real_model_handles_paraphrase(settings, chunks):
    pytest.importorskip("sentence_transformers")
    r = hybrid.HybridRetriever(settings.retrieval, glossary={})
    r.build(chunks)
    if not r.has_dense:
        pytest.skip("embedding model could not be loaded (offline?)")
    out = r.retrieve("I forgot my login credentials, what now?", top_k=3, mode="dense")
    assert out[0].chunk.doc_id == "account-access"
