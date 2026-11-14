"""Member 3 - retrieval tests. Remove the xfail marker once a function is implemented."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.retrieval import hybrid, query_processing
from src.retrieval.dense_index import Embedder
from src.schemas import RetrieverName


class FakeEmbedder(Embedder):
    """Offline stand-in for SentenceTransformer: hashed bag-of-words, L2-normalised.
    Unit tests must not download models (CLAUDE.md section 7)."""

    def __init__(self) -> None:
        super().__init__("fake-hash-embedder")

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in query_processing.tokenize(text):
                out[row, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)



def test_rrf_rewards_agreement():
    scores = hybrid.reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert scores["a"] == pytest.approx(scores["b"])
    assert scores["a"] > scores["c"] > 0 and "d" in scores


def test_expand_query_adds_synonyms():
    glossary = {"password reset": ["forgot password"]}
    assert "password reset" in query_processing.expand_query("I forgot password", glossary)


def test_tokenize_lowercases_and_drops_punctuation():
    assert query_processing.tokenize("HTTP 429, Too Many!") == ["http", "429", "many"]


def test_hybrid_retrieve_returns_ranked_normalised_passages(settings, chunks):
    r = hybrid.HybridRetriever(settings.retrieval, embedder=FakeEmbedder())
    r.build(chunks)
    out = r.retrieve("How many requests per minute are allowed?", top_k=2)
    assert [p.rank for p in out] == [1, 2]
    assert all(0.0 <= p.score <= 1.0 for p in out)
    assert out[0].chunk.doc_id in {"api-usage-limits-v2", "faq-legacy"}


def test_retrieve_before_build_raises(settings):
    from src.errors import IndexNotBuiltError
    with pytest.raises(IndexNotBuiltError):
        hybrid.HybridRetriever(settings.retrieval).retrieve("anything")


def test_doc_id_filter_and_unknown_doc(settings, chunks):
    r = hybrid.HybridRetriever(settings.retrieval, embedder=FakeEmbedder())
    r.build(chunks)
    out = r.retrieve("password reset link", top_k=5, doc_ids=["account-access"])
    assert out and {p.chunk.doc_id for p in out} == {"account-access"}
    assert r.retrieve("password", doc_ids=["no-such-doc"]) == []


def test_bm25_finds_exact_error_code(settings, chunks):
    r = hybrid.HybridRetriever(settings.retrieval, embedder=FakeEmbedder(), mode=RetrieverName.BM25)
    r.build(chunks)
    out = r.retrieve("HTTP 429", top_k=1)
    assert out[0].retriever == RetrieverName.BM25 and "429" in out[0].chunk.text


def test_save_load_roundtrip(settings, chunks, tmp_path):
    r = hybrid.HybridRetriever(settings.retrieval, embedder=FakeEmbedder())
    r.build(chunks)
    r.save(tmp_path)
    r2 = hybrid.HybridRetriever(settings.retrieval, embedder=FakeEmbedder())
    r2.load(tmp_path)
    q = "reset my password"
    assert [p.chunk.chunk_id for p in r.retrieve(q)] == [p.chunk.chunk_id for p in r2.retrieve(q)]
