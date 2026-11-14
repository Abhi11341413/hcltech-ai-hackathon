"""Hybrid retriever: BM25 + dense fused with Reciprocal Rank Fusion (FR-03). OWNER: Member 3.

    rrf(d) = sum over retrievers r of 1 / (k + rank_r(d))      (k = 60 by default)

Why fuse by RANK, not score: BM25 scores are unbounded (0..~15) and cosine scores live
in [-1, 1]; adding them needs fragile normalisation. RRF only uses positions, so a chunk
ranked well by both retrievers wins, and one strong signal is still enough to surface it.

Optional second stage: cross-encoder re-ranking of the fused candidate pool.

``RetrievedPassage.score`` is normalised to [0, 1] because Member 1's confidence module
thresholds on it: for HYBRID it is the passage's dense cosine clipped to [0, 1] (an
absolute "how related is this text" signal, unlike RRF which is relative); for RERANK it
is the sigmoid of the cross-encoder logit. Raw per-signal scores go into ``score_breakdown``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT, RetrievalSettings, get_logger
from src.errors import ConfigError, IndexNotBuiltError, RetrievalError
from src.retrieval.bm25_index import BM25Index, index_text
from src.retrieval.dense_index import DenseIndex, Embedder
from src.retrieval.query_processing import expand_query, load_glossary
from src.schemas import Chunk, RetrievedPassage, RetrieverName

logger = get_logger(__name__)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Fuse several ranked lists of chunk_ids. Returns {chunk_id: rrf_score}.

    Rank positions start at 1. An id missing from a list simply gets nothing from it.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + position)
    return fused


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


class HybridRetriever:
    """BM25 + dense retrieval over the same chunks, fused with RRF.

    Args:
        settings: ``Settings.retrieval``.
        embedder: optional pre-built Embedder (tests inject a fake one; the API can share
            one loaded model between rebuilds). Defaults to ``Embedder(settings.embedding_model)``.
        mode: HYBRID (default), or BM25 / DENSE for the retrieval ablation (FR-08).
    """

    def __init__(self, settings: RetrievalSettings, embedder: Embedder | None = None,
                 mode: RetrieverName = RetrieverName.HYBRID) -> None:
        self.settings = settings
        self.mode = mode
        self._embedder = embedder or Embedder(settings.embedding_model)
        self._bm25 = BM25Index()
        self._dense = DenseIndex(self._embedder)
        self._chunks: list[Chunk] = []
        self._built = False
        self._reranker: Any = None
        try:
            self._glossary = load_glossary(_resolve(settings.glossary_path))
        except ConfigError as exc:
            logger.warning("Query expansion disabled: %s", exc.message)
            self._glossary = {}

    # ------------------------------------------------------------------ build / persist

    def build(self, chunks: list[Chunk]) -> None:
        """Build BM25 and dense indexes over the same chunks (replaces any previous index)."""
        self._chunks = list(chunks)
        self._bm25.build(self._chunks)
        self._dense.build(self._chunks)
        self._built = bool(self._chunks)
        logger.info("Hybrid index built over %d chunks", len(self._chunks))

    def save(self, directory: Path | None = None) -> Path:
        """Persist the dense index (BM25 is rebuilt from the saved chunks in < 1 s).
        Returns the directory written to."""
        directory = Path(directory) if directory else _resolve(self.settings.index_dir)
        self._dense.save(directory)
        return directory

    def load(self, directory: Path | None = None) -> None:
        """Load a saved index instead of re-embedding.

        Raises:
            IndexNotBuiltError, RetrievalError: see ``DenseIndex.load``.
        """
        self._dense.load(Path(directory) if directory else _resolve(self.settings.index_dir))
        self._chunks = self._dense.chunks
        self._bm25.build(self._chunks)
        self._built = bool(self._chunks)

    # ------------------------------------------------------------------ retrieve

    def _rank(self, scores: np.ndarray, allowed: np.ndarray, positive_only: bool) -> list[int]:
        idx = [int(i) for i in np.argsort(-scores) if allowed[i] and (scores[i] > 0 or not positive_only)]
        return idx[: self.settings.candidate_pool]

    def _rerank(self, question: str, candidates: list[int]) -> dict[int, float]:
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.settings.reranker_model, device="cpu")
            except Exception as exc:
                raise RetrievalError(f"Could not load reranker '{self.settings.reranker_model}': {exc}") from exc
        logits = self._reranker.predict([(question, index_text(self._chunks[i])) for i in candidates])
        return {i: float(s) for i, s in zip(candidates, logits, strict=True)}

    def retrieve(self, question: str, top_k: int = 5,
                 doc_ids: list[str] | None = None) -> list[RetrievedPassage]:
        """Expand query -> BM25 + dense (candidate_pool each) -> RRF -> optional rerank
        -> top_k RetrievedPassage with rank 1..top_k and retriever=HYBRID/RERANK.

        ``doc_ids`` restricts the search to those documents (unknown ids -> empty list).

        Raises:
            IndexNotBuiltError: build() was never called.
        """
        if not self._built:
            raise IndexNotBuiltError("Retriever has no index; call build() first")
        n = len(self._chunks)
        allowed = np.ones(n, dtype=bool)
        if doc_ids is not None:
            wanted = set(doc_ids)
            allowed = np.array([c.doc_id in wanted for c in self._chunks], dtype=bool)
        if not allowed.any():
            return []

        bm25_scores = np.asarray(self._bm25.scores(expand_query(question, self._glossary)), dtype=np.float32)
        dense_scores = np.asarray(self._dense.similarities(question), dtype=np.float32)
        bm25_rank = self._rank(bm25_scores, allowed, positive_only=True)
        dense_rank = self._rank(dense_scores, allowed, positive_only=False)

        if self.mode == RetrieverName.BM25:
            rankings = [bm25_rank]
        elif self.mode == RetrieverName.DENSE:
            rankings = [dense_rank]
        else:
            rankings = [bm25_rank, dense_rank]
        fused = reciprocal_rank_fusion([[str(i) for i in r] for r in rankings], k=self.settings.rrf_k)
        order = sorted((int(i) for i in fused), key=lambda i: (fused[str(i)], dense_scores[i]), reverse=True)

        retriever = self.mode
        rerank_scores: dict[int, float] = {}
        if self.settings.use_reranker and self.mode == RetrieverName.HYBRID and order:
            rerank_scores = self._rerank(question, order[: self.settings.candidate_pool])
            order = sorted(rerank_scores, key=rerank_scores.get, reverse=True)
            retriever = RetrieverName.RERANK

        bm25_pos = {i: p for p, i in enumerate(bm25_rank, start=1)}
        dense_pos = {i: p for p, i in enumerate(dense_rank, start=1)}
        passages: list[RetrievedPassage] = []
        for rank, i in enumerate(order[:top_k], start=1):
            breakdown = {"bm25": float(bm25_scores[i]), "dense": float(dense_scores[i]),
                         "rrf": fused.get(str(i), 0.0),
                         "bm25_rank": float(bm25_pos.get(i, 0)), "dense_rank": float(dense_pos.get(i, 0))}
            if i in rerank_scores:
                breakdown["rerank"] = rerank_scores[i]
                score = 1.0 / (1.0 + math.exp(-rerank_scores[i]))
            else:
                score = float(min(1.0, max(0.0, dense_scores[i])))
            passages.append(RetrievedPassage(chunk=self._chunks[i], score=score, rank=rank,
                                             retriever=retriever, score_breakdown=breakdown))
        return passages

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)
