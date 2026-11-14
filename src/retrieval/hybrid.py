"""Hybrid retriever: BM25 + dense fused with Reciprocal Rank Fusion (FR-03). OWNER: Member 3.

    rrf(d) = sum over retrievers r of 1 / (k + rank_r(d))      (k = 60 by default)

Why fuse by RANK, not score: BM25 scores are unbounded (0..20+) and cosine scores
live in [-1, 1], so adding them needs fragile normalisation. RRF only uses positions,
so a passage ranked high by either retriever rises, and one ranked high by both rises most.

Ordering vs. score (contract with Member 1):
* Passages are ORDERED by RRF (or by the reranker when enabled).
* ``RetrievedPassage.score`` is an ABSOLUTE relevance in [0, 1] that Member 1 can
  threshold for "not found": the dense cosine clipped to [0, 1] (or the reranker
  probability). RRF values are relative ranks and cannot say "nothing is relevant".
  In BM25-only mode the score is ``bm25 / (bm25 + BM25_SATURATION)``.
* Raw signals are kept in ``score_breakdown`` for transparency.

If the embedding model cannot be loaded, the retriever logs a warning and falls back
to BM25-only, so the demo keeps working offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.config import PROJECT_ROOT, RetrievalSettings, get_logger
from src.errors import IndexNotBuiltError, RetrievalError
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex, Embedder, TextEncoder
from src.retrieval.query_processing import expand_query, load_glossary
from src.retrieval.reranker import CrossEncoderReranker, Reranker
from src.schemas import Chunk, RetrievedPassage, RetrieverName

logger = get_logger(__name__)

Mode = Literal["bm25", "dense", "hybrid"]
BM25_SATURATION = 5.0  # BM25 score that maps to 0.5 in BM25-only mode


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Fuse several ranked lists of chunk_ids (best first). Returns ``{chunk_id: rrf_score}``.

    >>> s = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    >>> round(s["b"], 4), round(s["a"], 4)
    (0.0325, 0.0164)
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
    return scores


class HybridRetriever:
    def __init__(self, settings: RetrievalSettings, embedder: TextEncoder | None = None,
                 glossary: dict[str, list[str]] | None = None, reranker: Reranker | None = None) -> None:
        """
        Args:
            settings: ``Settings.retrieval``.
            embedder: inject a fake in tests; default is the SentenceTransformer from settings.
            glossary: default is loaded from ``settings.glossary_path``.
            reranker: default is a CrossEncoder if ``settings.use_reranker`` else none.
        """
        self.settings = settings
        self.glossary = glossary if glossary is not None else load_glossary(PROJECT_ROOT / settings.glossary_path)
        self.bm25 = BM25Index()
        self.dense: DenseIndex | None = DenseIndex(embedder or Embedder(settings.embedding_model))
        if reranker is None and settings.use_reranker:
            reranker = CrossEncoderReranker(settings.reranker_model)
        self.reranker = reranker
        self._chunks: list[Chunk] = []

    # ------------------------------------------------------------------ build
    def build(self, chunks: list[Chunk]) -> None:
        """Build BM25 and dense indexes over the same chunks (dense failure -> BM25-only)."""
        self._chunks = list(chunks)
        self.bm25.build(self._chunks)
        if self.dense is not None:
            try:
                self.dense.build(self._chunks)
            except RetrievalError as exc:
                logger.warning("Dense index unavailable, falling back to BM25-only: %s", exc)
                self.dense = None
        logger.info("Indexed %d chunks (dense=%s)", len(self._chunks), self.dense is not None)

    def save(self, directory: Path | None = None) -> Path:
        """Persist the dense index (BM25 is rebuilt from chunks on load, it takes milliseconds)."""
        directory = directory or PROJECT_ROOT / self.settings.index_dir
        if self.dense is None:
            raise RetrievalError("No dense index to save (running in BM25-only mode)")
        self.dense.save(directory)
        return directory

    def load(self, directory: Path | None = None) -> None:
        directory = directory or PROJECT_ROOT / self.settings.index_dir
        if self.dense is None:
            raise RetrievalError("Cannot load a dense index in BM25-only mode")
        self.dense.load(directory)
        self._chunks = self.dense.chunks
        self.bm25.build(self._chunks)

    # --------------------------------------------------------------- retrieve
    def retrieve(self, question: str, top_k: int = 5, doc_ids: list[str] | None = None,
                 mode: Mode | None = None) -> list[RetrievedPassage]:
        """Expand query -> BM25 + dense -> RRF -> optional rerank -> top_k passages.

        Args:
            question: the user's natural-language question.
            top_k: number of passages to return (ranks 1..top_k).
            doc_ids: optional filter; only chunks from these documents are returned.
            mode: "bm25", "dense" or "hybrid" (default: hybrid, or bm25 if dense is unavailable).
                  Single modes exist for the evaluation ablation.

        Raises:
            IndexNotBuiltError: ``build``/``load`` was never called.
            RetrievalError: ``mode="dense"`` requested but no dense index is available.
        """
        if not self._chunks:
            raise IndexNotBuiltError("Retriever has no chunks: ingest documents first")
        mode = mode or ("hybrid" if self.dense is not None else "bm25")
        if mode in ("dense", "hybrid") and self.dense is None:
            if mode == "dense":
                raise RetrievalError("Dense retrieval requested but the embedding model is unavailable")
            mode = "bm25"

        allowed = set(doc_ids) if doc_ids else None
        n_all = len(self._chunks)

        bm25_hits = self.bm25.search(expand_query(question, self.glossary), n_all) if mode != "dense" else []
        dense_hits = self.dense.search(question, n_all) if mode != "bm25" and self.dense is not None else []
        if allowed is not None:
            bm25_hits = [(c, s) for c, s in bm25_hits if c.doc_id in allowed]
            dense_hits = [(c, s) for c, s in dense_hits if c.doc_id in allowed]

        pool = self.settings.candidate_pool
        bm25_rank = {c.chunk_id: r for r, (c, _) in enumerate(bm25_hits[:pool], start=1)}
        dense_rank = {c.chunk_id: r for r, (c, _) in enumerate(dense_hits[:pool], start=1)}
        bm25_score = {c.chunk_id: s for c, s in bm25_hits}
        dense_score = {c.chunk_id: s for c, s in dense_hits}
        by_id = {c.chunk_id: c for c, _ in bm25_hits + dense_hits}

        rrf = reciprocal_rank_fusion(
            [[c.chunk_id for c, _ in bm25_hits[:pool]], [c.chunk_id for c, _ in dense_hits[:pool]]],
            k=self.settings.rrf_k,
        )
        if mode == "bm25":
            ordered = list(bm25_rank)
        elif mode == "dense":
            ordered = list(dense_rank)
        else:
            ordered = sorted(rrf, key=lambda cid: rrf[cid], reverse=True)

        retriever = {"bm25": RetrieverName.BM25, "dense": RetrieverName.DENSE, "hybrid": RetrieverName.HYBRID}[mode]
        rerank_score: dict[str, float] = {}
        if self.reranker is not None and ordered:
            candidates = ordered[:pool]
            try:
                probs = self.reranker.score(question, [by_id[cid] for cid in candidates])
                rerank_score = dict(zip(candidates, probs, strict=True))
                ordered = sorted(candidates, key=lambda cid: rerank_score[cid], reverse=True)
                retriever = RetrieverName.RERANK
            except RetrievalError as exc:
                logger.warning("Reranker failed, keeping fused order: %s", exc)

        passages: list[RetrievedPassage] = []
        for rank, cid in enumerate(ordered[:top_k], start=1):
            breakdown: dict[str, float] = {}
            if cid in bm25_score:
                breakdown["bm25"] = round(bm25_score[cid], 4)
                if cid in bm25_rank:
                    breakdown["bm25_rank"] = float(bm25_rank[cid])
            if cid in dense_score:
                breakdown["dense"] = round(dense_score[cid], 4)
                if cid in dense_rank:
                    breakdown["dense_rank"] = float(dense_rank[cid])
            if cid in rrf and mode == "hybrid":
                breakdown["rrf"] = round(rrf[cid], 6)
            if cid in rerank_score:
                breakdown["rerank"] = round(rerank_score[cid], 4)
            passages.append(RetrievedPassage(
                chunk=by_id[cid], score=self._absolute_score(cid, bm25_score, dense_score, rerank_score),
                rank=rank, retriever=retriever, score_breakdown=breakdown,
            ))
        return passages

    @staticmethod
    def _absolute_score(cid: str, bm25_score: dict[str, float], dense_score: dict[str, float],
                        rerank_score: dict[str, float]) -> float:
        if cid in rerank_score:
            value = rerank_score[cid]
        elif cid in dense_score:
            value = dense_score[cid]
        else:
            s = bm25_score.get(cid, 0.0)
            value = s / (s + BM25_SATURATION)
        return round(min(1.0, max(0.0, value)), 4)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def has_dense(self) -> bool:
        return self.dense is not None
