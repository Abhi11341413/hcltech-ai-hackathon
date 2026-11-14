"""Hybrid retriever: BM25 + dense fused with Reciprocal Rank Fusion (FR-03). OWNER: Member 3.

    rrf(d) = sum over retrievers r of 1 / (k + rank_r(d))      (k = 60 by default)

Optional second stage: cross-encoder re-ranking of the fused candidate pool.

``RetrievedPassage.score`` MUST be normalised to [0, 1] (e.g. the dense cosine of
the passage, or a min-max of the reranker score) because Member 1's confidence
module thresholds on it. Raw per-signal scores go into ``score_breakdown``.
"""

from __future__ import annotations

from src.config import RetrievalSettings
from src.schemas import Chunk, RetrievedPassage


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Fuse several ranked lists of chunk_ids. Returns {chunk_id: rrf_score}."""
    raise NotImplementedError("Member 3: implement reciprocal_rank_fusion")


class HybridRetriever:
    def __init__(self, settings: RetrievalSettings) -> None:
        self.settings = settings

    def build(self, chunks: list[Chunk]) -> None:
        """Build BM25 and dense indexes over the same chunks."""
        raise NotImplementedError("Member 3: implement HybridRetriever.build")

    def retrieve(self, question: str, top_k: int = 5,
                 doc_ids: list[str] | None = None) -> list[RetrievedPassage]:
        """Expand query -> BM25 + dense (candidate_pool each) -> RRF -> optional rerank
        -> top_k RetrievedPassage with rank 1..top_k and retriever=HYBRID/RERANK.

        Raises:
            IndexNotBuiltError: build() was never called.
        """
        raise NotImplementedError("Member 3: implement HybridRetriever.retrieve")

    @property
    def num_chunks(self) -> int:
        raise NotImplementedError("Member 3: implement HybridRetriever.num_chunks")
