"""Lexical retrieval with BM25 (``rank_bm25``). OWNER: Member 3.

Good at exact terms: error codes, product names, "HTTP 429", plan names.
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.errors import IndexNotBuiltError
from src.retrieval.query_processing import tokenize
from src.schemas import Chunk


def index_text(chunk: Chunk) -> str:
    """Text that both retrievers index: title + heading path + passage.

    Prepending the breadcrumb ("contextual header injection") lets a passage that never
    repeats its topic word still match: the TXT answer "A: Each account is limited to
    100 requests per minute" is found for "rate limit" via its heading "LIMITS".
    ``Chunk.text`` itself is NOT changed, so citation excerpts stay verbatim.
    """
    return f"{chunk.doc_title}\n{chunk.section}\n{chunk.text}"


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        """Index ``doc_title + section + text`` of each chunk using query_processing.tokenize."""
        self._chunks = list(chunks)
        corpus = [tokenize(index_text(c)) or ["<empty>"] for c in self._chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def scores(self, query: str) -> list[float]:
        """Raw BM25 score for every indexed chunk, in index order.

        Raises:
            IndexNotBuiltError: build() was not called or the index is empty.
        """
        if self._bm25 is None:
            raise IndexNotBuiltError("BM25 index is empty; call build() first")
        tokens = tokenize(query)
        if not tokens:
            return [0.0] * len(self._chunks)
        return [float(s) for s in self._bm25.get_scores(tokens)]

    def search(self, query: str, top_n: int) -> list[tuple[Chunk, float]]:
        """Return (chunk, raw_bm25_score) sorted desc; chunks with score <= 0 are omitted
        (they share no query term). Raise IndexNotBuiltError if empty."""
        scored = sorted(zip(self._chunks, self.scores(query), strict=True), key=lambda x: x[1], reverse=True)
        return [(c, s) for c, s in scored if s > 0][:top_n]

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks
