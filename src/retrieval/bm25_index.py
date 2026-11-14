"""Lexical retrieval with BM25 (``rank_bm25``). OWNER: Member 3.

Good at exact terms: error codes, product names, "HTTP 429", plan names.
"""

from __future__ import annotations

from src.schemas import Chunk


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Index ``section + ' ' + text`` of each chunk using query_processing.tokenize."""
        raise NotImplementedError("Member 3: implement BM25Index.build")

    def search(self, query: str, top_n: int) -> list[tuple[Chunk, float]]:
        """Return (chunk, raw_bm25_score) sorted desc. Raise IndexNotBuiltError if empty."""
        raise NotImplementedError("Member 3: implement BM25Index.search")
