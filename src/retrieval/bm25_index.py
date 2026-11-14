"""Lexical retrieval with BM25 (``rank_bm25.BM25Okapi``). OWNER: Member 3.

Good at exact terms: error codes ("ERR-403"), product names, "HTTP 429", plan names.

Each chunk is indexed as ``doc_title + section + text`` so that a passage whose body
never repeats the topic (e.g. "Each account is limited to 100...") can still be found
through its heading ("LIMITS").

IDF choice: rank_bm25's default IDF ``log((N - n + 0.5) / (n + 0.5))`` is exactly 0
for a term found in half of the documents, and it floors only negative values, so such a
term scores LOWER than a term found in every document. In small or filtered corpora that
silently hides real matches. We use the Lucene / Elasticsearch IDF
``log(1 + (N - n + 0.5) / (n + 0.5))``, which is always positive and still rewards rare terms.
"""

from __future__ import annotations

import math

from rank_bm25 import BM25Okapi

from src.errors import IndexNotBuiltError
from src.retrieval.query_processing import tokenize
from src.schemas import Chunk


class LuceneBM25(BM25Okapi):
    """BM25Okapi with the always-positive Lucene IDF (see module docstring)."""

    def _calc_idf(self, nd: dict[str, int]) -> None:
        for word, doc_freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5))
        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0


def chunk_index_text(chunk: Chunk) -> str:
    """Text used to index a chunk (shared by BM25 and dense indexes)."""
    return f"{chunk.doc_title}. {chunk.section}. {chunk.text}"


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._bm25: LuceneBM25 | None = None

    def build(self, chunks: list[Chunk]) -> None:
        """Index ``doc_title + section + text`` of each chunk using ``tokenize``."""
        self._chunks = list(chunks)
        if not self._chunks:
            self._bm25 = None
            return
        corpus = [tokenize(chunk_index_text(c)) or ["_empty_"] for c in self._chunks]
        self._bm25 = LuceneBM25(corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_n: int) -> list[tuple[Chunk, float]]:
        """Return ``(chunk, raw_bm25_score)`` sorted by score, descending.

        Chunks with score 0 (no query term in common) are not returned.

        Raises:
            IndexNotBuiltError: ``build`` was never called or was called with no chunks.
        """
        if self._bm25 is None:
            raise IndexNotBuiltError("BM25 index is empty: call build() with chunks first")
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self._chunks[i], float(scores[i])) for i in order[:top_n] if scores[i] > 0]

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)
