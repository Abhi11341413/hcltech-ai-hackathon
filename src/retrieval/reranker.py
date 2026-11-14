"""Optional second-stage re-ranking with a cross-encoder. OWNER: Member 3.

A bi-encoder (dense index) embeds question and passage separately: fast, but coarse.
A cross-encoder reads question + passage together and outputs one relevance logit:
slower, more accurate. We only apply it to the small fused candidate pool.
Disabled by default (``retrieval.use_reranker: false``); enable it if the
ablation shows it helps.
"""

from __future__ import annotations

import math
from typing import Protocol

from src.config import get_logger
from src.errors import RetrievalError
from src.retrieval.bm25_index import chunk_index_text
from src.schemas import Chunk

logger = get_logger(__name__)


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        """Return one relevance probability in [0, 1] per chunk."""
        ...


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RetrievalError("sentence-transformers is not installed") from exc
            logger.info("Loading reranker %s", self.model_name)
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                raise RetrievalError(f"Could not load reranker {self.model_name}: {exc}") from exc
        return self._model

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        logits = self._load().predict([(query, chunk_index_text(c)) for c in chunks], show_progress_bar=False)
        return [sigmoid(float(x)) for x in logits]
