"""Semantic retrieval with sentence-transformer embeddings. OWNER: Member 3.

Good at paraphrases ("forgot my credentials" ~ "password reset"). Embeddings are
L2-normalised, so cosine similarity = dot product, and a numpy matrix is enough for a
hackathon-sized corpus (no vector database needed: 10k chunks x 384 dims = 15 MB).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from src.config import get_logger
from src.errors import IndexNotBuiltError, RetrievalError
from src.retrieval.bm25_index import chunk_index_text
from src.schemas import Chunk

logger = get_logger(__name__)


class TextEncoder(Protocol):
    """Anything that turns texts into L2-normalised vectors (real model or test fake)."""

    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class Embedder:
    """Wraps a SentenceTransformer model, loaded lazily on first use (the first load
    downloads ~90 MB for all-MiniLM-L6-v2 and caches it in the HuggingFace cache)."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RetrievalError("sentence-transformers is not installed") from exc
            logger.info("Loading embedding model %s", self.model_name)
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # network/model errors vary by backend
                raise RetrievalError(f"Could not load embedding model {self.model_name}: {exc}") from exc
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return float32 array of shape ``(len(texts), dim)``, rows L2-normalised."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._load().encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


class DenseIndex:
    def __init__(self, embedder: TextEncoder) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        """Embed ``doc_title + section + text`` for every chunk."""
        self._chunks = list(chunks)
        self._matrix = self.embedder.encode([chunk_index_text(c) for c in self._chunks]) if chunks else None

    def search(self, query: str, top_n: int) -> list[tuple[Chunk, float]]:
        """Return ``(chunk, cosine_similarity)`` sorted descending. Cosine is in [-1, 1].

        Raises:
            IndexNotBuiltError: ``build``/``load`` was never called or the index is empty.
        """
        if self._matrix is None or not self._chunks:
            raise IndexNotBuiltError("Dense index is empty: call build() or load() first")
        query_vec = self.embedder.encode([query])[0]
        sims = self._matrix @ query_vec
        top = np.argsort(-sims)[:top_n]
        return [(self._chunks[i], float(sims[i])) for i in top]

    def save(self, directory: Path) -> None:
        """Persist embeddings (.npy), chunks (.jsonl) and metadata to ``directory``."""
        if self._matrix is None:
            raise IndexNotBuiltError("Nothing to save: dense index is empty")
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self._matrix)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(chunk.model_dump_json() + "\n")
        meta = {"model_name": self.embedder.model_name, "num_chunks": len(self._chunks),
                "dim": int(self._matrix.shape[1])}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, directory: Path) -> None:
        """Load an index saved by ``save``.

        Raises:
            IndexNotBuiltError: files are missing.
            RetrievalError: the index was built with a different embedding model.
        """
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise IndexNotBuiltError(f"No saved index in {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["model_name"] != self.embedder.model_name:
            raise RetrievalError(
                f"Index built with {meta['model_name']}, but embedder is {self.embedder.model_name}; rebuild it"
            )
        self._matrix = np.load(directory / "embeddings.npy")
        with (directory / "chunks.jsonl").open(encoding="utf-8") as fh:
            self._chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)
