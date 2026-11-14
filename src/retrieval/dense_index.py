"""Semantic retrieval with sentence-transformer embeddings. OWNER: Member 3.

Good at paraphrases ("forgot my credentials" ~ "password reset"). Embeddings are
L2-normalised so cosine similarity = dot product; a numpy matrix is enough for a
hackathon-sized corpus (no vector database needed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.schemas import Chunk


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return float32 array of shape (len(texts), dim), L2-normalised.
        Load the SentenceTransformer lazily on first call."""
        raise NotImplementedError("Member 3: implement Embedder.encode")


class DenseIndex:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError("Member 3: implement DenseIndex.build")

    def search(self, query: str, top_n: int) -> list[tuple[Chunk, float]]:
        """Return (chunk, cosine_similarity) sorted desc."""
        raise NotImplementedError("Member 3: implement DenseIndex.search")

    def save(self, directory: Path) -> None:
        """Persist embeddings (.npy) + chunks (.jsonl) to models/index/ (git-ignored)."""
        raise NotImplementedError("Member 3: implement DenseIndex.save")

    def load(self, directory: Path) -> None:
        raise NotImplementedError("Member 3: implement DenseIndex.load")
