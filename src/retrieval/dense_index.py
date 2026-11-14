"""Semantic retrieval with sentence-transformer embeddings. OWNER: Member 3.

Good at paraphrases ("forgot my credentials" ~ "password reset"). Embeddings are
L2-normalised so cosine similarity = dot product; a numpy matrix is enough for a
hackathon-sized corpus (no vector database needed: 72 chunks x 384 dims = ~110 KB).

On-disk layout written by ``DenseIndex.save`` (``models/index/``, git-ignored)::

    embeddings.npy   float32 (n_chunks, dim)
    chunks.jsonl     one Chunk per line, same order as the matrix rows
    meta.json        {"model": ..., "dim": ..., "n_chunks": ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_logger
from src.errors import IndexNotBuiltError, RetrievalError
from src.retrieval.bm25_index import index_text
from src.schemas import Chunk

logger = get_logger(__name__)

# Some retrieval models expect an instruction prefix on the QUERY side only.
_QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device="cpu")
            except Exception as exc:  # download / load failures of many kinds
                raise RetrievalError(f"Could not load embedding model '{self.model_name}': {exc}") from exc
            logger.info("Loaded embedding model %s", self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return float32 array of shape (len(texts), dim), L2-normalised.
        Load the SentenceTransformer lazily on first call.

        Raises:
            RetrievalError: the model cannot be downloaded or loaded.
        """
        model = self._load()
        if not texts:
            return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
        vectors = model.encode(texts, batch_size=32, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one query (with the model's query prefix, if it needs one) -> shape (dim,)."""
        return self.encode([_QUERY_PREFIXES.get(self.model_name, "") + query])[0]


class DenseIndex:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        """Embed ``doc_title + section + text`` of every chunk (same text BM25 indexes)."""
        self._chunks = list(chunks)
        self._matrix = self.embedder.encode([index_text(c) for c in self._chunks])
        logger.info("Dense index built: %d chunks, dim %d", *self._matrix.shape)

    def similarities(self, query: str) -> np.ndarray:
        """Cosine similarity of the query to every chunk, in index order (range [-1, 1]).

        Raises:
            IndexNotBuiltError: build()/load() was not called or the index is empty.
        """
        if self._matrix is None or len(self._chunks) == 0:
            raise IndexNotBuiltError("Dense index is empty; call build() or load() first")
        return self._matrix @ self.embedder.encode_query(query)

    def search(self, query: str, top_n: int) -> list[tuple[Chunk, float]]:
        """Return (chunk, cosine_similarity) sorted desc."""
        sims = self.similarities(query)
        order = np.argsort(-sims)[:top_n]
        return [(self._chunks[i], float(sims[i])) for i in order]

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def save(self, directory: Path) -> None:
        """Persist embeddings (.npy) + chunks (.jsonl) + meta.json to ``directory``.

        Raises:
            IndexNotBuiltError: nothing to save.
        """
        if self._matrix is None:
            raise IndexNotBuiltError("Nothing to save; call build() first")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self._matrix)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(chunk.model_dump_json() + "\n")
        meta = {"model": self.embedder.model_name, "dim": int(self._matrix.shape[1]),
                "n_chunks": len(self._chunks)}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Saved dense index (%d chunks) to %s", len(self._chunks), directory.name)

    def load(self, directory: Path) -> None:
        """Load an index written by ``save``.

        Raises:
            IndexNotBuiltError: files missing.
            RetrievalError: index was built with a different embedding model, or is corrupt.
        """
        directory = Path(directory)
        files = [directory / n for n in ("embeddings.npy", "chunks.jsonl", "meta.json")]
        if not all(f.is_file() for f in files):
            raise IndexNotBuiltError(f"No saved index in {directory}")
        meta = json.loads(files[2].read_text(encoding="utf-8"))
        if meta.get("model") != self.embedder.model_name:
            raise RetrievalError(f"Index was built with '{meta.get('model')}', "
                                 f"but the configured model is '{self.embedder.model_name}'; rebuild it")
        matrix = np.load(files[0])
        lines = files[1].read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
        if matrix.shape[0] != len(chunks):
            raise RetrievalError("Corrupt index: embeddings and chunks have different lengths")
        self._matrix, self._chunks = matrix.astype(np.float32), chunks
