"""End-to-end orchestrator. OWNER: Member 1.

    ingest(paths):  loaders (M2) -> chunking (M2) -> HybridRetriever.build (M3)
    ask(request):   retrieve (M3) -> [early NOT_FOUND / NEEDS_CLARIFICATION]
                    -> generate (M4) -> detect_conflicts (M1)
                    -> assess_confidence + decide_status (M1) -> QAResponse

Dependencies are injected so each stage can be swapped for a mock in tests.
The API layer (Member 2) only ever calls ``QAPipeline``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.config import Settings
from src.schemas import Chunk, GeneratedAnswer, IngestResult, QAResponse, QueryRequest, RetrievedPassage


class Retriever(Protocol):
    def build(self, chunks: list[Chunk]) -> None: ...
    def retrieve(self, question: str, top_k: int = 5,
                 doc_ids: list[str] | None = None) -> list[RetrievedPassage]: ...


class Generator(Protocol):
    def generate(self, question: str, passages: list[RetrievedPassage],
                 version: str | None = None) -> GeneratedAnswer: ...


class QAPipeline:
    def __init__(self, settings: Settings, retriever: Retriever | None = None,
                 generator: Generator | None = None) -> None:
        """None -> construct the real HybridRetriever / AnswerGenerator from settings."""
        self.settings = settings
        self.retriever = retriever
        self.generator = generator

    def ingest_paths(self, paths: list[Path]) -> IngestResult:
        """Load + chunk + (re)build the index. Partial failures are reported, not raised."""
        raise NotImplementedError("Member 1: implement QAPipeline.ingest_paths")

    def ingest_directory(self, directory: Path) -> IngestResult:
        raise NotImplementedError("Member 1: implement QAPipeline.ingest_directory")

    def ask(self, request: QueryRequest) -> QAResponse:
        """Never raises for 'no answer' situations - returns NOT_FOUND/NEEDS_CLARIFICATION.
        Measures latency_ms. Raises IndexNotBuiltError if nothing was ingested."""
        raise NotImplementedError("Member 1: implement QAPipeline.ask")
