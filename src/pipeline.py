"""End-to-end orchestrator. OWNER: Member 1.

    ingest(paths):  loaders (M2) -> chunking (M2) -> HybridRetriever.build (M3)
    ask(request):   retrieve (M3) -> [early NOT_FOUND / NEEDS_CLARIFICATION]
                    -> generate (M4) -> detect_conflicts (M1)
                    -> assess_confidence + decide_status (M1) -> QAResponse

Dependencies are injected so each stage can be swapped for a mock in tests.
The API layer (Member 2) only ever calls ``QAPipeline``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from src.config import Settings, get_logger
from src.errors import AssistantError, IndexNotBuiltError, ValidationFailedError
from src.reasoning.confidence import (
    NOT_FOUND_MESSAGE,
    assess_confidence,
    build_clarifying_question,
    decide_status,
    pre_generation_status,
)
from src.reasoning.conflicts import detect_conflicts
from src.schemas import (
    AnswerStatus,
    Chunk,
    GeneratedAnswer,
    IngestFailure,
    IngestResult,
    ParsedDocument,
    QAResponse,
    QueryRequest,
    RetrievedPassage,
)

logger = get_logger(__name__)


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
        """None -> construct the real HybridRetriever / AnswerGenerator from settings.

        Imports are local so tests that inject fakes never load sentence-transformers or anthropic.
        """
        self.settings = settings
        if retriever is None:
            from src.retrieval.hybrid import HybridRetriever
            retriever = HybridRetriever(settings.retrieval)
        if generator is None:
            from src.generation.generator import AnswerGenerator
            generator = AnswerGenerator(settings.generation)
        self.retriever: Retriever = retriever
        self.generator: Generator = generator
        self._documents: dict[str, ParsedDocument] = {}
        self._chunks: dict[str, list[Chunk]] = {}   # doc_id -> chunks (re-ingesting a doc replaces it)

    # ------------------------------------------------------------------ ingestion
    @property
    def documents(self) -> list[ParsedDocument]:
        """Currently indexed documents (for GET /documents)."""
        return list(self._documents.values())

    @property
    def num_chunks(self) -> int:
        return sum(len(c) for c in self._chunks.values())

    def all_chunks(self) -> list[Chunk]:
        return [c for chunks in self._chunks.values() for c in chunks]

    def _index(self, docs: list[ParsedDocument], failures: list[IngestFailure]) -> IngestResult:
        """Chunk the new documents, merge them into the corpus and rebuild the retriever index."""
        from src.ingestion.chunking import chunk_document

        ing = self.settings.ingestion
        new_chunks = new_docs = 0
        for doc in docs:
            try:
                chunks = chunk_document(doc, ing.chunk_max_tokens, ing.chunk_overlap_tokens, ing.min_chunk_chars)
            except AssistantError as exc:
                failures.append(IngestFailure(source=doc.source, error_code=exc.error_code, message=exc.message))
                continue
            if not chunks:
                failures.append(IngestFailure(source=doc.source, error_code="EMPTY_DOCUMENT",
                                              message="document produced no chunks"))
                continue
            self._documents[doc.doc_id] = doc
            self._chunks[doc.doc_id] = chunks
            new_chunks += len(chunks)
            new_docs += 1
        if new_chunks:
            self.retriever.build(self.all_chunks())
        logger.info("ingested %d docs, %d chunks, %d failures (corpus: %d docs, %d chunks)",
                    new_docs, new_chunks, len(failures), len(self._documents), self.num_chunks)
        return IngestResult(documents_ingested=new_docs, chunks_created=new_chunks, failures=failures)

    def ingest_paths(self, paths: list[Path]) -> IngestResult:
        """Load + chunk + (re)build the index. Partial failures are reported, not raised."""
        from src.ingestion.loaders import load_document

        docs: list[ParsedDocument] = []
        failures: list[IngestFailure] = []
        for path in paths:
            try:
                docs.append(load_document(Path(path)))
            except AssistantError as exc:
                failures.append(IngestFailure(source=str(path), error_code=exc.error_code, message=exc.message))
            except Exception as exc:  # noqa: BLE001 - one bad file must never abort the batch (CLAUDE.md §8)
                logger.warning("unexpected error loading %s: %s", Path(path).name, type(exc).__name__)
                failures.append(IngestFailure(source=str(path), error_code="PARSE_ERROR", message=str(exc)))
        return self._index(docs, failures)

    def ingest_directory(self, directory: Path) -> IngestResult:
        """Ingest every supported file in ``directory``.

        Raises:
            ValidationFailedError: the path is not an existing directory.
        """
        from src.ingestion.loaders import load_directory

        directory = Path(directory)
        if not directory.is_dir():
            raise ValidationFailedError(f"not a directory: {directory.name}")
        docs, failures = load_directory(directory)
        return self._index(docs, list(failures))

    # ------------------------------------------------------------------ question answering
    def ask(self, request: QueryRequest) -> QAResponse:
        """Never raises for 'no answer' situations - returns NOT_FOUND/NEEDS_CLARIFICATION.
        Measures latency_ms. Raises IndexNotBuiltError if nothing was ingested."""
        return self.ask_detailed(request)[0]

    def ask_detailed(self, request: QueryRequest) -> tuple[QAResponse, GeneratedAnswer | None]:
        """Like ``ask`` but also returns the raw ``GeneratedAnswer`` (None on early exit),
        which the evaluation needs for groundedness."""
        if not self._chunks:
            raise IndexNotBuiltError("No documents have been ingested yet. Ingest documents first.")
        started = time.perf_counter()
        reasoning = self.settings.reasoning

        passages = self.retriever.retrieve(request.question, top_k=request.top_k, doc_ids=request.doc_ids)
        early = pre_generation_status(request.question, passages, reasoning)
        if early is not None:
            logger.info("early exit %s after retrieval (%d passages)", early.value, len(passages))
            clarify = build_clarifying_question(request.question, passages) \
                if early == AnswerStatus.NEEDS_CLARIFICATION else None
            response = QAResponse(
                question=request.question, status=early, answer=clarify or NOT_FOUND_MESSAGE,
                confidence=assess_confidence(passages, None, [], reasoning),
                clarifying_question=clarify, retrieved=passages,
                latency_ms=_elapsed_ms(started),
            )
            return response, None

        generated = self.generator.generate(request.question, passages, version=request.version)
        conflicts = detect_conflicts(request.question, passages, generated,
                                     min_score=reasoning.min_top_score_for_answer)
        status = decide_status(request.question, passages, generated, conflicts, reasoning)
        confidence = assess_confidence(passages, generated, conflicts, reasoning)

        if status == AnswerStatus.CONFLICT:
            notes = "\n".join(f"- {c.description}" for c in conflicts)
            answer = f"{generated.answer_text}\n\nThe sources disagree, so check which applies to you:\n{notes}"
            citations = generated.citations
        elif status == AnswerStatus.ANSWERED:
            answer, citations = generated.answer_text, generated.citations
        else:  # NOT_FOUND after generation: insufficient evidence or low groundedness -> do not bluff
            answer, citations, conflicts = NOT_FOUND_MESSAGE, [], []

        logger.info("status=%s confidence=%.2f conflicts=%d citations=%d",
                    status.value, confidence.score, len(conflicts), len(citations))
        response = QAResponse(
            question=request.question, status=status, answer=answer, citations=citations,
            conflicts=conflicts if status == AnswerStatus.CONFLICT else [],
            confidence=confidence, retrieved=passages, latency_ms=_elapsed_ms(started),
        )
        return response, generated


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
