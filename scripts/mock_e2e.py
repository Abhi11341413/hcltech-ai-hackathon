"""End-to-end MOCK walkthrough: shows how the four modules connect.

Runs today, before any real module exists, using only src/mocks.py.
Each stage prints what it receives and returns, in the same order as QAPipeline.ask.

    python -m scripts.mock_e2e
"""

from __future__ import annotations

from src import mocks
from src.schemas import AnswerStatus, Chunk, QAResponse, QueryRequest


def stage_ingest() -> list[Chunk]:                       # Member 2
    doc = mocks.mock_parsed_document()
    print(f"[M2 ingestion] ParsedDocument '{doc.doc_id}' with {len(doc.sections)} sections")
    chunks = mocks.mock_chunks()
    print(f"[M2 chunking ] -> {len(chunks)} Chunk objects: {[c.chunk_id for c in chunks]}")
    return chunks


def stage_retrieve(request: QueryRequest, chunks: list[Chunk]):  # Member 3
    passages = mocks.mock_passages()[: request.top_k]
    for p in passages:
        print(f"[M3 retrieval] rank {p.rank} score {p.score:.2f} {p.chunk.chunk_id}")
    return passages


def stage_generate(request: QueryRequest, passages):    # Member 4
    answer = mocks.mock_generated_answer()
    print(f"[M4 generation] {len(answer.claims)} claims, {len(answer.citations)} citations, "
          f"groundedness={answer.groundedness_score}")
    return answer


def stage_decide(request: QueryRequest, passages, answer) -> QAResponse:  # Member 1
    conflicts = mocks.mock_conflicts()
    confidence = mocks.mock_confidence()
    status = AnswerStatus.CONFLICT if conflicts else AnswerStatus.ANSWERED
    print(f"[M1 reasoning ] conflicts={len(conflicts)} confidence={confidence.level.value} status={status.value}")
    return QAResponse(question=request.question, status=status, answer=answer.answer_text,
                      citations=answer.citations, conflicts=conflicts, confidence=confidence,
                      retrieved=passages)


def run() -> QAResponse:
    request = QueryRequest(question=mocks.MOCK_QUESTION, top_k=3)
    chunks = stage_ingest()
    passages = stage_retrieve(request, chunks)
    answer = stage_generate(request, passages)
    response = stage_decide(request, passages, answer)
    print("\n[M2 API] POST /ask would return:\n" + response.model_dump_json(indent=2, exclude={"retrieved"}))
    return response


if __name__ == "__main__":
    run()
