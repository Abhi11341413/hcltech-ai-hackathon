"""Realistic mock objects so every member can develop and test in isolation.

Use these in unit tests and in ``scripts/mock_e2e.py`` until the real upstream
module is merged. They follow ``src/schemas.py`` exactly - if a schema changes,
update the mocks in the same PR.

The scenario: a question about the API rate limit, where v1 and v2 of the docs
disagree (100 vs 300 requests/minute). This exercises retrieval, citation,
conflict detection and confidence in one example.
"""

from __future__ import annotations

from datetime import date

from src.schemas import (
    AnswerStatus,
    Chunk,
    Citation,
    Claim,
    ConfidenceAssessment,
    ConfidenceLevel,
    Conflict,
    ConflictType,
    EvalQuestion,
    FileType,
    GeneratedAnswer,
    ParsedDocument,
    QAResponse,
    QuestionCategory,
    RelevanceLabel,
    RetrievedPassage,
    RetrieverName,
    Section,
)

MOCK_QUESTION = "How many API requests can I make per minute?"


def mock_parsed_document() -> ParsedDocument:
    return ParsedDocument(
        doc_id="api-usage-limits-v2",
        title="API Usage Limits (v2.0)",
        source="data/sample_docs/api_usage_limits_v2.md",
        file_type=FileType.MARKDOWN,
        version="2.0",
        effective_date=date(2026, 7, 1),
        sections=[
            Section(heading_path=["API Usage Limits"], text="This page describes quotas for the v2 API."),
            Section(
                heading_path=["API Usage Limits", "Rate limits"],
                text="Standard plan accounts may send up to 300 requests per minute. "
                "Requests above the limit receive HTTP 429.",
            ),
        ],
    )


def mock_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="api-usage-limits-v2::0001",
            doc_id="api-usage-limits-v2",
            doc_title="API Usage Limits (v2.0)",
            source="data/sample_docs/api_usage_limits_v2.md",
            section="API Usage Limits > Rate limits",
            text="Standard plan accounts may send up to 300 requests per minute. "
            "Requests above the limit receive HTTP 429.",
            version="2.0",
            effective_date=date(2026, 7, 1),
            ordinal=1,
        ),
        Chunk(
            chunk_id="faq-legacy::0004",
            doc_id="faq-legacy",
            doc_title="Support FAQ (v1.x)",
            source="data/sample_docs/support_faq_v1.txt",
            section="Limits",
            text="Q: What is the API rate limit? A: Each account is limited to 100 requests per minute.",
            version="1.x",
            effective_date=date(2025, 3, 15),
            ordinal=4,
        ),
        Chunk(
            chunk_id="account-access::0002",
            doc_id="account-access",
            doc_title="Account Access Guide",
            source="data/sample_docs/account_access.html",
            section="Account Access > Password reset",
            text="To reset your password, open Settings > Security and choose Reset password. "
            "A link valid for 30 minutes is emailed to you.",
            ordinal=2,
        ),
    ]


def mock_passages() -> list[RetrievedPassage]:
    c = mock_chunks()
    return [
        RetrievedPassage(chunk=c[0], score=0.82, rank=1, retriever=RetrieverName.HYBRID,
                         score_breakdown={"bm25": 6.4, "dense": 0.79, "rrf": 0.0325}),
        RetrievedPassage(chunk=c[1], score=0.74, rank=2, retriever=RetrieverName.HYBRID,
                         score_breakdown={"bm25": 7.1, "dense": 0.68, "rrf": 0.0320}),
        RetrievedPassage(chunk=c[2], score=0.12, rank=3, retriever=RetrieverName.HYBRID,
                         score_breakdown={"bm25": 0.4, "dense": 0.21, "rrf": 0.0154}),
    ]


def mock_citations() -> list[Citation]:
    c = mock_chunks()
    return [
        Citation(citation_id=1, chunk_id=c[0].chunk_id, doc_id=c[0].doc_id, doc_title=c[0].doc_title,
                 section=c[0].section, source=c[0].source,
                 excerpt="Standard plan accounts may send up to 300 requests per minute."),
        Citation(citation_id=2, chunk_id=c[1].chunk_id, doc_id=c[1].doc_id, doc_title=c[1].doc_title,
                 section=c[1].section, source=c[1].source,
                 excerpt="Each account is limited to 100 requests per minute."),
    ]


def mock_generated_answer() -> GeneratedAnswer:
    return GeneratedAnswer(
        answer_text="Under the v2.0 API documentation, Standard plan accounts can send up to 300 "
        "requests per minute [1]. The older v1.x FAQ states a limit of 100 requests per minute [2].",
        claims=[
            Claim(text="Standard plan accounts can send up to 300 requests per minute (v2.0).",
                  citation_ids=[1], supported=True, support_score=0.93),
            Claim(text="The v1.x FAQ states a limit of 100 requests per minute.",
                  citation_ids=[2], supported=True, support_score=0.91),
        ],
        citations=mock_citations(),
        insufficient_evidence=False,
        llm_reported_conflict=True,
        model="mock-llm",
        groundedness_score=0.92,
    )


def mock_conflicts() -> list[Conflict]:
    return [
        Conflict(
            conflict_type=ConflictType.VERSION,
            description="Rate limit differs between v2.0 docs (300/min) and v1.x FAQ (100/min).",
            chunk_ids=["api-usage-limits-v2::0001", "faq-legacy::0004"],
            values={"api-usage-limits-v2::0001": "300 requests/minute (v2.0)",
                    "faq-legacy::0004": "100 requests/minute (v1.x)"},
        )
    ]


def mock_confidence() -> ConfidenceAssessment:
    return ConfidenceAssessment(
        score=0.71, level=ConfidenceLevel.MEDIUM,
        reasons=["strong top retrieval score", "all claims supported", "version conflict present"],
    )


def mock_qa_response() -> QAResponse:
    return QAResponse(
        question=MOCK_QUESTION,
        status=AnswerStatus.CONFLICT,
        answer=mock_generated_answer().answer_text,
        citations=mock_citations(),
        conflicts=mock_conflicts(),
        confidence=mock_confidence(),
        retrieved=mock_passages(),
        latency_ms=850.0,
    )


def mock_eval_questions() -> list[EvalQuestion]:
    return [
        EvalQuestion(qid="q01", question="How do I reset my password?", category=QuestionCategory.DIRECT,
                     expected_status=AnswerStatus.ANSWERED,
                     relevant=[RelevanceLabel(doc_id="account-access", section="Account Access > Password reset")]),
        EvalQuestion(qid="q02", question="I forgot my login credentials, what now?",
                     category=QuestionCategory.PARAPHRASE, expected_status=AnswerStatus.ANSWERED,
                     relevant=[RelevanceLabel(doc_id="account-access", section="Account Access > Password reset")]),
        EvalQuestion(qid="q03", question=MOCK_QUESTION, category=QuestionCategory.CONFLICT,
                     expected_status=AnswerStatus.CONFLICT,
                     relevant=[RelevanceLabel(doc_id="api-usage-limits-v2"), RelevanceLabel(doc_id="faq-legacy")]),
        EvalQuestion(qid="q04", question="What is the limit?", category=QuestionCategory.AMBIGUOUS,
                     expected_status=AnswerStatus.NEEDS_CLARIFICATION),
        EvalQuestion(qid="q05", question="Does the product support on-premise Kubernetes deployment?",
                     category=QuestionCategory.UNANSWERABLE, expected_status=AnswerStatus.NOT_FOUND),
    ]
