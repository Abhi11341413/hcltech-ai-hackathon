"""Shared data contracts for the Enterprise Documentation Q&A Assistant.

EVERY module communicates through the models in this file. Changing a field name,
type or meaning here is a TEAM decision (see CLAUDE.md -> "Shared contract changes").

Data flow (who produces what):

    Member 2  files  ──► ParsedDocument ──► list[Chunk]
    Member 3  Chunk + question ──► list[RetrievedPassage]
    Member 4  question + passages ──► GeneratedAnswer (claims + citations)
    Member 1  passages + GeneratedAnswer ──► Conflict[], ConfidenceAssessment ──► QAResponse
    Member 1  EvalQuestion[] + pipeline ──► EvalReport
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    """Base model: forbid unknown fields so typos in field names fail loudly."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FileType(str, Enum):
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    TXT = "txt"


class RetrieverName(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    RERANK = "rerank"


class ConflictType(str, Enum):
    VERSION = "version"          # passages apply to different product versions
    DATE = "date"                # passages have different effective dates
    VALUE = "value"              # passages state different numbers/limits/prices
    CONTRADICTION = "contradiction"  # passages directly contradict each other


class AnswerStatus(str, Enum):
    ANSWERED = "answered"                       # supported answer with citations
    CONFLICT = "conflict"                       # answer shown together with visible conflict
    NEEDS_CLARIFICATION = "needs_clarification"  # ambiguous question -> ask user
    NOT_FOUND = "not_found"                     # knowledge base does not support an answer


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QuestionCategory(str, Enum):
    DIRECT = "direct"
    PARAPHRASE = "paraphrase"
    CONFLICT = "conflict"
    AMBIGUOUS = "ambiguous"
    UNANSWERABLE = "unanswerable"


# ---------------------------------------------------------------------------
# Ingestion (Member 2)
# ---------------------------------------------------------------------------


class Section(_Strict):
    """One logical section of a source document, as extracted by a loader."""

    heading_path: list[str] = Field(
        default_factory=list,
        description='Heading hierarchy, e.g. ["Billing", "Refund policy"]. Empty = preamble.',
    )
    text: str
    page: int | None = Field(default=None, ge=1, description="1-based page (PDF only).")


class ParsedDocument(_Strict):
    """A whole source document after loading + cleaning, before chunking."""

    doc_id: str = Field(description="Stable id, e.g. 'billing-policy-v2'. Derived from file name.")
    title: str
    source: str = Field(description="Relative path or URL of the original file.")
    file_type: FileType
    sections: list[Section]
    version: str | None = Field(default=None, description="Product/doc version if detectable, e.g. '2.0'.")
    effective_date: date | None = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class IngestFailure(_Strict):
    source: str
    error_code: str  # e.g. "UNSUPPORTED_FORMAT", "EMPTY_DOCUMENT", "PARSE_ERROR"
    message: str


class Chunk(_Strict):
    """A searchable passage. The unit that retrieval ranks and citations point to."""

    chunk_id: str = Field(description="'{doc_id}::{ordinal:04d}', e.g. 'billing-policy-v2::0003'.")
    doc_id: str
    doc_title: str
    source: str
    section: str = Field(description="Heading path joined with ' > ', e.g. 'Billing > Refunds'.")
    page: int | None = None
    text: str = Field(min_length=1)
    version: str | None = None
    effective_date: date | None = None
    ordinal: int = Field(ge=0, description="Position of the chunk inside its document.")


class IngestResult(_Strict):
    documents_ingested: int
    chunks_created: int
    failures: list[IngestFailure] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval (Member 3)
# ---------------------------------------------------------------------------


class RetrievedPassage(_Strict):
    chunk: Chunk
    score: float = Field(description="Final score used for ranking (higher = better).")
    rank: int = Field(ge=1)
    retriever: RetrieverName
    score_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-signal scores, e.g. {'bm25': 7.1, 'dense': 0.82, 'rrf': 0.031}.",
    )


# ---------------------------------------------------------------------------
# Generation (Member 4)
# ---------------------------------------------------------------------------


class Citation(_Strict):
    citation_id: int = Field(ge=1, description="Number shown in the answer text as [1], [2] ...")
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    page: int | None = None
    source: str
    excerpt: str = Field(description="Short verbatim quote from the chunk that supports the claim.")


class Claim(_Strict):
    text: str
    citation_ids: list[int] = Field(default_factory=list)
    supported: bool | None = Field(default=None, description="Set by groundedness verifier.")
    support_score: float | None = Field(default=None, ge=0.0, le=1.0)


class GeneratedAnswer(_Strict):
    answer_text: str = Field(description="Concise answer with inline [n] citation markers.")
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    insufficient_evidence: bool = Field(
        default=False, description="True when the LLM itself says evidence does not answer the question."
    )
    llm_reported_conflict: bool = False
    model: str = Field(description="Model id or 'extractive-fallback'.")
    groundedness_score: float | None = Field(default=None, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Reasoning / decision layer (Member 1)
# ---------------------------------------------------------------------------


class Conflict(_Strict):
    conflict_type: ConflictType
    description: str
    chunk_ids: list[str] = Field(min_length=2)
    values: dict[str, str] = Field(
        default_factory=dict, description="chunk_id -> the conflicting value/version/date it states."
    )


class ConfidenceAssessment(_Strict):
    score: float = Field(ge=0.0, le=1.0)
    level: ConfidenceLevel
    reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API boundary (Member 2 exposes, Member 1 orchestrates)
# ---------------------------------------------------------------------------


class QueryRequest(_Strict):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_ids: list[str] | None = Field(default=None, description="Optional filter.")
    version: str | None = Field(default=None, description="Optional product-version context.")

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question too short")
        return v


class QAResponse(_Strict):
    question: str
    status: AnswerStatus
    answer: str = Field(description="Answer text, conflict summary, clarifying question or not-found message.")
    citations: list[Citation] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    confidence: ConfidenceAssessment
    clarifying_question: str | None = None
    retrieved: list[RetrievedPassage] = Field(default_factory=list, description="Top passages, for transparency.")
    latency_ms: float | None = None


# ---------------------------------------------------------------------------
# Evaluation (Member 1 runs, everyone contributes questions)
# ---------------------------------------------------------------------------


class RelevanceLabel(_Strict):
    """Gold label at doc+section granularity so labels survive re-chunking."""

    doc_id: str
    section: str | None = Field(default=None, description="Section path; None = any section of the doc.")
    grade: int = Field(default=1, ge=1, le=3, description="1 relevant, 2 highly, 3 essential.")


class EvalQuestion(_Strict):
    qid: str
    question: str
    category: QuestionCategory
    expected_status: AnswerStatus
    relevant: list[RelevanceLabel] = Field(default_factory=list)
    reference_answer: str | None = None
    notes: str | None = None


class EvalQuestionResult(_Strict):
    qid: str
    category: QuestionCategory
    expected_status: AnswerStatus
    predicted_status: AnswerStatus
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    ndcg_at_k: float | None = None
    groundedness: float | None = None
    citation_precision: float | None = None
    latency_ms: float | None = None
    error: str | None = None


class EvalReport(_Strict):
    k: int
    n_questions: int
    metrics: dict[str, float] = Field(description="Aggregates, e.g. {'recall@5': 0.9, 'mrr': 0.78, ...}.")
    per_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    results: list[EvalQuestionResult]
    retriever_config: dict[str, str | int | float | bool] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Standard error envelope returned by the API
# ---------------------------------------------------------------------------


class ErrorResponse(_Strict):
    error_code: str
    message: str
    details: dict[str, str] | None = None
