"""Integration tests. Run after every important merge to main."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts import mock_e2e
from src import mocks
from src.api.app import app
from src.errors import IndexNotBuiltError, ValidationFailedError
from src.evaluation import run_eval
from src.pipeline import QAPipeline
from src.schemas import AnswerStatus, Chunk, GeneratedAnswer, QueryRequest, RetrievedPassage


class FakeRetriever:
    """Returns the mock passages; low-scoring / spread passages for special questions."""

    def __init__(self) -> None:
        self.built_with: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.built_with = chunks

    def retrieve(self, question: str, top_k: int = 5, doc_ids: list[str] | None = None) -> list[RetrievedPassage]:
        passages = mocks.mock_passages()
        if "mars" in question.lower():
            return [p.model_copy(update={"score": 0.1}) for p in passages][:top_k]
        if question == "What is the limit?":
            return [passages[0], passages[1].model_copy(update={"score": 0.8})][:top_k]
        if "password" in question.lower():
            pw = passages[2].model_copy(update={"score": 0.9, "rank": 1})
            return [pw][:top_k]
        return passages[:top_k]


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question: str, passages: list[RetrievedPassage], version: str | None = None) -> GeneratedAnswer:
        self.calls += 1
        if "password" in question.lower():
            c = passages[0].chunk
            return GeneratedAnswer(
                answer_text="Open Settings > Security and choose Reset password [1].",
                claims=[{"text": "Open Settings > Security and choose Reset password.", "citation_ids": [1],
                         "supported": True, "support_score": 0.9}],
                citations=[{"citation_id": 1, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "doc_title": c.doc_title,
                            "section": c.section, "source": c.source,
                            "excerpt": "open Settings > Security and choose Reset password"}],
                model="fake", groundedness_score=1.0)
        return mocks.mock_generated_answer()


@pytest.fixture
def pipeline(settings, monkeypatch) -> QAPipeline:
    pipe = QAPipeline(settings, retriever=FakeRetriever(), generator=FakeGenerator())
    monkeypatch.setattr("src.ingestion.loaders.load_document", lambda path: mocks.mock_parsed_document())
    monkeypatch.setattr("src.ingestion.chunking.chunk_document", lambda doc, *a, **k: mocks.mock_chunks())
    return pipe


def test_mock_pipeline_contract_connects_all_modules():
    response = mock_e2e.run()
    assert response.status == AnswerStatus.CONFLICT
    assert response.citations and response.conflicts


def test_health_endpoint():
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_ask_before_ingest_raises(settings):
    pipe = QAPipeline(settings, retriever=FakeRetriever(), generator=FakeGenerator())
    with pytest.raises(IndexNotBuiltError):
        pipe.ask(QueryRequest(question="How do I reset my password?"))


def test_ingest_builds_index_and_reports_failures(pipeline, monkeypatch):
    def load(path: Path):
        if path.suffix == ".exe":
            from src.errors import UnsupportedFormatError
            raise UnsupportedFormatError("unsupported")
        return mocks.mock_parsed_document()

    monkeypatch.setattr("src.ingestion.loaders.load_document", load)
    result = pipeline.ingest_paths([Path("a.md"), Path("bad.exe")])
    assert result.documents_ingested == 1 and result.chunks_created == 3
    assert [f.error_code for f in result.failures] == ["UNSUPPORTED_FORMAT"]
    assert len(pipeline.retriever.built_with) == 3
    # re-ingesting the same document replaces it instead of duplicating chunks
    pipeline.ingest_paths([Path("a.md")])
    assert pipeline.num_chunks == 3 and len(pipeline.documents) == 1


def test_ingest_directory_rejects_missing_dir(pipeline, tmp_path):
    with pytest.raises(ValidationFailedError):
        pipeline.ingest_directory(tmp_path / "missing")


def test_conflict_question_end_to_end(pipeline):
    pipeline.ingest_paths([Path("a.md")])
    resp = pipeline.ask(QueryRequest(question=mocks.MOCK_QUESTION))
    assert resp.status == AnswerStatus.CONFLICT
    assert resp.conflicts and set(resp.conflicts[0].chunk_ids) == {"api-usage-limits-v2::0001", "faq-legacy::0004"}
    assert resp.citations and "disagree" in resp.answer
    assert resp.latency_ms is not None and resp.retrieved


def test_answered_question_end_to_end(pipeline):
    pipeline.ingest_paths([Path("a.md")])
    resp = pipeline.ask(QueryRequest(question="How do I reset my password?"))
    assert resp.status == AnswerStatus.ANSWERED and resp.citations and not resp.conflicts


def test_not_found_skips_llm(pipeline):
    pipeline.ingest_paths([Path("a.md")])
    resp = pipeline.ask(QueryRequest(question="Does it run on a Mars rover?"))
    assert resp.status == AnswerStatus.NOT_FOUND and not resp.citations
    assert pipeline.generator.calls == 0


def test_ambiguous_question_asks_back(pipeline):
    pipeline.ingest_paths([Path("a.md")])
    resp = pipeline.ask(QueryRequest(question="What is the limit?"))
    assert resp.status == AnswerStatus.NEEDS_CLARIFICATION
    assert resp.clarifying_question and resp.answer == resp.clarifying_question
    assert pipeline.generator.calls == 0


def test_run_evaluation_and_report(pipeline, tmp_path):
    pipeline.ingest_paths([Path("a.md")])
    questions = mocks.mock_eval_questions()
    questions[4] = questions[4].model_copy(update={"question": "Does it run on a Mars rover?"})
    report = run_eval.run_evaluation(pipeline, questions, k=3)
    by_qid = {r.qid: r for r in report.results}
    assert report.n_questions == 5
    assert by_qid["q01"].predicted_status == AnswerStatus.ANSWERED and by_qid["q01"].recall_at_k == 1.0
    assert by_qid["q03"].predicted_status == AnswerStatus.CONFLICT
    assert by_qid["q04"].predicted_status == AnswerStatus.NEEDS_CLARIFICATION
    assert by_qid["q05"].predicted_status == AnswerStatus.NOT_FOUND
    assert report.metrics["conflict_detection_recall"] == 1.0
    assert set(report.per_category) == {"direct", "paraphrase", "conflict", "ambiguous", "unanswerable"}

    json_path, md_path = run_eval.write_report(report, tmp_path, questions=questions)
    md = md_path.read_text(encoding="utf-8")
    assert json_path.exists() and "## Overall metrics" in md and "## Failure cases" in md


def test_run_evaluation_records_crash(settings, tmp_path):
    class Boom(FakeRetriever):
        def retrieve(self, *a, **k):
            raise RuntimeError("index exploded")

    pipe = QAPipeline(settings, retriever=Boom(), generator=FakeGenerator())
    pipe._chunks = {"x": mocks.mock_chunks()}
    report = run_eval.run_evaluation(pipe, mocks.mock_eval_questions()[:1], k=3)
    assert report.results[0].error and report.metrics["errors"] == 1.0
    assert report.metrics["status_accuracy"] == 0.0


def test_load_questions_real_file(settings):
    questions = run_eval.load_questions(settings.resolve(settings.evaluation.questions_path))
    assert len(questions) >= 20 and len({q.qid for q in questions}) == len(questions)


def test_load_questions_rejects_bad_line(tmp_path):
    bad = tmp_path / "q.jsonl"
    bad.write_text('{"qid": "x"}\n', encoding="utf-8")
    with pytest.raises(ValidationFailedError):
        run_eval.load_questions(bad)


def test_ablation_with_fake_retrievers():
    questions = mocks.mock_eval_questions()
    table = run_eval.run_ablation(mocks.mock_chunks(), questions, 3, {"fake": FakeRetriever()})
    assert table["fake"]["n"] == 3.0 and 0.0 <= table["fake"]["recall@3"] <= 1.0


@pytest.mark.xfail(raises=NotImplementedError, reason="needs M2+M3+M4 merged", strict=False)
def test_real_pipeline_answers_sample_question(settings, sample_docs_dir):
    settings.generation.provider = "extractive"  # no network in CI
    pipe = QAPipeline(settings)
    result = pipe.ingest_directory(sample_docs_dir)
    assert result.documents_ingested >= 3
    resp = pipe.ask(QueryRequest(question="How do I reset my password?"))
    assert resp.status == AnswerStatus.ANSWERED and resp.citations
    assert pipe.ask(QueryRequest(question="Does it run on a Mars rover?")).status == AnswerStatus.NOT_FOUND
