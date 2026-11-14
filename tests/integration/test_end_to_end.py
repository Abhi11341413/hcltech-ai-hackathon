"""Integration tests. Run after every important merge to main."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scripts import mock_e2e
from src.api.app import app
from src.schemas import AnswerStatus


def test_mock_pipeline_contract_connects_all_modules():
    response = mock_e2e.run()
    assert response.status == AnswerStatus.CONFLICT
    assert response.citations and response.conflicts


def test_health_endpoint():
    assert TestClient(app).get("/health").json()["status"] == "ok"


@pytest.mark.xfail(raises=NotImplementedError, reason="needs M1+M2+M3+M4 merged", strict=False)
def test_real_pipeline_answers_sample_question(settings, sample_docs_dir):
    from src.pipeline import QAPipeline
    from src.schemas import QueryRequest

    settings.generation.provider = "extractive"  # no network in CI
    pipe = QAPipeline(settings)
    result = pipe.ingest_directory(sample_docs_dir)
    assert result.documents_ingested >= 3
    resp = pipe.ask(QueryRequest(question="How do I reset my password?"))
    assert resp.status == AnswerStatus.ANSWERED and resp.citations
    assert pipe.ask(QueryRequest(question="Does it run on a Mars rover?")).status == AnswerStatus.NOT_FOUND
