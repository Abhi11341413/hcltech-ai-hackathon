"""Contract tests for src/schemas.py (shared, must always pass on main)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src import mocks
from src.config import load_settings
from src.schemas import Chunk, Conflict, ConflictType, QAResponse, QueryRequest


def test_mocks_validate_against_schemas():
    assert len(mocks.mock_chunks()) == 3
    assert mocks.mock_passages()[0].rank == 1
    resp = mocks.mock_qa_response()
    assert QAResponse.model_validate_json(resp.model_dump_json()) == resp


def test_unknown_fields_are_rejected():
    data = mocks.mock_chunks()[0].model_dump()
    data["sectoin"] = "typo"
    with pytest.raises(ValidationError):
        Chunk.model_validate(data)


def test_query_request_validation():
    assert QueryRequest(question="  reset password  ").question == "reset password"
    with pytest.raises(ValidationError):
        QueryRequest(question="  ")
    with pytest.raises(ValidationError):
        QueryRequest(question="valid question", top_k=0)


def test_conflict_needs_two_chunks():
    with pytest.raises(ValidationError):
        Conflict(conflict_type=ConflictType.VALUE, description="x", chunk_ids=["only-one"])


def test_mock_citations_reference_mock_chunks():
    chunk_ids = {c.chunk_id for c in mocks.mock_chunks()}
    for cit in mocks.mock_generated_answer().citations:
        assert cit.chunk_id in chunk_ids


def test_settings_load_from_yaml():
    s = load_settings()
    assert s.retrieval.top_k >= 1
    assert s.generation.provider in {"anthropic", "extractive"}
