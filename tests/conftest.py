"""Shared pytest fixtures (mock data from src/mocks.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import mocks
from src.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def sample_docs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest.fixture
def chunks():
    return mocks.mock_chunks()


@pytest.fixture
def passages():
    return mocks.mock_passages()


@pytest.fixture
def generated_answer():
    return mocks.mock_generated_answer()


@pytest.fixture
def eval_questions():
    return mocks.mock_eval_questions()
