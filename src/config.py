"""Configuration loading.

* Non-secret tunables live in ``config/settings.yaml`` (committed).
* Secrets live in environment variables / a local ``.env`` (NEVER committed).
* Environment variables override YAML: ``QA_RETRIEVAL__TOP_K=8`` overrides
  ``retrieval.top_k``.

Usage::

    from src.config import get_settings
    settings = get_settings()
    settings.retrieval.top_k
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class IngestionSettings(BaseModel):
    supported_extensions: list[str] = [".pdf", ".html", ".htm", ".md", ".markdown", ".txt"]
    chunk_max_tokens: int = 300
    chunk_overlap_tokens: int = 50
    min_chunk_chars: int = 40


class RetrievalSettings(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    candidate_pool: int = 30
    rrf_k: int = 60
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    glossary_path: str = "config/glossary.yaml"
    index_dir: str = "models/index"


class GenerationSettings(BaseModel):
    provider: str = "anthropic"  # "anthropic" | "gemini" | "sarvam" | "extractive" (offline, no key needed)
    model: str = "claude-opus-5-5"
    max_tokens: int = 4096
    effort: str = "low"
    max_context_passages: int = 6
    timeout_seconds: float = 60.0


class ReasoningSettings(BaseModel):
    min_top_score_for_answer: float = 0.35
    high_confidence_threshold: float = 0.75
    medium_confidence_threshold: float = 0.5
    min_groundedness: float = 0.7


class EvaluationSettings(BaseModel):
    k: int = 5
    questions_path: str = "data/eval/questions.jsonl"
    reports_dir: str = "reports"


class Settings(BaseModel):
    log_level: str = "INFO"
    data_dir: str = "data/sample_docs"
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    reasoning: ReasoningSettings = Field(default_factory=ReasoningSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    def resolve(self, relative: str) -> Path:
        """Resolve a repo-relative path from settings to an absolute path."""
        return (PROJECT_ROOT / relative).resolve()


def _apply_env_overrides(raw: dict, prefix: str = "QA_") -> dict:
    """Map QA_SECTION__KEY=value environment variables onto the settings dict."""
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower().split("__")
        node = raw
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return raw


def load_settings(path: Path | None = None) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings_path = path or DEFAULT_SETTINGS_PATH
    raw: dict = {}
    if settings_path.exists():
        try:
            raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {settings_path}: {exc}") from exc
    raw = _apply_env_overrides(raw)
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def get_logger(name: str) -> logging.Logger:
    """Project-wide logger. Never log secrets or full document contents."""
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(name)
