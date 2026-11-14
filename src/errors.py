"""Shared exception hierarchy. Raise these, never bare Exception.

Each error carries a stable machine-readable ``error_code`` that the API layer
maps to an HTTP status and returns inside ``schemas.ErrorResponse``.
"""

from __future__ import annotations


class AssistantError(Exception):
    """Base class for all project errors."""

    error_code: str = "INTERNAL_ERROR"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# --- Ingestion (Member 2) ---------------------------------------------------
class IngestionError(AssistantError):
    error_code = "INGESTION_ERROR"
    http_status = 422


class UnsupportedFormatError(IngestionError):
    error_code = "UNSUPPORTED_FORMAT"
    http_status = 415


class EmptyDocumentError(IngestionError):
    error_code = "EMPTY_DOCUMENT"


# --- Retrieval (Member 3) ---------------------------------------------------
class RetrievalError(AssistantError):
    error_code = "RETRIEVAL_ERROR"


class IndexNotBuiltError(RetrievalError):
    error_code = "INDEX_NOT_BUILT"
    http_status = 409


# --- Generation (Member 4) --------------------------------------------------
class LLMError(AssistantError):
    """Network / API / refusal problems talking to the LLM. Caller may fall back."""

    error_code = "LLM_ERROR"
    http_status = 502


class LLMOutputParseError(LLMError):
    """LLM returned output that does not match the expected JSON schema."""

    error_code = "LLM_OUTPUT_INVALID"


# --- Config / validation ----------------------------------------------------
class ConfigError(AssistantError):
    error_code = "CONFIG_ERROR"


class ValidationFailedError(AssistantError):
    error_code = "VALIDATION_FAILED"
    http_status = 400
