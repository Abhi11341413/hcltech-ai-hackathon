"""Multi-format document loaders (FR-01). OWNER: Member 2.

Turn PDF / HTML / Markdown / TXT files into ``ParsedDocument`` objects while
preserving title, section (heading hierarchy), page and source metadata.

Suggested libraries: ``pypdf`` (PDF, page-aware), ``beautifulsoup4`` (HTML h1-h6),
plain regex for Markdown ``#`` headings, heuristics for TXT (ALL-CAPS / "Q:" lines).
"""

from __future__ import annotations

from pathlib import Path

from src.schemas import FileType, IngestFailure, ParsedDocument

EXTENSION_TO_TYPE: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".html": FileType.HTML,
    ".htm": FileType.HTML,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".txt": FileType.TXT,
}


def detect_file_type(path: Path) -> FileType:
    """Return the FileType for ``path``.

    Raises:
        UnsupportedFormatError: extension not in EXTENSION_TO_TYPE.
    """
    raise NotImplementedError("Member 2: implement detect_file_type")


def make_doc_id(path: Path) -> str:
    """Stable, URL-safe id from the file name, e.g. 'API Usage v2.md' -> 'api-usage-v2'."""
    raise NotImplementedError("Member 2: implement make_doc_id")


def load_pdf(path: Path) -> ParsedDocument:
    """One Section per detected heading (fallback: per page). Set ``Section.page``."""
    raise NotImplementedError("Member 2: implement load_pdf")


def load_html(path: Path) -> ParsedDocument:
    """Use <title>/<h1> as title, h1-h6 as heading_path. Drop nav/script/style/footer."""
    raise NotImplementedError("Member 2: implement load_html")


def load_markdown(path: Path) -> ParsedDocument:
    """Split on '#' headings, keep the heading hierarchy in ``heading_path``."""
    raise NotImplementedError("Member 2: implement load_markdown")


def load_txt(path: Path) -> ParsedDocument:
    """Heuristic headings (ALL CAPS lines, 'Section:' lines); else a single section."""
    raise NotImplementedError("Member 2: implement load_txt")


def load_document(path: Path) -> ParsedDocument:
    """Dispatch to the correct loader, then apply ``cleaning.clean_document``.

    Raises:
        UnsupportedFormatError, EmptyDocumentError, IngestionError
    """
    raise NotImplementedError("Member 2: implement load_document")


def load_directory(directory: Path) -> tuple[list[ParsedDocument], list[IngestFailure]]:
    """Load every supported file under ``directory``.

    One bad file must NOT stop the batch: record it as an IngestFailure and continue.
    """
    raise NotImplementedError("Member 2: implement load_directory")
