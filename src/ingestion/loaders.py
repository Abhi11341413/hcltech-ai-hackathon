"""Multi-format document loaders (FR-01). OWNER: Member 2.

Turn PDF / HTML / Markdown / TXT files into ``ParsedDocument`` objects while
preserving title, section (heading hierarchy), page and source metadata.

Libraries: ``pypdf`` (PDF, page-aware), ``beautifulsoup4`` (HTML h1-h6),
plain regex for Markdown ``#`` headings, heuristics for TXT (ALL-CAPS lines).

Heading-path conventions (the eval labels in ``data/eval/questions.jsonl`` depend on them):

* Markdown / HTML: the H1 is the first element of the path -> ``"Account Access > Password reset"``.
* PDF: the first line is the title and ``"1. Overview"``-style lines are headings -> ``[title, "1. Overview"]``.
* TXT: the first ALL-CAPS line is the title and is NOT part of the path; later ALL-CAPS lines
  are top-level headings -> ``"LIMITS"``.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from src.config import PROJECT_ROOT, get_logger
from src.errors import EmptyDocumentError, IngestionError, UnsupportedFormatError
from src.ingestion.cleaning import clean_document
from src.schemas import FileType, IngestFailure, ParsedDocument, Section

logger = get_logger(__name__)

EXTENSION_TO_TYPE: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".html": FileType.HTML,
    ".htm": FileType.HTML,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".txt": FileType.TXT,
}

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PDF_HEADING = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][^.!?]{1,78}$")
_HTML_HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_HTML_BLOCKS = ("p", "li", "pre", "blockquote", "td", "th", "dt", "dd")
_HTML_DROP = ("script", "style", "nav", "footer", "header", "noscript", "aside", "form", "svg")


def detect_file_type(path: Path) -> FileType:
    """Return the FileType for ``path`` (case-insensitive extension).

    Raises:
        UnsupportedFormatError: extension not in EXTENSION_TO_TYPE.
    """
    file_type = EXTENSION_TO_TYPE.get(path.suffix.lower())
    if file_type is None:
        raise UnsupportedFormatError(f"Unsupported file type: '{path.suffix or path.name}'",
                                     details={"source": path.name})
    return file_type


def make_doc_id(path: Path) -> str:
    """Stable, URL-safe id from the file name, e.g. 'API Usage v2.md' -> 'api-usage-v2'."""
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    return slug or "document"


def _source_of(path: Path) -> str:
    """Repo-relative path with forward slashes when possible, else the file name."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _new_doc(path: Path, title: str, file_type: FileType, sections: list[Section]) -> ParsedDocument:
    return ParsedDocument(doc_id=make_doc_id(path), title=title.strip() or path.stem,
                          source=_source_of(path), file_type=file_type, sections=sections)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def load_pdf(path: Path) -> ParsedDocument:
    """One Section per numbered heading (``"2. Configuring SAML"``); a section that spans a page
    break is split so every Section has a single ``page``. Pages without headings become one
    Section each. The first non-empty line of page 1 is the title.

    Raises:
        IngestionError: the PDF cannot be read.
        EmptyDocumentError: no extractable text (e.g. scanned image-only PDF).
    """
    from pypdf import PdfReader  # imported lazily: only PDF ingestion needs it

    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # pypdf raises many different exception types
        raise IngestionError(f"Could not read PDF: {exc}", details={"source": path.name}) from exc

    if not any(p.strip() for p in pages):
        raise EmptyDocumentError("PDF has no extractable text (scanned or image-only?)",
                                 details={"source": path.name})

    title = ""
    heading_path: list[str] = []
    sections: list[Section] = []
    buffer: list[str] = []

    def flush(page_no: int) -> None:
        if buffer:
            # PDF lines are hard-wrapped at the page width: re-join them into one paragraph.
            text = re.sub(r"(\w)-\s+(?=[a-z])", r"\1", " ".join(buffer))
            sections.append(Section(heading_path=list(heading_path), text=text, page=page_no))
            buffer.clear()

    for page_no, page_text in enumerate(pages, start=1):
        for raw in page_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if not title:
                title = line
                heading_path = [title]
                continue
            if _PDF_HEADING.match(line):
                flush(page_no)
                heading_path = [title, line]
                continue
            buffer.append(line)
        flush(page_no)  # never let a section cross a page boundary

    return _new_doc(path, title, FileType.PDF, sections)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _inside_block(el: Tag) -> bool:
    """True if ``el`` is nested inside another block we already collect (avoid double text)."""
    return any(parent.name in _HTML_BLOCKS for parent in el.parents)


def load_html(path: Path) -> ParsedDocument:
    """Use <title>/<h1> as title, h1-h6 as heading_path. Drop nav/script/style/footer."""
    soup = BeautifulSoup(_read_text(path), "html.parser")
    title_tag = soup.find("title")
    page_title = title_tag.get_text(" ", strip=True) if title_tag else ""
    for tag in soup.find_all(_HTML_DROP):
        tag.decompose()

    root = soup.body or soup
    stack: list[tuple[int, str]] = []  # (level, heading text)
    sections: list[Section] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            sections.append(Section(heading_path=[h for _, h in stack], text="\n\n".join(buffer)))
            buffer.clear()

    for el in root.find_all(_HTML_HEADINGS + _HTML_BLOCKS):
        if el.name in _HTML_HEADINGS:
            flush()
            level = int(el.name[1])
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, el.get_text(" ", strip=True)))
        elif not _inside_block(el):
            text = el.get_text(" ", strip=True)
            if text:
                buffer.append(f"- {text}" if el.name == "li" else text)
    flush()

    h1 = root.find("h1")
    title = page_title or (h1.get_text(" ", strip=True) if h1 else "")
    return _new_doc(path, title, FileType.HTML, sections)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def load_markdown(path: Path) -> ParsedDocument:
    """Split on '#' headings, keep the heading hierarchy in ``heading_path``.
    Headings inside fenced code blocks are ignored. Title = first H1, else file stem."""
    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    buffer: list[str] = []
    title = ""
    in_fence = False

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(Section(heading_path=[h for _, h in stack], text=text))
        buffer.clear()

    for line in _read_text(path).splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        match = None if in_fence else _MD_HEADING.match(line)
        if match is None:
            buffer.append(re.sub(r"^\s*>\s?", "", line))  # unwrap blockquotes, keep their text
            continue
        flush()
        level, heading = len(match.group(1)), match.group(2).strip()
        if level == 1 and not title:
            title = heading
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
    flush()

    return _new_doc(path, title or path.stem, FileType.MARKDOWN, sections)


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------


def _is_caps_heading(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return (3 <= len(line) <= 80 and len(letters) >= 3 and line == line.upper()
            and not line.endswith((".", ":", "?")))


def load_txt(path: Path) -> ParsedDocument:
    """Heuristic headings: ALL-CAPS lines. The first ALL-CAPS line (if it is the first
    non-empty line) is the document title; later ones start top-level sections.
    Without any heading the whole file is a single section titled after the file."""
    lines = _read_text(path).splitlines()
    title = ""
    heading_path: list[str] = []
    sections: list[Section] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(Section(heading_path=list(heading_path), text=text))
        buffer.clear()

    first_content_seen = False
    for raw in lines:
        line = raw.strip()
        if line and not first_content_seen:
            first_content_seen = True
            if _is_caps_heading(line):
                title = line
                continue
        if line and _is_caps_heading(line):
            flush()
            heading_path = [line]
            continue
        buffer.append(raw)
    flush()

    return _new_doc(path, title or path.stem.replace("_", " ").title(), FileType.TXT, sections)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_LOADERS = {
    FileType.PDF: load_pdf,
    FileType.HTML: load_html,
    FileType.MARKDOWN: load_markdown,
    FileType.TXT: load_txt,
}


def load_document(path: Path) -> ParsedDocument:
    """Dispatch to the correct loader, then apply ``cleaning.clean_document``.

    Raises:
        UnsupportedFormatError, EmptyDocumentError, IngestionError
    """
    path = Path(path)
    file_type = detect_file_type(path)
    if not path.is_file():
        raise IngestionError(f"File not found: {path.name}", details={"source": path.name})
    try:
        doc = _LOADERS[file_type](path)
    except IngestionError:
        raise
    except Exception as exc:  # any parser crash becomes a typed ingestion error
        raise IngestionError(f"Failed to parse {path.name}: {exc}", details={"source": path.name}) from exc
    if not doc.sections:
        raise EmptyDocumentError(f"No text found in {path.name}", details={"source": path.name})
    return clean_document(doc)


def load_directory(directory: Path) -> tuple[list[ParsedDocument], list[IngestFailure]]:
    """Load every file under ``directory`` (recursive, sorted for determinism).

    One bad file does NOT stop the batch: it is recorded as an IngestFailure.
    Hidden files and ``.gitkeep`` are skipped silently.
    """
    docs: list[ParsedDocument] = []
    failures: list[IngestFailure] = []
    seen_ids: set[str] = set()
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            doc = load_document(path)
        except IngestionError as exc:
            failures.append(IngestFailure(source=path.name, error_code=exc.error_code, message=exc.message))
            logger.warning("Ingest failed for %s: %s", path.name, exc.error_code)
            continue
        if doc.doc_id in seen_ids:
            failures.append(IngestFailure(source=path.name, error_code="DUPLICATE_DOC_ID",
                                          message=f"doc_id '{doc.doc_id}' already loaded from another file"))
            continue
        seen_ids.add(doc.doc_id)
        docs.append(doc)
    logger.info("Loaded %d documents, %d failures from %s", len(docs), len(failures), Path(directory).name)
    return docs, failures
