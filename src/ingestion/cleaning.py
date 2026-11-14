"""Text cleaning and metadata extraction (FR-02, supports FR-06). OWNER: Member 2.

Version and effective date are extracted here because conflict detection (Member 1)
needs them to tell "v1.x FAQ says 100/min" apart from "v3.0 release notes say 500/min".
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime

from src.errors import EmptyDocumentError
from src.schemas import ParsedDocument, Section

_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")

# 'v2.0', 'Version 3.0', 'VERSION 1.X', 'Policy version 2026.1', 'Release 2024.2', 'v3'
_VERSION = re.compile(r"(?<![\w-])(?:v|version|release)\s*(\d+(?:\.(?:\d+|x))*)(?!\w|\.\w)", re.IGNORECASE)

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_DATE_KEYWORD = r"(?:effective(?:\s+(?:from|as\s+of|date))?|last\s+updated|updated|released|valid\s+from|as\s+of)"
_DATE_ISO = re.compile(_DATE_KEYWORD + r"\s*:?\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
_DATE_DMY = re.compile(_DATE_KEYWORD + r"\s*:?\s*(\d{1,2}\s+(?:" + _MONTHS + r")\s+\d{4})", re.IGNORECASE)
_DATE_MDY = re.compile(_DATE_KEYWORD + r"\s*:?\s*((?:" + _MONTHS + r")\s+\d{1,2},?\s+\d{4})", re.IGNORECASE)

_META_SCAN_CHARS = 1500  # version/date statements live at the top of a document


def clean_text(text: str) -> str:
    """Normalise unicode (NFKC), fix hyphenated line breaks, collapse whitespace,
    remove page-number lines and collapse 3+ newlines to one blank line.

    Paragraph breaks (blank lines) and single line breaks are kept: TXT FAQs rely on
    "Q:"/"A:" lines and lists rely on "- " lines. Headings are not in ``text`` (they live
    in ``Section.heading_path``), so nothing here can drop them.
    """
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("​", "")
    text = _HYPHEN_BREAK.sub(r"\1", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if not _PAGE_NUMBER_LINE.match(line)]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_version(title: str, text: str) -> str | None:
    """Detect a product/doc version, e.g. 'v2.0', 'Version 3.1', 'Release 2024.2' -> '2.0'.

    The title is checked first because body text often mentions *other* versions
    ("migrate to the v2 API"). Returned lower-cased so '1.X' and '1.x' compare equal.
    """
    for source in (title, text[:_META_SCAN_CHARS]):
        match = _VERSION.search(source)
        if match:
            return match.group(1).lower()
    return None


def _parse_date(raw: str) -> date | None:
    raw = re.sub(r"\s+", " ", raw.replace(",", "")).strip()
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%B %d %Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def extract_effective_date(text: str) -> date | None:
    """Detect 'Effective from 1 July 2026', 'Last updated: 2026-07-01', 'Released 1 October 2026'.

    Only dates introduced by one of these keywords count, so an arbitrary date in the body
    (e.g. "switched off on 31 March 2027") is never mistaken for the document date.
    The earliest keyword match in the text wins.
    """
    head = text[:_META_SCAN_CHARS]
    matches = [m for pattern in (_DATE_ISO, _DATE_DMY, _DATE_MDY) for m in pattern.finditer(head)]
    for match in sorted(matches, key=lambda m: m.start()):
        parsed = _parse_date(match.group(1))
        if parsed:
            return parsed
    return None


def clean_document(doc: ParsedDocument) -> ParsedDocument:
    """Return a new ParsedDocument with cleaned sections, version and effective_date filled.

    Values already set on ``doc`` are kept. Empty sections are removed.

    Raises:
        EmptyDocumentError: no section has any text left after cleaning.
    """
    sections = []
    for section in doc.sections:
        text = clean_text(section.text)
        if text:
            heading_path = [clean_text(h) for h in section.heading_path if clean_text(h)]
            sections.append(Section(heading_path=heading_path, text=text, page=section.page))
    if not sections:
        raise EmptyDocumentError(f"Document '{doc.doc_id}' is empty after cleaning",
                                 details={"source": doc.source})

    title = clean_text(doc.title) or doc.doc_id
    head_text = "\n".join(s.text for s in sections)
    return doc.model_copy(update={
        "title": title,
        "sections": sections,
        "version": doc.version or extract_version(title, head_text),
        "effective_date": doc.effective_date or extract_effective_date(head_text),
    })
