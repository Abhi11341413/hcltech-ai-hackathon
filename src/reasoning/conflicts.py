"""Conflict and version-sensitivity detection (FR-06). OWNER: Member 1.

Deterministic and explainable to the panel:

1. Relevance filter: only passages with ``score >= min_score`` take part, and inside each
   passage only the sentences that talk about the question (they share a content term with
   the question, or the passage's own heading does).
2. Quantities: numbers with a unit are extracted from those sentences with a regex
   (``300 requests per minute`` -> ``(300.0, 'requests/minute')``). "from X to Y" phrasing in
   release notes keeps only the new value Y.
3. VALUE disagreement: two passages from DIFFERENT documents, about the same topic
   (``same_topic``), contain two similar sentences (>= 2 shared content terms) that state
   different numbers for the same unit, and neither document states the other's number.
   (So "Enterprise: 1,000/min" in both docs is agreement, not conflict.)
4. The conflict is then labelled from chunk metadata:
   VERSION if both chunks carry different ``version`` values, else DATE if their
   ``effective_date`` differ, else VALUE.
5. CONTRADICTION: two near-identical sentences (Jaccard >= 0.6 after removing negation
   words) where exactly one is negated ("is supported" vs "is not supported").
6. ``GeneratedAnswer.llm_reported_conflict`` is a secondary signal only: used when the rules
   above find nothing but the LLM cited >= 2 different documents and flagged a disagreement.

Never silently pick one side: the pipeline surfaces every Conflict to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import get_logger
from src.reasoning.text_utils import (
    NEGATIONS,
    content_terms,
    last_heading,
    split_sentences,
    words,
)
from src.schemas import Chunk, Conflict, ConflictType, GeneratedAnswer, RetrievedPassage

logger = get_logger(__name__)

# (regex for the unit as written, normalised unit). Longer / more specific patterns first.
_UNIT_PATTERNS: list[tuple[str, str]] = [
    (r"requests?\s*(?:per|/|a|each)\s*sec(?:ond)?s?", "requests/second"),
    (r"requests?\s*(?:per|/|a|each)\s*min(?:ute)?s?|rpm", "requests/minute"),
    (r"requests?\s*(?:per|/|a|each)\s*h(?:ou)?rs?", "requests/hour"),
    (r"requests?\s*(?:per|/|a|each)\s*day", "requests/day"),
    (r"business\s+days?", "business days"),
    (r"minutes?|mins?", "minutes"),
    (r"hours?|hrs?", "hours"),
    (r"days?", "days"),
    (r"weeks?", "weeks"),
    (r"months?", "months"),
    (r"years?", "years"),
    (r"gb|gigabytes?", "GB"),
    (r"mb|megabytes?", "MB"),
    (r"%|percent", "%"),
    (r"seats?", "seats"),
    (r"users?", "users"),
    (r"(?:failed\s+)?attempts?", "attempts"),
    (r"times", "times"),
]
_NUMBER = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
_UNIT_ALT = "|".join(f"(?:{pattern})" for pattern, _ in _UNIT_PATTERNS)
_QUANTITY_RE = re.compile(rf"{_NUMBER}\s*({_UNIT_ALT})(?![a-z])", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"(?:\$|usd\s*)\s?" + _NUMBER, re.IGNORECASE)
# "increased from 300 to 500 requests per minute" -> "increased to 500 requests per minute"
_FROM_TO_RE = re.compile(r"\bfrom\s+\$?[\d,.]+\s*(?:%|[a-z/ ]{0,25}?)\s+to\s+", re.IGNORECASE)
_COMPILED_UNITS = [(re.compile(rf"^(?:{p})$", re.IGNORECASE), unit) for p, unit in _UNIT_PATTERNS]

SAME_TOPIC_MIN_OVERLAP = 0.2       # overlap coefficient of content terms between two passages
SIMILAR_SENTENCE_MIN_SHARED = 2    # shared content terms for two sentences to be "about the same thing"
CONTRADICTION_MIN_JACCARD = 0.6


@dataclass(frozen=True)
class _Fact:
    """One number+unit stated in one sentence of one passage."""

    value: float
    unit: str
    sentence: str
    terms: frozenset[str]


def _normalise_unit(raw: str) -> str:
    raw = " ".join(raw.lower().split())
    for pattern, unit in _COMPILED_UNITS:
        if pattern.match(raw):
            return unit
    return raw


def extract_quantities(text: str) -> list[tuple[float, str]]:
    """Return (value, normalised_unit) pairs in order of appearance.

    Example: ``"up to 300 requests per minute"`` -> ``[(300.0, 'requests/minute')]``.
    Thousands separators are removed (``1,000`` -> 1000.0); "$49" -> ``(49.0, 'USD')``;
    in "from X to Y" only the new value Y is kept.
    """
    text = _FROM_TO_RE.sub("to ", text)
    found: list[tuple[int, float, str]] = []
    for m in _QUANTITY_RE.finditer(text):
        found.append((m.start(), float(m.group(1).replace(",", "")), _normalise_unit(m.group(2))))
    for m in _CURRENCY_RE.finditer(text):
        found.append((m.start(), float(m.group(1).replace(",", "")), "USD"))
    return [(value, unit) for _, value, unit in sorted(found)]


def _passage_terms(passage: RetrievedPassage) -> set[str]:
    return content_terms(f"{passage.chunk.section} {passage.chunk.text}")


def same_topic(a: RetrievedPassage, b: RetrievedPassage) -> bool:
    """Heuristic: do two passages talk about the same subject?

    Uses the overlap coefficient |A ∩ B| / min(|A|, |B|) of their content terms (section +
    text), which is not penalised when one chunk is much longer than the other.
    """
    terms_a, terms_b = _passage_terms(a), _passage_terms(b)
    if not terms_a or not terms_b:
        return False
    return len(terms_a & terms_b) / min(len(terms_a), len(terms_b)) >= SAME_TOPIC_MIN_OVERLAP


def _relevant_sentences(question_terms: set[str], chunk: Chunk) -> list[str]:
    """Sentences of the chunk that talk about the question.

    If the chunk's own heading matches the question (e.g. heading 'Rate limits' for a
    rate-limit question) the whole chunk is relevant; otherwise keep only sentences that
    share at least one content term with the question.
    """
    sentences = split_sentences(chunk.text)
    if not question_terms or content_terms(last_heading(chunk.section)) & question_terms:
        return sentences
    return [s for s in sentences if content_terms(s) & question_terms]


def _facts(question_terms: set[str], chunk: Chunk) -> list[_Fact]:
    facts: list[_Fact] = []
    for sentence in _relevant_sentences(question_terms, chunk):
        terms = frozenset(content_terms(sentence))
        for value, unit in extract_quantities(sentence):
            facts.append(_Fact(value=value, unit=unit, sentence=sentence, terms=terms))
    return facts


def _label(chunk: Chunk) -> str:
    """Human-readable source label: title plus version/date when known."""
    meta = [m for m in (f"v{chunk.version}" if chunk.version else None,
                        str(chunk.effective_date) if chunk.effective_date else None) if m]
    return f"'{chunk.doc_title}'" + (f" ({', '.join(meta)})" if meta else "")


def _classify(chunks: list[Chunk]) -> ConflictType:
    versions = {c.version for c in chunks if c.version}
    if len(versions) >= 2 and all(c.version for c in chunks):
        return ConflictType.VERSION
    dates = {c.effective_date for c in chunks if c.effective_date}
    if len(dates) >= 2:
        return ConflictType.DATE
    if len(versions) >= 2:
        return ConflictType.VERSION
    return ConflictType.VALUE


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if value == int(value) else f"{value:g}"


def _value_disagreements(facts_a: list[_Fact], facts_b: list[_Fact]) -> dict[str, tuple[set[float], set[float]]]:
    """For each unit, the values of A and of B that disagree (see module docstring, rule 3)."""
    out: dict[str, tuple[set[float], set[float]]] = {}
    for unit in {f.unit for f in facts_a} & {f.unit for f in facts_b}:
        values_a = {f.value for f in facts_a if f.unit == unit}
        values_b = {f.value for f in facts_b if f.unit == unit}
        for fa in (f for f in facts_a if f.unit == unit and f.value not in values_b):
            for fb in (f for f in facts_b if f.unit == unit and f.value not in values_a):
                if len(fa.terms & fb.terms) >= SIMILAR_SENTENCE_MIN_SHARED:
                    va, vb = out.setdefault(unit, (set(), set()))
                    va.add(fa.value)
                    vb.add(fb.value)
    return out


def _value_conflicts(question_terms: set[str], passages: list[RetrievedPassage]) -> list[Conflict]:
    facts = {p.chunk.chunk_id: _facts(question_terms, p.chunk) for p in passages}
    # unit -> {chunk_id: disagreeing values}; unit -> union-find parent map over chunk ids
    disagreeing: dict[str, dict[str, set[float]]] = {}
    parent: dict[str, dict[str, str]] = {}

    def find(unit: str, x: str) -> str:
        while parent[unit][x] != x:
            x = parent[unit][x]
        return x

    for i, a in enumerate(passages):
        for b in passages[i + 1:]:
            if a.chunk.doc_id == b.chunk.doc_id or not same_topic(a, b):
                continue
            for unit, (va, vb) in _value_disagreements(facts[a.chunk.chunk_id], facts[b.chunk.chunk_id]).items():
                per_unit = disagreeing.setdefault(unit, {})
                per_unit.setdefault(a.chunk.chunk_id, set()).update(va)
                per_unit.setdefault(b.chunk.chunk_id, set()).update(vb)
                p = parent.setdefault(unit, {})
                for cid in (a.chunk.chunk_id, b.chunk.chunk_id):
                    p.setdefault(cid, cid)
                p[find(unit, a.chunk.chunk_id)] = find(unit, b.chunk.chunk_id)

    by_id = {p.chunk.chunk_id: p.chunk for p in passages}
    order = {p.chunk.chunk_id: i for i, p in enumerate(passages)}
    conflicts: list[Conflict] = []
    for unit, per_chunk in disagreeing.items():
        groups: dict[str, list[str]] = {}
        for cid in per_chunk:
            groups.setdefault(find(unit, cid), []).append(cid)
        for members in groups.values():
            members.sort(key=order.__getitem__)
            chunks = [by_id[cid] for cid in members]
            values = {cid: " / ".join(f"{_fmt(v)} {unit}" for v in sorted(per_chunk[cid])) for cid in members}
            parts = [f"{_label(by_id[cid])} says {values[cid]}" for cid in members]
            conflicts.append(Conflict(
                conflict_type=_classify(chunks),
                description=f"Sources disagree on {unit}: " + "; ".join(parts) + ".",
                chunk_ids=members,
                values={cid: f"{values[cid]}{' (v' + by_id[cid].version + ')' if by_id[cid].version else ''}"
                        for cid in members},
            ))
    return conflicts


def _strip_negation(sentence: str) -> tuple[bool, set[str]]:
    tokens = words(sentence.replace("n't", " not"))
    negated = any(t in NEGATIONS for t in tokens)
    return negated, content_terms(" ".join(t for t in tokens if t not in NEGATIONS))


def _contradictions(question_terms: set[str], passages: list[RetrievedPassage]) -> list[Conflict]:
    conflicts: list[Conflict] = []
    sentences = {p.chunk.chunk_id: [(s, *_strip_negation(s)) for s in _relevant_sentences(question_terms, p.chunk)]
                 for p in passages}
    for i, a in enumerate(passages):
        for b in passages[i + 1:]:
            if a.chunk.doc_id == b.chunk.doc_id:
                continue
            for sa, neg_a, terms_a in sentences[a.chunk.chunk_id]:
                hit = next(((sb, terms_b) for sb, neg_b, terms_b in sentences[b.chunk.chunk_id]
                            if neg_a != neg_b and len(terms_a) >= 3 and len(terms_b) >= 3
                            and len(terms_a & terms_b) / len(terms_a | terms_b) >= CONTRADICTION_MIN_JACCARD),
                           None)
                if hit:
                    conflicts.append(Conflict(
                        conflict_type=ConflictType.CONTRADICTION,
                        description=f"{_label(a.chunk)} and {_label(b.chunk)} make opposite statements.",
                        chunk_ids=[a.chunk.chunk_id, b.chunk.chunk_id],
                        values={a.chunk.chunk_id: sa[:200], b.chunk.chunk_id: hit[0][:200]},
                    ))
                    break
    return conflicts


def _llm_signal(generated: GeneratedAnswer | None, relevant: list[RetrievedPassage]) -> list[Conflict]:
    """Secondary signal: trust the LLM's conflict flag only if it cited >= 2 different docs."""
    if not generated or not generated.llm_reported_conflict:
        return []
    allowed = {p.chunk.chunk_id for p in relevant}
    cited: dict[str, str] = {}  # doc_id -> first cited chunk_id
    for c in generated.citations:
        if c.chunk_id in allowed:
            cited.setdefault(c.doc_id, c.chunk_id)
    if len(cited) < 2:
        return []
    chunk_ids = list(cited.values())
    by_id = {p.chunk.chunk_id: p.chunk for p in relevant}
    return [Conflict(
        conflict_type=_classify([by_id[c] for c in chunk_ids]),
        description="The answer model flagged a possible disagreement between "
                    + ", ".join(_label(by_id[c]) for c in chunk_ids) + "; check the cited passages.",
        chunk_ids=chunk_ids,
    )]


def detect_conflicts(question: str, passages: list[RetrievedPassage],
                     generated: GeneratedAnswer | None = None,
                     min_score: float = 0.3) -> list[Conflict]:
    """Return all detected conflicts among relevant passages (empty list if none).

    Inputs: the user question, the retrieved passages (any order) and, optionally, the
    generated answer (its ``llm_reported_conflict`` flag is a secondary signal).
    Output: one ``Conflict`` per group of disagreeing passages; never raises for normal input.
    """
    relevant = [p for p in passages if p.score >= min_score]
    if len(relevant) < 2:
        return []
    question_terms = content_terms(question)
    conflicts = _value_conflicts(question_terms, relevant) + _contradictions(question_terms, relevant)
    if not conflicts:
        conflicts = _llm_signal(generated, relevant)
    logger.info("conflict detection: %d relevant passages -> %d conflicts", len(relevant), len(conflicts))
    return conflicts
