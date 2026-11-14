"""Build the retrieval index from a document folder and report a retrieval ablation.

    python -m scripts.build_index                         # data/sample_docs -> models/index
    python -m scripts.build_index --model BAAI/bge-small-en-v1.5 --no-save
    python -m scripts.build_index --show "How long do deleted files stay in the trash?"

Steps: load_directory -> chunk_documents -> HybridRetriever.build -> save (models/index/,
git-ignored) -> Recall@k / MRR / Hit@1 for BM25, dense and hybrid on the answerable
questions of data/eval/questions.jsonl.

The metric helpers here are a quick developer check for the indexing step only; the
official evaluation (nDCG, groundedness, status accuracy, report) is Member 1's
``src.evaluation.run_eval``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import get_settings
from src.ingestion.chunking import chunk_documents
from src.ingestion.loaders import load_directory
from src.retrieval.dense_index import Embedder
from src.retrieval.hybrid import HybridRetriever
from src.schemas import Chunk, EvalQuestion, RetrieverName


def is_relevant(chunk: Chunk, question: EvalQuestion) -> bool:
    """Doc must match; a label's section (if any) must be a prefix of the chunk's section."""
    return any(chunk.doc_id == lab.doc_id and (lab.section is None or chunk.section.startswith(lab.section))
               for lab in question.relevant)


def evaluate(retriever: HybridRetriever, questions: list[EvalQuestion], k: int) -> dict[str, float]:
    recall, rr, hit1 = [], [], []
    for q in questions:
        ranked = retriever.retrieve(q.question, top_k=k)
        flags = [is_relevant(p.chunk, q) for p in ranked]
        found_docs = {p.chunk.doc_id for p, f in zip(ranked, flags, strict=True) if f}
        recall.append(len(found_docs) / len({lab.doc_id for lab in q.relevant}))
        rr.append(next((1.0 / (i + 1) for i, f in enumerate(flags) if f), 0.0))
        hit1.append(1.0 if flags and flags[0] else 0.0)
    n = len(questions)
    return {f"recall@{k}": sum(recall) / n, "mrr": sum(rr) / n, "hit@1": sum(hit1) / n}


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs", default=settings.data_dir)
    parser.add_argument("--model", default=settings.retrieval.embedding_model)
    parser.add_argument("--k", type=int, default=settings.evaluation.k)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show", help="print the top-k passages for one question")
    args = parser.parse_args()

    docs, failures = load_directory(settings.resolve(args.docs))
    ic = settings.ingestion
    chunks = chunk_documents(docs, ic.chunk_max_tokens, ic.chunk_overlap_tokens, ic.min_chunk_chars)
    print(f"documents={len(docs)} failures={len(failures)} chunks={len(chunks)}")
    for f in failures:
        print(f"  FAILED {f.source}: {f.error_code} {f.message}")

    rsettings = settings.retrieval.model_copy(update={"embedding_model": args.model})
    embedder = Embedder(args.model)
    t0 = time.perf_counter()
    hybrid = HybridRetriever(rsettings, embedder=embedder)
    hybrid.build(chunks)
    print(f"embedded with {args.model} in {time.perf_counter() - t0:.1f}s")
    if not args.no_save:
        print(f"saved index to {hybrid.save()}")

    if args.show:
        for p in hybrid.retrieve(args.show, top_k=args.k):
            b = p.score_breakdown
            print(f"#{p.rank} score={p.score:.3f} bm25={b['bm25']:.2f} dense={b['dense']:.3f} "
                  f"rrf={b['rrf']:.4f}  {p.chunk.chunk_id}  [{p.chunk.section}]")
        return

    qpath = settings.resolve(settings.evaluation.questions_path)
    questions = [EvalQuestion.model_validate_json(line)
                 for line in Path(qpath).read_text(encoding="utf-8").splitlines() if line.strip()]
    answerable = [q for q in questions if q.relevant]
    print(f"\nRetrieval ablation on {len(answerable)} answerable questions (k={args.k}):")
    results = {}
    for mode in (RetrieverName.BM25, RetrieverName.DENSE, RetrieverName.HYBRID):
        r = HybridRetriever(rsettings, embedder=embedder, mode=mode)
        r.build(chunks)  # embeddings are cheap to recompute for 72 chunks
        results[mode.value] = evaluate(r, answerable, args.k)
        print(f"  {mode.value:<7} " + "  ".join(f"{m}={v:.3f}" for m, v in results[mode.value].items()))

    misses = [(q.qid, q.question) for q in answerable
              if not any(is_relevant(p.chunk, q) for p in hybrid.retrieve(q.question, top_k=args.k))]
    print(f"\nhybrid misses at k={args.k}: {misses or 'none'}")
    print(json.dumps({"model": args.model, "results": results}))


if __name__ == "__main__":
    main()
