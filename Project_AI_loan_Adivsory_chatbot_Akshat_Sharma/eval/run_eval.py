"""
Runs the ground-truth eval set against the retriever and reports:
- Top-1 hit rate: was the correct document the #1 retrieved result?
- Top-k hit rate: was the correct document anywhere in the top-k results?
- Section match rate: among correct-doc hits, did the section also look right?

This is the "does retrieval actually work" check that should be run any
time the chunking or retrieval logic changes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.parser import parse_directory
from src.ingestion.chunker import chunk_pages
from src.retrieval.retriever import HybridRetriever
from eval.eval_set import EVAL_SET


def run_eval(top_k: int = 3):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base, "data", "raw_pdfs")
    pages = parse_directory(raw_dir)
    chunks = chunk_pages(pages)
    retriever = HybridRetriever(chunks)

    top1_hits = 0
    topk_hits = 0
    section_match_hits = 0
    rows = []

    for item in EVAL_SET:
        results = retriever.retrieve(item["query"], top_k=top_k)
        doc_names = [r.chunk.doc_name for r in results]

        top1 = doc_names[0] == item["expected_doc"] if doc_names else False
        topk = item["expected_doc"] in doc_names

        section_match = False
        if topk:
            for r in results:
                if r.chunk.doc_name == item["expected_doc"]:
                    section_lower = r.chunk.section_title.lower()
                    if any(kw in section_lower for kw in item["expected_section_keywords"]):
                        section_match = True
                        break

        top1_hits += int(top1)
        topk_hits += int(topk)
        section_match_hits += int(section_match)

        rows.append({
            "query": item["query"],
            "expected_doc": item["expected_doc"],
            "top1_doc": doc_names[0] if doc_names else None,
            "top1_correct": top1,
            "topk_correct": topk,
            "section_correct": section_match,
        })

    n = len(EVAL_SET)
    print(f"Eval set size: {n}, top_k={top_k}\n")
    print(f"{'Query':<65} {'Top1':<6} {'Top-k':<6} {'Section':<8}")
    print("-" * 90)
    for r in rows:
        q_display = (r["query"][:62] + "...") if len(r["query"]) > 62 else r["query"]
        print(f"{q_display:<65} {str(r['top1_correct']):<6} {str(r['topk_correct']):<6} {str(r['section_correct']):<8}")

    print("\n" + "=" * 40)
    print(f"Top-1 doc hit rate:     {top1_hits}/{n}  ({100*top1_hits/n:.0f}%)")
    print(f"Top-{top_k} doc hit rate:    {topk_hits}/{n}  ({100*topk_hits/n:.0f}%)")
    print(f"Section match rate:     {section_match_hits}/{n}  ({100*section_match_hits/n:.0f}%)")
    return rows


if __name__ == "__main__":
    run_eval()
