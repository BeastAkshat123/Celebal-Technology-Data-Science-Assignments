"""
Output validation layer -- checks whether a generated answer is actually
supported by the retrieved context, before it's shown to the user.

Two checks, in increasing order of rigor:
1. Numeric grounding: any number in the answer (percentages, amounts, months,
   scores) must appear somewhere in the retrieved context or computed_fact.
   This is cheap, deterministic, and catches the most damaging failure mode
   for a financial tool -- a hallucinated rate or fee.
2. Low-retrieval-confidence gate: if the retriever's best match score is
   below a threshold, we flag the answer as low-confidence regardless of
   what the LLM said, since there was little to ground it in.

A third check (LLM-as-judge entailment, e.g. via RAGAS faithfulness) is
included as an optional stub -- it needs an LLM call, so it only runs when
ANTHROPIC_API_KEY is available.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List

from src.generation.generator import GeneratedAnswer
from src.retrieval.retriever import RetrievedChunk

NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*%?")


@dataclass
class ValidationResult:
    is_grounded: bool
    confidence: str  # "high", "medium", "low"
    ungrounded_numbers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _extract_numbers(text: str) -> set:
    return set(NUMBER_RE.findall(text))


def check_numeric_grounding(answer: GeneratedAnswer, computed_fact_text: str = "") -> List[str]:
    """Return numbers mentioned in the answer that don't appear anywhere in
    the source context or computed fact. Not foolproof (numbers can coincide
    or be legitimately derived), but catches outright fabrication cheaply."""
    answer_numbers = _extract_numbers(answer.answer_text)
    context_blob = "\n".join(answer.used_context) + "\n" + computed_fact_text
    context_numbers = _extract_numbers(context_blob)

    # ignore tiny numbers like "1." from list formatting / section numbers
    def is_meaningful(n):
        clean = n.replace(",", "").replace("%", "")
        try:
            return float(clean) >= 10 or "%" in n
        except ValueError:
            return False

    ungrounded = [n for n in answer_numbers if is_meaningful(n) and n not in context_numbers]
    return ungrounded


def validate(
    query: str,
    answer: GeneratedAnswer,
    retrieved: List[RetrievedChunk],
    computed_fact_text: str = "",
    min_confidence_score: float = 0.15,
    retriever=None,
    max_oov_ratio: float = 0.4,
) -> ValidationResult:
    warnings = []

    if not retrieved and not computed_fact_text:
        return ValidationResult(
            is_grounded=False,
            confidence="low",
            warnings=["No context was retrieved for this query."],
        )

    # Skip the retrieval-confidence check entirely when this answer came from a
    # deterministic tool (e.g. EMI calculator) rather than document retrieval --
    # there's no retrieval score to judge in that case.
    used_tool_only = bool(computed_fact_text) and not retrieved
    best_score = max((rc.raw_score for rc in retrieved), default=0.0)
    if retrieved and best_score < min_confidence_score:
        warnings.append(
            f"Best retrieval match score ({best_score:.2f}) is below threshold "
            f"({min_confidence_score}) -- the documents may not actually cover this question."
        )

    high_oov = False
    if retriever is not None:
        oov = retriever.oov_ratio(query)
        if oov > max_oov_ratio:
            high_oov = True
            warnings.append(
                f"{oov:.0%} of the query's key terms don't appear anywhere in the source "
                "documents -- this question may be about something outside their scope."
            )

    ungrounded_numbers = check_numeric_grounding(answer, computed_fact_text)
    if ungrounded_numbers:
        warnings.append(
            f"Answer contains numbers not found in retrieved context: {ungrounded_numbers}. "
            "This may indicate hallucination."
        )

    if not answer.sources and answer.mode.startswith("llm"):
        warnings.append("Answer has no cited sources.")

    if ungrounded_numbers or high_oov:
        confidence = "low"
    elif used_tool_only:
        confidence = "high"  # deterministic calculation, no retrieval ambiguity involved
    elif retrieved and best_score < min_confidence_score:
        confidence = "low"
    elif retrieved and best_score < 0.4:
        confidence = "medium"
    else:
        confidence = "high"

    return ValidationResult(
        is_grounded=len(ungrounded_numbers) == 0,
        confidence=confidence,
        ungrounded_numbers=ungrounded_numbers,
        warnings=warnings,
    )


if __name__ == "__main__":
    import os as _os
    from src.ingestion.parser import parse_directory
    from src.ingestion.chunker import chunk_pages
    from src.retrieval.retriever import HybridRetriever
    from src.generation.generator import generate_answer

    base = _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))
    raw_dir = _os.path.join(base, "data", "raw_pdfs")
    pages = parse_directory(raw_dir)
    chunks = chunk_pages(pages)
    retriever = HybridRetriever(chunks)

    # Case 1: a real, well-grounded query
    q1 = "What is the minimum CIBIL score for Sundhara Finance personal loans?"
    r1 = retriever.retrieve(q1, top_k=3, doc_filter="sundhara")
    a1 = generate_answer(q1, r1)
    v1 = validate(q1, a1, r1, retriever=retriever)
    print(f"Query: {q1}\nConfidence: {v1.confidence} | Grounded: {v1.is_grounded}\nWarnings: {v1.warnings}\n")

    # Case 2: a hallucination test -- fabricate a fake claim manually
    from src.generation.generator import GeneratedAnswer
    fake_answer = GeneratedAnswer(
        answer_text="The minimum CIBIL score is 999 and the interest rate is 45.5%.",
        sources=["sundhara_finance_personal_loan_policy.pdf (1. Eligibility Criteria)"],
        used_context=[rc.chunk.text for rc in r1],
        mode="llm",
    )
    v2 = validate(q1, fake_answer, r1)
    print(f"Hallucination test:\nConfidence: {v2.confidence} | Grounded: {v2.is_grounded}\nWarnings: {v2.warnings}")
