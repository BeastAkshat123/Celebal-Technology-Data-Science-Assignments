"""
Loan Advisory Agent -- main orchestrator.

Flow:
  query -> [is it an EMI calc?] -> yes -> deterministic EMI tool -> generation (explain numbers) -> validation
                                -> no  -> retrieval -> generation (grounded answer) -> validation
"""

import os
from dataclasses import dataclass
from typing import List, Optional

from src.ingestion.parser import parse_directory
from src.ingestion.chunker import chunk_pages, Chunk
from src.retrieval.retriever import HybridRetriever, RetrievedChunk
from src.generation.router import is_emi_calculation_query, extract_emi_params
from src.generation.emi_calculator import calculate_emi
from src.generation.generator import generate_answer, GeneratedAnswer
from src.validation.faithfulness import validate, ValidationResult

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_pdfs")


@dataclass
class AgentResponse:
    query: str
    answer: str
    sources: List[str]
    confidence: str
    warnings: List[str]
    mode: str
    retrieved_chunks: List[RetrievedChunk]


class LoanAdvisoryAgent:
    def __init__(self, pdf_dir: str = DATA_DIR):
        pages = parse_directory(pdf_dir)
        self.chunks: List[Chunk] = chunk_pages(pages)
        self.retriever = HybridRetriever(self.chunks)
        print(f"[Agent] Indexed {len(self.chunks)} chunks from {pdf_dir}")

    def ask(self, query: str, doc_filter: Optional[str] = None, top_k: int = 4) -> AgentResponse:
        if is_emi_calculation_query(query):
            return self._handle_emi_query(query)
        return self._handle_policy_query(query, doc_filter, top_k)

    def _handle_policy_query(self, query: str, doc_filter: Optional[str], top_k: int) -> AgentResponse:
        retrieved = self.retriever.retrieve(query, top_k=top_k, doc_filter=doc_filter)
        answer = generate_answer(query, retrieved)
        v = validate(query, answer, retrieved, retriever=self.retriever)
        return AgentResponse(
            query=query,
            answer=answer.answer_text,
            sources=answer.sources,
            confidence=v.confidence,
            warnings=v.warnings,
            mode=answer.mode,
            retrieved_chunks=retrieved,
        )

    def _handle_emi_query(self, query: str) -> AgentResponse:
        params = extract_emi_params(query)
        if not params.is_complete():
            missing = ", ".join(params.missing_fields())
            return AgentResponse(
                query=query,
                answer=(f"To calculate your EMI I still need: {missing}. "
                        f"Please provide the loan amount, annual interest rate, and tenure "
                        f"(in months or years)."),
                sources=[],
                confidence="low",
                warnings=["Incomplete parameters for EMI calculation."],
                mode="param-request",
                retrieved_chunks=[],
            )

        result = calculate_emi(params.principal, params.annual_rate_pct, params.tenure_months)
        computed_fact = (
            f"Principal: INR {result.principal:,.2f}\n"
            f"Annual interest rate: {result.annual_rate_pct}%\n"
            f"Tenure: {result.tenure_months} months\n"
            f"Monthly EMI: INR {result.emi:,.2f}\n"
            f"Total payment over tenure: INR {result.total_payment:,.2f}\n"
            f"Total interest payable: INR {result.total_interest:,.2f}"
        )
        # still generate a natural-language explanation via the LLM (or template fallback),
        # but with the computed numbers locked in as ground truth
        answer = generate_answer(query, retrieved=[], computed_fact=computed_fact)
        v = validate(query, answer, retrieved=[], computed_fact_text=computed_fact)
        return AgentResponse(
            query=query,
            answer=answer.answer_text,
            sources=["EMI Calculator (deterministic tool)"],
            confidence=v.confidence,
            warnings=v.warnings,
            mode=f"emi-tool+{answer.mode}",
            retrieved_chunks=[],
        )


if __name__ == "__main__":
    agent = LoanAdvisoryAgent()
    demo_queries = [
        "What is the minimum CIBIL score for a personal loan at Sundhara Finance?",
        "What is the EMI for a loan of 5 lakh at 11.75% for 48 months?",
        "What is the LTV ratio for a Vistara home loan above 75 lakhs?",
        "Does Aarna Bank offer loans for cryptocurrency trading?",  # should trigger low confidence
    ]
    for q in demo_queries:
        print(f"\n{'='*70}\nQ: {q}")
        resp = agent.ask(q)
        print(f"\n{resp.answer}")
        print(f"\n[confidence: {resp.confidence} | mode: {resp.mode}]")
        if resp.warnings:
            print(f"[warnings: {resp.warnings}]")
