"""
Answer generation for the Loan Advisory Agent.

Design principles:
1. The LLM only ever sees retrieved chunks as context -- never asked to
   answer from parametric/world knowledge about loans.
2. The prompt requires the model to say "not found in the provided
   documents" if the answer isn't supported, rather than guessing.
3. Every answer must cite which document/section it came from.
4. Numeric loan math (EMI etc.) is never computed by the LLM -- it's passed
   in as an already-computed fact for the model to present/explain.

PROVIDER SUPPORT: works with a free-tier provider or a paid Anthropic key,
so the pipeline is usable at $0 cost:
  - GROQ_API_KEY       -> free, no card, OpenAI-compatible (recommended default)
  - OPENROUTER_API_KEY -> free, no card, wide model variety, OpenAI-compatible
  - ANTHROPIC_API_KEY  -> paid, highest quality
Priority: ANTHROPIC_API_KEY > GROQ_API_KEY > OPENROUTER_API_KEY > template fallback.

KEY LOADING: reads a .env file in the project root automatically (via
python-dotenv), with override=True so the .env file always wins over any
stale value left behind by an old terminal `export`. If you don't have
python-dotenv installed, it silently falls back to whatever's already in
the environment (e.g. a real `export` in your current shell).
"""

import os
from dataclasses import dataclass
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from src.retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a Loan Advisory Assistant. You answer questions about loan \
eligibility, EMIs, interest rates, fees, and documentation using ONLY the context \
provided below, which is extracted from official bank/NBFC loan policy documents.

Rules you must follow strictly:
1. Answer only using facts present in the provided context. Do not use outside knowledge \
about loans, banks, or interest rates.
2. If the context does not contain enough information to answer the question, say clearly: \
"I don't have enough information in the provided documents to answer this." Do not guess \
or extrapolate.
3. If a pre-computed EMI/loan calculation is provided in the context, present it exactly as \
given -- do not recompute or alter the numbers yourself.
4. Every factual claim in your answer must be attributable to a specific source document \
and section. End your answer with a "Sources:" line listing the document name(s) and \
section(s) used.
5. Be concise and direct. Do not add disclaimers beyond what's asked, and do not offer \
financial advice beyond what the documents state -- present facts, not recommendations.
"""

GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
OPENROUTER_DEFAULT_MODEL = "deepseek/deepseek-r1:free"


@dataclass
class GeneratedAnswer:
    answer_text: str
    sources: List[str]
    used_context: List[str]
    mode: str


def _format_context(chunks: List[RetrievedChunk], computed_fact: Optional[str] = None) -> str:
    blocks = []
    for i, rc in enumerate(chunks, 1):
        blocks.append(
            f"[Context {i} | Source: {rc.chunk.doc_name} | Section: {rc.chunk.section_title}]\n"
            f"{rc.chunk.text}"
        )
    context_str = "\n\n".join(blocks)
    if computed_fact:
        context_str += f"\n\n[Pre-computed calculation -- present these numbers exactly, do not recompute]\n{computed_fact}"
    return context_str


def generate_answer(
    query: str,
    retrieved: List[RetrievedChunk],
    computed_fact: Optional[str] = None,
    model: Optional[str] = None,
) -> GeneratedAnswer:
    context_str = _format_context(retrieved, computed_fact)
    sources = sorted({f"{rc.chunk.doc_name} ({rc.chunk.section_title})" for rc in retrieved})
    user_msg = f"Context:\n{context_str}\n\nQuestion: {query}"

    if os.environ.get("ANTHROPIC_API_KEY"):
        text = _call_anthropic(user_msg, model or "claude-sonnet-4-6")
        mode = "llm:anthropic"
    elif os.environ.get("GROQ_API_KEY"):
        text = _call_openai_compatible(
            user_msg,
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
            model=model or GROQ_DEFAULT_MODEL,
        )
        mode = "llm:groq"
    elif os.environ.get("OPENROUTER_API_KEY"):
        text = _call_openai_compatible(
            user_msg,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=model or OPENROUTER_DEFAULT_MODEL,
        )
        mode = "llm:openrouter"
    else:
        return _template_fallback(query, retrieved, computed_fact, sources)

    return GeneratedAnswer(
        answer_text=text,
        sources=sources,
        used_context=[rc.chunk.text for rc in retrieved],
        mode=mode,
    )


def _call_anthropic(user_msg: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_openai_compatible(user_msg: str, base_url: str, api_key: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content


def _template_fallback(query, retrieved, computed_fact, sources) -> GeneratedAnswer:
    """No API key available -- produce a grounded but non-LLM answer so the
    rest of the pipeline (retrieval, EMI tool, validation) can still be
    tested end-to-end. This is NOT the production answer path."""
    if not retrieved and not computed_fact:
        return GeneratedAnswer(
            answer_text="I don't have enough information in the provided documents to answer this.",
            sources=[],
            used_context=[],
            mode="template-fallback",
        )

    lines = ["[TEMPLATE MODE -- no LLM API key set, showing raw retrieved facts]\n"]
    if computed_fact:
        lines.append(computed_fact)
    for rc in retrieved:
        lines.append(f"\nFrom {rc.chunk.doc_name} ({rc.chunk.section_title}):")
        body = rc.chunk.text.split("]\n", 1)[-1]
        lines.append(body.strip()[:400])
    lines.append(f"\nSources: {', '.join(sources) if sources else 'N/A'}")

    return GeneratedAnswer(
        answer_text="\n".join(lines),
        sources=sources,
        used_context=[rc.chunk.text for rc in retrieved],
        mode="template-fallback",
    )


if __name__ == "__main__":
    import os as _os
    from src.ingestion.parser import parse_directory
    from src.ingestion.chunker import chunk_pages
    from src.retrieval.retriever import HybridRetriever

    base = _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))
    raw_dir = _os.path.join(base, "data", "raw_pdfs")
    pages = parse_directory(raw_dir)
    chunks = chunk_pages(pages)
    retriever = HybridRetriever(chunks)

    q = "What is the minimum CIBIL score needed for a personal loan at Aarna Bank?"
    results = retriever.retrieve(q, top_k=3, doc_filter="aarna")
    ans = generate_answer(q, results)
    print(f"Mode: {ans.mode}\n")
    print(ans.answer_text)
