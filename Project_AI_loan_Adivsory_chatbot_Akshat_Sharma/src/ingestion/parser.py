"""
PDF parsing for the Loan Advisory Agent.

Loan policy PDFs are table-heavy (interest rate slabs, LTV ratios, fee
schedules). We use pdfplumber because its table extraction is more reliable
than pure text extraction for this kind of document. Tables are converted to
a normalized text form (not left as raw cell grids) so they chunk and embed
well downstream.
"""

import os
from dataclasses import dataclass, field
from typing import List

import pdfplumber


@dataclass
class Page:
    doc_name: str
    page_number: int
    text: str
    tables_as_text: List[str] = field(default_factory=list)


def table_to_text(table: List[List[str]]) -> str:
    """Turn a raw table (list of rows) into a readable sentence-per-row block.
    e.g. 'CIBIL Score: 750 and above | Interest Rate (p.a.): 10.50%'
    This reads naturally for both keyword search and LLM context.
    """
    if not table or len(table) < 2:
        return ""
    headers = [h.strip() if h else "" for h in table[0]]
    lines = []
    for row in table[1:]:
        row = [c.strip() if c else "" for c in row]
        pairs = [f"{h}: {c}" for h, c in zip(headers, row) if h or c]
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)


def parse_pdf(path: str) -> List[Page]:
    doc_name = os.path.basename(path)
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            raw_tables = page.extract_tables() or []
            tables_as_text = [table_to_text(t) for t in raw_tables]
            tables_as_text = [t for t in tables_as_text if t]
            pages.append(Page(doc_name=doc_name, page_number=i + 1,
                               text=text, tables_as_text=tables_as_text))
    return pages


def parse_directory(dir_path: str) -> List[Page]:
    all_pages = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.lower().endswith(".pdf"):
            all_pages.extend(parse_pdf(os.path.join(dir_path, fname)))
    return all_pages


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw_dir = os.path.join(base, "data", "raw_pdfs")
    pages = parse_directory(raw_dir)
    print(f"Parsed {len(pages)} pages from {raw_dir}")
    for p in pages[:1]:
        print(f"\n--- {p.doc_name} page {p.page_number} ---")
        print(p.text[:300])
        print(f"\nTables found: {len(p.tables_as_text)}")
        if p.tables_as_text:
            print(p.tables_as_text[0])
