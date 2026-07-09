"""
Section-aware chunking for loan policy documents.

Fixed-size chunking (e.g. every 500 characters) tends to slice a bullet point
in half or separate a table's header from its rows. These documents have
clear numbered section headings ("1. Eligibility Criteria"), so we chunk on
those boundaries instead, and keep any table extracted from that section
attached to the same chunk.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List

from src.ingestion.parser import Page

SECTION_HEADING_RE = re.compile(r"^\s*\d+\.\s+[A-Z][A-Za-z ()/&-]+\s*$")


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    section_title: str
    page_number: int
    text: str

    def to_metadata(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "section_title": self.section_title,
            "page_number": self.page_number,
        }


def split_into_sections(page_text: str):
    """Yields (heading, body_text) tuples from a page's raw text."""
    lines = page_text.split("\n")
    current_heading = "Document Header"
    current_body = []
    sections = []
    for line in lines:
        if SECTION_HEADING_RE.match(line.strip()):
            if current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))
    return sections


def chunk_pages(pages: List[Page], max_chars: int = 1200) -> List[Chunk]:
    chunks = []
    # group pages by document so tables/sections don't get mixed across docs
    by_doc = {}
    for p in pages:
        by_doc.setdefault(p.doc_name, []).append(p)

    for doc_name, doc_pages in by_doc.items():
        for page in doc_pages:
            sections = split_into_sections(page.text)
            table_text_blob = "\n\n".join(page.tables_as_text)

            for heading, body in sections:
                if not body and not table_text_blob:
                    continue
                full_text = body
                # attach tables to the section they most likely belong to
                # (interest rate / fee sections in our synthetic docs)
                if table_text_blob and any(
                    kw in heading.lower() for kw in ["interest", "rate", "fee", "charge", "ltv", "loan-to-value"]
                ):
                    full_text = f"{body}\n\n{table_text_blob}".strip()
                    table_text_blob = ""  # consumed

                if not full_text:
                    continue

                # further split if a section is unusually long
                for sub in _split_long_text(full_text, max_chars):
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4())[:8],
                        doc_name=doc_name,
                        section_title=heading,
                        page_number=page.page_number,
                        text=f"[{doc_name} | {heading}]\n{sub}",
                    ))

            # any leftover table not matched to a heading keyword still gets included
            if table_text_blob:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4())[:8],
                    doc_name=doc_name,
                    section_title="Table Data",
                    page_number=page.page_number,
                    text=f"[{doc_name} | Table Data]\n{table_text_blob}",
                ))
    return chunks


def _split_long_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    # split on sentence-ish boundaries to avoid cutting mid-fact
    parts = re.split(r"(?<=[.\n])\s+", text)
    out, cur = [], ""
    for part in parts:
        if len(cur) + len(part) > max_chars and cur:
            out.append(cur.strip())
            cur = part
        else:
            cur += " " + part
    if cur.strip():
        out.append(cur.strip())
    return out


if __name__ == "__main__":
    import os
    from src.ingestion.parser import parse_directory

    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw_dir = os.path.join(base, "data", "raw_pdfs")
    pages = parse_directory(raw_dir)
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks from {len(pages)} pages\n")
    for c in chunks[:4]:
        print(f"--- chunk {c.chunk_id} | {c.doc_name} | {c.section_title} ---")
        print(c.text[:300])
        print()
