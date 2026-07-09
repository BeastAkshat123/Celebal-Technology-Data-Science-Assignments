"""
Hybrid retriever for the Loan Advisory Agent.

NOTE ON EMBEDDINGS: in a full deployment you'd typically use dense embeddings
(sentence-transformers, OpenAI/Anthropic embeddings, or a hosted vector DB
like Chroma/Qdrant). This dev sandbox has no network access to
huggingface.co, so downloading an embedding model isn't possible here.
Instead we use a hybrid of TF-IDF cosine similarity + BM25 keyword scoring,
both fully local via scikit-learn/rank_bm25. This is a legitimate retrieval
approach on its own (it's what "sparse retrieval" means in RAG literature),
and it's easy to swap for dense embeddings later -- see `embed_dense()`
stub below for where that would plug in.
"""

import pickle
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.ingestion.chunker import Chunk


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9%.]+", text.lower())


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float       # combined BM25+TFIDF score, normalized per-query -- use for RANKING only
    raw_score: float    # unnormalized TFIDF cosine similarity -- use for absolute CONFIDENCE checks


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], bm25_weight: float = 0.5):
        self.chunks = chunks
        self.bm25_weight = bm25_weight
        self._texts = [c.text for c in chunks]

        tokenized = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized)

        self._tfidf = TfidfVectorizer(tokenizer=_tokenize, lowercase=False, token_pattern=None)
        self._tfidf_matrix = self._tfidf.fit_transform(self._texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        doc_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[RetrievedChunk]:
        query_tokens = _tokenize(query)

        bm25_scores = np.array(self._bm25.get_scores(query_tokens))
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        query_vec = self._tfidf.transform([query])
        raw_tfidf_scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        tfidf_scores = raw_tfidf_scores.copy()
        if tfidf_scores.max() > 0:
            tfidf_scores = tfidf_scores / tfidf_scores.max()

        combined = self.bm25_weight * bm25_scores + (1 - self.bm25_weight) * tfidf_scores

        candidates = []
        for idx, score in enumerate(combined):
            chunk = self.chunks[idx]
            if doc_filter and doc_filter.lower() not in chunk.doc_name.lower():
                continue
            if score < min_score:
                continue
            candidates.append(RetrievedChunk(
                chunk=chunk, score=float(score), raw_score=float(raw_tfidf_scores[idx])
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def oov_ratio(self, query: str) -> float:
        """Fraction of meaningful query tokens that never appear anywhere in
        the corpus vocabulary. High values mean the query is likely about
        something the documents simply don't cover (e.g. asking about
        cryptocurrency when the corpus is only loan policies), even if a
        few common words (like 'bank', 'loan') still produce a nonzero
        cosine similarity."""
        tokens = [t for t in _tokenize(query) if len(t) > 2]
        if not tokens:
            return 0.0
        vocab = self._tfidf.vocabulary_
        oov = [t for t in tokens if t not in vocab]
        return len(oov) / len(tokens)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "HybridRetriever":
        with open(path, "rb") as f:
            return pickle.load(f)


def embed_dense(texts: List[str]):
    """Placeholder for swapping in dense embeddings (sentence-transformers,
    OpenAI, or Anthropic embeddings) once network access to a model host is
    available. Would return an (n_texts, dim) array to combine with BM25
    the same way tfidf_scores is combined above."""
    raise NotImplementedError(
        "Dense embeddings not wired up in this sandbox (no huggingface.co access). "
        "Swap in sentence-transformers or an embeddings API here for production."
    )


if __name__ == "__main__":
    import os
    from src.ingestion.parser import parse_directory
    from src.ingestion.chunker import chunk_pages

    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw_dir = os.path.join(base, "data", "raw_pdfs")
    pages = parse_directory(raw_dir)
    chunks = chunk_pages(pages)
    retriever = HybridRetriever(chunks)

    test_queries = [
        "What is the minimum CIBIL score for a personal loan?",
        "What are the foreclosure charges if I pay off my loan early?",
        "What is the LTV ratio for a home loan above 75 lakhs?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, top_k=3)
        for r in results:
            print(f"  [{r.score:.3f}] {r.chunk.doc_name} | {r.chunk.section_title}")
            print(f"      {r.chunk.text[:150].splitlines()[-1][:150]}")
