import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Set

def tokenize_text(text: str) -> List[str]:
    """Extracts lowercase alphanumeric tokens and code symbols."""
    return re.findall(r"\w+", text.lower())

class BM25Scorer:
    """
    Okapi BM25 Sparse Keyword Scorer.
    Computes exact term frequency and inverse document frequency (IDF) relevance ranking.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        self.doc_term_freqs: List[Counter] = []
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.documents: List[Dict[str, Any]] = []

    def fit(self, documents: List[Dict[str, Any]]):
        """Indexes documents and pre-calculates IDF dictionary."""
        self.documents = documents
        self.doc_count = len(documents)
        self.doc_len = []
        self.doc_term_freqs = []
        self.df = defaultdict(int)

        total_len = 0
        for doc in documents:
            text = doc.get("text", doc.get("content", ""))
            tokens = tokenize_text(text)
            term_counts = Counter(tokens)
            self.doc_term_freqs.append(term_counts)
            self.doc_len.append(len(tokens))
            total_len += len(tokens)

            for term in term_counts:
                self.df[term] += 1

        self.avg_doc_len = total_len / max(self.doc_count, 1)

        # Pre-compute Okapi BM25 IDF for each term
        self.idf = {}
        for term, freq in self.df.items():
            # Standard Okapi BM25 IDF formula
            idf_val = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = max(idf_val, 1e-4)

    def score(self, query: str) -> List[float]:
        """Calculates BM25 relevance scores across all indexed documents."""
        query_tokens = tokenize_text(query)
        scores = [0.0] * self.doc_count

        for i in range(self.doc_count):
            doc_tf = self.doc_term_freqs[i]
            d_len = self.doc_len[i]
            doc_score = 0.0

            for q_term in query_tokens:
                if q_term not in doc_tf:
                    continue
                tf = doc_tf[q_term]
                idf = self.idf.get(q_term, 0.0)

                # Okapi BM25 term score calculation
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (d_len / max(self.avg_doc_len, 1e-5)))
                doc_score += idf * (num / denom)

            scores[i] = doc_score

        return scores

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Returns top_k highest BM25 scoring documents with metadata."""
        if not self.documents:
            return []

        scores = self.score(query)
        indexed_scores = [(idx, score) for idx, score in enumerate(scores) if score > 0.0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            doc_copy = dict(self.documents[idx])
            doc_copy["bm25_score"] = round(score, 4)
            results.append(doc_copy)

        return results
