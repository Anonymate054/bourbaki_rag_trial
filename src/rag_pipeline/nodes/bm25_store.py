import re
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

class BM25StoreNode:
    """Node 3B: Sparse Okapi BM25 Store"""
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def tokenize(self, text: str) -> List[str]:
        clean = re.sub(r'[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ]', ' ', text.lower())
        # Preserve numbers of any length (e.g., "2", "17", "62") and words > 2 chars
        return [w for w in clean.split() if w.isdigit() or len(w) > 2]

    def build_index(self, chunks: List[Dict]):
        self.chunks = chunks
        corpus_tokens = [self.tokenize(c["texto"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
