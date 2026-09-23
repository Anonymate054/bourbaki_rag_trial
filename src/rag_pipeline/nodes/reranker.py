import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class CrossEncoderRerankerNode:
    """Node 5: Cross-Encoder & Fine Semantic Reranker"""
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder

    def rerank(self, query: str, candidate_indices: List[int], chunks: List[Dict], top_n: int = 3) -> List[int]:
        if not candidate_indices:
            return []
            
        passages = [chunks[idx]["texto"] for idx in candidate_indices]
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        p_embs = self.embedder.encode(passages, convert_to_numpy=True, normalize_embeddings=True)
        
        scores = np.dot(p_embs, q_emb)
        ranked_order = np.argsort(scores)[::-1]
        
        return [candidate_indices[i] for i in ranked_order[:top_n]]
