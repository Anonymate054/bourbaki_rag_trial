from typing import List, Tuple, Dict

class ReciprocalRankFusionNode:
    """Node 4: Reciprocal Rank Fusion (RRF) Scorer"""
    def run(self, dense_results: List[Tuple[int, float]], sparse_results: List[Tuple[int, float]], k: int = 60, top_n: int = 5) -> List[int]:
        rrf_scores: Dict[int, float] = {}
        
        # Dense ranks
        for rank, (doc_idx, _) in enumerate(dense_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))
            
        # Sparse ranks
        for rank, (doc_idx, _) in enumerate(sparse_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))
            
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_idx for doc_idx, _ in sorted_candidates[:top_n]]
