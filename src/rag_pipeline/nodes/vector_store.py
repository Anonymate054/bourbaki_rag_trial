import os
import faiss
import numpy as np
import polars as pl
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

class FAISSVectorStoreNode:
    """Node 3A: Dense FAISS GPU Store & Parquet Persistence"""
    def __init__(self, model_name: str, device: str = "cuda"):
        self.embedder = SentenceTransformer(model_name, device=device)
        self.index = None
        self.chunks = []

    def build_index(self, chunks: List[Dict], parquet_path: str = "rag_chunks.parquet"):
        self.chunks = chunks
        texts = [c["texto"] for c in chunks]
        embeddings = self.embedder.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        # Parquet Export
        df_data = []
        for i, c in enumerate(chunks):
            df_data.append({
                "chunk_id": c["chunk_id"],
                "tipo": c.get("tipo", "Estructural"),
                "articulo": c["articulo"],
                "texto": c["texto"],
                "length": len(c["texto"]),
                "embedding": embeddings[i].tolist()
            })
        pl.DataFrame(df_data).write_parquet(parquet_path)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(q_vec, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
