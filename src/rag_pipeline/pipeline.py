import time
import unicodedata
import re
import numpy as np
from typing import List, Dict, Any
from .config import PipelineConfig
from .nodes.loader import DocumentLoaderNode
from .nodes.chunker import StructuralChunkerNode
from .nodes.vector_store import FAISSVectorStoreNode
from .nodes.bm25_store import BM25StoreNode
from .nodes.fusion import ReciprocalRankFusionNode
from .nodes.reranker import CrossEncoderRerankerNode
from .nodes.llm import LocalLLMNode

class RAGPipelineRunner:
    """DAG Orchestrator for the RAG Playground System"""
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Instantiate Nodes
        self.loader = DocumentLoaderNode(self.config.document_path)
        self.chunker = StructuralChunkerNode()
        self.vector_store = FAISSVectorStoreNode(self.config.embedding_model_name, device=self.config.device)
        self.bm25_store = BM25StoreNode()
        self.fusion = ReciprocalRankFusionNode()
        self.reranker = CrossEncoderRerankerNode(self.vector_store.embedder)
        self.llm = LocalLLMNode(self.config.ollama_host, self.config.ollama_model)
        
        self.is_initialized = False

    def initialize(self):
        if self.is_initialized:
            return
        # DAG Execution Flow
        raw_text = self.loader.run()
        self.chunks = self.chunker.run(raw_text)
        self.vector_store.build_index(self.chunks)
        self.bm25_store.build_index(self.chunks)
        self.is_initialized = True

    def _apply_preprocessing(self, text: str, mode: str) -> str:
        if mode == "clean":
            text_low = text.lower()
            text_norm = unicodedata.normalize('NFD', text_low)
            clean_str = "".join(c for c in text_norm if unicodedata.category(c) != 'Mn')
            return re.sub(r'[^a-z0-9\s]', ' ', clean_str)
        return text

    def _extract_explicit_article_indices(self, query: str) -> List[int]:
        """Detect explicit article references in user query (e.g. 'Artículo 17', 'Art. 62')"""
        matches = re.findall(r'(?:artículo|art\.?)\s*(\d+|único)', query.lower())
        boosted_indices = []
        for target_num in matches:
            for idx, c in enumerate(self.chunks):
                if f"Artículo {target_num}" in c["articulo"]:
                    if idx not in boosted_indices:
                        boosted_indices.append(idx)
        return boosted_indices

    def query(self, user_prompt: str, override_config: Dict[str, Any] = None) -> Dict[str, Any]:
        self.initialize()
        
        cfg = self.config
        mode = override_config.get("retrieval_mode", cfg.retrieval_mode) if override_config else cfg.retrieval_mode
        enable_rerank = override_config.get("enable_rerank", cfg.enable_rerank) if override_config else cfg.enable_rerank
        prep_mode = override_config.get("preprocessing_mode", cfg.preprocessing_mode) if override_config else cfg.preprocessing_mode
        top_k = override_config.get("top_k", cfg.top_k) if override_config else cfg.top_k
        model_name = override_config.get("ollama_model", cfg.ollama_model) if override_config else cfg.ollama_model
        enable_router = override_config.get("enable_router", True) if override_config else True
        api_key = override_config.get("api_key", None) if override_config else None
        provider = override_config.get("provider", "ollama") if override_config else "ollama"
        api_base = override_config.get("api_base", None) if override_config else None
        
        t0 = time.time()
        processed_prompt = self._apply_preprocessing(user_prompt, prep_mode)
        
        # Step 1: Explicit Article Router & Retrieval Candidate Collection
        boosted_indices = self._extract_explicit_article_indices(user_prompt) if enable_router else []
        
        # Retrieve candidate pools with raw scores for transparency
        dense_candidates = self.vector_store.search(processed_prompt, top_k=10) # [(idx, cos_sim)]
        sparse_candidates = self.bm25_store.search(processed_prompt, top_k=10)  # [(idx, bm25_score)]
        
        dense_dict = {idx: score for idx, score in dense_candidates}
        sparse_dict = {idx: score for idx, score in sparse_candidates}
        
        # Dense vs Sparse vs Hybrid Candidate Selection
        if mode == "dense":
            candidate_indices = [idx for idx, _ in dense_candidates[:top_k]]
        elif mode == "sparse":
            candidate_indices = [idx for idx, _ in sparse_candidates[:top_k]]
        else: # Hybrid RRF
            candidate_indices = self.fusion.run(dense_candidates, sparse_candidates, k=cfg.rrf_k, top_n=10)
            
        # Priority Boosting for Explicitly Referenced Articles if router enabled
        router_applied = False
        if enable_router and boosted_indices:
            router_applied = True
            for b_idx in reversed(boosted_indices):
                if b_idx in candidate_indices:
                    candidate_indices.remove(b_idx)
                candidate_indices.insert(0, b_idx)
            
        # Step 2: Re-ranking with Cross-Encoder Fine Scoring
        rerank_scores_dict = {}
        if enable_rerank and candidate_indices:
            if router_applied and boosted_indices:
                non_boosted = [idx for idx in candidate_indices if idx not in boosted_indices]
                reranked_rest = self.reranker.rerank(user_prompt, non_boosted, self.chunks, top_n=max(1, top_k - len(boosted_indices)))
                final_indices = (boosted_indices + reranked_rest)[:top_k]
            else:
                final_indices = self.reranker.rerank(user_prompt, candidate_indices, self.chunks, top_n=top_k)
                
            # Compute fine cross-encoder scores for inspection UI
            q_emb = self.vector_store.embedder.encode([user_prompt], convert_to_numpy=True, normalize_embeddings=True)[0]
            for idx in candidate_indices:
                p_emb = self.vector_store.embedder.encode([self.chunks[idx]["texto"]], convert_to_numpy=True, normalize_embeddings=True)[0]
                rerank_scores_dict[idx] = float(np.dot(p_emb, q_emb))
        else:
            final_indices = candidate_indices[:top_k]
            
        retrieval_time = time.time() - t0
        
        # Build Detailed Diagnostic Objects for each selected chunk
        inspected_chunks = []
        for rank_pos, idx in enumerate(final_indices, 1):
            c_info = self.chunks[idx].copy()
            c_info["rank"] = rank_pos
            c_info["faiss_cosine_sim"] = round(float(dense_dict.get(idx, 0.0)), 4)
            c_info["bm25_score"] = round(float(sparse_dict.get(idx, 0.0)), 4)
            c_info["cross_encoder_score"] = round(float(rerank_scores_dict.get(idx, dense_dict.get(idx, 0.0))), 4) if enable_rerank else "N/A"
            c_info["is_boosted_by_router"] = (idx in boosted_indices) if enable_router else False
            inspected_chunks.append(c_info)
            
        # Step 3: Grounded Prompt Building
        ctx_str = ""
        for i, c in enumerate(inspected_chunks, 1):
            ctx_str += f"\n[FUENTE #{i} - {c['articulo']} (Similitud: {c['faiss_cosine_sim']}, BM25: {c['bm25_score']})]\n{c['texto']}\n"
            
        prompt = (
            "Eres un asistente legal experto en la Ley Federal para la Prevención e Identificación de Operaciones con Recursos de Procedencia Ilícita (LFPIORPI) de México.\n\n"
            "Instrucciones:\n"
            "1. Responde a la pregunta del usuario de forma completa, clara y detallada.\n"
            "2. BÁSATE ÚNICAMENTE en el contexto legal proporcionado a continuación. Cita explícitamente el o los artículos correspondientes.\n\n"
            f"CONTEXTO LEGAL RECUPERADO:\n{ctx_str}\n\n"
            f"PREGUNTA DEL USUARIO:\n{user_prompt}\n\n"
            "RESPUESTA FUNDAMENTADA:"
        )
        
        # Step 4: Generation (Supports Local Ollama GPU & Cloud API Providers)
        t1 = time.time()
        llm_response = self.llm.generate(
            prompt,
            model_name=model_name,
            temperature=cfg.temperature,
            api_key=api_key,
            api_base=api_base,
            provider=provider
        )
        generation_time = time.time() - t1
        
        return {
            "prompt": user_prompt,
            "response": llm_response,
            "contextos": inspected_chunks,
            "prompt_inyectado": prompt,
            "metrics": {
                "retrieval_time_ms": round(retrieval_time * 1000, 2),
                "generation_time_sec": round(generation_time, 2),
                "retrieval_mode": mode,
                "rerank_enabled": enable_rerank,
                "router_enabled": enable_router,
                "preprocessing": prep_mode,
                "model_used": model_name,
                "provider": provider,
                "total_chunks_indexed": len(self.chunks)
            }
        }
