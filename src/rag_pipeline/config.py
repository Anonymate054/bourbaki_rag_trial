import os
from dataclasses import dataclass, field

@dataclass
class PipelineConfig:
    # Document Path
    document_path: str = "Compilado_LFPIORPI20mayo2021.txt"
    
    # Embedding Config
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cuda"
    
    # Retrieval Config
    retrieval_mode: str = "hybrid"  # "dense", "sparse", "hybrid"
    top_k: int = 3
    rrf_k: int = 60
    
    # Re-ranking Config
    enable_rerank: bool = True
    rerank_candidate_count: int = 8
    
    # Preprocessing Config
    preprocessing_mode: str = "raw"  # "raw", "clean"
    
    # LLM Config
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"  # or llama3.2:3b, phi3.5:latest
    temperature: float = 0.2
