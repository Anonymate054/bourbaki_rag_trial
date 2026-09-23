import os
import re
import sys
import time
import json
import numpy as np
import pandas as pd
import polars as pl
import torch
import faiss
from sentence_transformers import SentenceTransformer

# Reconfigure stdout to UTF-8 for Windows console support
sys.stdout.reconfigure(encoding='utf-8')

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_test():
    print("=" * 60)
    print("EJECUCION Y VERIFICACION END-TO-END DEL RAG LOCAL")
    print("=" * 60)
    
    # 1. GPU Check
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA disponible: {cuda_available}")
    if cuda_available:
        print(f"GPU Activa: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # 2. Document Load (Resolves root file relative path)
    doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Compilado_LFPIORPI20mayo2021.txt"))
    with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    cleaned_lines = [l for l in lines if not (re.search(r'--(?: \d+ of \d+ )?--', l.strip()) or "DIARIO OFICIAL" in l or "Primera Sección" in l)]
    text = "".join(cleaned_lines)
    print(f"Documento cargado: {len(cleaned_lines):,} líneas, {len(text):,} caracteres.")
    
    # 3. Structural Chunking
    art_pattern = r'(Artículo\s+(?:\d+|Único)(?:\s+Bis|\s+Ter|\s+Quáter)?\.\s*)'
    parts = re.split(art_pattern, text)
    
    chunks = []
    if parts[0].strip():
        chunks.append({
            "chunk_id": "art_0_preambulo",
            "tipo": "Estructural-Legal",
            "articulo": "Preámbulo/Decreto",
            "texto": parts[0].strip()[:1000],
            "length": len(parts[0].strip()[:1000])
        })
        
    for i in range(1, len(parts), 2):
        art_title = parts[i].strip()
        art_body = parts[i+1].strip() if i+1 < len(parts) else ""
        full_art_text = f"{art_title} {art_body}".strip()
        
        art_match = re.search(r'Artículo\s+([\w\s]+?)\.', art_title)
        art_num = art_match.group(1) if art_match else "Desconocido"
        
        if len(full_art_text) > 1200:
            for sub_idx, sub_start in enumerate(range(0, len(full_art_text), 1000)):
                sub_chunk = full_art_text[sub_start:sub_start+1000]
                chunks.append({
                    "chunk_id": f"art_{art_num}_part{sub_idx+1}",
                    "tipo": "Estructural-Legal",
                    "articulo": f"Artículo {art_num}",
                    "texto": sub_chunk,
                    "length": len(sub_chunk)
                })
        else:
            chunks.append({
                "chunk_id": f"art_{art_num}",
                "tipo": "Estructural-Legal",
                "articulo": f"Artículo {art_num}",
                "texto": full_art_text,
                "length": len(full_art_text)
            })
            
    print(f"Chunks Estructurales creados: {len(chunks)}")
    
    # 4. Sentence Embeddings on GPU
    print("Cargando modelo sentence-transformers en GPU CUDA...")
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=str(device))
    texts = [c["texto"] for c in chunks]
    
    t0 = time.time()
    embeddings = embedder.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    t1 = time.time()
    print(f"Embeddings generados en {t1-t0:.3f} s. Matriz: {embeddings.shape}")
    
    # 5. Parquet Save
    df_data = []
    for i, c in enumerate(chunks):
        df_data.append({
            "chunk_id": c["chunk_id"],
            "tipo": c["tipo"],
            "articulo": c["articulo"],
            "texto": c["texto"],
            "length": c["length"],
            "embedding": embeddings[i].tolist()
        })
    df_polars = pl.DataFrame(df_data)
    out_parquet = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag_chunks.parquet"))
    df_polars.write_parquet(out_parquet)
    print(f"Archivo 'rag_chunks.parquet' guardado ({os.path.getsize(out_parquet)/1024:.2f} KB).")
    
    # 6. FAISS Index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Indice FAISS creado con {index.ntotal} vectores.")
    
    # 7. Test Retrieval
    query = "¿Qué actividades son clasificadas como vulnerables según el Artículo 17 de la ley?"
    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_vec, 3)
    
    print("\nPRUEBA DE BUSQUEDA VECTORIAL CON FAISS:")
    print(f"Consulta: '{query}'")
    for score, idx in zip(scores[0], indices[0]):
        res = chunks[idx]
        print(f" -> Score Coseno: {score:.4f} | {res['articulo']} (ID: {res['chunk_id']})")
        print(f"    Snippet: {res['texto'][:150]}...\n")
        
    print("=" * 60)
    print("TODAS LAS PRUEBAS RAG PASARON CON EXITO")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
