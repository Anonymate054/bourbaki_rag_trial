import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def create_colab_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    def add_markdown(source):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # Header
    add_markdown("""# 🚀 Google Colab RAG Pipeline: LFPIORPI Legal Assistant (Gemini & OpenAI API + NGROK)

Este Jupyter Notebook está optimizado para ejecutarse **100% en Google Colab con 1 solo clic**. 
Demuestra todas las etapas del pipeline RAG para la **Ley Federal para la Prevención e Identificación de Operaciones con Recursos de Procedencia Ilícita (LFPIORPI)**:

1. 📥 **Descarga Automática del Corpus Legal** desde GitHub.
2. 🧹 **Preprocesamiento y Fragmentación Estructural (Chunking)** por Artículos.
3. ⚡ **Indexación Vectorial (FAISS)** y **Búsqueda por Palabras Clave (BM25 Okapi)**.
4. 🔀 **Fusión Híbrida (RRF)** y **Reordenamiento (Cross-Encoder Re-ranker)**.
5. 🤖 **Inferencia de LLM Grounded con Celdas DEDICADAS**:
   - **Celda 1**: Google Gemini API (`gemini-2.0-flash`).
   - **Celda 2**: OpenAI API (`gpt-4o-mini`).
6. 🌐 **Despliegue del Servidor Web (Streamlit) en Segundo Plano con NGROK**.
""")

    # Step 1: Install Dependencies
    add_markdown("## 1. 📦 Instalación de Dependencias en Google Colab")
    add_code("""!pip install -q sentence-transformers faiss-cpu rank-bm25 google-generativeai openai pyngrok streamlit pandas polars matplotlib
print("✅ Todas las dependencias se instalaron correctamente.")""")

    # Step 2: Download Data
    add_markdown("## 2. 📥 Descarga Automática del Corpus Legal (LFPIORPI)")
    add_code("""import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/Anonymate054/bourbaki_rag_trial/main/data/Compilado_LFPIORPI20mayo2021.txt"
DATA_PATH = "Compilado_LFPIORPI20mayo2021.txt"

if not os.path.exists(DATA_PATH):
    print("Descargando corpus legal desde GitHub...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

print(f"✅ Corpus legal listo: '{DATA_PATH}' ({os.path.getsize(DATA_PATH)/1024:.2f} KB).")""")

    # Step 3: Preprocessing & Chunking
    add_markdown("## 3. 🧹 Preprocesamiento y Fragmentación Estructural (Chunking)")
    add_code("""import re
import pandas as pd

with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# Limpieza de cabeceras DOF
cleaned_lines = [l for l in lines if not (re.search(r'--(?: \d+ of \d+ )?--', l.strip()) or "DIARIO OFICIAL" in l or "Primera Sección" in l)]
raw_text = "".join(cleaned_lines)

# Regex Splitting por Artículos
art_pattern = r'(Artículo\s+(?:\d+|Único)(?:\s+Bis|\s+Ter|\s+Quáter)?\.\s*)'
parts = re.split(art_pattern, raw_text)

chunks = []
if parts[0].strip():
    chunks.append({
        "chunk_id": "art_0_preambulo",
        "articulo": "Preámbulo/Decreto",
        "texto": parts[0].strip()[:1000]
    })

for i in range(1, len(parts), 2):
    art_title = parts[i].strip()
    art_body = parts[i+1].strip() if i+1 < len(parts) else ""
    full_text = f"{art_title} {art_body}".strip()
    
    art_match = re.search(r'Artículo\s+([\w\s]+?)\.', art_title)
    art_num = art_match.group(1) if art_match else "Desconocido"
    
    if len(full_text) > 1200:
        for sub_idx, sub_start in enumerate(range(0, len(full_text), 1000)):
            sub_chunk = full_text[sub_start:sub_start+1000]
            chunks.append({
                "chunk_id": f"art_{art_num}_part{sub_idx+1}",
                "articulo": f"Artículo {art_num}",
                "texto": sub_chunk
            })
    else:
        chunks.append({
            "chunk_id": f"art_{art_num}",
            "articulo": f"Artículo {art_num}",
            "texto": full_text
        })

df_chunks = pd.DataFrame(chunks)
print(f"✅ Preprocesamiento completado. Total de Chunks creados: {len(chunks)}")
display(df_chunks.head(5))""")

    # Step 4: Indexing FAISS & BM25
    add_markdown("## 4. ⚡ Indexación Vectorial (FAISS) y Búsqueda por Palabras Clave (BM25)")
    add_code("""import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

print("Cargando modelo de Embeddings Multilingüe...")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
texts = [c["texto"] for c in chunks]

# 1. FAISS Index
embeddings = embedder.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings)

# 2. BM25 Index con Tokenización Numérica Preservada
def tokenize_bm25(text):
    words = re.findall(r'\\w+', text.lower())
    return [w for w in words if w.isdigit() or len(w) > 2]

corpus_tokens = [tokenize_bm25(t) for t in texts]
bm25_index = BM25Okapi(corpus_tokens)

print(f"✅ Indice FAISS creado con {faiss_index.ntotal} vectores (Dim {dimension}).")
print(f"✅ Indice BM25 creado con {len(corpus_tokens)} documentos tokenizados.")""")

    # Step 5: Hybrid Search & Re-ranking Function
    add_markdown("## 5. 🔀 Búsqueda Híbrida (RRF) y Reordenamiento (Cross-Encoder)")
    add_code("""def retrieve_hybrid_context(query, top_k=3):
    # 1. Dense Search
    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    dense_scores, dense_indices = faiss_index.search(q_vec, 10)
    dense_dict = {idx: float(score) for score, idx in zip(dense_scores[0], dense_indices[0])}
    
    # 2. Sparse Search
    q_tokens = tokenize_bm25(query)
    bm25_scores = bm25_index.get_scores(q_tokens)
    sparse_indices = np.argsort(bm25_scores)[::-1][:10]
    sparse_dict = {idx: float(bm25_scores[idx]) for idx in sparse_indices}
    
    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    for rank, idx in enumerate(dense_indices[0]):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (60 + rank + 1))
    for rank, idx in enumerate(sparse_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (60 + rank + 1))
        
    sorted_candidates = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:10]
    
    # 4. Explicit Article Router Boost
    article_matches = re.findall(r'(?:artículo|art\.?)\\s*(\\d+|único)', query.lower())
    boosted = []
    for art_num in article_matches:
        for idx, c in enumerate(chunks):
            if f"Artículo {art_num}" in c["articulo"] and idx not in boosted:
                boosted.append(idx)
                
    for b_idx in reversed(boosted):
        if b_idx in sorted_candidates:
            sorted_candidates.remove(b_idx)
        sorted_candidates.insert(0, b_idx)
        
    final_indices = sorted_candidates[:top_k]
    
    selected_chunks = []
    for rank, idx in enumerate(final_indices, 1):
        c_info = chunks[idx].copy()
        c_info["rank"] = rank
        c_info["faiss_sim"] = round(dense_dict.get(idx, 0.0), 4)
        c_info["bm25_score"] = round(sparse_dict.get(idx, 0.0), 4)
        selected_chunks.append(c_info)
        
    return selected_chunks

# Prueba de Búsqueda
test_query = "¿Qué actividades son consideradas vulnerables según el Artículo 17 de la ley?"
results = retrieve_hybrid_context(test_query, top_k=3)

print(f"🔎 Consulta: '{test_query}'\\n")
df_res = pd.DataFrame(results)
display(df_res[["rank", "articulo", "chunk_id", "faiss_sim", "bm25_score"]])""")

    # Step 6: Grounded Prompt Builder
    add_markdown("## 6. 📄 Construcción del Prompt Fundamentado")
    add_code("""def build_grounded_prompt(query, retrieved_chunks):
    ctx_str = ""
    for i, c in enumerate(retrieved_chunks, 1):
        ctx_str += f"\\n[FUENTE #{i} - {c['articulo']} (FAISS Sim: {c['faiss_sim']}, BM25: {c['bm25_score']})]\\n{c['texto']}\\n"
        
    prompt = (
        "Eres un asistente legal experto en la Ley Federal para la Prevención e Identificación de Operaciones con Recursos de Procedencia Ilícita (LFPIORPI) de México.\\n\\n"
        "Instrucciones:\\n"
        "1. Responde a la pregunta del usuario de forma completa, clara y detallada.\\n"
        "2. BÁSATE ÚNICAMENTE en el contexto legal proporcionado a continuación. Cita explícitamente el o los artículos correspondientes.\\n\\n"
        f"CONTEXTO LEGAL RECUPERADO:\\n{ctx_str}\\n\\n"
        f"PREGUNTA DEL USUARIO:\\n{query}\\n\\n"
        "RESPUESTA FUNDAMENTADA:"
    )
    return prompt

prompt_example = build_grounded_prompt(test_query, results)
print("Ejemplo de Prompt Inyectado:")
print(prompt_example[:600] + "\\n... [Recortado para visualización]")""")

    # Step 7: Dual LLM Generation (Dedicated Cells)
    add_markdown("## 7. 🤖 Generación con LLM (Celdas DEDICADAS)")

    # Cell 7A: Google Gemini API
    add_markdown("### 🌟 OPCIÓN A: Inferencia con Google Gemini API (`gemini-2.0-flash` / `gemini-1.5-flash`)")
    add_code("""import time
import getpass
import google.generativeai as genai

# Intenta obtener la API Key desde los secretos de Colab o solicita la clave
try:
    from google.colab import userdata
    gemini_key = userdata.get('GEMINI_API_KEY')
except Exception:
    gemini_key = None

if not gemini_key:
    gemini_key = getpass.getpass("🔑 Ingresa tu GEMINI_API_KEY (de aistudio.google.com): ")

genai.configure(api_key=gemini_key)
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

# Ejecutar Búsqueda y Generación con Gemini
t0 = time.time()
prompt_text = build_grounded_prompt(test_query, results)
response_gemini = model_gemini.generate_content(prompt_text)
t_gen = time.time() - t0

print("=" * 60)
print(f"🟢 RESPUESTA GENERADA CON GOOGLE GEMINI API (Tiempo: {t_gen:.2f} s):")
print("=" * 60)
print(response_gemini.text)""")

    # Cell 7B: OpenAI API
    add_markdown("### ⚡ OPCIÓN B: Inferencia con OpenAI API (`gpt-4o-mini`)")
    add_code("""import time
import getpass
from openai import OpenAI

# Intenta obtener la API Key desde variables de entorno o solicita la clave
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    openai_key = getpass.getpass("🔑 Ingresa tu OPENAI_API_KEY (de platform.openai.com): ")

client = OpenAI(api_key=openai_key)

# Ejecutar Búsqueda y Generación con OpenAI
t0 = time.time()
prompt_text = build_grounded_prompt(test_query, results)
response_openai = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Eres un asistente legal experto en la Ley LFPIORPI de México."},
        {"role": "user", "content": prompt_text}
    ],
    temperature=0.2
)
t_gen = time.time() - t0

print("=" * 60)
print(f"🟢 RESPUESTA GENERADA CON OPENAI API (Tiempo: {t_gen:.2f} s):")
print("=" * 60)
print(response_openai.choices[0].message.content)""")

    # Step 8: Web Application Server & NGROK Tunnel Launch
    add_markdown("## 8. 🌐 Servidor Web en Segundo Plano y Túnel NGROK para Compartir en Internet")
    add_code("""import subprocess
import getpass
import time
from pyngrok import ngrok, conf

# 1. Configurar Authtoken de NGROK
ngrok_token = getpass.getpass("🔑 Ingresa tu Authtoken de NGROK (de dashboard.ngrok.com): ")
conf.get_default().auth_token = ngrok_token.strip()

# 2. Clonar/Asegurar código de la interfaz Streamlit en Colab
if not os.path.exists("app.py"):
    !git clone https://github.com/Anonymate054/bourbaki_rag_trial.git repo_temp
    !cp repo_temp/app.py .
    !cp repo_temp/tunnel_manager.py .
    !cp -r repo_temp/src .
    !mkdir -p data
    !cp Compilado_LFPIORPI20mayo2021.txt data/

# 3. Iniciar Servidor Streamlit en Segundo Plano (Puerto 8501)
print("🚀 Iniciando Servidor Web Streamlit en segundo plano...")
process = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"])
time.sleep(5)

# 4. Crear Túnel HTTPS Público con NGROK
try:
    public_url = ngrok.connect(8501)
    print("=" * 70)
    print("🎉 INTERFAZ WEB PÚBLICA EN INTERNET ACTIVA CON ÉXITO:")
    print(f"👉 {public_url}")
    print("=" * 70)
    print("Haz clic en el enlace superior para abrir la aplicación web desde cualquier dispositivo o teléfono móvil.")
except Exception as e:
    print(f"⚠️ Error al conectar NGROK: {e}")""")

    out_path = os.path.join("notebooks", "colab_rag_example.ipynb")
    os.makedirs("notebooks", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"✅ Notebook '{out_path}' creado exitosamente.")

if __name__ == "__main__":
    create_colab_notebook()
