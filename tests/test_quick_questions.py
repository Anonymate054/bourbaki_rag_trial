import sys
import os
import re
import unicodedata
from rank_bm25 import BM25Okapi
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

with open("Compilado_LFPIORPI20mayo2021.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

art_pattern = r'(Artículo\s+(?:\d+|Único)(?:\s+Bis|\s+Ter|\s+Quáter)?\.\s*)'
parts = re.split(art_pattern, text)

chunks = []
for i in range(1, len(parts), 2):
    art_title = parts[i].strip()
    art_body = parts[i+1].strip() if i+1 < len(parts) else ""
    full_text = f"{art_title} {art_body}".strip()
    art_match = re.search(r'Artículo\s+([\w\s]+?)\.', art_title)
    art_num = art_match.group(1) if art_match else "Desconocido"
    chunks.append({"articulo": f"Artículo {art_num}", "texto": full_text[:1000]})

def tokenize(t):
    clean = re.sub(r'[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ]', ' ', t.lower())
    return [w for w in clean.split() if len(w) > 2]

corpus_tokens = [tokenize(c["texto"]) for c in chunks]
bm25 = BM25Okapi(corpus_tokens)

questions = [
    ("¿Cuál es el objeto de la Ley según el Artículo 2?", "Artículo 2"),
    ("¿Qué actividades se consideran vulnerables según el Artículo 17?", "Artículo 17"),
    ("¿Cuáles son las sanciones del Artículo 62?", "Artículo 62"),
    ("¿Por cuánto tiempo se deben conservar los documentos de clientes según el Artículo 18?", "Artículo 18")
]

print("=" * 60)
print("🔍 COMPROBACIÓN DE RECUPERACIÓN PARA PREGUNTAS SUGERIDAS")
print("=" * 60)

for q, expected_art in questions:
    tokens = tokenize(q)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:3]
    top_arts = [chunks[idx]["articulo"] for idx in top_indices]
    
    print(f"\nPregunta: {q}")
    print(f"Esperado: {expected_art}")
    print(f"Recuperado por BM25: {top_arts}")
    print(f"¿Éxito BM25?: {'✅ SÍ' if expected_art in top_arts else '❌ NO'}")

print("=" * 60)
