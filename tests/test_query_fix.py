import re
import sys

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

def extract_explicit_article(query):
    # Detect "Artículo X" or "Art X"
    match = re.search(r'(?:artículo|art\.?)\s*(\d+|único)', query.lower())
    if match:
        target_num = match.group(1)
        for idx, c in enumerate(chunks):
            if f"Artículo {target_num}" in c["articulo"]:
                return idx
    return None

questions = [
    ("¿Cuál es el objeto de la Ley según el Artículo 2?", "Artículo 2"),
    ("¿Qué actividades se consideran vulnerables según el Artículo 17?", "Artículo 17"),
    ("¿Cuáles son las sanciones del Artículo 62?", "Artículo 62"),
    ("¿Por cuánto tiempo se deben conservar los documentos de clientes según el Artículo 18?", "Artículo 18")
]

print("=" * 60)
print("🎯 PRUEBA DE ENROUTADOR EXPLÍCITO DE ARTÍCULOS LEGALES")
print("=" * 60)

for q, expected in questions:
    idx = extract_explicit_article(q)
    retrieved = chunks[idx]["articulo"] if idx is not None else "No detectado"
    print(f"Pregunta: {q}")
    print(f"Esperado: {expected} | Detectado: {retrieved} | ¿Éxito?: {'✅ SÍ' if expected == retrieved else '❌ NO'}\n")
