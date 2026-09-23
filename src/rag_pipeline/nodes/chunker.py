import re
from typing import List, Dict

class StructuralChunkerNode:
    """Node 2: Structural Legal Chunker with Metadata Richness"""
    def run(self, text: str) -> List[Dict]:
        art_pattern = r'(Artículo\s+(?:\d+|Único)(?:\s+Bis|\s+Ter|\s+Quáter)?\.\s*)'
        parts = re.split(art_pattern, text)
        
        chunks = []
        if parts[0].strip():
            raw_t = parts[0].strip()[:1000]
            chunks.append({
                "chunk_id": "art_0_preambulo",
                "tipo": "Estructural-Legal",
                "articulo": "Preámbulo/Decreto",
                "longitud_caracteres": len(raw_t),
                "total_palabras": len(raw_t.split()),
                "texto": raw_t
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
                        "longitud_caracteres": len(sub_chunk),
                        "total_palabras": len(sub_chunk.split()),
                        "texto": sub_chunk
                    })
            else:
                chunks.append({
                    "chunk_id": f"art_{art_num}",
                    "tipo": "Estructural-Legal",
                    "articulo": f"Artículo {art_num}",
                    "longitud_caracteres": len(full_art_text),
                    "total_palabras": len(full_art_text.split()),
                    "texto": full_art_text
                })
        return chunks
