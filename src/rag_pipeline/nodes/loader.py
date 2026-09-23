import os
import re

class DocumentLoaderNode:
    """Node 1: Document Loader & Cleaner"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def run(self) -> str:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Archivo no encontrado en {self.file_path}")
            
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        cleaned_lines = [
            l for l in lines 
            if not (re.search(r'--(?: \d+ of \d+ )?--', l.strip()) or "DIARIO OFICIAL" in l or "Primera Sección" in l)
        ]
        return "".join(cleaned_lines)
