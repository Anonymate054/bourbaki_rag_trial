import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.rag_pipeline.pipeline import RAGPipelineRunner

def main():
    print("PROBANDO CONSULTAS DE PREGUNTAS SUGERIDAS...")
    runner = RAGPipelineRunner()
    runner.initialize()
    
    q = "¿Qué actividades se consideran vulnerables según el Artículo 17?"
    print(f"\nConsulta: {q}")
    res = runner.query(q)
    print("Artículos Recuperados:", [c["articulo"] for c in res["contextos"]])
    print("Respuesta Ollama:\n", res["response"])

if __name__ == "__main__":
    main()
