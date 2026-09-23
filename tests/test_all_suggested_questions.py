import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.rag_pipeline.pipeline import RAGPipelineRunner

def test_questions():
    print("=" * 60)
    print("🔍 DIAGNÓSTICO DE PREGUNTAS SUGERIDAS")
    print("=" * 60)
    
    runner = RAGPipelineRunner()
    runner.initialize()
    
    questions = [
        "¿Cuál es el objeto de la Ley según el Artículo 2?",
        "¿Qué actividades se consideran vulnerables según el Artículo 17?",
        "¿Cuáles son las sanciones del Artículo 62?",
        "¿Por cuánto tiempo se deben conservar los documentos de clientes según el Artículo 18?"
    ]
    
    for idx, q in enumerate(questions, 1):
        print(f"\n--- PREGUNTA #{idx}: {q} ---")
        res = runner.query(q, override_config={"retrieval_mode": "hybrid", "enable_rerank": True, "top_k": 3})
        print(f"📌 Chunks Recuperados:")
        for c in res["contextos"]:
            print(f"  - {c['articulo']} (ID: {c['chunk_id']})")
        print(f"🤖 Respuesta Generada por Ollama:\n{res['response']}\n")
        
if __name__ == "__main__":
    test_questions()
