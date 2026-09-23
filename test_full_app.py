import sys
import os

# Reconfigure stdout to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.rag_pipeline.pipeline import RAGPipelineRunner, PipelineConfig

def test_pipeline():
    print("=" * 60)
    print("PROBANDO PIPELINE DAG MODULAR RAG Y MEMORIA")
    print("=" * 60)
    
    config = PipelineConfig()
    runner = RAGPipelineRunner(config)
    runner.initialize()
    print("Pipeline DAG inicializado exitosamente.")
    
    query = "¿Qué actividades se consideran vulnerables según el Artículo 17?"
    
    # Test 1: Híbrido con Re-ranking ON
    res1 = runner.query(query, override_config={"retrieval_mode": "hybrid", "enable_rerank": True})
    print("\n--- PRUEBA 1: Modo Hibrido + Re-ranking ON ---")
    print(f"Latencia Busqueda: {res1['metrics']['retrieval_time_ms']} ms")
    print(f"Respuesta:\n{res1['response'][:200]}...")
    print(f"Top Fuente: {res1['contextos'][0]['articulo']}")
    
    # Test 2: Densa FAISS con Re-ranking OFF
    res2 = runner.query(query, override_config={"retrieval_mode": "dense", "enable_rerank": False})
    print("\n--- PRUEBA 2: Modo Denso FAISS + Re-ranking OFF ---")
    print(f"Latencia Busqueda: {res2['metrics']['retrieval_time_ms']} ms")
    print(f"Top Fuente: {res2['contextos'][0]['articulo']}")
    
    print("=" * 60)
    print("PIPELINE COMPLETO VERIFICADO CON EXITO")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
