import os
import sys
import shutil

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.rag_pipeline.pipeline import RAGPipelineRunner
from src.rag_pipeline.memory import MultiChatMemoryManager

def refresh_chats():
    print("=" * 60)
    print("🧹 LIMPIANDO CHATS ANTIGUOS Y SEMBRANDO CONVERSACIONES FRESCAS CON OLLAMA REAL")
    print("=" * 60)
    
    memory_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "chat_sessions"))
    if os.path.exists(memory_dir):
        shutil.rmtree(memory_dir)
    os.makedirs(memory_dir, exist_ok=True)
    
    memory_mgr = MultiChatMemoryManager()
    runner = RAGPipelineRunner()
    runner.initialize()
    
    # 1. Chat 1: Consulta del Artículo 2 (Objeto de la Ley)
    s1_id = memory_mgr.create_session(title="Objeto de la Ley (Art 2)")
    q1 = "¿Cuál es el objeto de la Ley según el Artículo 2?"
    print(f"📝 Generando Inferencia Real para: '{q1}'...")
    res1 = runner.query(q1, override_config={"retrieval_mode": "hybrid", "enable_rerank": True, "enable_router": True})
    memory_mgr.add_message(s1_id, "user", q1)
    memory_mgr.add_message(s1_id, "assistant", res1["response"], metadata=res1)
    
    # 2. Chat 2: Actividades Vulnerables (Artículo 17)
    s2_id = memory_mgr.create_session(title="Actividades Vulnerables (Art 17)")
    q2 = "¿Qué actividades se consideran vulnerables según el Artículo 17?"
    print(f"📝 Generando Inferencia Real para: '{q2}'...")
    res2 = runner.query(q2, override_config={"retrieval_mode": "hybrid", "enable_rerank": True, "enable_router": True})
    memory_mgr.add_message(s2_id, "user", q2)
    memory_mgr.add_message(s2_id, "assistant", res2["response"], metadata=res2)

    print("=" * 60)
    print("✅ CHATS FRESCOS GENERADOS CON ÉXITO Y CONECTADOS A OLLAMA REAL")
    print("=" * 60)

if __name__ == "__main__":
    refresh_chats()
