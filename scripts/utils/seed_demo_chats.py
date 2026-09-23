import os
import sys

# Reconfigure stdout to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.rag_pipeline.pipeline import RAGPipelineRunner, PipelineConfig
from src.rag_pipeline.memory import MultiChatMemoryManager

def seed_demo_chats():
    print("=" * 60)
    print("SEMBRANDO CONVERSACIONES DE DEMOSTRACION PARA LA UI WEB")
    print("=" * 60)
    
    memory_mgr = MultiChatMemoryManager()
    runner = RAGPipelineRunner()
    runner.initialize()
    
    # Session 1: Búsqueda Híbrida con Re-ranking
    s1_id = memory_mgr.create_session(title="Actividades Vulnerables (Art 17)")
    q1 = "¿Qué actividades se consideran vulnerables según el Artículo 17 de la LFPIORPI?"
    res1 = runner.query(q1, override_config={"retrieval_mode": "hybrid", "enable_rerank": True})
    memory_mgr.add_message(s1_id, "user", q1)
    memory_mgr.add_message(s1_id, "assistant", res1["response"], metadata=res1)
    
    # Session 2: Objeto de la Ley y Autoridades Competentes
    s2_id = memory_mgr.create_session(title="Objeto de la Ley (Art 2 y 5)")
    q2 = "¿Cuál es el objeto de la Ley y qué autoridades son competentes en el ámbito administrativo?"
    res2 = runner.query(q2, override_config={"retrieval_mode": "hybrid", "enable_rerank": True})
    memory_mgr.add_message(s2_id, "user", q2)
    memory_mgr.add_message(s2_id, "assistant", res2["response"], metadata=res2)
    
    print("Conversaciones sembradas exitosamente en chat_sessions/")

if __name__ == "__main__":
    seed_demo_chats()
