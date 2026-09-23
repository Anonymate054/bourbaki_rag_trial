import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.rag_pipeline.memory import MultiChatMemoryManager

def test_lock_and_delete():
    print("=" * 60)
    print("PROBANDO FUNCIONALIDADES DE BLOQUEO Y BORRADO DE CHATS")
    print("=" * 60)
    
    memory_mgr = MultiChatMemoryManager()
    
    # 1. Crear sesión de prueba
    s_id = memory_mgr.create_session(title="Chat de Prueba Bloqueo")
    print(f"✅ Sesión creada: {s_id}")
    
    # 2. Agregar mensaje inicial
    memory_mgr.add_message(s_id, "user", "Hola, prueba de mensaje")
    s_data = memory_mgr.get_session(s_id)
    print(f"Mensajes antes de bloquear: {len(s_data['messages'])}")
    
    # 3. Bloquear sesión
    is_locked = memory_mgr.toggle_lock(s_id)
    print(f"🔒 Estado de bloqueo: {is_locked}")
    
    # 4. Intentar agregar mensaje a chat bloqueado (Debe lanzar ValueError)
    try:
        memory_mgr.add_message(s_id, "user", "Este mensaje no debe agregarse")
        print("❌ Error: Se permitió escribir en un chat bloqueado.")
    except ValueError as e:
        print(f"✅ Bloqueo Funcional: {e}")
        
    # 5. Desbloquear sesión
    is_locked_after = memory_mgr.toggle_lock(s_id)
    print(f"🔓 Estado de desbloqueo: {is_locked_after}")
    
    # 6. Borrar sesión
    memory_mgr.delete_session(s_id)
    s_deleted = memory_mgr.get_session(s_id)
    print(f"🗑️ Sesión borrada correctamente: {len(s_deleted['messages']) == 0}")
    
    print("=" * 60)
    print("PRUEBA DE BLOQUEO Y BORRADO COMPLETADA CON ÉXITO")
    print("=" * 60)

if __name__ == "__main__":
    test_lock_and_delete()
