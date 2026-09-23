import os
import json
import time
from typing import List, Dict

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chat_sessions")

class MultiChatMemoryManager:
    def __init__(self, memory_dir: str = MEMORY_DIR):
        self.memory_dir = os.path.abspath(memory_dir)
        os.makedirs(self.memory_dir, exist_ok=True)
        
    def list_sessions(self) -> List[Dict]:
        sessions = []
        for fname in os.listdir(self.memory_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.memory_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        sessions.append({
                            "session_id": data.get("session_id", fname.replace(".json", "")),
                            "title": data.get("title", "Conversación"),
                            "created_at": data.get("created_at", time.time()),
                            "is_locked": data.get("is_locked", False),
                            "message_count": len(data.get("messages", []))
                        })
                except Exception:
                    continue
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions

    def create_session(self, title: str = "Nueva Conversación") -> str:
        session_id = f"session_{int(time.time()*1000)}"
        data = {
            "session_id": session_id,
            "title": title,
            "created_at": time.time(),
            "is_locked": False,
            "messages": []
        }
        self.save_session(session_id, data)
        return session_id

    def get_session(self, session_id: str) -> Dict:
        fpath = os.path.join(self.memory_dir, f"{session_id}.json")
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"session_id": session_id, "title": "Conversación", "created_at": time.time(), "is_locked": False, "messages": []}

    def save_session(self, session_id: str, session_data: Dict):
        fpath = os.path.join(self.memory_dir, f"{session_id}.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

    def toggle_lock(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        session["is_locked"] = not session.get("is_locked", False)
        self.save_session(session_id, session)
        return session["is_locked"]

    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        session = self.get_session(session_id)
        if session.get("is_locked", False):
            raise ValueError("No se pueden añadir mensajes a una conversación bloqueada.")
            
        msg = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        session["messages"].append(msg)
        if len(session["messages"]) == 2 and session.get("title") == "Nueva Conversación":
            first_prompt = content[:30].strip() + "..."
            session["title"] = first_prompt
        self.save_session(session_id, session)

    def delete_session(self, session_id: str):
        fpath = os.path.join(self.memory_dir, f"{session_id}.json")
        if os.path.exists(fpath):
            os.remove(fpath)
