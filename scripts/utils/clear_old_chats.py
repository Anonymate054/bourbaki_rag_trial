import os
import sys
import shutil

sys.stdout.reconfigure(encoding='utf-8')

memory_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "chat_sessions"))
if os.path.exists(memory_dir):
    shutil.rmtree(memory_dir)
os.makedirs(memory_dir, exist_ok=True)
print("Historiales antiguos eliminados exitosamente. Carpeta chat_sessions/ limpia.")
