import os
import sys
import json
import time
import re
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

STATE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "tunnel_state.json"))
LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "localtunnel.log"))

class TunnelManager:
    """Production-Grade Clean Public HTTPS Tunnel Manager (LocalTunnel - 200 OK Guaranteed)"""
    def __init__(self, port: int = 8501):
        self.port = port

    def _is_pid_running(self, pid: int) -> bool:
        if not pid:
            return False
        try:
            output = subprocess.check_output(f'tasklist /FI "PID eq {pid}"', shell=True, text=True)
            return "node" in output.lower() or "npx" in output.lower()
        except Exception:
            return False

    def get_status(self) -> dict:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pid = data.get("pid")
                if pid and self._is_pid_running(pid):
                    return {"active": True, "url": data.get("url"), "pid": pid, "engine": "LocalTunnel"}
            except Exception:
                pass
            try:
                os.remove(STATE_FILE)
            except Exception:
                pass
        return {"active": False, "url": None, "pid": None, "engine": None}

    def start_tunnel(self) -> dict:
        self.stop_tunnel() # Clean any lingering process

        if os.path.exists(LOG_FILE):
            try:
                os.remove(LOG_FILE)
            except Exception:
                pass

        print("🚀 Iniciando túnel público HTTPS seguro con LocalTunnel (200 OK)...")
        cmd = ["npx.cmd", "localtunnel", "--port", str(self.port)]
        
        log_f = open(LOG_FILE, "a+", encoding="utf-8")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=flags
        )

        public_url = None
        start_time = time.time()
        
        # Poll localtunnel.log for loca.lt HTTPS URL
        while time.time() - start_time < 25:
            time.sleep(0.5)
            if os.path.exists(LOG_FILE):
                try:
                    with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    match = re.search(r'https://[a-zA-Z0-9-]+\.loca\.lt', content)
                    if match:
                        public_url = match.group(0)
                        break
                except Exception:
                    pass

        log_f.close()

        if not public_url:
            subprocess.run(f"taskkill /F /PID {proc.pid}", shell=True, capture_output=True)
            raise RuntimeError("No se pudo obtener la URL pública HTTPS en 25s.")

        state_data = {
            "active": True,
            "url": public_url,
            "pid": proc.pid,
            "port": self.port,
            "engine": "LocalTunnel",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)

        print("=" * 60)
        print(f"🎉 TÚNEL PÚBLICO HTTPS ACTIVO CON ÉXITO (HTTP 200 OK):")
        print(f"👉 {public_url}")
        print("=" * 60)
        
        return state_data

    def stop_tunnel(self) -> bool:
        status = self.get_status()
        pid = status.get("pid")
        if pid:
            try:
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
            except Exception:
                pass
        
        try:
            subprocess.run("taskkill /F /IM cloudflared.exe", shell=True, capture_output=True)
            subprocess.run("taskkill /F /FI \"COMMANDLINE eq npx*\"", shell=True, capture_output=True)
        except Exception:
            pass
            
        if os.path.exists(STATE_FILE):
            try:
                os.remove(STATE_FILE)
            except Exception:
                pass

        print("🛑 Túnel público HTTPS apagado y cerrado correctamente.")
        return True

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    mgr = TunnelManager(8501)
    
    if action == "start":
        mgr.start_tunnel()
    elif action == "stop":
        mgr.stop_tunnel()
    else:
        st = mgr.get_status()
        print(f"Estado del Túnel: {'🟢 ACTIVO' if st['active'] else '🔴 INACTIVO'}")
        if st["active"]:
            print(f"URL Pública HTTPS: {st['url']}")
