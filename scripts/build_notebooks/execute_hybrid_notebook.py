import json
import os
import sys
import io
import contextlib
import traceback

sys.stdout.reconfigure(encoding='utf-8')

def execute_notebook():
    nb_path = "reto3_busqueda_hibrida_rerank.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    global_env = {"__name__": "__main__"}
    print("🚀 Ejecutando celdas del notebook de búsqueda híbrida y re-ranking...")

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source_code = "".join(cell["source"])
            print(f"\n--- Ejecutando Celda #{idx+1} ---")
            
            output_capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(output_capture):
                    exec(source_code, global_env)
                cell_output = output_capture.getvalue()
                print(f"Vista previa de salida:\n{cell_output[:350]}...")
                cell["outputs"] = [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [line + "\n" for line in cell_output.splitlines()]
                    }
                ]
                cell["execution_count"] = idx + 1
            except Exception as e:
                err_msg = traceback.format_exc()
                print(f"❌ Error en celda #{idx+1}: {e}")
                cell["outputs"] = [
                    {
                        "name": "stderr",
                        "output_type": "stream",
                        "text": [line + "\n" for line in err_msg.splitlines()]
                    }
                ]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Cuaderno '{nb_path}' ejecutado y actualizado exitosamente con todas las salidas!")

if __name__ == "__main__":
    execute_notebook()
