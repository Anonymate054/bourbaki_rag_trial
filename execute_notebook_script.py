import json
import os
import sys
import io
import contextlib
import traceback
import matplotlib.pyplot as plt

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

def execute_notebook():
    nb_path = "reto3_rag_local.ipynb"
    if not os.path.exists(nb_path):
        print(f"Error: {nb_path} no existe.")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Global execution namespace
    global_env = {"__name__": "__main__"}

    print("🚀 Ejecutando celdas del notebook...")
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source_code = "".join(cell["source"])
            print(f"\n--- Executing Code Cell #{idx+1} ---")
            
            output_capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(output_capture):
                    exec(source_code, global_env)
                cell_output_text = output_capture.getvalue()
                print(f"Output preview ({len(cell_output_text)} chars):\n{cell_output_text[:300]}...")
                
                # Format output into notebook format
                cell["outputs"] = [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [line + "\n" for line in cell_output_text.splitlines()]
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

    out_nb_path = "reto3_rag_local.ipynb"
    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Cuaderno '{out_nb_path}' ejecutado y actualizado exitosamente con todas sus salidas!")

if __name__ == "__main__":
    execute_notebook()
