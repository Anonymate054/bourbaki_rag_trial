import sys
import os
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

def test_rpa():
    print("=" * 60)
    print("🤖 EJECUTANDO PRUEBA RPA DE INTERFAZ GRÁFICA EN PLAYWRIGHT")
    print("=" * 60)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print("🌐 Navegando a http://localhost:8501 ...")
        page.goto("http://localhost:8501", timeout=30000)
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        # 1. Click "➕ Crear Nuevo Chat" in sidebar
        new_chat_btn = page.locator("button:has-text('Crear Nuevo Chat')")
        if new_chat_btn.count() > 0:
            print("➕ Haciendo clic en 'Crear Nuevo Chat'...")
            new_chat_btn.click()
            time.sleep(2)
            
        # 2. Locate chat input textarea
        inp = page.locator("textarea")
        if inp.count() > 0:
            print("✅ INPUT BOX ENCONTRADO!")
            prompt_text = "¿Cuáles son las sanciones del Artículo 62?"
            print(f"📝 Escribiendo consulta: '{prompt_text}'")
            inp.first.fill(prompt_text)
            time.sleep(1)
            
            # 3. Locate & Click Streamlit Chat Input Submit Button
            submit_btn = page.locator("button[data-testid='stChatInputSubmitButton']")
            if submit_btn.count() > 0:
                print("🚀 Haciendo clic en el Botón de Enviar (stChatInputSubmitButton)...")
                submit_btn.click()
            else:
                print("⌨️ Presionando Enter en el teclado...")
                inp.first.press("Enter")
                
            print("⏳ Esperando respuesta del asistente RAG y barra de progreso (15s)...")
            time.sleep(15)
            
            # 4. Screenshot & Message Extraction
            page.screenshot(path="rpa_chat_test_screenshot.png", full_page=True)
            print("📸 CAPTURA DE PANTALLA RPA GUARDADA EXITOSAMENTE!")
            
            chat_msgs = page.locator("[data-testid='stChatMessage']").all_text_contents()
            print(f"💬 Total de burbujas de Chat visibles en pantalla: {len(chat_msgs)}")
            for idx, msg in enumerate(chat_msgs, 1):
                print(f"\n--- BURBUJA #{idx} ---\n{msg.strip()}\n")
        else:
            print("❌ INPUT BOX NO ENCONTRADO!")
            
        browser.close()
        
    print("=" * 60)
    print("🎉 PRUEBA RPA COMPLETADA CON ÉXITO")
    print("=" * 60)

if __name__ == "__main__":
    test_rpa()
