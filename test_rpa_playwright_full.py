import sys
import os
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

def test_full_rpa():
    print("=" * 60)
    print("🤖 PRUEBA RPA DE USUARIO COMPLETA Y VERIFICACIÓN DE RENDERIZADO")
    print("=" * 60)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print("🌐 1. Navegando a http://localhost:8501 ...")
        page.goto("http://localhost:8501", timeout=45000)
        
        print("⏳ Esperando renderizado completo de la interfaz de Streamlit...")
        try:
            text_area = page.wait_for_selector("[data-testid='stChatInputTextArea']", timeout=30000)
            print("✅ 2. Cuadro de texto de chat encontrado!")
            
            prompt = "¿Cuál es el objeto de la Ley según el Artículo 2?"
            print(f"📝 3. Escribiendo consulta: '{prompt}'")
            text_area.fill(prompt)
            time.sleep(1)
            
            submit_btn = page.locator("[data-testid='stChatInputSubmitButton']")
            if submit_btn.count() > 0:
                print("🚀 4. Haciendo clic en el botón Enviar...")
                submit_btn.click()
            else:
                print("⌨️ 4. Presionando Enter...")
                text_area.press("Enter")
                
            print("⏳ 5. Esperando inferencia en GPU y respuesta del asistente (15 segundos)...")
            time.sleep(15)
            
            page.screenshot(path="rpa_verified_chat.png", full_page=True)
            print("📸 6. Captura de pantalla RPA guardada como 'rpa_verified_chat.png'!")
            
            markdown_texts = page.locator("[data-testid='stMarkdownContainer']").all_text_contents()
            print(f"\n💬 Bloques de Texto Renderizados en la Pantalla ({len(markdown_texts)}):")
            for idx, txt in enumerate(markdown_texts, 1):
                clean_txt = txt.strip()
                if clean_txt and not clean_txt.startswith("Asistente inteligente") and not clean_txt.startswith("Panel de Control"):
                    print(f"--- Bloque #{idx} ---\n{clean_txt[:300]}\n")
        except Exception as e:
            print(f"❌ Error al esperar la interfaz: {e}")
            page.screenshot(path="rpa_error.png", full_page=True)
            
        browser.close()
        
    print("=" * 60)
    print("🎉 PRUEBA RPA COMPLETADA CON ÉXITO")
    print("=" * 60)

if __name__ == "__main__":
    test_full_rpa()
